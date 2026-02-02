# benchmark/data_generator_real.py
"""
Real KV Cache Data Generator using MLPerf datasets + LLaMA model.

Generates actual KV cache from:
1. LLaMA-2-70B or LLaMA-3-70B model
2. MLPerf Inference datasets (OpenORCA, CNN/DailyMail, SCROLLS)
3. ShareGPT conversations (shared prefix scenario)

Usage:
    srun -N1 --gpus=4 python -m benchmark.data_generator_real --dataset openorca --num_samples 10000
"""
import os
import sys
import json
import hashlib
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Generator
from dataclasses import dataclass, asdict
import time
import torch

# ============================================================================
# MLPerf Dataset Loaders
# ============================================================================

@dataclass
class BenchmarkSample:
    """A single benchmark sample with prompt and expected KV blocks"""
    sample_id: str
    dataset: str
    prompt: str
    prompt_tokens: List[int]
    num_tokens: int
    is_shared_prefix: bool
    prefix_group: Optional[str] = None  # For shared prefix grouping


class MLPerfDatasetLoader:
    """
    Load MLPerf Inference datasets for LLM benchmarking.
    
    Supported datasets:
    - OpenORCA: Instruction-following (MLPerf v3.1+)
    - CNN/DailyMail: Summarization
    - SCROLLS: Long context QA
    - ShareGPT: Multi-turn conversations (shared system prompts)
    """
    
    DATASET_CONFIGS = {
        "openorca": {
            "hf_path": "Open-Orca/OpenOrca",
            "split": "train",
            "prompt_field": "question",
            "max_samples": 50000,
        },
        "cnn_dailymail": {
            "hf_path": "cnn_dailymail",
            "hf_config": "3.0.0",
            "split": "validation",
            "prompt_field": "article",
            "max_samples": 10000,
        },
        "scrolls": {
            "hf_path": "tau/scrolls",
            "hf_config": "qasper",
            "split": "validation",
            "prompt_field": "input",
            "max_samples": 5000,
        },
        "sharegpt": {
            "hf_path": "anon8231489123/ShareGPT_Vicuna_unfiltered",
            "split": "train",
            "prompt_field": "conversations",
            "max_samples": 20000,
        },
    }
    
    # System prompts for shared prefix testing (realistic production prompts)
    SYSTEM_PROMPTS = [
        # Claude-style
        "You are Claude, an AI assistant created by Anthropic to be helpful, harmless, and honest. You should give concise responses to very simple questions, but provide thorough responses to more complex and open-ended questions.",
        
        # GPT-style
        "You are a helpful assistant. You answer questions accurately and concisely. If you don't know the answer, you say so.",
        
        # Code assistant
        "You are an expert programming assistant. You help users write, debug, and optimize code. Always explain your reasoning and provide working examples.",
        
        # RAG-style with context
        "You are a knowledgeable assistant with access to the following context. Use this context to answer the user's question accurately. If the answer is not in the context, say so.\\n\\nContext:\\n{context}",
        
        # Long document QA
        "You are analyzing the following document. Answer questions based solely on the information provided in the document. Be precise and cite relevant sections when possible.\\n\\nDocument:\\n{document}",
    ]
    
    def __init__(self, cache_dir: str = "/pscratch/sd/s/sgkim/Skim-cascade/benchmark/datasets"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def load_dataset(self, name: str, num_samples: int = 1000) -> List[BenchmarkSample]:
        """Load a dataset and convert to benchmark samples."""
        if name not in self.DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset: {name}. Available: {list(self.DATASET_CONFIGS.keys())}")
        
        config = self.DATASET_CONFIGS[name]
        
        try:
            from datasets import load_dataset
            
            print(f"Loading {name} from HuggingFace...")
            
            if "hf_config" in config:
                ds = load_dataset(
                    config["hf_path"],
                    config["hf_config"],
                    split=config["split"],
                    cache_dir=str(self.cache_dir)
                )
            else:
                ds = load_dataset(
                    config["hf_path"],
                    split=config["split"],
                    cache_dir=str(self.cache_dir)
                )
            
            samples = []
            num_to_load = min(num_samples, config["max_samples"], len(ds))
            
            for i in range(num_to_load):
                item = ds[i]
                
                if name == "sharegpt":
                    # ShareGPT has conversation format
                    prompt = self._extract_sharegpt_prompt(item)
                else:
                    prompt = item[config["prompt_field"]]
                
                if not prompt or len(prompt) < 10:
                    continue
                    
                samples.append(BenchmarkSample(
                    sample_id=f"{name}_{i:06d}",
                    dataset=name,
                    prompt=prompt,
                    prompt_tokens=[],
                    num_tokens=0,
                    is_shared_prefix=False
                ))
            
            return samples
            
        except Exception as e:
            print(f"Error loading dataset {name}: {e}")
            print("Using cached samples if available...")
            return self._load_cached(name, num_samples)
    
    def _extract_sharegpt_prompt(self, item: Dict) -> str:
        """Extract prompt from ShareGPT conversation format."""
        conversations = item.get("conversations", [])
        if not conversations:
            return ""
        
        # Get first human turn
        for turn in conversations:
            if turn.get("from") == "human":
                return turn.get("value", "")
        return ""
    
    def _load_cached(self, name: str, num_samples: int) -> List[BenchmarkSample]:
        """Load from local cache if HuggingFace not available."""
        cache_file = self.cache_dir / f"{name}_samples.json"
        if cache_file.exists():
            with open(cache_file) as f:
                data = json.load(f)
            return [BenchmarkSample(**s) for s in data[:num_samples]]
        return []
    
    def create_shared_prefix_workload(
        self,
        base_samples: List[BenchmarkSample],
        num_prefix_groups: int = 10,
        sessions_per_prefix: int = 100
    ) -> List[BenchmarkSample]:
        """
        Create workload with shared system prompts.
        
        This simulates production scenario where many users share
        the same system prompt (e.g., ChatGPT, Claude).
        """
        samples = []
        
        for group_id in range(num_prefix_groups):
            system_prompt = self.SYSTEM_PROMPTS[group_id % len(self.SYSTEM_PROMPTS)]
            
            for session_id in range(sessions_per_prefix):
                # Pick a user query from base samples
                base_idx = (group_id * sessions_per_prefix + session_id) % len(base_samples)
                base = base_samples[base_idx]
                
                # Combine system prompt + user query
                full_prompt = f"{system_prompt}\\n\\nUser: {base.prompt}\\n\\nAssistant:"
                
                samples.append(BenchmarkSample(
                    sample_id=f"shared_g{group_id:02d}_s{session_id:04d}",
                    dataset=f"shared_prefix_{base.dataset}",
                    prompt=full_prompt,
                    prompt_tokens=[],
                    num_tokens=0,
                    is_shared_prefix=True,
                    prefix_group=f"prefix_group_{group_id:02d}",
                ))
        
        return samples


# ============================================================================
# Real KV Cache Extractor (using vLLM or HuggingFace)
# ============================================================================

class RealKVCacheExtractor:
    """
    Extract actual KV cache from LLaMA model inference.
    
    Uses vLLM for efficient batched inference on Perlmutter A100s.
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-70b-hf",
        tensor_parallel: int = 4,  # 4 A100s per node
        max_model_len: int = 4096,
        block_size: int = 256,     # Tokens per KV block (matches LMCache)
    ):
        self.model_name = model_name
        self.tensor_parallel = tensor_parallel
        self.max_model_len = max_model_len
        self.block_size = block_size
        
        self.model = None
        self.tokenizer = None
        self._initialized = False
    
    def initialize(self) -> bool:
        """Initialize model for KV extraction."""
        try:
            # Try vLLM first (faster)
            return self._init_vllm()
        except ImportError:
            print("vLLM not available, trying HuggingFace...")
            return self._init_hf()
    
    def _init_vllm(self) -> bool:
        """Initialize using vLLM."""
        from vllm import LLM, SamplingParams
        
        print(f"Loading {self.model_name} with vLLM (TP={self.tensor_parallel})...")
        
        self.model = LLM(
            model=self.model_name,
            tensor_parallel_size=self.tensor_parallel,
            max_model_len=self.max_model_len,
            trust_remote_code=True,
            # Enable KV cache extraction
            enforce_eager=True,  # Disable CUDA graphs for KV access
        )
        
        self.tokenizer = self.model.get_tokenizer()
        self._use_vllm = True
        self._initialized = True
        
        print(f"Model loaded successfully")
        return True
    
    def _init_hf(self) -> bool:
        """Initialize using HuggingFace transformers."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        print(f"Loading {self.model_name} with HuggingFace...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        self._use_vllm = False
        self._initialized = True
        
        print(f"Model loaded successfully")
        return True
    
    def extract_kv_cache(
        self,
        samples: List[BenchmarkSample],
        output_dir: Path,
        batch_size: int = 8,
    ) -> Dict[str, any]:
        """
        Run inference and extract KV cache for each sample.
        
        Returns metadata about extracted blocks.
        """
        if not self._initialized:
            raise RuntimeError("Model not initialized")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_blocks = []
        prefix_to_blocks = {}  # For deduplication analysis
        
        # Process in batches
        for batch_start in range(0, len(samples), batch_size):
            batch_end = min(batch_start + batch_size, len(samples))
            batch = samples[batch_start:batch_end]
            
            print(f"Processing batch {batch_start//batch_size + 1}/{(len(samples) + batch_size - 1)//batch_size}")
            
            for sample in batch:
                blocks = self._extract_single(sample, output_dir, prefix_to_blocks)
                all_blocks.extend(blocks)
            
            # Periodic flush
            if (batch_start // batch_size) % 10 == 0:
                self._save_metadata(output_dir, all_blocks, prefix_to_blocks)
        
        # Final save
        self._save_metadata(output_dir, all_blocks, prefix_to_blocks)
        
        return {
            "total_blocks": len(all_blocks),
            "unique_blocks": len(set(b["block_id"] for b in all_blocks)),
            "prefix_groups": len(prefix_to_blocks),
            "dedup_ratio": 1 - len(set(b["block_id"] for b in all_blocks)) / max(1, len(all_blocks)),
        }
    
    def _extract_single(
        self,
        sample: BenchmarkSample,
        output_dir: Path,
        prefix_to_blocks: Dict
    ) -> List[Dict]:
        """Extract KV cache for a single sample."""
        
        # Tokenize
        tokens = self.tokenizer.encode(sample.prompt, add_special_tokens=True)
        sample.prompt_tokens = tokens
        sample.num_tokens = len(tokens)
        
        if len(tokens) > self.max_model_len:
            tokens = tokens[:self.max_model_len]
        
        # Run inference to generate KV cache
        if self._use_vllm:
            kv_cache = self._extract_vllm(tokens)
        else:
            kv_cache = self._extract_hf(tokens)
        
        # Split into blocks
        blocks = []
        num_blocks = (len(tokens) + self.block_size - 1) // self.block_size
        
        for block_idx in range(num_blocks):
            start_token = block_idx * self.block_size
            end_token = min((block_idx + 1) * self.block_size, len(tokens))
            
            # Extract this block's KV data
            key_data, value_data = self._slice_kv(kv_cache, start_token, end_token)
            
            # Content-addressed block ID
            content = key_data.tobytes() + value_data.tobytes()
            block_id = hashlib.sha256(content).hexdigest()[:32]
            
            # Check for deduplication
            is_duplicate = block_id in prefix_to_blocks
            
            # Save block (skip if duplicate and we want dedup)
            if not is_duplicate:
                block_path = output_dir / "blocks" / f"{block_id}.npz"
                block_path.parent.mkdir(exist_ok=True)
                np.savez_compressed(block_path, key=key_data, value=value_data)
            
            # Track prefix groups
            if sample.is_shared_prefix and block_idx < (len(tokens) // self.block_size) // 2: # heuristic: first half is logic
                if sample.prefix_group not in prefix_to_blocks:
                    prefix_to_blocks[sample.prefix_group] = []
                prefix_to_blocks[sample.prefix_group].append(block_id)
            
            blocks.append({
                "block_id": block_id,
                "sample_id": sample.sample_id,
                "block_idx": block_idx,
                "num_tokens": end_token - start_token,
                "size_bytes": key_data.nbytes + value_data.nbytes,
                "is_prefix": sample.is_shared_prefix,
                "prefix_group": sample.prefix_group
            })
        
        return blocks
    
    def _extract_vllm(self, tokens: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract KV cache using vLLM internals."""
        from vllm import SamplingParams
        
        # Run single forward pass
        sampling_params = SamplingParams(max_tokens=1, temperature=0)
        
        # Access internal KV cache
        # Note: This requires vLLM internals access, may need modification
        outputs = self.model.generate(
            prompt_token_ids=[tokens],
            sampling_params=sampling_params,
            use_tqdm=False
        )
        
        # Extract KV from vLLM's cache manager
        # This is vLLM version-specific
        kv_cache = self._get_vllm_kv_cache(outputs[0])
        
        return kv_cache
    
    def _extract_hf(self, tokens: List[int]) -> Dict:
        """Extract KV cache using HuggingFace."""
        import torch
        
        input_ids = torch.tensor([tokens], device=self.model.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids,
                use_cache=True,
                return_dict=True,
            )
        
        # past_key_values: tuple of (key, value) for each layer
        # Each key/value: [batch, num_heads, seq_len, head_dim]
        past_kv = outputs.past_key_values
        
        return past_kv
    
    def _slice_kv(
        self,
        kv_cache,
        start_token: int,
        end_token: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Slice KV cache for a specific token range."""
        
        if self._use_vllm:
            # vLLM format
            keys = []
            values = []
            for layer_kv in kv_cache:
                k, v = layer_kv
                keys.append(k[:, start_token:end_token, :].cpu().numpy())
                values.append(v[:, start_token:end_token, :].cpu().numpy())
            
            key_data = np.stack(keys, axis=1).astype(np.float16)
            value_data = np.stack(values, axis=1).astype(np.float16)
        else:
            # HuggingFace format: tuple of (key, value) per layer
            keys = []
            values = []
            for layer_idx, (k, v) in enumerate(kv_cache):
                # k, v: [batch, num_heads, seq_len, head_dim]
                keys.append(k[0, :, start_token:end_token, :].cpu().numpy())
                values.append(v[0, :, start_token:end_token, :].cpu().numpy())
            
            # Stack: [num_tokens, num_layers, num_heads, head_dim]
            key_data = np.stack(keys, axis=0).transpose(2, 0, 1, 3).astype(np.float16)
            value_data = np.stack(values, axis=0).transpose(2, 0, 1, 3).astype(np.float16)
        
        return key_data, value_data
    
    def _save_metadata(self, output_dir: Path, blocks: List[Dict], prefix_to_blocks: Dict):
        """Save metadata to disk."""
        metadata = {
            "model": self.model_name,
            "block_size": self.block_size,
            "total_blocks": len(blocks),
            "unique_blocks": len(set(b["block_id"] for b in blocks)),
            "prefix_groups": {k: list(v) for k, v in prefix_to_blocks.items()},
            "blocks": blocks,
        }
        
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)


# ============================================================================
# Main Generator
# ============================================================================

class RealDataGenerator:
    """
    End-to-end real KV cache data generator.
    
    Usage:
        generator = RealDataGenerator(model="meta-llama/Llama-2-70b-hf")
        generator.generate(
            datasets=["openorca", "sharegpt"],
            output_dir="/path/to/output",
            total_samples=10000,
            include_shared_prefix=True
        )
    """
    
    def __init__(
        self,
        model: str = "meta-llama/Llama-2-70b-hf",
        tensor_parallel: int = 4,
        block_size: int = 256,
    ):
        self.dataset_loader = MLPerfDatasetLoader()
        self.kv_extractor = RealKVCacheExtractor(
            model_name=model,
            tensor_parallel=tensor_parallel,
            block_size=block_size,
        )
    
    def generate(
        self,
        datasets: List[str] = ["openorca"],
        output_dir: str = "/pscratch/sd/s/sgkim/Skim-cascade/benchmark/data_real",
        total_samples: int = 5000,
        include_shared_prefix: bool = True,
        shared_prefix_ratio: float = 0.3,  # 30% shared prefix workload
    ):
        """Generate complete benchmark dataset."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("=" * 60)
        print("Real KV Cache Data Generator")
        print("=" * 60)
        print(f"Model: {self.kv_extractor.model_name}")
        print(f"Datasets: {datasets}")
        print(f"Total samples: {total_samples}")
        print(f"Output: {output_dir}")
        
        # 1. Load datasets
        print("\\n[1/4] Loading datasets...")
        all_samples = []
        samples_per_dataset = total_samples // len(datasets)
        
        for ds_name in datasets:
            samples = self.dataset_loader.load_dataset(ds_name, samples_per_dataset)
            all_samples.extend(samples)
        
        print(f"Loaded {len(all_samples)} base samples")
        
        # 2. Create shared prefix workload
        if include_shared_prefix:
            print("\\n[2/4] Creating shared prefix workload...")
            num_shared = int(total_samples * shared_prefix_ratio)
            shared_samples = self.dataset_loader.create_shared_prefix_workload(
                all_samples,
                num_prefix_groups=10,
                sessions_per_prefix=num_shared // 10
            )
            all_samples.extend(shared_samples)
            print(f"Added {len(shared_samples)} shared prefix samples")
        
        # 3. Initialize model
        print("\\n[3/4] Initializing model...")
        if not self.kv_extractor.initialize():
            raise RuntimeError("Failed to initialize model")
        
        # 4. Extract KV cache
        print("\\n[4/4] Extracting KV cache...")
        stats = self.kv_extractor.extract_kv_cache(all_samples, output_dir)
        
        print("\\n" + "=" * 60)
        print("Generation Complete!")
        print("=" * 60)
        print(f"Total blocks: {stats['total_blocks']:,}")
        print(f"Unique blocks: {stats['unique_blocks']:,}")
        print(f"Dedup ratio: {stats['dedup_ratio']:.1%}")
        print(f"Output: {output_dir}")
        
        return stats


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate real KV cache data")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-70b-hf")
    parser.add_argument("--datasets", type=str, default="openorca,sharegpt")
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--output", type=str, 
                        default="/pscratch/sd/s/sgkim/Skim-cascade/benchmark/data_real")
    parser.add_argument("--tp", type=int, default=4, help="Tensor parallel size")
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--shared_prefix", action="store_true", default=True)
    args = parser.parse_args()
    
    generator = RealDataGenerator(
        model=args.model,
        tensor_parallel=args.tp,
        block_size=args.block_size,
    )
    
    generator.generate(
        datasets=args.datasets.split(","),
        output_dir=args.output,
        total_samples=args.num_samples,
        include_shared_prefix=args.shared_prefix,
    )


if __name__ == "__main__":
    main()
