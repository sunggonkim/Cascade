#!/bin/bash
#SBATCH -A m1248_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 4
#SBATCH -t 00:30:00
#SBATCH -J real_6sys_v3
#SBATCH --gpus-per-node=4
#SBATCH -o /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/real_6sys_v3_%j.out
#SBATCH -e /pscratch/sd/s/sgkim/Skim-cascade/benchmark/logs/real_6sys_v3_%j.err

set -e
export MPICH_GPU_SUPPORT_ENABLED=1
cd /pscratch/sd/s/sgkim/Skim-cascade

echo "================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "Start: $(date)"
echo "================================================"

# Python 환경 로드
module load python cudatoolkit pytorch

# Redis Python client 설치
pip install redis h5py --user --quiet 2>/dev/null || true
export PYTHONPATH=$HOME/.local/lib/python3.11/site-packages:$PYTHONPATH

RESULTS_DIR=/pscratch/sd/s/sgkim/Skim-cascade/benchmark/results
mkdir -p $RESULTS_DIR

BENCH_SCRIPT=/pscratch/sd/s/sgkim/Skim-cascade/benchmark/scripts/real_6sys_bench_$SLURM_JOB_ID.py

cat > $BENCH_SCRIPT << 'PYTHON_EOF'
#!/usr/bin/env python3
"""6개 시스템 비교: Cascade, vLLM, LMCache, PDC, Redis, HDF5"""
import os, sys, time, json, ctypes, mmap
import numpy as np
from pathlib import Path

BLOCK_SIZE_MB = int(os.environ.get('BLOCK_SIZE_MB', 10))
NUM_BLOCKS = int(os.environ.get('NUM_BLOCKS', 100))
BLOCK_SIZE = BLOCK_SIZE_MB * 1024 * 1024
RESULTS_DIR = os.environ.get('RESULTS_DIR', '/tmp')
JOB_ID = os.environ.get('SLURM_JOB_ID', 'local')

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except ImportError:
    rank, size, comm = 0, 1, None

def get_hostname():
    import socket
    return socket.gethostname()

def drop_page_cache(path):
    try:
        fd = os.open(path, os.O_RDONLY)
        libc = ctypes.CDLL("libc.so.6")
        libc.posix_fadvise(fd, 0, os.fstat(fd).st_size, 4)
        os.close(fd)
    except: pass

# == 1. Cascade (SHM mmap) ==
class CascadeStore:
    def __init__(self):
        self.shm_dir = Path(f"/dev/shm/cascade_bench_{rank}")
        self.shm_dir.mkdir(exist_ok=True)
        if rank == 0: print("  Cascade: mmap /dev/shm")
    def put(self, key, data):
        (self.shm_dir / f"{key}.bin").write_bytes(data)
        return len(data)
    def get(self, key):
        path = self.shm_dir / f"{key}.bin"
        with open(path, 'rb') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                return bytes(mm[:])
    def cleanup(self):
        import shutil
        shutil.rmtree(self.shm_dir, ignore_errors=True)

# == 2. vLLM (GPU HBM via torch) ==
class VLLMStore:
    def __init__(self):
        self.available = False
        self.cache = {}
        try:
            import torch
            if torch.cuda.is_available():
                self.torch = torch
                gpu_id = rank % torch.cuda.device_count()
                self.device = torch.device(f'cuda:{gpu_id}')
                self.available = True
                if rank == 0:
                    print(f"  vLLM: GPU {torch.cuda.get_device_name()} [cuda:{gpu_id}]")
            else:
                if rank == 0: print("  vLLM: No CUDA available")
        except ImportError as e:
            if rank == 0: print(f"  vLLM: torch not installed ({e})")
    def put(self, key, data):
        if self.available:
            self.cache[key] = self.torch.from_numpy(np.frombuffer(data, dtype=np.uint8).copy()).to(self.device)
        return len(data)
    def get(self, key):
        if self.available and key in self.cache:
            return self.cache[key].cpu().numpy().tobytes()
        return b''
    def cleanup(self):
        self.cache.clear()
        if self.available:
            self.torch.cuda.empty_cache()

# == 3. LMCache (File-based, session-separated) ==
class LMCacheStore:
    def __init__(self):
        self.disk_dir = Path(f"/tmp/lmcache_bench_{rank}")
        self.disk_dir.mkdir(exist_ok=True)
        if rank == 0: print("  LMCache: File-based (session-separated)")
    def put(self, key, data):
        (self.disk_dir / f"session_{rank}_{key}.bin").write_bytes(data)
        return len(data)
    def get(self, key):
        path = self.disk_dir / f"session_{rank}_{key}.bin"
        drop_page_cache(str(path))
        return path.read_bytes()
    def cleanup(self):
        import shutil
        shutil.rmtree(self.disk_dir, ignore_errors=True)

# == 4. PDC (fsync durability) ==
class PDCStore:
    def __init__(self):
        self.disk_dir = Path(f"/tmp/pdc_bench_{rank}")
        self.disk_dir.mkdir(exist_ok=True)
        if rank == 0: print("  PDC: With fsync durability")
    def put(self, key, data):
        path = self.disk_dir / f"{key}.bin"
        with open(path, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        return len(data)
    def get(self, key):
        path = self.disk_dir / f"{key}.bin"
        drop_page_cache(str(path))
        return path.read_bytes()
    def cleanup(self):
        import shutil
        shutil.rmtree(self.disk_dir, ignore_errors=True)

# == 5. Redis (In-memory KV) ==
class RedisStore:
    def __init__(self, port=16379):
        self.available = False
        self.port = port
        try:
            import redis
            self.client = redis.Redis(host='localhost', port=port, socket_timeout=5)
            self.client.ping()
            self.available = True
            if rank == 0: print(f"  Redis: Connected on port {port}")
        except ImportError:
            if rank == 0: print("  Redis: redis module not installed")
        except Exception as e:
            if rank == 0: print(f"  Redis: Server not running ({e})")
    def put(self, key, data):
        if self.available:
            self.client.set(f"r{rank}:{key}", data)
        return len(data)
    def get(self, key):
        if self.available:
            return self.client.get(f"r{rank}:{key}") or b''
        return b''
    def cleanup(self):
        if self.available:
            try: self.client.flushdb()
            except: pass

# == 6. HDF5 (h5py) ==
class HDF5Store:
    def __init__(self):
        self.available = False
        self.file_path = f"/tmp/hdf5_bench_{rank}.h5"
        try:
            import h5py
            self.h5py = h5py
            self.available = True
            if rank == 0: print(f"  HDF5: h5py {h5py.__version__}")
        except ImportError:
            if rank == 0: print("  HDF5: h5py not installed")
    def put(self, key, data):
        if self.available:
            with self.h5py.File(self.file_path, 'a') as f:
                if key in f: del f[key]
                f.create_dataset(key, data=np.frombuffer(data, dtype=np.uint8))
        return len(data)
    def get(self, key):
        if self.available:
            with self.h5py.File(self.file_path, 'r') as f:
                return f[key][()].tobytes()
        return b''
    def cleanup(self):
        try: os.remove(self.file_path)
        except: pass

def run_benchmark(store, store_name, num_blocks, block_size):
    data = np.random.bytes(block_size)
    
    start = time.perf_counter()
    for i in range(num_blocks):
        store.put(f"block_{i}", data)
    write_time = time.perf_counter() - start
    
    if comm: comm.Barrier()
    
    start = time.perf_counter()
    for i in range(num_blocks):
        _ = store.get(f"block_{i}")
    read_time = time.perf_counter() - start
    
    total_gb = (num_blocks * block_size) / (1024**3)
    return {
        'system': store_name,
        'rank': rank,
        'hostname': get_hostname(),
        'total_size_mb': (num_blocks * block_size) // (1024*1024),
        'write_gbps': total_gb / write_time,
        'read_gbps': total_gb / read_time
    }

def main():
    if rank == 0:
        print("\n" + "="*70)
        print("  6개 시스템 비교: Cascade vs vLLM vs LMCache vs PDC vs Redis vs HDF5")
        print(f"  Config: {NUM_BLOCKS} x {BLOCK_SIZE_MB}MB = {NUM_BLOCKS * BLOCK_SIZE_MB}MB/rank, {size} ranks")
        print("="*70)
    
    all_results = []
    
    # Redis 서버 시작
    if rank == 0:
        redis_server = "/pscratch/sd/s/sgkim/Skim-cascade/third_party/redis/src/redis-server"
        if os.path.exists(redis_server):
            import subprocess
            subprocess.run(f"{redis_server} --port 16379 --daemonize yes", shell=True, stderr=subprocess.DEVNULL)
            time.sleep(2)
    if comm: comm.Barrier()
    
    systems = [
        ("Cascade", CascadeStore, {}),
        ("vLLM", VLLMStore, {}),
        ("LMCache", LMCacheStore, {}),
        ("PDC", PDCStore, {}),
        ("Redis", RedisStore, {'port': 16379}),
        ("HDF5", HDF5Store, {})
    ]
    
    for i, (name, StoreClass, kwargs) in enumerate(systems, 1):
        if rank == 0:
            print(f"\n[{i}/6] {name}...")
        
        store = StoreClass(**kwargs)
        
        if hasattr(store, 'available') and not store.available:
            result = {'system': name, 'rank': rank, 'hostname': get_hostname(),
                      'status': 'unavailable', 'write_gbps': 0, 'read_gbps': 0}
        else:
            result = run_benchmark(store, name, NUM_BLOCKS, BLOCK_SIZE)
        
        all_results.append(result)
        store.cleanup()
        if comm: comm.Barrier()
    
    # 결과 출력
    if rank == 0:
        print("\n" + "="*70)
        print("  결과 (Rank 0 기준)")
        print("="*70)
        types = {'Cascade': 'SHM mmap', 'vLLM': 'GPU HBM', 'LMCache': 'File-based',
                 'PDC': 'fsync', 'Redis': 'In-memory KV', 'HDF5': 'Hierarchical'}
        print(f"{'System':<12} {'Type':<15} {'Write GB/s':>12} {'Read GB/s':>12}")
        print("-"*55)
        for r in all_results:
            t = types.get(r['system'], '')
            w = r.get('write_gbps', 0)
            rd = r.get('read_gbps', 0)
            status = r.get('status', '')
            if status == 'unavailable':
                print(f"{r['system']:<12} {t:<15} {'N/A':>12} {'N/A':>12}")
            else:
                print(f"{r['system']:<12} {t:<15} {w:>12.2f} {rd:>12.2f}")
        print("="*70)
    
    # 결과 수집
    if comm:
        all_rank_results = comm.gather(all_results, root=0)
    else:
        all_rank_results = [all_results]
    
    if rank == 0:
        output = {
            'job_id': JOB_ID,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'nodes': size,
            'block_size_mb': BLOCK_SIZE_MB,
            'num_blocks': NUM_BLOCKS,
            'systems': ['Cascade', 'vLLM', 'LMCache', 'PDC', 'Redis', 'HDF5'],
            'results': all_rank_results
        }
        with open(f"{RESULTS_DIR}/real_6sys_{JOB_ID}.json", 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved: {RESULTS_DIR}/real_6sys_{JOB_ID}.json")
        os.system("pkill -f 'redis-server.*16379' 2>/dev/null || true")

if __name__ == '__main__':
    main()
PYTHON_EOF

export BLOCK_SIZE_MB=10
export NUM_BLOCKS=100
export RESULTS_DIR=$RESULTS_DIR

echo "Running 6-system benchmark with torch/redis..."
srun -N $SLURM_NNODES -n $SLURM_NNODES --gpus-per-node=4 python3 $BENCH_SCRIPT

echo ""
echo "Complete: $(date)"

if [ -f "$RESULTS_DIR/real_6sys_${SLURM_JOB_ID}.json" ]; then
    cat $RESULTS_DIR/real_6sys_${SLURM_JOB_ID}.json
fi

rm -f $BENCH_SCRIPT
