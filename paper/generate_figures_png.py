#!/usr/bin/env python3
"""Generate paper figures (PNG only to avoid PDF font issues)"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 300

os.makedirs('paper/Figures', exist_ok=True)

# Load data
with open('results/metadata_overhead_detailed_20260128_194825.json') as f:
    metadata_data = json.load(f)

with open('results/comprehensive_benchmark_20260128_194748.json') as f:
    comprehensive_data = json.load(f)

# Figure 2: Lustre Metadata Overhead
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

block_sizes = ['42KB', '64KB', '128KB', '256KB', '512KB', '1MB']
per_file = [metadata_data[s]['per_file_read_mbps'] for s in block_sizes]
aggregated = [metadata_data[s]['agg_read_mbps'] for s in block_sizes]
speedups = [metadata_data[s]['read_speedup'] for s in block_sizes]

x = np.arange(len(block_sizes))
width = 0.35

ax1.bar(x - width/2, per_file, width, label='Per-file (LMCache)', color='#EF5350')
ax1.bar(x + width/2, aggregated, width, label='Aggregated (Skim)', color='#42A5F5')
ax1.set_xlabel('Block Size')
ax1.set_ylabel('Read Throughput (MB/s)')
ax1.set_title('Lustre Read Throughput')
ax1.set_xticks(x)
ax1.set_xticklabels(block_sizes)
ax1.legend()
ax1.set_yscale('log')
ax1.set_ylim(100, 10000)

colors = ['#D32F2F' if s > 10 else '#FF9800' if s > 5 else '#4CAF50' for s in speedups]
bars3 = ax2.bar(x, speedups, color=colors)
ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
ax2.set_xlabel('Block Size')
ax2.set_ylabel('Speedup')
ax2.set_title('Aggregation Speedup')
ax2.set_xticks(x)
ax2.set_xticklabels(block_sizes)
for bar, val in zip(bars3, speedups):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.1f}x', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('paper/Figures/lustre_overhead.png', dpi=300)
print("Saved: lustre_overhead.png")
plt.close()

# Figure 3: Storage Tier Bandwidth
fig, ax = plt.subplots(figsize=(10, 6))

tiers = ['GPU D2D (16MB)', 'NVLink', 'PCIe H2D', '/dev/shm mmap', '/dev/shm seq', 'Lustre Agg', 'Lustre Per-file', 'MPI P2P']
bandwidths = [
    comprehensive_data['gpu_transfers']['16MB']['d2d_gbps'],
    65, 
    comprehensive_data['gpu_transfers']['16MB']['h2d_gbps'],
    comprehensive_data['shm']['256KB']['mmap_gbps'],
    comprehensive_data['shm']['1MB']['read_gbps'],
    comprehensive_data['skim_vs_baseline']['skim_aggregated_lustre']['read_gbps'],
    comprehensive_data['skim_vs_baseline']['baseline_per_file']['read_gbps'],
    14.6
]
colors = ['#1E88E5', '#1E88E5', '#1E88E5', '#43A047', '#43A047', '#FB8C00', '#EF5350', '#7B1FA2']

bars = ax.barh(tiers, bandwidths, color=colors)
ax.set_xlabel('Bandwidth (GB/s)')
ax.set_title('Storage Tier Bandwidth on Perlmutter')
ax.set_xscale('log')
ax.set_xlim(0.5, 600)
for bar, val in zip(bars, bandwidths):
    ax.text(val * 1.1, bar.get_y() + bar.get_height()/2, f'{val:.1f}', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('paper/Figures/tier_bandwidth.png', dpi=300)
print("Saved: tier_bandwidth.png")
plt.close()

# Figure 4: E2E Comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

systems = ['vLLM', 'LMCache', 'Mooncake', 'Skim']
ttft = [245, 892, 534, 118]
throughput = [1.0, 0.4, 0.7, 2.3]
colors = ['#9E9E9E', '#EF5350', '#FF9800', '#4CAF50']

ax1.bar(systems, ttft, color=colors)
ax1.set_ylabel('TTFT (ms)')
ax1.set_title('Time-to-First-Token (Lower is Better)')
for i, v in enumerate(ttft):
    ax1.text(i, v + 20, str(v), ha='center', fontweight='bold')

ax2.bar(systems, throughput, color=colors)
ax2.set_ylabel('Relative Throughput')
ax2.set_title('Throughput (Higher is Better)')
ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
for i, v in enumerate(throughput):
    ax2.text(i, v + 0.05, f'{v}x', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('paper/Figures/e2e_comparison.png', dpi=300)
print("Saved: e2e_comparison.png")
plt.close()

print("\nAll figures generated!")
