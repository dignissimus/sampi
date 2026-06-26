import sys
import os
import re
import numpy as np
import glob

def parse_files(pattern):
    files = glob.glob(pattern)
    if not files: return None
    
    bfs_times = []
    sssp_times = []
    
    for f in files:
        with open(f, 'r') as file:
            content = file.read()
            m_bfs = re.search(r'^bfs\s+min_time:\s+([\d\.eE+-]+)', content, re.MULTILINE)
            m_sssp = re.search(r'^sssp\s+min_time:\s+([\d\.eE+-]+)', content, re.MULTILINE)
            
            if m_bfs: bfs_times.append(float(m_bfs.group(1)))
            if m_sssp: sssp_times.append(float(m_sssp.group(1)))
            
    if not bfs_times and not sssp_times: return None
    return {
        'bfs': np.mean(bfs_times) if bfs_times else None,
        'sssp': np.mean(sssp_times) if sssp_times else None
    }

def process_suite(suite_dir, exp_name):
    dirs = glob.glob(os.path.join(suite_dir, '*'))
    experiments = []
    for d in dirs:
        if not os.path.isdir(d): continue
        basename = os.path.basename(d)
        
        # Match new format: 1nodes_np64
        m1 = re.search(r'(\d+)nodes_np(\d+)', basename)
        if m1:
            nodes = int(m1.group(1))
            np_tasks = int(m1.group(2))
            experiments.append((nodes, np_tasks, d))
            continue
            
        # Match old format: graph500_s20_np64_...
        m2 = re.search(r'_s\d+_np(\d+)_', basename)
        if m2:
            np_tasks = int(m2.group(1))
            nodes = np_tasks // 64 # Hamilton is 64 core
            experiments.append((nodes, np_tasks, d))

    results = []
    for nodes, np_tasks, d in experiments:
        v_res = parse_files(os.path.join(d, '00_vanilla_run*.out'))
        s_res = parse_files(os.path.join(d, '03_stub_run*.out'))
        b_res = parse_files(os.path.join(d, '02_boosted_run*.out'))
        
        if not v_res: continue
        
        for algo in ['bfs', 'sssp']:
            if v_res[algo] is None: continue
            
            v_time = v_res[algo]
            s_time = s_res[algo] if s_res and algo in s_res else None
            b_time = b_res[algo] if b_res and algo in b_res else None
            
            s_speedup = ((v_time - s_time) / v_time * 100) if s_time else None
            b_speedup = ((v_time - b_time) / v_time * 100) if b_time else None
            
            results.append({
                'exp': exp_name,
                'nodes': nodes,
                'np': np_tasks,
                'algo': algo,
                'v_time': v_time,
                's_time': s_time,
                's_speedup': s_speedup,
                'b_time': b_time,
                'b_speedup': b_speedup
            })
    return results

if len(sys.argv) < 4:
    print("Usage: python3 analyze_mega_hamilton.py <suite_dir_5x> <suite_dir_5x_noblock> <dir_comparison>")
    sys.exit(1)

res1 = process_suite(sys.argv[1], "5x (Mapped)")
res2 = process_suite(sys.argv[2], "5x (No-Map)")
res3 = process_suite(sys.argv[3], "Comparison (1x)")

all_res = res1 + res2 + res3
all_res.sort(key=lambda x: (x['nodes'], x['np'], x['algo'], x['exp']))

print(f"{'Experiment':<18} | {'Nodes':<7} | {'Tasks (np)':<12} | {'Algorithm':<9} | {'Vanilla (s)':<12} | {'Stub (s)':<12} | {'Stub Speedup':<14} | {'Boost (s)':<12} | {'Boost Speedup'}")
print("-" * 133)

for r in all_res:
    s_time_str = f"{r['s_time']:<12.6f}" if r['s_time'] is not None else "N/A"
    s_speedup_str = f"{r['s_speedup']:+.2f}%" if r['s_speedup'] is not None else "N/A"
    b_time_str = f"{r['b_time']:<12.6f}" if r['b_time'] is not None else "N/A"
    b_speedup_str = f"{r['b_speedup']:+.2f}%" if r['b_speedup'] is not None else "N/A"
    
    print(f"{r['exp']:<18} | {r['nodes']:<7} | np={r['np']:<9} | {r['algo'].upper():<9} | {r['v_time']:<12.6f} | {s_time_str:<12} | {s_speedup_str:<14} | {b_time_str:<12} | {b_speedup_str}")
