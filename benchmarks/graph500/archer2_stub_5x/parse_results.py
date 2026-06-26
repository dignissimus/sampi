import sys
import glob
import os
import re

def parse_files(file_pattern):
    files = glob.glob(file_pattern)
    if not files:
        return None
        
    all_bfs_times = []
    all_sssp_times = []
    
    for filepath in files:
        with open(filepath, 'r') as f:
            for line in f:
                m_bfs = re.search(r'Time for BFS \d+ is ([\d\.]+)', line)
                if m_bfs:
                    all_bfs_times.append(float(m_bfs.group(1)))
                    
                m_sssp = re.search(r'Time for SSSP \d+ is ([\d\.]+)', line)
                if m_sssp:
                    all_sssp_times.append(float(m_sssp.group(1)))
                    
    results = {}
    if all_bfs_times:
        results['bfs_time'] = sum(all_bfs_times) / len(all_bfs_times)
    if all_sssp_times:
        results['sssp_time'] = sum(all_sssp_times) / len(all_sssp_times)
        
    if results:
        return results
    return None

def analyze_directory(base_dir):
    print(f"\n=== Analyzing {base_dir} ===")
    dirs = glob.glob(os.path.join(base_dir, '*'))
    
    experiments = []
    for d in dirs:
        if not os.path.isdir(d): continue
        basename = os.path.basename(d)
        m = re.search(r'_s(\d+)_np(\d+)_', basename)
        if m:
            scale = int(m.group(1))
            np = int(m.group(2))
            experiments.append((scale, np, d))
    
    experiments.sort()
    
    print(f"{'Scale':<7} | {'Nodes (np)':<12} | {'Algorithm':<9} | {'Vanilla (s)':<12} | {'Stub (s)':<12} | {'Stub Speedup':<14} | {'Boost (s)':<12} | {'Boost Speedup'}")
    print("-" * 110)
    
    for scale, np, d in experiments:
        vanilla_res = parse_files(os.path.join(d, '00_vanilla_run_*.out'))
        stub_res = parse_files(os.path.join(d, '03_stub_run_*.out'))
        boost_res = parse_files(os.path.join(d, '02_boosted_run_*.out'))
        
        # Fallback to non-repeated names
        if not vanilla_res:
            vanilla_res = parse_files(os.path.join(d, '00_vanilla_run.out'))
        if not stub_res:
            stub_res = parse_files(os.path.join(d, '03_stub_run.out'))
        if not boost_res:
            boost_res = parse_files(os.path.join(d, '02_boosted_run.out'))
            
        if not vanilla_res:
            continue
            
        for algo in ['bfs', 'sssp']:
            k_time = f'{algo}_time'
            
            if k_time not in vanilla_res:
                continue
                
            v_time = vanilla_res[k_time]
            
            s_time_str = "N/A"
            s_speedup_str = "N/A"
            if stub_res and k_time in stub_res:
                s_time = stub_res[k_time]
                s_time_str = f"{s_time:.6f}"
                s_speedup = (v_time / s_time - 1) * 100
                s_speedup_str = f"{s_speedup:+.2f}%"
                
            b_time_str = "N/A"
            b_speedup_str = "N/A"
            if boost_res and k_time in boost_res:
                b_time = boost_res[k_time]
                b_time_str = f"{b_time:.6f}"
                b_speedup = (v_time / b_time - 1) * 100
                b_speedup_str = f"{b_speedup:+.2f}%"
            
            if algo == 'bfs':
                print(f"{scale:<7} | np={np:<9} | {algo.upper():<9} | {v_time:<12.6f} | {s_time_str:<12} | {s_speedup_str:<14} | {b_time_str:<12} | {b_speedup_str}")
            else:
                print(f"{'':<7} | {'':<12} | {algo.upper():<9} | {v_time:<12.6f} | {s_time_str:<12} | {s_speedup_str:<14} | {b_time_str:<12} | {b_speedup_str}")
        print("-" * 110)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        for d in sys.argv[1:]:
            analyze_directory(d)
    else:
        print("Usage: python parse_results.py <path_to_results_dir>")
