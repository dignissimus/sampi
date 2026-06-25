import os
import re
import scipy.stats as stats
import numpy as np
import glob

def parse_file(filename):
    bfs = []
    sssp = []
    with open(filename, 'r') as f:
        for line in f:
            m = re.match(r'Time for BFS \d+ is ([\d\.]+)', line)
            if m:
                bfs.append(float(m.group(1)))
            m = re.match(r'Time for SSSP \d+ is ([\d\.]+)', line)
            if m:
                sssp.append(float(m.group(1)))
    return np.array(bfs), np.array(sssp)

def analyze(b, t):
    n1, n2 = len(b), len(t)
    if n1 == 0 or n2 == 0: return 0, 0, 0, 0, (0,0)
    
    mean_b, mean_t = np.mean(b), np.mean(t)
    var_b, var_t = np.var(b, ddof=1), np.var(t, ddof=1)
    df = n1 + n2 - 2
    
    t_stat, p_val = stats.ttest_ind(b, t, equal_var=True)
    if df <= 0: return t_stat, p_val, 0, 0, (0,0)
    
    pooled_var = ((n1 - 1) * var_b + (n2 - 1) * var_t) / df
    s = np.sqrt(pooled_var)
    cd = (mean_b - mean_t) / s if s != 0 else 0
    
    sp = (mean_b / mean_t) - 1
    
    se = np.sqrt(pooled_var * (1/n1 + 1/n2))
    diff = mean_b - mean_t
    me_99 = stats.t.ppf(1 - 0.01/2, df) * se
    ci_99 = (((diff - me_99) / mean_t) * 100, ((diff + me_99) / mean_t) * 100)
    
    return t_stat, p_val, cd, sp, ci_99

def run_analysis_for_dir(results_dir, tasks_per_node, label):
    if not os.path.exists(results_dir):
        print(f"Directory {results_dir} not found. Skipping {label}.")
        return

    dirs = sorted(glob.glob(os.path.join(results_dir, 'graph500_s*')))
    valid_runs = []
    for d in dirs:
        dirname = os.path.basename(d)
        m = re.match(r'graph500_s(\d+)_np(\d+)_.*', dirname)
        if not m: continue
        scale = int(m.group(1))
        np_count = int(m.group(2))
        node_count = np_count // tasks_per_node
        
        try:
            bfs_v, sssp_v = parse_file(os.path.join(d, '00_vanilla_run.out'))
            bfs_p, sssp_p = parse_file(os.path.join(d, '01_profile_run.out'))
            bfs_b, sssp_b = parse_file(os.path.join(d, '02_boosted_run.out'))
        except FileNotFoundError:
            continue
        
        if len(bfs_v) == 0 or len(bfs_p) == 0 or len(bfs_b) == 0: continue
        valid_runs.append((scale, node_count, bfs_v, sssp_v, bfs_p, sssp_p, bfs_b, sssp_b))

    if not valid_runs:
        print(f"No valid runs found in {results_dir}")
        return

    print(f"## ARCHER2 Results: {label}")
    print("### Rank Reordering Performance\n")
    for scale, node_count, bfs_v, sssp_v, bfs_p, sssp_p, bfs_b, sssp_b in valid_runs:
        print(f"#### Scale: {scale} | Node Count: {node_count}\n")
        print("| Algorithm | Mode | Speedup | 99% CI | t-stat | p-value | Cohen's d |")
        print("| --------- | ---- | ------- | ------ | ------ | ------- | --------- |")
        for algo, v, p_run, b in [("BFS", bfs_v, bfs_p, bfs_b), ("SSSP", sssp_v, sssp_p, sssp_b)]:
            t, p_val, cd, sp, ci99 = analyze(p_run, b)
            sign = "+" if sp > 0 else ""
            ci_str = f"[{ci99[0]:.1f}%, {ci99[1]:.1f}%]"
            speedup_str = f"**{sign}{sp*100:.1f}%**" if sp > 0 else f"{sign}{sp*100:.1f}%"
            print(f"| {algo} | Automated Rank Reordering | {speedup_str} | {ci_str} | {t:.2f} | {p_val:.2e} | {cd:.3f} |")
        print()

    print("### Profiling Overhead\n")
    for scale, node_count, bfs_v, sssp_v, bfs_p, sssp_p, bfs_b, sssp_b in valid_runs:
        print(f"#### Scale: {scale} | Node Count: {node_count}\n")
        print("| Algorithm | Mode | Speedup | 99% CI | t-stat | p-value | Cohen's d |")
        print("| --------- | ---- | ------- | ------ | ------ | ------- | --------- |")
        for algo, v, p_run, b in [("BFS", bfs_v, bfs_p, bfs_b), ("SSSP", sssp_v, sssp_p, sssp_b)]:
            t, p_val, cd, sp, ci99 = analyze(v, p_run)
            sign = "+" if sp > 0 else ""
            ci_str = f"[{ci99[0]:.1f}%, {ci99[1]:.1f}%]"
            print(f"| {algo} | Profile Run | {sign}{sp*100:.1f}% | {ci_str} | {t:.2f} | {p_val:.2e} | {cd:.3f} |")
        print()

def main():
    base_dir = os.path.join(os.path.dirname(__file__), '../experiments')
    run_analysis_for_dir(os.path.join(base_dir, 'results_archer2_64core'), 64, "64-Core Weak Scaling")
    print("\n======================================================\n")
    run_analysis_for_dir(os.path.join(base_dir, 'results_archer2_128core'), 128, "128-Core Weak Scaling")

if __name__ == "__main__":
    main()
