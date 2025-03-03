import os

top_k_values = [2, 4, 6, 8, 10, 12, 32, 64]

for top_k in top_k_values:
    command = f"python ../../megablocks/layers/benchmark.py --top_k {top_k} --e 128 --s 4096 --hid_dim 7168 --bs 1"
    print(f"Running: {command}")
    os.system(command)         
