import os

expert = [1, 2, 4, 8]

for ex in expert:
    command = f"python ../megablocks/layers/benchmark.py --top_k 1 --e {ex} --s 4096 --hid_dim 7168 --bs 1"
    print(f"Running: {command}")
    os.system(command)         
