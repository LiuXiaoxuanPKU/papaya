import matplotlib.pyplot as plt
import json

results = {}
filename = '../results/speed_results.json'
with open(filename, 'r') as f:
    lines = f.readlines()
    for line in lines:
        if line.startswith("/"):
            continue
        obj = json.loads(line)
        if obj['ips'] == -1:
            continue
        fp16, ckpt, bz, ips = obj['fp16'], obj['ckpt'], obj['batch_size'], obj['ips']
        alg = "fp16=" + fp16 + ", ckpt="+str(ckpt)
        if alg is None:
            alg = 'exact'
        if alg not in results:
            results[alg] = {}
        results[alg][bz] = ips


fig, ax = plt.subplots()
for alg in results:
    ax.plot(results[alg].keys(), results[alg].values(), label=alg, marker='o')
    ax.legend()
    # ax.set_yscale("log")
ax.set_title("Swin Large on V100", size=16)
ax.set_xlabel("batch size", size=16)
ax.set_ylabel("throughput (record/s)", size=16)
fig.savefig('mem_optimization')