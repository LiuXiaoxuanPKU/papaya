import matplotlib.pyplot as plt
import json

results = {}
filename = '../results/V100/speed_results_no_limit_36.json'
with open(filename, 'r') as f:
    lines = f.readlines()
    for line in lines:
        if line.startswith("/"):
            continue
        obj = json.loads(line)
        alg, layer_num, bz, ips = obj['algorithm'], obj['layer_num'], obj['batch_size'], obj['ips']
        if ips == -1:
            continue
        if alg is None:
            alg = 'exact'
        if alg not in results:
            results[alg] = {}
        results[alg][bz] = ips

alg_map = {
    "exact" : "exact",
    "swap" : "swap",
    "L4bit-swap" : "quantize+swap",
    "swap-lz4" : "lz4+swap",
    "ckpt" : "checkpoint",
    "L1" : "quantize",
    "L1_ckpt" : "quantize+checkpoint",
    "swap_ckpt" : "swap+checkpoint"
}

fig, ax = plt.subplots()
for alg in results:
    print(alg)
    ax.plot(results[alg].keys(), results[alg].values(), label=alg_map[alg], marker='o')
    ax.legend(loc="lower right", prop={'size': 16})
    # ax.set_yscale("log")
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

ax.set_title("Bert 36 with FP16 on V100", size=22)
ax.set_xlabel("batch size", size=22)
ax.set_ylabel("throughput (record/s)", size=22)
fig.savefig('../graphs/mem_optimization_36.pdf', bbox_inches='tight')