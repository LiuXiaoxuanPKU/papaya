import matplotlib.pyplot as plt
import json

results = {}
filename = '../results/speed_results_no_limit.json'
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
    ax.plot(results[alg].keys(), results[alg].values(), label=alg_map[alg], marker='o')
    ax.legend(prop={"size":13})
    # ax.set_yscale("log")
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
ax.set_title("Bert Large on V100", size=18)
ax.set_xlabel("batch size", size=18)
ax.set_ylabel("throughput (record/s)", size=18)
fig.savefig('../graphs/mem_optimization.pdf', bbox_inches='tight')