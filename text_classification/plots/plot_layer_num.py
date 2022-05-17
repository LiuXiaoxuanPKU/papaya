import matplotlib.pyplot as plt
import json

results = {}
filename = '../results/speed_results_quantize_all.json'
with open(filename, 'r') as f:
    lines = f.readlines()
    for line in lines:
        if line.startswith("/"):
            continue
        obj = json.loads(line)
        # alg, bz, ips = obj['algorithm'], obj['batch_size'], obj['ips']
        alg, layer_num, bz, ips = obj['algorithm'], obj['layer_num'], obj['batch_size'], obj['ips']
        if layer_num < 20 or layer_num % 8 == 0 or layer_num > 40:
            continue
        if alg is None:
            alg = "exact"
        alg = alg + "_" + str(layer_num)
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
    # ax.plot(results[alg].keys(), results[alg].values(), label=alg_map[alg], marker='o')
    if alg.startswith("L1"):
        marker = "x"
        alg_name = alg.replace("L1", "quantize")
    else:
        marker = "o"
        alg_name = alg
    ax.plot(results[alg].keys(), results[alg].values(), label=alg_name, marker=marker, lw=1, ms=7)
    ax.legend()
    # ax.set_yscale("log")
ax.set_title("Bert of different depths on V100", size=16)
ax.set_xlabel("batch size", size=16)
ax.set_ylabel("throughput (records/s)", size=16)
fig.savefig('../graphs/layer_num.pdf')