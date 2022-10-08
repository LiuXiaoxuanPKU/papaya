import json
import matplotlib.pyplot as plt

NAMES = {
    "L1" : "quantize",
    "ckpt": "checkpoint",
    None: "exact"
}

COLORS = {
    "L1" : "orange",
    "ckpt": "green",
    None: "#2596be"
}

data = {}
file_dir = "speed_results.json"
with open(file_dir, "r") as f:
    lines = f.readlines()
    for line in lines:
        obj = json.loads(line)
        alg, ips, layer_num = obj["algorithm"], obj["ips"], obj["layer_num"]
        if ips == -1:
            continue
        if alg == "swap":
            continue
        if layer_num <= 10:
            continue
        if alg not in data:
            data[alg] = {}
        if layer_num not in data[alg]:
            data[alg][layer_num] = ips
        else:
            data[alg][layer_num] = max(data[alg][layer_num], ips)


fig, ax = plt.subplots(1, 1)
fig.set_size_inches(4, 4)
for alg in data:
    xs = []
    ys = []
    for k in sorted(data[alg].keys()):
        xs.append(k)
        ys.append(data[alg][k])
    plt.plot(xs, ys, "-o", label=NAMES[alg])

plt.xlabel('model depth (# of layers)', size=15)
plt.ylabel('max throughput (records/s)', size=15)
plt.legend(fontsize='large')
ax.tick_params(axis='both', which='major', labelsize=15)
plt.tight_layout()
plt.xlim(8, 70)
plt.ylim(0, 280)
plt.legend()
plt.savefig("depth.png")