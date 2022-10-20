import json
import matplotlib.pyplot as plt
import sys
sys.path.append('.')
from plot_case_study.plot_util import ALG_COLOR, ALG_MAP, ALG_MARKER
from util import Util

suffix = "pdf"
data = {}
file_dir = "text_classification_fp16/v100/results/speed_results_depth.json"
with open(file_dir, "r") as f:
    lines = f.readlines()
    for line in lines:
        obj = json.loads(line)
        alg, ips, layer_num = obj["algorithm"], obj["ips"], obj["layer_num"]
        if ips == -1:
            continue
        if alg == "swap":
            continue
        if alg is None:
            alg = 'exact'
        if layer_num <= 10:
            continue
        if alg not in data:
            data[alg] = {}
        if layer_num not in data[alg]:
            data[alg][layer_num] = ips
        else:
            data[alg][layer_num] = max(data[alg][layer_num], ips)


fig, ax = plt.subplots(1, 1)
fig.set_size_inches(6, 6)
for alg in data:
    xs = []
    ys = []
    for k in sorted(data[alg].keys()):
        xs.append(k)
        ys.append(data[alg][k])
    plt.plot(xs, ys, marker=ALG_MARKER[alg], label=ALG_MAP[alg], color=ALG_COLOR[alg], ms=12, lw=3)

plt.xlabel('Model depth (# of layers)')
plt.grid()
# plt.ylabel('max throughput (records/s)', size=15)
# ax.tick_params(axis='both', which='major', labelsize=15)
Util.set_tick_label_size([ax])
plt.xlim(8, 70)
plt.ylim(0, 350)
plt.legend(fontsize=20)
plt.tight_layout()
plt.savefig(f'graphs/case_study/large_model_depth.{suffix}')