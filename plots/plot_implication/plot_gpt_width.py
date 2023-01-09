import matplotlib.pyplot as plt
import sys
sys.path.append('.')
from plots.plot_util import ALG_COLOR, ALG_MAP, ALG_MARKER, NET_TO_FOLDER
from util import Util

import csv
import matplotlib.pyplot as plt

data_dir = "benchmarks/Megatron-LM/benchmarks/model_width_data"

def parse_row(tokens):
    bz, model_info, _, _, dist_info = eval(tokens[1])
    seq_len, hidden_size, num_layers, num_heads, vocab_size = model_info
    _, remat, dp, op, pp, _ = dist_info
    is_dist = (op > 1) or (pp > 1)
    batch_time = float(tokens[7])
    return bz, hidden_size, remat, is_dist, batch_time

reader = csv.reader(open(data_dir), delimiter="\t")
data = {}
for tokens in reader:
    bz, hidden_size, remat, is_dist, batch_time = parse_row(tokens)
    if hidden_size % 2 != 0:
        continue
    tpt = bz / batch_time
    alg = None
    if remat and is_dist:
        alg = "ckpt_dis"
    elif is_dist:
        alg = "dis"
    elif remat:
        alg = "ckpt"
    else:
        alg = "exact"
    if alg not in data:
        data[alg] = {}
    if hidden_size not in data[alg]:
        data[alg][hidden_size] = tpt
    else:
        data[alg][hidden_size] = max(data[alg][hidden_size], tpt)

print(data)
print(list(data.keys()))
fig, ax = plt.subplots()
fig.set_size_inches(6, 6)
for alg in ["exact", "ckpt"]:
    ax.plot(list(data[alg].keys()), list(data[alg].values()), color=ALG_COLOR[alg],\
        marker=ALG_MARKER[alg], label=ALG_MAP[alg],  ms=12, lw=3)
plt.xlabel("Model width (hidden size)", size=16)
# plt.ylabel("max throughput (records/s)", size=16)
plt.grid()
plt.legend(fontsize=20)
Util.set_tick_label_size([ax])
plt.tight_layout()
suffix = "pdf"
plt.savefig(f"graphs/implications/GPT_width.{suffix}")
