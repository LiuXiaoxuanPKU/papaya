from cProfile import label
import csv
import matplotlib.pyplot as plt

data_dir = "all_data"

def parse_row(tokens):
    bz, model_info, _, _, dist_info = eval(tokens[1])
    seq_len, hidden_size, num_layers, num_heads, vocab_size = model_info
    _, remat, dp, op, pp, _ = dist_info
    is_dist = (op > 1) or (pp > 1)
    batch_time = float(tokens[7])
    return bz, num_layers, remat, is_dist, batch_time

reader = csv.reader(open(data_dir), delimiter="\t")
data = {}
for tokens in reader:
    bz, num_layers, remat, is_dist, batch_time = parse_row(tokens)
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
    if num_layers not in data[alg]:
        data[alg][num_layers] = tpt
    else:
        data[alg][num_layers] = max(data[alg][num_layers], tpt)

LABLES = {
    "ckpt_dis" : "mp + ckpt (4GPU)",
    "dis" : "mp (4GPU)",
    "ckpt": "ckpt (1GPU)",
    "exact": "exact (1GPU)"
}
print(list(data.keys()))
fig, ax = plt.subplots()
fig.set_size_inches(4, 4)
for alg in ["ckpt_dis", "dis", "ckpt", "exact"]:
    ax.plot(list(data[alg].keys()), list(data[alg].values()), "o", label=LABLES[alg], markersize=5)
plt.xlabel("model depth", size=16)
plt.ylabel("max throughput (records/s)", size=16)
plt.legend()
plt.tight_layout()
suffix = "pdf"
plt.savefig(f"/Users/xiaoxuanliu/Documents/2022fall/papaya/graphs/implications/GPT_dis.{suffix}")
