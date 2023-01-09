import sys
sys.path.append('.')
from plots.plot_util import ALG_COLOR, ALG_MAP, ALG_MARKER, NET_TO_ALGS, NET_TO_FOLDER
from util import Util
import matplotlib.pyplot as plt


GPU_CNT_TO_NUM = {
    1 : "",
    2 : 2,
    4 : 4,
    8 : 8
}
labels = [str(k) for k in GPU_CNT_TO_NUM]

suffix = "pdf"
def collect_max_tpt(hardware, model):
    data = {}
    for gpu_cnt in GPU_CNT_TO_NUM.keys():
        ips_dir = f"benchmarks/{NET_TO_FOLDER[model]}/{GPU_CNT_TO_NUM[gpu_cnt]}{hardware}/results/speed_results.json"
        for alg in NET_TO_ALGS[model]:
            if alg not in data:
                data[alg] = {}
            
            if model == "swin_large":
                if alg == "ckpt":
                    cond = lambda obj: obj["ckpt"] == True
                else:
                    cond = lambda obj: obj['algorithm'] == alg and obj["ckpt"] == False
            elif model in ["transformer_lm_gpt3_small"]:
                cond = lambda obj : (obj["alg"] == alg)
            else:
                cond = lambda obj : (obj['algorithm'] == alg)
            print(alg)
            btimes = Util.load_data(ips_dir, "batch_size", "ips", cond)
            max_tpt = max(btimes.values())
            data[alg][gpu_cnt] = max_tpt
    return data

def normalize_tpt(data):
    base_tpt = data[None]
    for alg in data:
        if alg is None:
            continue
        for gpu_cnt in data[alg]:
            data[alg][gpu_cnt] /= base_tpt[gpu_cnt]
    return data

def plot_gpu_tpt(model, hardware, data):
    fig, ax = plt.subplots()
    fig.set_size_inches(4, 4)
    print(data)
    for alg in NET_TO_ALGS[model]:
        if alg is None:
            continue
        color = ALG_COLOR[alg]
        recs = plt.bar(range(len(labels)), (data[alg].values()), 
                label = alg, color = color)
        ax.set_xticks(range(len(labels)), labels)
    plt.grid(axis='y')
    plt.xlabel("Number of GPUs", size=16)
    # plt.legend()
    Util.set_tick_label_size([ax])
    plt.tight_layout()
    plt.savefig(f"graphs/implications/{model}_{hardware}_multi_gpu.{suffix}")

if __name__ == "__main__":
    hardware = "v100"
    networks = ["swin_large", "bert-large-cased", "transformer_lm_gpt3_small"]
    for network in networks:
        data = collect_max_tpt(hardware, network)
        normalized_data = normalize_tpt(data)
        plot_gpu_tpt(network, hardware, normalized_data)
