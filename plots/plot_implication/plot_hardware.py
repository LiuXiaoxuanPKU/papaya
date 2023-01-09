import matplotlib.pyplot as plt
import sys
sys.path.append('.')
from plots.plot_util import ALG_COLOR, ALG_MAP, ALG_MARKER, NET_TO_ALGS, NET_TO_FOLDER
from util import Util
import numpy as np


GPU_CNT_TO_NUM = {
    1 : "",
    2 : 2,
    4 : 4
}


suffix = "pdf"
def collect_max_tpt(model):
    data = {}
    gpu_cnt = 1
    hardwares = ["t4", "v100"] if model in ["transformer_lm_gpt3_small"] else ["k80", "t4", "v100"]
    for hardware in hardwares:
        if hardware == "v100-fp32":
            ips_dir = "benchmarks/text_classification/results/speed_results.json"
        else:
            ips_dir = f"benchmarks/{NET_TO_FOLDER[model]}/{GPU_CNT_TO_NUM[gpu_cnt]}{hardware}/results/speed_results.json"
        for alg in NET_TO_ALGS[model]:
            if alg not in data:
                data[alg] = {}
            
            if model == "swin":
                if alg == "ckpt":
                    cond = lambda obj: obj["ckpt"] == True
                else:
                    cond = lambda obj: obj['algorithm'] == alg and obj["ckpt"] == False
            elif model == "transformer_lm_gpt3_small":
                cond = lambda obj: obj["alg"] == alg
            else:
                cond = lambda obj : obj['algorithm'] == alg
            btimes = Util.load_data(ips_dir, "batch_size", "ips", cond)
            print(alg, btimes)
            if len(btimes) == 0:
                continue
            max_tpt = max(btimes.values())
            data[alg][hardware] = max_tpt
    return data

def plot_gpu_tpt(model, data):
    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(8, 4)
    algs = [x for x in NET_TO_ALGS[model] if x not in ["dtr"]]
    width = 0.1
    offset = -(len(algs) - 1) * width / 2
    hardwares = ["t4", "v100"] if model in ["transformer_lm_gpt3_small"] else ["k80", "t4", "v100"]
    for i, alg in enumerate(algs):
        if alg in ["dtr"]:
            continue
        if alg is None:
            alg_name = "exact"
        else:
            alg_name = alg
        marker = ALG_MARKER[alg_name]
        color = ALG_COLOR[alg_name]
        normalized = np.array(list(data[alg].values())) / \
                        np.array(list(data[None].values()))
        axes[0].plot(hardwares, (data[alg].values()), 
                f"-{marker}", label = alg, color = color, markersize=10)

        if alg is None:
            continue
        if model in ["transformer_lm_gpt3_small"]:
            axes[1].bar([i-width/2, i+width/2], normalized, width, label=alg)
        else:
            axes[1].bar([i-width, i, i+width], normalized, width, label=alg)
        offset += width
          
    axes[1].set_xticks(range(1, len(algs)), [ALG_MAP[alg] for alg in algs if alg is not None])
    axes[1].set_ylim(0, 1.1)
    axes[0].set_title("Maximum throughput", size=16)
    axes[1].set_title("Maximum throughput ratio", size=16)
    Util.set_tick_label_size(axes, 14)
    plt.tight_layout()
    plt.savefig(f"graphs/implications/{model}_hardware.{suffix}")

if __name__ == "__main__":
    networs = ["resnet50", "bert-large-cased", "transformer_lm_gpt3_small"]
    for network in networs:
        data = collect_max_tpt(network)
        print(data)
        plot_gpu_tpt(network, data)
