from util import Util, colors, markers
import matplotlib.pyplot as plt
import numpy as np

MODEL_TO_DIR = {
    "bert" : "text_classification_fp16",
    "swin" : "Swin-Transformer"
}

GPU_CNT_TO_NUM = {
    1 : "",
    2 : 2,
    4 : 4
}

ALGOS = [None, "ckpt", "L1"]

ALG_TO_NAME = {
    None : "exact",
    "ckpt" : "checkpoint",
    "L1" : "quantize"
}

ALGNAME_NORMALIZE = {
    "exact": "org",
    "checkpoint": "ckpt",
    "quantize": "quantize"
}

suffix = "pdf"
def collect_max_tpt(model):
    data = {}
    gpu_cnt = 1
    for hardware in ['k80', 't4', 'v100']:
        if hardware == "v100-fp32":
            ips_dir = "text_classification/results/speed_results.json"
        else:
            ips_dir = f"{MODEL_TO_DIR[model]}/{GPU_CNT_TO_NUM[gpu_cnt]}{hardware}/results/speed_results.json"
        for alg in ALGOS:
            if ALG_TO_NAME[alg] not in data:
                data[ALG_TO_NAME[alg]] = {}
            
            if model == "swin":
                if alg == "ckpt":
                    cond = lambda obj: obj["ckpt"] == True
                else:
                    cond = lambda obj: obj['algorithm'] == alg and obj["ckpt"] == False
            else:
                cond = lambda obj : obj['algorithm'] == alg
            btimes = Util.load_data(ips_dir, "batch_size", "ips", cond)
            print(alg, btimes)
            if len(btimes) == 0:
                continue
            max_tpt = max(btimes.values())
            data[ALG_TO_NAME[alg]][hardware] = max_tpt
    return data

def plot_gpu_tpt(model, data):
    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(8, 4)
    for alg in ALG_TO_NAME.values():
        # if alg == "checkpoint":
        #     continue
        marker = markers[ALGNAME_NORMALIZE[alg]]
        color = colors[ALGNAME_NORMALIZE[alg]]
        normalized = np.array(list(data[alg].values())) / \
                        np.array(list(data['exact'].values()))
        axes[0].plot(["k80", "t4", "v100"], (data[alg].values()), 
                f"-{marker}", label = alg, color = color, markersize=10)
        width = 0.35
        x = np.arange(len(list(data[alg].values())))
        if alg == "checkpoint":
            rects1 = axes[1].bar(x - width/2, normalized, width, label=alg)
        elif alg == "quantize":
            rects2 = axes[1].bar(x + width/2, normalized, width, label=alg)
    axes[1].set_xticks(x, ["k80", "t4", "v100"])
    axes[1].set_ylim(0, 1.1)
    for ax in axes:
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)
        ax.legend() 
        # plt.xlabel("GPU generation", size=16)
    axes[0].set_title("max throughput", size=16)
    axes[1].set_title("max throughput ratio", size=16)
    plt.tight_layout()
    plt.savefig(f"graphs/implications/{model}_hardware.{suffix}")

if __name__ == "__main__":
    model = "bert"
    data = collect_max_tpt(model)
    print(data)
    plot_gpu_tpt(model, data)
