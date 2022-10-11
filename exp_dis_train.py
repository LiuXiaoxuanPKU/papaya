from util import Util, colors, markers
import matplotlib.pyplot as plt

MODEL_TO_DIR = {
    "bert" : "text_classification_fp16",
    "swin" : "Swin-Transformer",
    "GPT" : "GPT"
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
def collect_max_tpt(hardware, model):
    data = {}
    for gpu_cnt in [1, 2, 4]:
        ips_dir = f"{MODEL_TO_DIR[model]}/{GPU_CNT_TO_NUM[gpu_cnt]}{hardware}/results/speed_results.json"
        for alg in ALGOS:
            if ALG_TO_NAME[alg] not in data:
                data[ALG_TO_NAME[alg]] = {}
            
            if model == "swin":
                if alg == "ckpt":
                    cond = lambda obj: obj["ckpt"] == True
                else:
                    cond = lambda obj: obj['algorithm'] == alg and obj["ckpt"] == False
            elif model == "GPT":
                cond = lambda obj : (obj["alg"] == alg)
            else:
                cond = lambda obj : (obj['algorithm'] == alg)
            btimes = Util.load_data(ips_dir, "batch_size", "ips", cond)
            print(alg, btimes)
            max_tpt = max(btimes.values())
            data[ALG_TO_NAME[alg]][gpu_cnt] = max_tpt
    return data

def plot_gpu_tpt(model, hardware, data):
    fig, ax = plt.subplots()
    fig.set_size_inches(4, 4)
    for alg in ALG_TO_NAME.values():
        marker = markers[ALGNAME_NORMALIZE[alg]]
        color = colors[ALGNAME_NORMALIZE[alg]]
        plt.plot(["1", "2", "4"], (data[alg].values()), 
                f"-{marker}", label = alg, color = color, markersize=10)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)  
    plt.xlabel("number of GPUs", size=16)
    plt.ylabel("max throughput (records/s)", size=16)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"graphs/implications/{model}_{hardware}_gpu.{suffix}")

if __name__ == "__main__":
    hardware = "v100"
    model = "GPT"
    data = collect_max_tpt(hardware, model)
    print(data)
    plot_gpu_tpt(model, hardware, data)
