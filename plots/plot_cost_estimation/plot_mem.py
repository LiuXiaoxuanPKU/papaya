import sys
sys.path.append('.')
from plots.plot_util import ALG_COLOR, ALG_MAP, ALG_MARKER

import numpy as np
from pathlib import Path
from fitter import FitterPool, ModelFnPool
from util import Viewer, Util
import random
import matplotlib.pyplot as plt

random.seed(0)

suffix = "pdf"

NET_TO_FOLER = {
    "resnet50" : "resnet",
    "wide_resnet50_2" : "resnet",
    "resnet152" : "resnet",
    "bert-large-cased" : "text_classification_fp16",
    "swin_large" : "Swin-Transformer",
    "transformer_lm_gpt3_small" : "GPT"
}

GB_NORMALIZE = {
    "resnet50" : 1e3,
    "wide_resnet50_2" : 1e3,
    "resnet152" : 1e3,
    "bert-large-cased" : 1e3,
    "swin_large" : 1e9,
    "transformer_lm_gpt3_small" : 1e9
}

NET_TO_ALGS = {
    # resnet should have ["None", "L1", "swap", "dtr", "L4bit-swap"]
    "resnet50" : ["None", "L1", "swap"],
    "wide_resnet50_2" : ["None", "L1"],
    # bert should have [None, "L1", "swap", "L14bit-swap"]
    "bert-large-cased" : [None, "L1", "swap"],
    # swin should have [None, "L1", "swap", "L14bit-swap", "ckpt"]
    "swin_large" : [None, "L1", "swap", "ckpt"],
    "transformer_lm_gpt3_small" : [None, "L1", "ckpt"]
}

def cond_gen(network, alg):
    if network in ["resnet50", "wide_resnet50_2", "resnet152"]:
        return lambda obj : obj["algorithm"] == alg and obj["network"] == network
    elif network in ["bert-large-cased"]:
        return lambda obj : obj["algorithm"] == alg and obj["network"] == network
    elif network in ["swin_large"]:
        if alg == "ckpt":
            return lambda obj : obj["fp16"] == 'O1' and obj["network"] == network and obj["ckpt"] == True
        else:
            return lambda obj : obj["fp16"] == 'O1' and obj["network"] == network \
                                and obj["algorithm"] == alg and obj["ckpt"] == False
    elif network in ["transformer_lm_gpt3_small"]:
        return lambda obj : "alg" in obj and obj["alg"] == alg and obj["network"] == network
    else:
        print(f"[Error] Unsupport network {network}, alg {alg}")
        exit(0)

def plot(hardware, network):
    mem_dir = f"benchmarks/{NET_TO_FOLER[network]}/{hardware}/results/mem_results.json"
    result_file = f"graphs/cost_estimation/{network}.{suffix}"
    if not Path(mem_dir).is_file():
        print("Error: No experiment data found. Pease run expriment from \
            scratch with --run-new for {}@{}".format(network,hardware))
        return
    
    algs = NET_TO_ALGS[network]
    fig, ax = plt.subplots(1, 1)
    for alg in algs:
        is_alg = cond_gen(network, alg)
        alg_mem = Util.load_data(mem_dir, "batch_size", "peak", is_alg)   
        for k in alg_mem:
            alg_mem[k] /= GB_NORMALIZE[network]
        xs, ys = list(alg_mem.keys()), list(alg_mem.values())
        sample_cnt = 10
        xs, ys = Util.sample_data(xs, sample_cnt), Util.sample_data(ys, sample_cnt)

        if alg is None or alg == "None":
             alg = "exact"
        ax.scatter(xs, ys, label=alg, 
                    marker=ALG_MARKER[alg], s=100, c=ALG_COLOR[alg])
        
        sample_per = 0.2
        alg_sample = Util.sample_dict(alg_mem, sample_per)
        mem_model, mem_score, alpha, beta = FitterPool.fit_leastsq_verbose(alg_sample, ModelFnPool.linear)
        xs = np.array(xs)
        ax.plot(xs, mem_model(xs), c=ALG_COLOR[alg], linewidth=3)

    plt.grid()
    ax.set_xlabel("Batch Size")
    Util.set_tick_label_size([ax])
    plt.tight_layout()
    fig.savefig(result_file)




if __name__ == "__main__": 
    hardware = 'v100'
    networks = ['resnet50', 'wide_resnet50_2', 'bert-large-cased', 'swin_large', 'transformer_lm_gpt3_small']
    for net in networks:
        plot(hardware, net)
