import sys
sys.path.append('.')
from plots.plot_util import ALG_COLOR, ALG_MAP, ALG_MARKER, NET_TO_ALGS, NET_TO_FOLER
from util import Util
from fitter import FitterPool, ModelFnPool

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

suffix = "pdf"

ratio = 0.8
ALG_HW = {
    "resnet50" : 16000 * ratio,
    "wide_resnet50_2" : 16000 * ratio,
    # bert should have [None, "L1", "swap", "L4bit-swap"]
    "bert-large-cased" : 16000 * ratio,
    "swin_large" : 16 * 1e9 * ratio,
    # gpt should have [None, "L1", "swap", "L4bit-swap"]
    "transformer_lm_gpt3_small" : 16 * 1e9 * ratio
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

def get_pscore_param(mem_dir, speed_dir, cond):
    mem_data = Util.load_data(mem_dir, "batch_size", "peak", cond)
    speed_data =  Util.load_data(speed_dir, "batch_size", "batch_time", cond)
    print(mem_dir, mem_data)
    print(mem_dir, speed_data)
    max_tpt = max([x / speed_data[x] for x in speed_data])

    sample_per = 0.25    
    mem_sample = Util.sample_dict(mem_data, sample_per)
    mem_model, mem_score, mem_alpha, mem_beta = FitterPool.fit_leastsq_verbose(mem_sample, ModelFnPool.linear)
    speed_sample = Util.sample_dict(speed_data, sample_per)
    speed_model, speed_score, speed_alpha, speed_beta = FitterPool.fit_leastsq_verbose(speed_sample, ModelFnPool.linear)

    return mem_alpha, mem_beta, speed_alpha, speed_beta, max_tpt

def get_p_point(network, org_mem_alpha, org_mem_beta, org_speed_alpha, org_speed_beta):
    HW_MEM = ALG_HW[network]
    print("p score,", org_mem_beta, org_speed_beta)
    return (HW_MEM - org_mem_beta) / org_speed_beta

def get_p_score(org_mem_alpha, org_mem_beta, org_speed_alpha, org_speed_beta,
                mem_alpha, mem_beta, speed_alpha, speed_beta):
    return (org_mem_alpha - mem_alpha) / (speed_alpha - org_speed_alpha)

def plot(hardware, network):
    speed_dir = f"benchmarks/{NET_TO_FOLER[network]}/{hardware}/results/speed_results.json"
    mem_dir = f"benchmarks/{NET_TO_FOLER[network]}/{hardware}/results/mem_results.json"
    result_file = f"graphs/cost_estimation/papaya_score/{network}.{suffix}"
    if not Path(speed_dir).is_file():
        print(f"Error: No experiment data found for {network} on {hardware}")
        return
    org_alg = None
    is_org_alg = cond_gen(network, org_alg)
    org_mem_alpha, org_mem_beta, org_speed_alpha, org_speed_beta, org_max_tpt \
                = get_pscore_param(mem_dir, speed_dir, is_org_alg)
    p_point = get_p_point(network, org_mem_alpha, org_mem_beta, org_speed_alpha, org_speed_beta)

    p_scores = {}
    max_tpts = {}
    for alg in NET_TO_ALGS[network]:
        if alg is None:
            continue
        print(f"========={alg}===========")
        is_alg = cond_gen(network, alg)
        mem_alpha, mem_beta, speed_alpha, speed_beta, max_tpt \
                = get_pscore_param(mem_dir, speed_dir, is_alg)
        p_scores[alg] = get_p_score(org_mem_alpha, org_mem_beta, org_speed_alpha, org_speed_beta,
                                    mem_alpha, mem_beta, speed_alpha, speed_beta)
        max_tpts[alg] = max_tpt
    
    vis_method = 2
    fig, ax_pscore = plt.subplots(1, 1)
    plt.grid()
    print(max_tpts)
    print(p_point, p_scores)
    # normalize p score
    for k in p_scores:
        p_scores[k] /= p_point
    # normalize throughput
    for k in max_tpts:
        max_tpts[k] /= org_max_tpt

    if vis_method == 1:
        fig.set_size_inches(4, 6)
        ax_pscore.scatter(range(len(p_scores)), list(p_scores.values()), color='green', marker='x', s=300)
        ax_tpt = ax_pscore.twinx()
        ax_tpt.scatter(range(len(p_scores)), list(max_tpts.values()), color='blue', s=150)
        ax_pscore.set_xticks(range(len(p_scores)), [ALG_MAP[alg] for alg in p_scores.keys()], rotation=90)
        ax_tpt.axhline(y=org_max_tpt, color="red", lw=3)
    
        Util.set_tick_label_size([ax_pscore, ax_tpt], 16)
    elif vis_method == 2:
        fig.set_size_inches(2, 2)
        for alg in p_scores:
            ax_pscore.scatter(np.array([p_scores[alg]]), np.array([max_tpts[alg]]), \
                marker=ALG_MARKER[alg], color=ALG_COLOR[alg], s=100)
        # ax_pscore.set_yticks([])
        ax_pscore.axhline(y=1, color="blue", lw=3)
        ax_pscore.axvline(x=1, color="blue", lw=3)
        ax_pscore.set_ylim(0,1.2)
        ax_pscore.set_xlabel("Normalized\nP Score")
        Util.set_tick_label_size([ax_pscore], 8)

    plt.tight_layout()
    fig.savefig(result_file)

    



if __name__ == "__main__": 
    hardware = 'v100'
    networks = ['resnet50', 'wide_resnet50_2', 'bert-large-cased', 'swin_large', 'transformer_lm_gpt3_small']
    networks = ['bert-large-cased']
    for net in networks:
        plot(hardware, net)