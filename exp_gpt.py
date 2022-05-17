from typing import Callable
import numpy as np

from cost_model import Model
from fitter import FitterPool, ModelFnPool
from util import Viewer, Util
import matplotlib.pyplot as plt

def plot_helper(cond, mem_dir, ips_dir):
    # mem = Util.load_data(mem_dir, "batch_size", "peak", cond)
    # mem_model = FitterPool.fit_leastsq(mem, ModelFnPool.linear)
    mem, mem_model = None, None
    btime = Util.load_data(ips_dir, "batch_size", "batch_time", cond)
    btime_model = FitterPool.fit_leastsq(btime, ModelFnPool.linear)
    ips_model = lambda bsize: bsize / btime_model(bsize)
    # print("[predict mem] ", mem_model(np.array(list(mem.keys()))))
    return mem, btime, mem_model, btime_model, ips_model

if __name__ == "__main__":
    mem_dir = "GPT/results/mem_results.json"
    ips_dir = "GPT/results/speed_results.json"
    result_dir = "graphs/GPT/"
    suffix = "pdf"

    is_org = lambda obj : obj['alg'] == None and obj["network"] == "transformer_lm_gpt3_small"
    org_mem, org_btime, org_mem_model, org_btime_model, org_ips_model = \
        plot_helper(is_org, mem_dir, ips_dir)

    is_swap = lambda obj : obj['alg'] == "swap" and obj['network'] == "transformer_lm_gpt3_small"
    swap_mem, swap_btime, swap_mem_model, swap_btime_model, swap_ips_model = \
        plot_helper(is_swap, mem_dir, ips_dir)

    is_ckpt = lambda obj : obj['alg'] == "ckpt" and  obj['network'] == "transformer_lm_gpt3_small"
    ckpt_mem, ckpt_btime, ckpt_mem_model, ckpt_btime_model, ckpt_ips_model = \
         plot_helper(is_ckpt, mem_dir, ips_dir) 
    
    is_quantize = lambda obj : obj['alg'] == "L1" and obj['network'] == "transformer_lm_gpt3_small"
    quantize_mem, quantize_btime, quantize_mem_model, quantize_btime_model, quantize_ips_model = \
         plot_helper(is_quantize, mem_dir, ips_dir) 


    fig, axes = plt.subplots(4, 1, sharex=True)
    fig.set_size_inches(4, 6)
    # plot batch time
    Viewer.plot_fit(axes[0], "org", org_btime_model, np.array(list(org_btime.keys())), np.array(
        list(org_btime.values())), None, False)
    Viewer.plot_fit(axes[1], "swap", swap_btime_model, np.array(list(swap_btime.keys())), np.array(
        list(swap_btime.values())), None, False)
    Viewer.plot_fit(axes[2], "ckpt", ckpt_btime_model, np.array(list(ckpt_btime.keys())), np.array(
        list(ckpt_btime.values())), None, False) 
    Viewer.plot_fit(axes[3], "quantize", quantize_btime_model, np.array(list(quantize_btime.keys())), np.array(
        list(quantize_btime.values())), None, False) 
    plt.xlabel("Batch size", size=16)  
    for ax in axes: 
        # ax.legend(loc="lower right")
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)  
    # fig.text(-0.05, 0.5, 'Time (s)', va='center', rotation='vertical', size=16)
    plt.savefig(result_dir + "gpt3_batch_time.%s" % suffix, bbox_inches="tight")
    plt.close()

    # # plot memory
    # Viewer.plot_fit("org", org_mem_model, np.array(list(org_mem.keys())), np.array(
    #     list(org_mem.values())), None, False)
    # Viewer.plot_fit("swap", swap_mem_model, np.array(list(swap_mem.keys())), np.array(
    #     list(swap_mem.values())), None, False)
    # Viewer.plot_fit("ckpt", ckpt_mem_model, np.array(list(ckpt_mem.keys())), np.array(
    #     list(ckpt_mem.values())), None, False) 
    # Viewer.plot_fit("quantize", quantize_mem_model, np.array(list(quantize_mem.keys())), np.array(
    #     list(quantize_mem.values())), None, False) 
    # plt.savefig(result_dir + "mem.png")
    # plt.close()

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(4, 4)
    Viewer.plot_fit(ax, "org", org_ips_model, np.array(list(org_btime.keys())), np.array(
        [bsize / org_btime[bsize] for bsize in org_btime]), None, False)
    Viewer.plot_fit(ax, "swap", swap_ips_model, np.array(list(swap_btime.keys())), np.array(
        [bsize / swap_btime[bsize] for bsize in swap_btime]), None, False)
    Viewer.plot_fit(ax, "ckpt", ckpt_ips_model, np.array(list(ckpt_btime.keys())), np.array(
        [bsize / ckpt_btime[bsize] for bsize in ckpt_btime]), None, False) 
    Viewer.plot_fit(ax, "quantize", quantize_ips_model, np.array(list(quantize_btime.keys())), np.array(
        [bsize / quantize_btime[bsize] for bsize in quantize_btime]), None, False) 
    plt.ylabel("Throughput (record/s)", size=16)
    plt.xlabel("Batch size", size=16)
    # plt.legend(prop={'size': 14})    
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.savefig(result_dir + "gpt3_ips.%s" % suffix, bbox_inches="tight")
    plt.close()
    
