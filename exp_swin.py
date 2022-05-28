from typing import Callable
import numpy as np

from cost_model import Model
from fitter import FitterPool, ModelFnPool
from util import Viewer, Util
import matplotlib.pyplot as plt

import random
random.seed(0)

def sample_dict(dic, percentenge):
    sample_data = {}
    sample_num = max(2, int(len(dic) * percentenge))
    remove_num = 3
    if len(dic) - sample_num < remove_num:
        return dic
    sample_keys = random.sample(list(dic.keys())[remove_num:], sample_num)
    for k in sample_keys:
        sample_data[k] = dic[k]
    return sample_data

def plot_helper(cond, mem_dir, ips_dir):
    mem = Util.load_data(mem_dir, "batch_size", "peak", cond)
    for k in mem:
        mem[k] /= 1000000000
    btime = Util.load_data(ips_dir, "batch_size", "batch_time", cond)
    # use only 20% data to fit the model
    mem_sample= sample_dict(mem, 0.2)
    btime_sample= sample_dict(btime, 0.2)
    mem_model = FitterPool.fit_leastsq(mem_sample, ModelFnPool.linear)
    btime_model = FitterPool.fit_leastsq(btime_sample, ModelFnPool.linear)
    ips_model = lambda bsize: bsize / btime_model(bsize)
    print("[predict mem] ", mem_model(np.array(list(mem.keys()))))
    return mem, btime, mem_model, btime_model, ips_model

if __name__ == "__main__":
    mem_dir = "Swin-Transformer/results/mem_results.json"
    ips_dir = "Swin-Transformer/results/speed_results.json"
    result_dir = "graphs/Swin-Transformer/"
    suffix = "pdf"

    print("------------------Org---------------")
    is_org = lambda obj : obj['algorithm'] == None and obj['fp16'] == "O1" and obj['ckpt'] == False 
    org_mem, org_btime, org_mem_model, org_btime_model, org_ips_model = \
        plot_helper(is_org, mem_dir, ips_dir)
    
    print("------------------Swap---------------")
    is_swap = lambda obj : obj['algorithm'] == "swap" and obj['fp16'] == "O1"
    swap_mem, swap_btime, swap_mem_model, swap_btime_model, swap_ips_model = \
        plot_helper(is_swap, mem_dir, ips_dir)

    print("------------------Ckpt---------------")
    is_ckpt = lambda obj : obj['ckpt'] == True and obj['fp16'] == "O1"
    ckpt_mem, ckpt_btime, ckpt_mem_model, ckpt_btime_model, ckpt_ips_model = \
         plot_helper(is_ckpt, mem_dir, ips_dir) 

    print("------------------Quantize---------------")
    is_quantize = lambda obj : obj['algorithm'] == "L1" and obj['fp16'] == "O1"
    quantize_mem, quantize_btime, quantize_mem_model, quantize_btime_model, quantize_ips_model = \
         plot_helper(is_quantize, mem_dir, ips_dir) 


    fig, axes = plt.subplots(4, 1, sharex=True)
    fig.set_size_inches(4, 6)
    # plot batch time
    Viewer.plot_fit(axes[0], "org", org_btime_model, np.array(list(org_btime.keys())), np.array(
        list(org_btime.values())), None, False)
    Viewer.plot_fit(axes[1],"swap", swap_btime_model, np.array(list(swap_btime.keys())), np.array(
        list(swap_btime.values())), None, False)
    Viewer.plot_fit(axes[2],"ckpt", ckpt_btime_model, np.array(list(ckpt_btime.keys())), np.array(
        list(ckpt_btime.values())), None, False) 
    Viewer.plot_fit(axes[3],"quantize", quantize_btime_model, np.array(list(quantize_btime.keys())), np.array(
        list(quantize_btime.values())), None, False) 
    
    plt.xlabel("Batch size", size=16)  
    for ax in axes: 
        # ax.legend(loc="lower right")
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        
    # fig.text(-0.02, 0.5, 'Time (s)', va='center', rotation='vertical', size=16)
    plt.savefig(result_dir + "swin_batch_time.%s" % suffix, bbox_inches="tight")
    plt.close()

    # plot memory
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(4, 4)
    Viewer.plot_fit(ax, "org", org_mem_model, np.array(list(org_mem.keys())), np.array(
        list(org_mem.values())), None, False)
    Viewer.plot_fit(ax, "swap", swap_mem_model, np.array(list(swap_mem.keys())), np.array(
        list(swap_mem.values())), None, False)
    Viewer.plot_fit(ax, "ckpt", ckpt_mem_model, np.array(list(ckpt_mem.keys())), np.array(
        list(ckpt_mem.values())), None, False) 
    Viewer.plot_fit(ax, "quantize", quantize_mem_model, np.array(list(quantize_mem.keys())), np.array(
        list(quantize_mem.values())), None, False) 
    plt.ylabel("Memory (GB)", size=16)
    plt.xlabel("Batch size", size=16)
    # plt.legend(prop={'size': 14})    
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.savefig(result_dir + "swin_mem.%s" % suffix, bbox_inches="tight")
    plt.close()

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
    plt.ylabel("Throughput (image/s)", size=16)
    plt.xlabel("Batch size", size=16)
    # plt.legend(prop={'size': 14})    
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.savefig(result_dir + "swin_ips.%s" % suffix, bbox_inches="tight")
    plt.close()
    
