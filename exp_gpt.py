from typing import Callable
import numpy as np
from pathlib import Path
from cost_model import Model
from fitter import FitterPool, ModelFnPool
from util import Viewer, Util

suffix = "png"

class Experiment:
    def plot_helper(cond, mem_dir, ips_dir):
        mem = Util.load_data(mem_dir, "batch_size", "peak_mem", cond)
        for k in mem:
            mem[k] /= 1000000000
        mem_model,mem_score,alpha,beta = FitterPool.fit_leastsq_verbose(mem, ModelFnPool.linear)
        btime = Util.load_data(ips_dir, "batch_size", "batch_time", cond)
        btime_model,btime_score,gamma,delta = FitterPool.fit_leastsq_verbose(btime, ModelFnPool.linear)
        ips_model = lambda bsize: bsize / btime_model(bsize)
        # print("[predict mem] ", mem_model(np.array(list(mem.keys()))))
        return mem, btime, mem_model, btime_model, ips_model, alpha, beta, gamma, delta, mem_score, btime_score

    def do_plot(machine_tag, to_plot):
        algo = "GPT"
        mem_dir = "{}/{}/results/mem_results.json".format(algo,machine_tag)
        ips_dir = "{}/{}/results/speed_results.json".format(algo,machine_tag)
        result_dir = "graphs/{}/{}/".format(algo,machine_tag)
        if not Path(mem_dir).is_file() or not Path(ips_dir).is_file():
            print("Error: No experiment data found. Pease run expriment from scratch with --run-new for {}@{}".format(algo,machine_tag))
            return
        Path(result_dir).mkdir(parents=True, exist_ok=True)


        #print("-----------------Org-----------------")
        is_org = lambda obj : obj['alg'] == None and obj["network"] == "transformer_lm_gpt3_small"
        org_mem, org_btime, org_mem_model, org_btime_model, org_ips_model,\
        alpha, beta, gamma, delta, mem_score, btime_score = Experiment.plot_helper(is_org, mem_dir, ips_dir)
        print("-----------------{}@{} Params-----------------".format(algo,machine_tag))
        print ("{:<8} {:<10} {:<10} {:<10} {:<10} {:<12} {:<12}".format('Method','Alpha','Beta','Gamma','Delta','Mem R','Latency R'))
        print ("{:<8} {:<10g} {:<10g} {:<10g} {:<10g} {:<12g} {:<12g}".format('Org',\
        alpha,beta,gamma,delta,mem_score,btime_score))

        #print("-----------------Swap-----------------")
        is_swap = lambda obj : obj['alg'] == "swap" and obj['network'] == "transformer_lm_gpt3_small"
        swap_mem, swap_btime, swap_mem_model, swap_btime_model, swap_ips_model,\
        alpha, beta, gamma, delta, mem_score, btime_score = Experiment.plot_helper(is_swap, mem_dir, ips_dir)
        print ("{:<8} {:<10g} {:<10g} {:<10g} {:<10g} {:<12g} {:<12g}".format('Swap',alpha,beta,gamma,delta,mem_score,btime_score))

        #print("-----------------Ckpt-----------------")
        is_ckpt = lambda obj : obj['alg'] == "ckpt" and  obj['network'] == "transformer_lm_gpt3_small"
        ckpt_mem, ckpt_btime, ckpt_mem_model, ckpt_btime_model, ckpt_ips_model,\
        alpha, beta, gamma, delta, mem_score, btime_score = Experiment.plot_helper(is_ckpt, mem_dir, ips_dir) 
        print ("{:<8} {:<10g} {:<10g} {:<10g} {:<10g} {:<12g} {:<12g}".format('Ckpt',alpha,beta,gamma,delta,mem_score,btime_score))
        
        #print("-----------------Quantize-----------------")
        is_quantize = lambda obj : obj['alg'] == "L1" and obj['network'] == "transformer_lm_gpt3_small"
        quantize_mem, quantize_btime, quantize_mem_model, quantize_btime_model, quantize_ips_model,\
        alpha, beta, gamma, delta, mem_score, btime_score = Experiment.plot_helper(is_quantize, mem_dir, ips_dir) 
        print ("{:<8} {:<10g} {:<10g} {:<10g} {:<10g} {:<12g} {:<12g}".format('Quantize',alpha,beta,gamma,delta,mem_score,btime_score))

        if to_plot:
            import matplotlib
            matplotlib.rc('axes',edgecolor='silver')
            matplotlib.pyplot.style.use(['science','ieee'])
            fig, axes = matplotlib.pyplot.subplots(4, 1, sharex=True)
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
            matplotlib.pyplot.xlabel("Batch Size", size=18)  
            for ax in axes: 
                # ax.legend(loc="lower right")
                ax.tick_params(axis='x', labelsize=14)
                ax.tick_params(axis='y', labelsize=14)  
            # fig.text(-0.05, 0.5, 'Time (s)', va='center', rotation='vertical', size=18)
            matplotlib.pyplot.savefig(result_dir + "gpt3_batch_time.%s" % suffix, bbox_inches="tight")
            matplotlib.pyplot.close()

            # plot memory
            fig, ax = matplotlib.pyplot.subplots(1, 1)
            fig.set_size_inches(4, 4)
            Viewer.plot_fit(ax, "org", org_mem_model, np.array(list(org_mem.keys())), np.array(
                list(org_mem.values())), None, False)
            Viewer.plot_fit(ax, "swap", swap_mem_model, np.array(list(swap_mem.keys())), np.array(
                list(swap_mem.values())), None, False)
            Viewer.plot_fit(ax, "ckpt", ckpt_mem_model, np.array(list(ckpt_mem.keys())), np.array(
                list(ckpt_mem.values())), None, False) 
            Viewer.plot_fit(ax, "quantize", quantize_mem_model, np.array(list(quantize_mem.keys())), np.array(
                list(quantize_mem.values())), None, False)
            matplotlib.pyplot.ylabel("Memory (GB)", size=18)
            matplotlib.pyplot.xlabel("Batch Size", size=18)
            # matplotlib.pyplot.legend(prop={'size': 14})    
            matplotlib.pyplot.yticks(fontsize=15)
            matplotlib.pyplot.xticks(fontsize=15) 
            matplotlib.pyplot.savefig(result_dir + "gpt3_mem.%s" % suffix, bbox_inches="tight")
            matplotlib.pyplot.close()

            fig, ax = matplotlib.pyplot.subplots(1, 1)
            fig.set_size_inches(4, 4)
            Viewer.plot_fit(ax, "org", org_ips_model, np.array(list(org_btime.keys())), np.array(
                [bsize / org_btime[bsize] for bsize in org_btime]), None, False)
            Viewer.plot_fit(ax, "swap", swap_ips_model, np.array(list(swap_btime.keys())), np.array(
                [bsize / swap_btime[bsize] for bsize in swap_btime]), None, False)
            Viewer.plot_fit(ax, "ckpt", ckpt_ips_model, np.array(list(ckpt_btime.keys())), np.array(
                [bsize / ckpt_btime[bsize] for bsize in ckpt_btime]), None, False) 
            Viewer.plot_fit(ax, "quantize", quantize_ips_model, np.array(list(quantize_btime.keys())), np.array(
                [bsize / quantize_btime[bsize] for bsize in quantize_btime]), None, False) 
            matplotlib.pyplot.ylabel("Throughput (record/s)", size=18)
            matplotlib.pyplot.xlabel("Batch Size", size=18)
            # matplotlib.pyplot.legend(prop={'size': 14})    
            matplotlib.pyplot.yticks(fontsize=15)
            matplotlib.pyplot.xticks(fontsize=15)
            matplotlib.pyplot.savefig(result_dir + "gpt3_ips.%s" % suffix, bbox_inches="tight")
            matplotlib.pyplot.close()

if __name__=="__main__":
    Experiment.do_plot("v100",True)
