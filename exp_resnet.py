from typing import Callable
import numpy as np
from pathlib import Path
from cost_model import Model
from fitter import FitterPool, ModelFnPool
from util import Viewer, Util
import random,os

random.seed(0)

suffix = "pdf"
algo = "resnet"

class Experiment:
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

    def plot_helper(cond, mem_dir, ips_dir, offset = None):
        # mem = Util.load_data(mem_dir, "batch_size", "peak", cond)
        # for k in mem:
            # mem[k] /= 1000000000
        btime = Util.load_data(ips_dir, "batch_size", "batch_time", cond)
        # use only 20% data to fit the model
        # mem_sample= Experiment.sample_dict(mem, 0.4)
        btime_sample= Experiment.sample_dict(btime, 0.4)
        # mem_model,mem_score,alpha,beta = FitterPool.fit_leastsq_verbose(mem_sample, ModelFnPool.linear)
        mem, mem_model, alpha, beta, mem_score = None, None, None, None, None
        
        btime_model,btime_score,gamma,delta = FitterPool.fit_leastsq_verbose(btime_sample, ModelFnPool.linear)
        while offset is None and delta<0:
            retry += 1
            btime_sample= Experiment.sample_dict(btime, 0.4)
            btime_model,btime_score,gamma,delta = FitterPool.fit_leastsq_verbose(btime_sample, ModelFnPool.linear)
            if retry>3: break
        if delta<0 and offset: btime_model,btime_score,gamma,delta = FitterPool.fit_leastsq_verbose_offset(btime_sample, ModelFnPool.linear,offset)
        ips_model = lambda bsize: bsize / btime_model(bsize)
        # print("[predict mem] ", mem_model(np.array(list(mem.keys()))))
        return mem, btime, mem_model, btime_model, ips_model, alpha, beta, gamma, delta, mem_score, btime_score

    def do_plot(machine_tag,to_plot):
        algo = "ResNet"
        mem_dir = "{}/{}/results/mem_results.json".format(algo,machine_tag)
        ips_dir = "{}/{}/results/speed_results.json".format(algo,machine_tag)
        result_dir = "graphs/{}/{}/".format(algo,machine_tag)
        if not Path(mem_dir).is_file() or not Path(ips_dir).is_file():
            print("Error: No experiment data found. Pease run expriment from scratch with --run-new for {}@{}".format(algo,machine_tag))
            return
        Path(result_dir).mkdir(parents=True, exist_ok=True)

        #print("------------------Org---------------")
        is_org = lambda obj : obj['algorithm'] == "None" and obj['network'] == "resnet152"
        org_mem, org_btime, org_mem_model, org_btime_model, org_ips_model,\
        alpha, beta, gamma, delta, mem_score, btime_score = Experiment.plot_helper(is_org, mem_dir, ips_dir)
        offset = delta
        print("-----------------{}@{} Params-----------------".format(algo,machine_tag))
        print ("{:<8} {:<10} {:<10} {:<10} {:<10} {:<12} {:<12}".\
        format('Method','Alpha','Beta','Gamma','Delta','Mem R','Latency R'))
        # print ("{:<8} {:<10g} {:<10g} {:<10g} {:<10g} {:<12g} {:<12g}".format('Org',alpha,beta,gamma,delta,mem_score,btime_score))
        
        # #print("------------------Swap---------------")
        # is_swap = lambda obj : obj['algorithm'] == "swap" and  obj['network'] == "resnet50"
        # swap_mem, swap_btime, swap_mem_model, swap_btime_model, swap_ips_model,\
        # alpha, beta, gamma, delta, mem_score, btime_score = Experiment.plot_helper(is_swap, mem_dir, ips_dir, offset)
        # print ("{:<8} {:<10g} {:<10g} {:<10g} {:<10g} {:<12g} {:<12g}".format('Swap',alpha,beta,gamma,delta,mem_score,btime_score))

        # #print("------------------Ckpt---------------")
        # is_ckpt = lambda obj : obj['ckpt'] == True and obj['fp16'] == "O1"
        # ckpt_mem, ckpt_btime, ckpt_mem_model, ckpt_btime_model, ckpt_ips_model,\
        # alpha, beta, gamma, delta, mem_score, btime_score = Experiment.plot_helper(is_ckpt, mem_dir, ips_dir, offset) 
        # print ("{:<8} {:<10g} {:<10g} {:<10g} {:<10g} {:<12g} {:<12g}".format('Ckpt',alpha,beta,gamma,delta,mem_score,btime_score))

        #print("------------------Quantize---------------")
        is_quantize = lambda obj : obj['algorithm'] == "L1" and  obj['network'] == "resnet152"
        quantize_mem, quantize_btime, quantize_mem_model, quantize_btime_model, quantize_ips_model,\
        alpha, beta, gamma, delta, mem_score, btime_score = Experiment.plot_helper(is_quantize, mem_dir, ips_dir, offset) 
        # print ("{:<8} {:<10g} {:<10g} {:<10g} {:<10g} {:<12g} {:<12g}".format('Quantize',alpha,beta,gamma,delta,mem_score,btime_score))

        if to_plot:
            import matplotlib
            # matplotlib.rc('axes',edgecolor='silver')
            import matplotlib.pyplot as plt
            # plt.style.use(['grid'])
            fig, axes = plt.subplots(2, 1, sharex=True)
            fig.set_size_inches(4, 6)
            # plot batch time
            Viewer.plot_fit(axes[0], "org", org_btime_model, np.array(list(org_btime.keys())), np.array(
                list(org_btime.values())), None, False)
            # Viewer.plot_fit(axes[1],"swap", swap_btime_model, np.array(list(swap_btime.keys())), np.array(
            #     list(swap_btime.values())), None, False)
            # Viewer.plot_fit(axes[2],"ckpt", ckpt_btime_model, np.array(list(ckpt_btime.keys())), np.array(
            #     list(ckpt_btime.values())), None, False) 
            Viewer.plot_fit(axes[1],"quantize", quantize_btime_model, np.array(list(quantize_btime.keys())), np.array(
                list(quantize_btime.values())), None, False) 
            
            plt.xlabel("Batch Size", size=22)  
            for ax in axes: 
                # ax.legend(loc="lower right")
                ax.tick_params(axis='x', labelsize=18)
                ax.tick_params(axis='y', labelsize=18)
                
            # fig.text(-0.02, 0.5, 'Time (s)', va='center', rotation='vertical', size=22)
            plt.savefig(result_dir + "resnet152_batch_time.%s" % suffix, bbox_inches="tight")
            plt.close()

            # # plot memory
            # fig, ax = plt.subplots(1, 1)
            # fig.set_size_inches(4, 4)
            # Viewer.plot_fit(ax, "org", org_mem_model, np.array(list(org_mem.keys())), np.array(
            #     list(org_mem.values())), None, False)
            # Viewer.plot_fit(ax, "swap", swap_mem_model, np.array(list(swap_mem.keys())), np.array(
            #     list(swap_mem.values())), None, False)
            # Viewer.plot_fit(ax, "ckpt", ckpt_mem_model, np.array(list(ckpt_mem.keys())), np.array(
            #     list(ckpt_mem.values())), None, False) 
            # Viewer.plot_fit(ax, "quantize", quantize_mem_model, np.array(list(quantize_mem.keys())), np.array(
            #     list(quantize_mem.values())), None, False) 
            # plt.ylabel("Memory (GB)", size=22)
            # plt.xlabel("Batch Size", size=22)
            # # plt.legend(prop={'size': 14})    
            # plt.yticks(fontsize=15)
            # plt.xticks(fontsize=15)
            # plt.savefig(result_dir + "resnet_mem.%s" % suffix, bbox_inches="tight")
            # plt.close()

            fig, ax = plt.subplots(1, 1)
            fig.set_size_inches(4, 4)
            Viewer.plot_fit(ax, "org", org_ips_model, np.array(list(org_btime.keys())), np.array(
                [bsize / org_btime[bsize] for bsize in org_btime]), None, False)
            # Viewer.plot_fit(ax, "swap", swap_ips_model, np.array(list(swap_btime.keys())), np.array(
            #     [bsize / swap_btime[bsize] for bsize in swap_btime]), None, False)
            # Viewer.plot_fit(ax, "ckpt", ckpt_ips_model, np.array(list(ckpt_btime.keys())), np.array(
            #     [bsize / ckpt_btime[bsize] for bsize in ckpt_btime]), None, False) 
            Viewer.plot_fit(ax, "quantize", quantize_ips_model, np.array(list(quantize_btime.keys())), np.array(
                [bsize / quantize_btime[bsize] for bsize in quantize_btime]), None, False) 
            # ax.set_yticks([20, 40, 60, 80])
            plt.ylabel("Throughput (image/s)", size=22)
            plt.xlabel("Batch Size", size=22)
            # plt.legend(prop={'size': 14})    
            plt.yticks(fontsize=15)
            plt.xticks(fontsize=15)
            plt.savefig(result_dir + "resnet152_ips.%s" % suffix, bbox_inches="tight")
            plt.close()

if __name__ == "__main__": Experiment.do_plot("v100",True)