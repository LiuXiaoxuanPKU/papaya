from typing import Callable
import numpy as np
from pathlib import Path
from cost_model import Model
from fitter import FitterPool, ModelFnPool
from util import Viewer, Util
import os

suffix = "pdf"
algo = "GPT"
class Experiment:
    def run_experiment(machine_tag, network = None):
        mem_dir = "{}/{}/results/mem_results.json".format(algo,machine_tag)
        ips_dir = "{}/{}/results/speed_results.json".format(algo,machine_tag)
        cnt = 1
        ips_archived, mem_archived = False, False
        while True:
            mem_dir_arc = "{}/{}/results/mem_archive_{}.json".format(algo,machine_tag,cnt)
            ips_dir_arc = "{}/{}/results/speed_archive_{}.json".format(algo,machine_tag,cnt)
            if (not os.path.exists(mem_dir_arc)) and (not os.path.exists(ips_dir_arc)):
                break
            cnt += 1
        if Path(mem_dir).is_file():
            os.rename(mem_dir, mem_dir_arc)
            mem_archived = True
        if Path(ips_dir).is_file():
            os.rename(ips_dir, ips_dir_arc)
            ips_archived = True
        cmd = '''cd ./GPT/{}/ && 
        python exp_mem_speed.py'''.format(machine_tag)
        if network: cmd += " --network {} ".format(network)
        ret = os.system(cmd)
        if ret!=0:
            print("[Error] Failed to run new experiments, restoring experiment data")
        if ret!=0 and mem_archived:
            if Path(mem_dir).is_file():
                os.remove(mem_dir)
            os.rename(mem_dir_arc, mem_dir)
        if ret!=0 and ips_archived:
            if Path(ips_dir).is_file():
                os.remove(ips_dir)
            os.rename(ips_dir_arc, ips_dir)
        return ret

    def plot_helper(cond, mem_dir, ips_dir, offset = None):
        mem = Util.load_data(mem_dir, "batch_size", "peak_mem", cond)
        for k in mem:
            mem[k] /= 1000000000
        mem_model,mem_score,alpha,beta = FitterPool.fit_leastsq_verbose(mem, ModelFnPool.linear)
        btime = Util.load_data(ips_dir, "batch_size", "batch_time", cond)
        btime_model,btime_score,gamma,delta = FitterPool.fit_leastsq_verbose(btime, ModelFnPool.linear)
        if delta<0 and offset: btime_model,btime_score,gamma,delta = FitterPool.fit_leastsq_verbose_offset(btime, ModelFnPool.linear,offset)
        ips_model = lambda bsize: bsize / btime_model(bsize)
        # print("[predict mem] ", mem_model(np.array(list(mem.keys()))))
        return mem, btime, mem_model, btime_model, ips_model, alpha, beta, gamma, delta, mem_score, btime_score

    def do_plot(machine_tag, to_plot):
        
        algo = "GPT"
        ips_dir = "{}/{}/results/speed_results.json".format(algo,machine_tag)
        mem_dir = ips_dir
        result_dir = "graphs/{}/{}/".format(algo,machine_tag)
        if not Path(mem_dir).is_file() or not Path(ips_dir).is_file():
            print("Error: No experiment data found. Pease run expriment from scratch with --run-new for {}@{}".format(algo,machine_tag))
            return
        Path(result_dir).mkdir(parents=True, exist_ok=True)
        for network in ["transformer_lm_gpt3_small","transformer_lm_gpt3_medium"]:
            human_name = network.split("_")[-1];
            try:
                #print("-----------------Org-----------------")
                is_org = lambda obj : obj['alg'] == None and obj["network"] == network
                org_mem, org_btime, org_mem_model, org_btime_model, org_ips_model,\
                alpha, beta, gamma, delta, mem_score, btime_score = Experiment.plot_helper(is_org, mem_dir, ips_dir)
                offset = delta
                print("-----------------{}({})@{} Params-----------------".format(algo,human_name,machine_tag))
                print ("{:<8} {:<10} {:<10} {:<10} {:<10} {:<12} {:<12}".format('Method','Alpha','Beta','Gamma','Delta','Mem R','Latency R'))
                print ("{:<8} {:<10g} {:<10g} {:<10g} {:<10g} {:<12g} {:<12g}".format('Org',\
                alpha,beta,gamma,delta,mem_score,btime_score))

                #print("-----------------Swap-----------------")
                # is_swap = lambda obj : obj['alg'] == "swap" and obj['network'] == "transformer_lm_gpt3_small"
                # swap_mem, swap_btime, swap_mem_model, swap_btime_model, swap_ips_model,\
                # alpha, beta, gamma, delta, mem_score, btime_score = Experiment.plot_helper(is_swap, mem_dir, ips_dir, offset)


                # print ("{:<8} {:<10g} {:<10g} {:<10g} {:<10g} {:<12g} {:<12g}".format('Swap',alpha,beta,gamma,delta,mem_score,btime_score))

                #print("-----------------Ckpt-----------------")
                is_ckpt = lambda obj : obj['alg'] == "ckpt" and  obj['network'] == network
                ckpt_mem, ckpt_btime, ckpt_mem_model, ckpt_btime_model, ckpt_ips_model,\
                alpha, beta, gamma, delta, mem_score, btime_score = Experiment.plot_helper(is_ckpt, mem_dir, ips_dir, offset) 
                print ("{:<8} {:<10g} {:<10g} {:<10g} {:<10g} {:<12g} {:<12g}".format('Ckpt',alpha,beta,gamma,delta,mem_score,btime_score))
                
                #print("-----------------Quantize-----------------")
                is_quantize = lambda obj : obj['alg'] == "L1" and obj['network'] == network
                quantize_mem, quantize_btime, quantize_mem_model, quantize_btime_model, quantize_ips_model,\
                alpha, beta, gamma, delta, mem_score, btime_score = Experiment.plot_helper(is_quantize, mem_dir, ips_dir, offset) 
                print ("{:<8} {:<10g} {:<10g} {:<10g} {:<10g} {:<12g} {:<12g}".format('Quantize',alpha,beta,gamma,delta,mem_score,btime_score))
            except IndexError:
                print("[Error] Data missing for {}({})@{}.".format(algo,human_name,machine_tag))
                continue

            if to_plot:
                import matplotlib
                # matplotlib.rc('axes',edgecolor='silver')
                import matplotlib.pyplot as plt
                # plt.style.use(['grid'])
                fig, axes = plt.subplots(4, 1, sharex=True)
                fig.set_size_inches(4, 6)
                # plot batch time
                sample_cnt = 5
                x, y= Util.sample_data(list(org_btime.keys()), sample_cnt), Util.sample_data(list(org_btime.values()), sample_cnt) 
                Viewer.plot_fit(axes[0], "org", org_btime_model, np.array(x), np.array(y), None, False)
                # Viewer.plot_fit(axes[1], "swap", swap_btime_model, np.array(list(swap_btime.keys())), np.array(
                #     list(swap_btime.values())), None, False)
                
                x, y= Util.sample_data(list(ckpt_btime.keys()), sample_cnt), Util.sample_data(list(ckpt_btime.values()), sample_cnt) 
                Viewer.plot_fit(axes[2], "ckpt", ckpt_btime_model, np.array(x), np.array(y), None, False) 

                x, y= Util.sample_data(list(quantize_btime.keys()), sample_cnt), Util.sample_data(list(quantize_btime.values()), sample_cnt) 
                Viewer.plot_fit(axes[3], "quantize", quantize_btime_model, np.array(x), np.array(y), None, False) 
                plt.xlabel("Batch Size")  
                Util.set_tick_label_size(axes)

                # fig.text(-0.05, 0.5, 'Time (s)', va='center', rotation='vertical', size=22)
                plt.savefig(result_dir + "gpt3_%s_batch_time.%s" % (human_name,suffix), bbox_inches="tight")
                plt.close()

                # plot memory
                fig, ax = plt.subplots(1, 1)
                fig.set_size_inches(4, 4)
                x, y= Util.sample_data(list(org_mem.keys()), sample_cnt), Util.sample_data(list(org_mem.values()), sample_cnt) 
                Viewer.plot_fit(ax, "org", org_mem_model, np.array(x), np.array(y), None, False)
                # Viewer.plot_fit(ax, "swap", swap_mem_model, np.array(list(swap_mem.keys())), np.array(
                #     list(swap_mem.values())), None, False)
                x, y= Util.sample_data(list(ckpt_mem.keys()), sample_cnt), Util.sample_data(list(ckpt_mem.values()), sample_cnt) 
                Viewer.plot_fit(ax, "ckpt", ckpt_mem_model, np.array(x), np.array(y), None, False) 
                x, y= Util.sample_data(list(quantize_mem.keys()), sample_cnt), Util.sample_data(list(quantize_mem.values()), sample_cnt) 
                Viewer.plot_fit(ax, "quantize", quantize_mem_model, np.array(x), np.array(y), None, False)
               
                # plt.ylabel("Memory (GB)", size=22)
                plt.xlabel("Batch Size")
                Util.set_tick_label_size([ax])
                plt.savefig(result_dir + "gpt3_%s_mem.%s" % (human_name,suffix), bbox_inches="tight")
                plt.close()

                fig, ax = plt.subplots(1, 1)
                fig.set_size_inches(4, 4)
                Viewer.plot_fit(ax, "org", org_ips_model, np.array(list(org_btime.keys())), np.array(
                    [bsize / org_btime[bsize] for bsize in org_btime]), None, False)
                # Viewer.plot_fit(ax, "swap", swap_ips_model, np.array(list(swap_btime.keys())), np.array(
                #     [bsize / swap_btime[bsize] for bsize in swap_btime]), None, False)
                Viewer.plot_fit(ax, "ckpt", ckpt_ips_model, np.array(list(ckpt_btime.keys())), np.array(
                    [bsize / ckpt_btime[bsize] for bsize in ckpt_btime]), None, False) 
                Viewer.plot_fit(ax, "quantize", quantize_ips_model, np.array(list(quantize_btime.keys())), np.array(
                    [bsize / quantize_btime[bsize] for bsize in quantize_btime]), None, False) 
                
                ax.set_yticks([20, 40, 60, 80])
                plt.ylabel("Throughput (record/s)", size=22)
                plt.xlabel("Batch Size", size=22)
                # plt.legend(prop={'size': 14})    
                plt.yticks(fontsize=15)
                plt.xticks(fontsize=15)
                plt.savefig(result_dir + "gpt3_%s_ips.%s" % (human_name,suffix), bbox_inches="tight")
                plt.close()

if __name__=="__main__":
    Experiment.do_plot("v100",True)
