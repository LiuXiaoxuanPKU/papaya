from utilization.occupancy import utilization as occupancy
from utilization.smi import plot as smi
from pathlib import Path
from typing import Callable
import numpy as np
from cost_model import Model
from fitter import FitterPool, ModelFnPool
from util import Viewer, Util
import random

suffix = "png"
occupancy_data_path = "./utilization/occupancy/data"
smi_data_path = "./utilization/smi/data"
graph_path = "./graphs/utilization"

class Experiment:
    def sample_dict(dic, percentenge):
        sample_data = {}
        sample_keys = random.sample(list(dic.keys())[5:], int(len(dic) * percentenge))
        for k in sample_keys:
            sample_data[k] = dic[k]
        return sample_data

    def plot_helper(cond, mem_dir, ips_dir):
        # mem = Util.load_data(mem_dir, "batch_size", "peak", cond)
        # for k in mem:
        #     mem[k] /= 1000
        btime = Util.load_data(ips_dir, "batch_size", "batch_time", cond)
        # # use only 20% data to fit the model
        # mem_sample= Experiment.sample_dict(mem, 0.4)
        btime_sample= Experiment.sample_dict(btime, 0.4)
        # mem_model,mem_score,alpha,beta = FitterPool.fit_leastsq_verbose(mem_sample, ModelFnPool.linear)
        btime_model,btime_score,gamma,delta = FitterPool.fit_leastsq_verbose(btime_sample, ModelFnPool.linear)
        ips_model = lambda bsize: bsize / btime_model(bsize)
        # print("[predict mem] ", mem_model(np.array(list(mem.keys()))))
        return None, btime, None, btime_model, ips_model, None, None, gamma, delta, None, btime_score
    
    def get_plot_batch_time(machine_tag, algo = "text_classification_fp16"):
        if algo == "text_classification_fp16": algo_kw = "algorithm"
        elif algo == "GPT": algo_kw = "alg"
        else:
            print("Not supported.")
            return
        mem_dir = "{}/{}/results/mem_results.json".format(algo,machine_tag)
        ips_dir = "{}/{}/results/speed_results.json".format(algo,machine_tag)
        result_dir = "graphs/{}/{}/".format(algo,machine_tag)
        if not Path(mem_dir).is_file() or not Path(ips_dir).is_file():
            print("Error: No experiment data found. Pease run expriment from scratch with --run-new for {}@{}".format(algo,machine_tag))
            return
        Path(result_dir).mkdir(parents=True, exist_ok=True)

        is_org = lambda obj : obj[algo_kw] == None
        org_mem, org_btime, org_mem_model, org_btime_model, org_ips_model,\
        alpha, beta, gamma, delta, mem_score, btime_score = Experiment.plot_helper(is_org, mem_dir, ips_dir)
        return org_btime_model, org_btime

    def do_occupancy():
        Path(graph_path).mkdir(parents=True, exist_ok=True)
        occupancy.plotUtil("{}/occupancy_bz_4.json".format(occupancy_data_path),"{}/occupancy_bz_4.pdf".format(graph_path))
        occupancy.plotUtil("{}/occupancy_bz_64.json".format(occupancy_data_path),"{}/occupancy_bz_64.pdf".format(graph_path))
    
    def do_smi(model, btime, alg):
        Path(graph_path).mkdir(parents=True, exist_ok=True)
        method = "none"
        smi.plot_average(method,smi_data_path,graph_path,alg, model, btime)


if __name__=="__main__":
    model, btime = Experiment.get_plot_batch_time("v100_util")
    Experiment.do_smi(model, btime, "bert")
    model, btime = Experiment.get_plot_batch_time("v100_util", "GPT")
    Experiment.do_smi(model, btime, "gpt")