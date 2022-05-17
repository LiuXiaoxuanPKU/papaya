import numpy as np

from cost_model import Model
from fitter import FitterPool, ModelFnPool
from util import Viewer, Util

if __name__ == "__main__":
    mem_dir = "text_classification/results/mem_results.json"
    ips_dir = "text_classification/results/speed_results_quantize_all.json"

    
    org_mem = Util.load_data(mem_dir, "batch_size", "peak", lambda obj : obj['algorithm'] == None)
    org_btime = Util.load_data(ips_dir, "batch_size", "bacth_time", lambda obj : obj['algorithm'] == None and obj["layer_num"] == 24)
    org_mem_model = FitterPool.fit_leastsq(org_mem, ModelFnPool.linear)
    org_btime_model = FitterPool.fit_leastsq(org_btime, ModelFnPool.linear)
    org_ips_model = lambda bsize: bsize / org_btime_model(bsize)
    # plot ips fit
    Viewer.plot_fit(org_ips_model, np.array(list(org_btime.keys())), np.array(
        [bsize / org_btime[bsize] for bsize in org_btime]), "org_ips.pdf")
    # plot memory fit
    Viewer.plot_fit(org_mem_model, np.array(list(org_mem.keys())), np.array(
        list(org_mem.values())), "org_mem.pdf") 

    # ckpt_mem = Util.load_data(mem_dir, "batch_size", "peak", lambda obj : obj['algorithm'] == 'ckpt')
    # ckpt_btime = Util.load_data(ips_dir, "batch_size", "bacth_time", lambda obj : obj['algorithm'] == 'ckpt')
    ckpt_mem = Util.load_data(mem_dir, "batch_size", "peak", lambda obj : obj['algorithm'] == 'L1')
    ckpt_btime = Util.load_data(ips_dir, "batch_size", "bacth_time", lambda obj : obj['algorithm'] == 'L1' and obj["layer_num"] == 24)
    ckpt_mem_model = FitterPool.fit_leastsq(ckpt_mem, ModelFnPool.linear)
    ckpt_btime_model = FitterPool.fit_leastsq(ckpt_btime, ModelFnPool.linear)
    ckpt_ips_model = lambda bsize: bsize / ckpt_btime_model(bsize)
    # plot ips fit
    Viewer.plot_fit(ckpt_ips_model, np.array(list(ckpt_btime.keys())), np.array(
        [bsize / ckpt_btime[bsize] for bsize in ckpt_btime]), "quantize_ips.pdf")
    # plot memory fit
    Viewer.plot_fit(ckpt_mem_model, np.array(list(ckpt_mem.keys())), np.array(
        list(ckpt_mem.values())), "quantize_mem.pdf")