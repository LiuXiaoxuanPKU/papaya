import sys
sys.path.append('.')
from plots.plot_util import ALG_COLOR, ALG_MARKER, NET_TO_FOLDER
from util import Util
from fitter import FitterPool, ModelFnPool

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


suffix = 'pdf'

def gen_cond(network, alg=None):
    if network in ["resnet50", "wide_resnet50_2", "resnet152"]:
        return lambda obj : obj["network"] == network
    elif network in ["bert-large-cased"]:
        return lambda obj : obj["algorithm"] == alg and obj["network"] == network
    elif network in ["swin_large"]:
        if alg == "ckpt":
            return lambda obj : obj["fp16"] == 'O1' and obj["network"] == network and obj["ckpt"] == True
        else:
            return lambda obj : obj["fp16"] == 'O1' and obj["network"] == network \
                                and obj["algorithm"] == alg and obj["ckpt"] == False
    elif network in ["transformer_lm_gpt3_small"]:
        return lambda obj : obj["network"] == network
    else:
        print(f"[Error] Unsupport network {network}, alg {alg}")
        exit(0)


def plot(hardware, network):
    util_dir = f"benchmarks/{NET_TO_FOLDER[network]}/{hardware}/results/utilization.json"
    result_file = f"graphs/cost_estimation/batch_time/{network}.{suffix}"
    if not Path(util_dir).is_file():
        print(f"Error: No utilization data found for {network} on {hardware}")
        return
    batch_time = Util.load_data(util_dir, "batch_size", "batch_time", gen_cond(network))
    xs, ys = list(batch_time.keys()), list(batch_time.values())
    sample_cnt = 10
    alg = "exact"
    xs, ys = Util.sample_data(xs, sample_cnt), Util.sample_data(ys, sample_cnt)
    fig, ax = plt.subplots(1, 1)
    ax.scatter(xs, ys, marker=ALG_MARKER[alg], s=100, c=ALG_COLOR[alg])

    sample_per = 0.2
    bt_sample = Util.sample_dict(batch_time, sample_per)
    bt_model, bt_score, alpha, beta = FitterPool.fit_leastsq_verbose(bt_sample, ModelFnPool.linear)
    xs = np.array(xs)
    ax.plot(xs, bt_model(xs), c=ALG_COLOR[alg], linewidth=3)
    ax.set_yticklabels([x.get_text() + "s" for x in ax.get_yticklabels()])

    utils = Util.load_data(util_dir, "batch_size", "utilization", gen_cond(network))
    xs, ys = list(utils.keys()), list(utils.values())
    sample_cnt = 10
    alg = "exact"
    xs, ys = Util.sample_data(xs, sample_cnt), Util.sample_data(ys, sample_cnt)
    # util_ax = ax.twinx()
    # util_ax.plot(xs, ys, c="green", linewidth=3)
    # util_ax.set_ylim(0, 100)
    # util_ax.set_yticklabels([x.get_text() + "%" for x in util_ax.get_yticklabels()])
    ax.set_xlabel("Batch Size")

    plt.grid()
    # Util.set_tick_label_size([ax, util_ax])
    Util.set_tick_label_size([ax])
    plt.tight_layout()
    fig.savefig(result_file)

    

if __name__ == "__main__": 
    hardware = 'v100'
    networks = ['resnet50', 'wide_resnet50_2', 'bert-large-cased', 'swin_large', 'transformer_lm_gpt3_small']
    for net in networks:
        plot(hardware, net)