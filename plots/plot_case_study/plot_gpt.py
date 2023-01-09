import matplotlib.pyplot as plt
import json
import sys
sys.path.append(".")
from plots.plot_util import ALG_COLOR, ALG_MAP, ALG_MARKER
from util import Util

filename = 'benchmarks/GPT/v100/results/speed_results.json'

NET_TO_NAME = {
    "transformer_lm_gpt3_small" : "GPT3-Small",
    "transformer_lm_gpt3_medium" : "GPT3-Medium",
    "transformer_lm_gpt3_large" : "GPT3-Large"
}

def plot(network):
    results = {}
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("/"):
                continue
            obj = json.loads(line)
            if network != obj['network']:
                continue
            if 'alg' not in obj:
                continue
            alg, bz, ips = obj['alg'], obj['batch_size'], obj['ips']
            if ips == -1:
                continue
            if alg is None:
                alg = 'exact'
            if alg not in results:
                results[alg] = {}
            results[alg][bz] = ips

    fig, ax = plt.subplots()
    for alg in results:
        xs, ys = list(results[alg].keys()), list(results[alg].values())
        max_x, max_y = max(xs), max(ys)
        sample_cnt = 10
        xs, ys = Util.sample_data(xs, sample_cnt), Util.sample_data(ys, sample_cnt)
        xs += [max_x]
        ys += [max_y]
        ax.plot(xs, ys, color=ALG_COLOR[alg], label=ALG_MAP[alg], marker=ALG_MARKER[alg],
            markersize=8, linewidth=3)
        ax.scatter([max_x+2], [max_y], marker='x', s=140, color='black', linewidths=3)
        # ax.legend(prop={"size":13})
        # ax.set_yscale("log")
    plt.grid()
    ax.set_axisbelow(True)
    ax.set_xlabel("Batch size",)
    # ax.set_ylabel("Throughput (record/s)")
    Util.set_tick_label_size([ax])
    fig.savefig(f'graphs/case_study/{network}.pdf', bbox_inches='tight')
    plt.close()
    
    
plot("transformer_lm_gpt3_small")
plot("transformer_lm_gpt3_medium")
plot("transformer_lm_gpt3_large")