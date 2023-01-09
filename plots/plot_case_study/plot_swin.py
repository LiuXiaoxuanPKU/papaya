import matplotlib.pyplot as plt
import json
import sys
sys.path.append(".")
from util import Util
from plots.plot_util import ALG_MAP, ALG_MARKER, ALG_COLOR

org_filename = 'benchmarks/Swin-Transformer/v100/results/speed_results.json'

NET_TO_NAME = {
    "swin_large" : "Swin-Large",
}

def load_data(network):
    results = {}
    with open(org_filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("/"):
                continue
            obj = json.loads(line)
            if network != obj['network']:
                continue
            alg, bz, ips = obj['algorithm'], obj['batch_size'], obj['ips']
            if ips == -1:
                continue
            if obj['fp16'] != 'O1':
                continue
            if alg is None and obj["ckpt"] == False:
                alg = 'exact'
            if alg is None and obj["ckpt"] == True:
                alg = 'ckpt'
            if alg == 'L1' and obj['ckpt'] == True:
                alg = 'L1_ckpt'
            if alg not in results:
                results[alg] = {}
            results[alg][bz] = ips
    return results

def plot(network):
    results = load_data(network)
    fig, ax = plt.subplots()
    for alg in results:
        sorted_x, sorted_y = Util.sort_dict(results[alg])
        max_x, max_y = sorted_x[-1], sorted_y[-1]
        sample_cnt = 10
        sorted_x = Util.sample_data(sorted_x, sample_cnt) + [max_x]
        sorted_y = Util.sample_data(sorted_y, sample_cnt) + [max_y]
        
        ax.plot(sorted_x, sorted_y, label=ALG_MAP[alg], marker=ALG_MARKER[alg], \
                color=ALG_COLOR[alg], markersize=8, linewidth=3)
        ax.scatter([max_x+10], [max_y], marker='x', s=140, color='black', linewidths=3, zorder=2)
        # ax.legend(prop={"size":13}, loc='upper right')
        # ax.legend(prop={"size":13}, loc=(-0.1, -0.6), ncol = 5)

    # ax.set_title(f"{NET_TO_NAME[network]}", size=22)
    plt.grid()
    ax.set_axisbelow(True)
    ax.set_xlabel("Batch Size")
    # ax.set_ylabel("Throughput (images/s)")
    Util.set_tick_label_size([ax])
    fig.savefig(f'graphs/case_study/{network}.pdf', bbox_inches='tight')
    plt.close()
    
    
plot("swin_large")