import matplotlib.pyplot as plt
import json
import sys
sys.path.append('.')
from plot_all.plot_util import ALG_MAP, ALG_MARKER, ALG_COLOR
from util import Util

filename = 'text_classification/results/speed_results_no_limit.json'

def load_data():
    results = {}
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("/"):
                continue
            obj = json.loads(line)
            alg, layer_num, bz, ips = obj['algorithm'], obj['layer_num'], obj['batch_size'], obj['ips']
            if ips == -1:
                continue
            if alg is None:
                alg = 'exact'
            if alg not in results:
                results[alg] = {}
            results[alg][bz] = ips
    return results

def plot():
    results = load_data()
    fig, ax = plt.subplots()
    for alg in results:
        sample_cnt = 10
        xs = Util.sample_data(list(results[alg].keys()), sample_cnt)
        ys = Util.sample_data(list(results[alg].values()), sample_cnt)
        ax.plot(xs, ys, label=ALG_MAP[alg], marker=ALG_MARKER[alg], color=ALG_COLOR[alg], markersize=8, linewidth=3)
        ax.scatter([max(xs)+30], [max(ys)], marker='x', s=140, color='black', linewidths=3)

    ax.set_xlabel("Batch size")
    plt.grid()
    Util.set_tick_label_size([ax])

    fig.savefig('graphs/case_study/bert.pdf', bbox_inches='tight')


plot()