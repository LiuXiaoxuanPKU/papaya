import sys
from unittest import result
sys.path.append('.')
from plot_case_study.plot_util import ALG_COLOR, ALG_MAP, ALG_MARKER
import matplotlib.pyplot as plt
import json
from util import Util


def load_data():
    results = {}
    filename = 'text_classification/results/speed_results_quantize_all.json'
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("/"):
                continue
            obj = json.loads(line)
            # alg, bz, ips = obj['algorithm'], obj['batch_size'], obj['ips']
            alg, layer_num, bz, ips = obj['algorithm'], obj['layer_num'], obj['batch_size'], obj['ips']
            if layer_num < 20 or layer_num % 8 == 0 or layer_num > 40:
                continue
            if alg is None:
                alg = "exact"
            alg = alg + "_" + str(layer_num)
            if ips == -1:
                continue
            if alg is None:
                alg = 'exact'
            if alg not in results:
                results[alg] = {}
            results[alg][bz] = ips
    return results

COLOR = {
    "20" : 'green',
    '28' : 'blue',
    '36' : 'orange'
}

def plot(results):
    fig, ax = plt.subplots()
    for alg in results:
        if alg.startswith("L1"):
            alg_name = alg.replace("L1", "quantize")
        else:
            alg_name = alg
        xs, ys = list(results[alg].keys()), list(results[alg].values())
        sample_cnt = 10
        xs, ys = Util.sample_data(xs, sample_cnt), Util.sample_data(ys, sample_cnt)
        ax.plot(xs, ys, label=alg_name, marker=ALG_MARKER[alg.split('_')[0]], markersize=8, linewidth=3, color=COLOR[alg.split('_')[-1]])
        ax.legend(prop={"size":14})
        
    ax.set_title("Bert of different layers on V100", size=24)
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Throughput (records/s)")
    Util.set_tick_label_size([ax])
    fig.savefig('graphs/case_study/layer_num.pdf', bbox_inches='tight')

results = load_data()
plot(results)
