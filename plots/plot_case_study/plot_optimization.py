import matplotlib.pyplot as plt
import json
import sys
sys.path.append(".")
from util import Util
from plots.plot_util import ALG_COLOR, ALG_MAP, ALG_MARKER


def load_data(layer_num):
    results = {}
    filename = f'benchmarks/text_classification_fp16/v100/results/speed_results_{layer_num}.json'
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("/"):
                continue
            obj = json.loads(line)
            alg, obj_layer_num, bz, ips = obj['algorithm'], obj['layer_num'], obj['batch_size'], obj['ips']
            assert(obj_layer_num == layer_num)
            if ips == -1:
                continue
            if alg is None:
                alg = 'exact'
            if alg not in results:
                results[alg] = {}
            results[alg][bz] = ips
    return results

def plot(results, layer_num):
    fig, ax = plt.subplots()
    for alg in results:
        xs, ys = list(results[alg].keys()), list(results[alg].values())
        sample_cnt = 10
        xs, ys = Util.sample_data(xs, sample_cnt), Util.sample_data(ys, sample_cnt)
        ax.plot(xs, ys, color=ALG_COLOR[alg], label=ALG_MAP[alg], marker=ALG_MARKER[alg], ms=8, lw=3)
        ax.legend(loc="lower right", prop={'size': 16})
        # ax.set_yscale("log")

    plt.grid()
    ax.set_title(f"Bert {layer_num}", size=24)
    ax.set_xlabel("Batch size")
    Util.set_tick_label_size([ax])
    # ax.set_ylabel("throughput (record/s)", size=22)
    fig.savefig(f'graphs/case_study/bert_{layer_num}.pdf', bbox_inches='tight')

results_36 = load_data(36)
plot(results_36, 36)
results_48 = load_data(48)
plot(results_48, 48)