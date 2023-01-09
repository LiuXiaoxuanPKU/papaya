import json
import matplotlib.pyplot as plt
import sys
sys.path.append('.')
from plots.plot_util import ALG_COLOR, ALG_MAP, ALG_MARKER, NET_TO_FOLDER
from util import Util

suffix = "pdf"

def parse_resnet_obj(line):
    obj = json.loads(line)
    prefix = "scaled_wide_resnet_"
    if obj["network"].startswith(prefix):
        layer_num = int(obj["network"][len(prefix):])
        return obj["algorithm"], obj["ips"], layer_num
    else:
        return None, -1, None


def collect_max_tpt(hardware, network):
    data = {}
    file_dir = f"benchmarks/{NET_TO_FOLDER[network]}/{hardware}/results/speed_results_depth.json"
    with open(file_dir, "r") as f:
        lines = f.readlines()
        for line in lines:
            if network in ["resnet50"]:
                alg, ips, layer_num = parse_resnet_obj(line)
            elif network in ["transformer_lm_gpt3_small"]:
                pass
            else:
                print(f"[Error] Unsupport network {network}")
                exit(0)
            if ips == -1 or alg == "swap":
                continue
            if alg in ["dtr", "L4bit-swap"]:
                continue
            if alg not in data:
                data[alg] = {}
            if layer_num not in data[alg]:
                data[alg][layer_num] = ips
            else:
                data[alg][layer_num] = max(data[alg][layer_num], ips)
    return data

def plot_model_depth(data):
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(6, 6)
    for alg in data:
        xs = []
        ys = []
        for k in sorted(data[alg].keys()):
            xs.append(k)
            ys.append(data[alg][k])
        if alg is None: alg = 'exact'
        plt.plot(xs, ys, marker=ALG_MARKER[alg], label=ALG_MAP[alg], color=ALG_COLOR[alg], ms=12, lw=3)

    plt.xlabel('Model width (hidden size)')
    plt.grid()
    # plt.ylabel('max throughput (records/s)', size=15)
    # ax.tick_params(axis='both', which='major', labelsize=15)
    Util.set_tick_label_size([ax])
    # plt.xlim(8, 70)
    # plt.ylim(0, 350)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig(f'graphs/implications/{network}_width.{suffix}')

if __name__ == "__main__":
    networks = ["resnet50"]
    hardware = 'v100'
    for network in networks:
        data = collect_max_tpt(hardware, network)
        plot_model_depth(data)
