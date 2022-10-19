import json

from collections import OrderedDict
from typing import Callable, Dict, Optional, List
from matplotlib import pyplot as plt
import numpy as np

from fitter import Data


class Util:
    def load_data(dir: str, key: str, val: str, \
                    is_valid: Optional[Callable[[Dict[str, float]], bool]] = None) -> Data:
        with open(dir, 'r') as f:
            lines = f.readlines()
        data = {}
        for line in lines:
            if line.startswith("//"):
                continue
            obj = json.loads(line)
            if "ips" in obj and obj["ips"] == -1:
                continue
            if filter is not None and not is_valid(obj):
                continue
            if val == "batch_time" and "batch_time" not in obj:
                obj["batch_time"] = obj["batch_size"] / obj["ips"]
            data[obj[key]] = obj[val]
        return data

    def sample_data(data, cnt):
        if cnt >= len(data):
            return data
        interval = max(len(data) // cnt, 2)
        # print(interval, cnt, len(data))
        i = 0
        sampled_data = []
        while i < len(data):
            sampled_data.append(data[i])
            i += interval
        return sampled_data

    def set_tick_label_size(axes):
        for ax in axes: 
            ax.tick_params(axis='x', labelsize=18)
            ax.tick_params(axis='y', labelsize=18)
            ax.xaxis.label.set_size(20)
            ax.yaxis.label.set_size(20)

    def sort_dict(dic):
        keys = sorted(list(dic.keys()))[:-1]
        sk, sv = keys, []
        for k in keys:
            sv.append(dic[k])
        return sk, sv

markers = {
    "org" : "o",
    "ckpt" : "p",
    "swap" : "D",
    "quantize" : "v"
}

lines = {
    "org" : "-",
    "ckpt" : "--",
    "swap" : "-",
    "quantize" : "-"
}

sizes = {
   "org" : 100,
    "ckpt" : 100,
    "swap" : 100,
    "quantize" : 100
}

colors = {
   "org" : '#4d72b0',
    "ckpt" : '#c44e52',
    "swap" : '#025839',
    "quantize" : '#e28743'  
}

class Viewer:
    def plot_fit(ax, label: str, model: Callable[[np.array], float], x: np.array, y: np.array, output_name: str, save_fig: bool = True) -> None:
        # if label == "swap":
        #     return
        
        ax.scatter(x, y, label=label, 
                    marker=markers[label], s=sizes[label],
                    c=colors[label])
        ax.plot(x, model(x), c=colors[label], linewidth=2.4)
        if save_fig:
            plt.savefig(output_name)
            plt.close()

class GlobalExpRecorder:
    def __init__(self):
        self.val_dict = OrderedDict()

    def record(self, key, value, float_round=6):
        if isinstance(value, (np.int32, np.int64)):
            value = int(value)
        if isinstance(value, (float, np.float32, np.float64)):
            value = round(value, float_round)

        self.val_dict[key] = value

    def dump(self, filename):
        with open(filename, "a") as fout:
            fout.write(json.dumps(self.val_dict) + '\n')
        print("Save exp results to %s" % filename)

    def clear(self):
        pass


exp_recorder = GlobalExpRecorder()