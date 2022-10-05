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
            obj = json.loads(line)
            if "ips" in obj and obj["ips"] == -1:
                continue
            if filter is not None and not is_valid(obj):
                continue
            if val == "batch_time" and "batch_time" not in obj:
                obj["batch_time"] = obj["batch_size"] / obj["ips"]
            data[obj[key]] = obj[val]
        return data

markers = {
    "org" : "^",
    "ckpt" : "*",
    "swap" : "x",
    "quantize" : "o"
}

lines = {
    "org" : "-",
    "ckpt" : "-",
    "swap" : "-",
    "quantize" : "-"
}

sizes = {
   "org" : 100,
    "ckpt" : 100,
    "swap" : 100,
    "quantize" : 60
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

        if label == "swap":
            x = x[::4]
            y = y[::4]
        else:
            x = x[::2]
            y = y[::2]
        
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