import json
from typing import Callable, Dict, Optional
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
    "org" : "s",
    "ckpt" : "v",
    "swap" : "o",
    "quantize" : "X"
}
sizes = {
   "org" : 20,
    "ckpt" : 20,
    "swap" : 20,
    "quantize" : 20 
}

colors = {
   "org" : 'black',
    "ckpt" : 'green',
    "swap" : 'red',
    "quantize" : 'orange'  
}

class Viewer:
    def plot_fit(ax, label: str, model: Callable[[np.array], float], x: np.array, y: np.array, output_name: str, save_fig: bool = True) -> None:
        # if label == "swap":
        #     return
        ax.scatter(x, y, label=label, 
                    marker=markers[label], s=sizes[label],
                    c=colors[label])
        ax.plot(x, model(x), c=colors[label])
        if save_fig:
            plt.savefig(output_name)
            plt.close()