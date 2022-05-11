import json
from typing import Callable
from matplotlib import pyplot as plt
import numpy as np

from fitter import Data


class Util:
    def load_data(dir: str, key: str, val: str) -> Data:
        with open(dir, 'r') as f:
            lines = f.readlines()
        data = {}
        for line in lines:
            obj = json.loads(line)
            data[obj[key]] = obj[val]
        return data

class Viewer:
    def plot_fit(model: Callable[[np.array], float], x: np.array, y: np.array, output_name: str) -> None:
        plt.scatter(x, y, label="actual")
        plt.plot(x, model(x), label="predict")
        plt.xlabel("batch size")
        plt.legend()
        plt.savefig(output_name)
        plt.close()