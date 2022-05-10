import json
from typing import Dict, Callable
import scipy.optimize as optimization
import matplotlib.pyplot as plt
import numpy as np


class Model:
    def __init__(self, mem_dir: str, ips_dir: str) -> None:
        self.fit_methods = {
            "leastsq": self.fit_leastsq
        }
        self.mem_data = self.load_mem_data(mem_dir)
        self.ips_data = self.load_ips_data(ips_dir)
        self.mem_model = self.fit_mem(self.mem_data)
        
        batch_time_data = {}
        for bz in self.ips_data:
            batch_time_data[bz] = self.ips_data[bz]["batch_time"]
        self.ips_model = self.fit_ips(batch_time_data)

        # plot ips fit
        self.plot_fit(self.ips_model, np.array(list(self.ips_data.keys())), np.array([self.ips_data[k]["ips"] for k in self.ips_data]), "ips")
        # plot memory fit
        self.plot_fit(self.mem_model, np.array(list(self.mem_data.keys())), np.array(list(self.mem_data.values())), "mem")

    def load_mem_data(self, mem_dir: str) -> Dict[int, int]:
        with open(mem_dir, 'r') as f:
            lines = f.readlines()
        data = {}
        for line in lines:
            obj = json.loads(line)
            data[obj["batch_size"]] = obj["peak"]
        return data

    def load_ips_data(self, ips_dir: str) -> Dict[int, int]:
        with open(ips_dir, 'r') as f:
            lines = f.readlines()
        data = {}
        for line in lines:
            obj = json.loads(line)
            data[obj["batch_size"]] = {"ips": obj["ips"], "batch_time": obj["bacth_time"]}
        return data

    def fit_ips(self, ips_data: Dict[int, int]) -> None:
        def func(x: np.array, a: float, b: float):
            return a * x + b
        batch_time_model = self.fit_methods["leastsq"](ips_data, func)
        def ips_model(bz):
            return bz / batch_time_model(bz)
        return ips_model

    def fit_mem(self, mem_data: Dict[int, int]) -> None:
        def func(x: np.array, a: float, b: float):
            return a * x + b
        return self.fit_methods["leastsq"](mem_data, func)

    def predict(self, bz: float) -> Dict[str, float]:
        predict_mem = self.mem_model(bz)
        predict_ips = self.ips_model(bz)
        return {"batch_size": bz, "mem": predict_mem, "ips": predict_ips}

    def fit_leastsq(self, 
                    data: Dict[int, int], 
                    func: Callable[[np.array, float, float], np.array]) -> Callable[[float], float]:
        xs, ys = np.array(list(data.keys())), np.array(list(data.values()))
        popt, pcov = optimization.curve_fit(func, xs, ys)

        def model(x):
            return func(x, *popt)
        return model
    
    def plot_fit(self, model: Callable[[np.array], float], x: np.array, y: np.array, output_name: str) -> None:
        plt.scatter(x, y, label="actual")
        plt.plot(x, model(x), label="predict")
        plt.xlabel("batch size")
        plt.legend()
        plt.savefig(output_name)
        plt.close()


if __name__ == "__main__":
    mem_dir = "text_classification/results/mem_results.json"
    ips_dir = "text_classification/results/speed_results.json"
    model = Model(mem_dir, ips_dir)
