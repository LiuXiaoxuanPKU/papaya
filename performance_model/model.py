from scipy import optimize
import numpy as np
from dataclasses import dataclass
import json
from sklearn.metrics import r2_score
from typing import Optional

@dataclass
class ProfileData:
    method: str
    latency_batch_size: list[int]
    latency: list[float]
    memory_batch_size: list[int]
    memory: list[float]
    
    depth_batch_size: int
    depth: list[int]
    depth_latency: list[float]

    width_batch_size: int
    width: list[int]
    width_latency: list[float]

    cost_batch_size: Optional[int]
    dist_gpu_cnt: Optional[int]
    dist_latency: Optional[float]

    def __post_init__(self):
        self.ips = []
        for bz, l in zip(self.latency_batch_size, self.latency):
            self.ips.append(bz/l)
        self.depth_ips = []
        for l in self.depth_latency:
            self.depth_ips.append(self.depth_batch_size / l)
        self.width_ips = []
        for l in self.width_latency:
            self.width_ips.append(self.width_batch_size / l)

class PerformanceModel:
    def __init__(self, profile_data: ProfileData):
        self.data = profile_data

    def fit_batch_latency(self, x, y):
        def batch_latency_func(x, a, k, b):
            return np.maximum(a, k * np.array(x)) + b
        popt, pcov = optimize.curve_fit(batch_latency_func, x, y, bounds=(0, [1, 1, 1]))
        return batch_latency_func, popt
    
    def fit_memory(self, x, y):
        def memory_func(x, a, b):
            return a * x + b            
        popt, pcov = optimize.curve_fit(memory_func, x, y)
        return memory_func, popt
    
    def predict_throughput(self, batch_size):
        func, popt = self.fit_batch_latency(self.data.latency_batch_size, self.data.latency)
        predict_latency = func([batch_size], *popt)[0]
        return batch_size / predict_latency

    def predict_max_throughput(self, device_limit):
        memory_func, memory_popt = self.fit_memory(self.data.memory_batch_size, self.data.memory)
        memory_frac = 0.8
        max_batch_size = (device_limit * memory_frac - memory_popt[1]) / memory_popt[0]
        return self.predict_throughput(max_batch_size)

    def fit_width_depth_throughput(self, type):
        func, lpopt = self.fit_batch_latency(self.data.latency_batch_size, self.data.latency)
        # we only model the model size amplification factor
        def model_latency(data, factor):
            return np.maximum(lpopt[0], factor * data[:, 0] * lpopt[1] * np.array(data[:, 1])) + lpopt[2]
        
        if type == "width":
            batch_size = self.data.width_batch_size
            x = self.data.width
            y = self.data.width_latency
        elif type == "depth":
            batch_size = self.data.depth_batch_size
            x = self.data.depth
            y = self.data.depth_latency
        else:
            raise ValueError()
        batch_size = np.repeat(batch_size, len(x)).reshape(-1, 1)
        width = np.array(x).reshape(-1, 1)
        data = np.concatenate((batch_size, width), 1)
        popt, pcov = optimize.curve_fit(model_latency, data, y)
        return model_latency, popt
    
    def predict_model_throughput(self, batch_size, width_depth, type):
        assert type in ["width", "depth"]
        func, popt = self.fit_width_depth_throughput(type)
        model_latency = func(np.array([[batch_size, width_depth]]), *popt)[0]
        return self.data.depth_batch_size / model_latency

    def fit_tradeoff_curve(self):
        func, lpopt = self.fit_batch_latency(self.data.latency_batch_size, self.data.latency)
        # we only model the gpu amplification factor
        def model_latency(data, factor):
            return np.maximum(lpopt[0], factor * data[:, 0] * lpopt[1] * np.array(data[:, 1])) + lpopt[2]
        batch_size = TODO
        x = TODO
        y = TODO
        batch_size = np.repeat(batch_size, len(x)).reshape(-1, 1)
        gpu_cnt = np.array(x).reshape(-1, 1)
        data = np.concatenate((batch_size, gpu_cnt), 1)
        popt, pcov = optimize.curve_fit(model_latency, data, y)
        k, b = lpopt[1], lpopt[2]
        return model_latency, popt, k, b

    def predict_tradeoff_curve(self, batch_size, gpu_cnt):
        func, popt, k, b = self.fit_tradeoff_curve()
        latency = func(np.array([[batch_size, gpu_cnt]]), *popt)[0]
        return latency / (latency - b) * batch_size * k


def prepare_bert_data():
    def preprocess_lines(lines):
        model_name = "bert-large-cased"
        process_lines = []
        for l in lines:
            if ('ips' in l and l["ips"] == -1) or l["network"] != model_name:
                continue
            if l["algorithm"] == "swap":
                continue
            l["algorithm"] = "exact" if l["algorithm"] is None else l["algorithm"]
            process_lines.append(l)
        return process_lines

    with open("benchmarks/text_classification_fp16/v100/results/mem_results.json", "r") as f:
        lines = f.readlines()
    memory_lines = preprocess_lines([json.loads(l) for l in lines])

    with open("benchmarks/text_classification_fp16/v100/results/speed_results.json", "r") as f:
        lines = f.readlines()
    batch_latency_lines = preprocess_lines([json.loads(l) for l in lines])

    with open("benchmarks/text_classification_fp16/v100/results/speed_results_model_size_depth.json", "r") as f:
        lines = f.readlines()
    model_depth_lines = preprocess_lines([json.loads(l) for l in lines])

    with open("benchmarks/text_classification_fp16/v100/results/speed_results_model_size_width.json", "r") as f:
        lines = f.readlines()
    model_width_lines = preprocess_lines([json.loads(l) for l in lines])
 
    all_data = {}
    all_algs = set([l["algorithm"] for l in batch_latency_lines])
    for alg in all_algs:
        alg_lines = [l for l in batch_latency_lines if l["algorithm"] == alg]   
        latency = [l["batch_time"] for l in alg_lines]
        latency_batch_size = [l["batch_size"] for l in alg_lines]
        
        depth_alg_lines = [l for l in model_depth_lines if l["algorithm"] == alg]
        depth_batch_size = list(set([l["batch_size"] for l in depth_alg_lines]))
        assert len(depth_batch_size) == 1
        depth_batch_size = depth_batch_size[0]
        depth = [l["layer_num"] for l in depth_alg_lines]
        depth_latency = [l["batch_time"] for l in depth_alg_lines]

        width_alg_lines = [l for l in model_width_lines if l["algorithm"] == alg]
        width_batch_size = list(set([l["batch_size"] for l in width_alg_lines]))
        assert len(width_batch_size) == 1
        width_batch_size = width_batch_size[0]
        width = [l["hidden_size"] for l in width_alg_lines]
        width_latency = [l["batch_time"] for l in width_alg_lines]

        memory_alg_lines = [l for l in memory_lines if l["algorithm"] == alg]
        memory_batch_size = [l["batch_size"] for l in memory_alg_lines]
        memory = [l["peak"] for l in memory_alg_lines]

        all_data[alg] = ProfileData(alg, latency_batch_size, latency, memory_batch_size, memory, 
                    depth_batch_size, depth, depth_latency,
                    width_batch_size, width, width_latency)
    return all_data


def get_R_score(x, y):
    return r2_score(x, y)  

if __name__ == "__main__":
    all_bert_data = prepare_bert_data()
    for method in all_bert_data:
        print(f"============{method}===============")
        model = PerformanceModel(all_bert_data[method])
        print("Throughput R^2 Score:")
        print(get_R_score(all_bert_data[method].ips, 
                          [model.predict_throughput(bz) for bz in all_bert_data[method].latency_batch_size]))
        print("Model Depth R^2 Score")
        print(get_R_score(all_bert_data[method].depth_ips,
                          [model.predict_model_throughput(all_bert_data[method].depth_batch_size, d, "depth") for d in all_bert_data[method].depth]))
        
        print("Model Width R^2 Score")
        print(get_R_score(all_bert_data[method].width_ips,
                          [model.predict_model_throughput(all_bert_data[method].width_batch_size, w, "width") for w in all_bert_data[method].width]))
        

        # print(model.predict_throughput(1))
        # print(model.predict_max_throughput(16*1024))
