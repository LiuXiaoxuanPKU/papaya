from abc import ABCMeta, abstractmethod
import sys

import torch

class local_sharing():
    def __init__(self, func):
        self._locals = {}
        self.func = func

    def __call__(self, *args, **kwargs):
        def tracer(frame, event, arg):
            if event=='return':
                self._locals = frame.f_locals.copy()
        sys.setprofile(tracer)
        try:
            res = self.func(*args, **kwargs)
        finally:
            sys.setprofile(None)
        return res

    def clear_locals(self):
        self._locals = {}

    @property
    def locals(self):
        return self._locals


class PapayaProfiler:
    __metaclass__ = ABCMeta

    @local_sharing
    def init(self):
        pass

    @abstractmethod
    def create_dataloader(self, batch_size):
        pass

    def post_create_dataloader():
        pass

    
    @abstractmethod
    def run_iter(self, batch):
        pass


    def profile(self):
        batch_times = []
        self.init()
        for batch_size in [4, 8, 12]:
            dataloader = self.create_dataloader(batch_size)
            self.post_create_dataloader()
            
            for i, batch in enumerate(dataloader):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                start.record()

                self.run_iter(batch)

                torch.cuda.synchronize()
                end.record()
                batch_time = start.elapsed_time(end) / 1000.0 # event in ms


    def get_p_point(self):
        pass

    def get_p_score(self):
        pass

    def predict_max_tpt(self, op, fragmentation_ratio):
        pass

    def predict_tpt(self, op, batch_size):
        pass

    def predict_peak_mem(self, op, batch_size):
        pass

class Papaya:
    def __init__(self) -> None:
        pass

    def set_optimization(self, op):
        pass

    def start_iter(self):
        pass

    def finish_iter(self):
        pass

    def get_point(self):
        pass

    def get_score(self, op):
        pass

    def predict_max_tpt(self, op, fragmentation_ratio):
        pass

    def predict_tpt(self, op, batch_size):
        pass

    def predict_peak_mem(self, op, batch_size):
        pass

    def dump_info():
        pass

    # ======================= Private Methods Below ===========================

import exp_bert, exp_gpt, exp_swin, argparse, utilizations
algo_dict = {
    "swin": exp_swin.Experiment,
    "bert": exp_bert.Experiment,
    "gpt": exp_gpt.Experiment
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--machine-tag",
        nargs='*',
        type=str,
        default=["v100"],
        help="tag for machine configuration, e.g. v100/t4/4v100",
    )
    parser.add_argument(
        "--run-new", help="run experiment from scratch,\
            otherwise using existing data", action='store_true'
    )
    parser.add_argument(
        "--plot-graph", help="plot graph for experiment data", action='store_true'
    )
    parser.add_argument('--algos', nargs='*', type=str)
    args = parser.parse_args()

    if args.algos and len(args.algos): algos = [a.lower() for a in args.algos]
    else: algos = list(algo_dict.keys())
    if not all(m in algo_dict for m in algos):
        print("Framework not covered in experiments.")
        return
    if "all" in args.machine_tag: args.machine_tag = ["t4","v100","4v100"]
    if args.run_new:
        # run experiment code to be filled
        if not all(mt[-1]==args.machine_tag[0][-1] for mt in args.machine_tag):
            print("[ERROR] Please specify a single tag for current machine configurations.")
            return
        else:
            for mt in args.machine_tag:
                for m in algos: 
                    print("================={}@{}=================".format(m,mt))
                    algo_dict[m].run_experiment(mt) 
    
    for tag in args.machine_tag:
        for m in algos: algo_dict[m].do_plot(tag,args.plot_graph)