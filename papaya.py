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
