import numpy as np
from sortedcontainers import SortedKeyList

from fitter import Fitter, FitterPool, ModelFnPool
from util import Util, Viewer


class Point:
    def __init__(self, bsize: int, mem: int, ips: int) -> None:
        self.bsize = bsize
        self.mem = mem
        self.ips = ips


class Model:
    def __init__(self, mem_dir: str, ips_dir: str, fit_method: Fitter) -> None:
        self.fit_method = fit_method

        self.mem_data = Util.load_data(mem_dir, "batch_size", "peak")
        self.btime_data = Util.load_data(ips_dir, "batch_size", "bacth_time")

        batches = self.mem_data.keys()
        self.range = (min(batches), max(batches))

        self.mem_model = fit_method(self.mem_data, ModelFnPool.linear)
        btime_model = fit_method(self.btime_data, ModelFnPool.linear)
        self.ips_model = lambda bsize: bsize / btime_model(bsize)

    def predict(self, bsize: float) -> Point:
        return Point(bsize, self.mem_model(bsize), self.ips_model(bsize))

    def __gt__(self, other) -> bool:
        other_sorted_points = SortedKeyList(key=lambda point: point.mem)
        MIN, MAX = other.range
        for bsize in range(MIN, MAX + 1):
            other_sorted_points.add(other.predict(bsize))

        MIN, MAX = self.range
        for bsize in range(MIN, MAX + 1):
            self_point = self.predict(bsize)
            nearest_other_point = other_sorted_points[
                other_sorted_points.bisect_key_left(
                    self_point.mem)]
            if nearest_other_point.ips > self_point.ips:
                return True

        return False


if __name__ == "__main__":
    mem_dir = "text_classification/results/mem_results.json"
    ips_dir = "text_classification/results/speed_results.json"

    model = Model(mem_dir, ips_dir, FitterPool.fit_leastsq)
    other = Model(mem_dir, ips_dir, FitterPool.fit_leastsq)

    assert (model > other) == False

    # plot ips fit
    Viewer.plot_fit(model.ips_model, np.array(list(model.btime_data.keys())), np.array(
        [bsize / model.btime_data[bsize] for bsize in model.btime_data]), "ips")
    # plot memory fit
    Viewer.plot_fit(model.mem_model, np.array(list(model.mem_data.keys())), np.array(
        list(model.mem_data.values())), "mem")
