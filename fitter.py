from typing import Callable, Dict
import scipy.optimize as optimization

import numpy as np
from sklearn.metrics import r2_score

Data = Dict[int, int]
ModelFn = Callable[[np.array, float, float], np.array]
Prediction = Callable[[float], float]
Fitter = Callable[[Data, ModelFn], Prediction]


class ModelFnPool:
    def linear(x: np.array, a: float, b: float):
        return a * x + b


class FitterPool:
    def fit_leastsq(data: Data, fn: ModelFn) -> Prediction:
        ret,_,_,_ = FitterPool.fit_leastsq_verbose(data,fn)
        return ret

    def fit_leastsq_verbose(data: Data, fn: ModelFn) -> (Prediction, int, int, int):
        xs, ys = np.array(list(data.keys())), np.array(list(data.values()))
        # print("[xs]", xs)
        # print("[ys]", ys)
        guess_k = (ys[-1]-ys[-2])/(xs[-1]-xs[-2])
        guess_b = ys[0] - guess_k * xs[0] 
        popt, _ = optimization.curve_fit(ModelFnPool.linear, xs, ys, \
            [guess_k, guess_b])
        # print(popt)
        ret = lambda x: fn(x, *popt)
        r2 = r2_score(ys, ret(xs))
        return ret,r2,popt[0],popt[1]

    def fit_leastsq_verbose_offset(data: Data, fn: ModelFn, offset: int) -> (Prediction, int, int, int):
        xs, ys = np.array(list(data.keys())), np.array(list(data.values()))
        # print("[xs]", xs)
        # print("[ys]", ys)
        pys = [y-offset for y in ys]
        guess_k = (pys[-1]-pys[-2])/(xs[-1]-xs[-2])
        popt, _ = optimization.curve_fit(lambda x,k: k*x, xs, pys, \
            [guess_k])
        # print(popt)
        ret = lambda x: fn(x, popt[0],offset)
        r2 = r2_score(ys, ret(xs))
        return ret,r2,popt[0],offset