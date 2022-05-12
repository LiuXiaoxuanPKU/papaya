from typing import Callable, Dict
import scipy.optimize as optimization

import numpy as np

Data = Dict[int, int]
ModelFn = Callable[[np.array, float, float], np.array]
Prediction = Callable[[float], float]
Fitter = Callable[[Data, ModelFn], Prediction]


class ModelFnPool:
    def linear(x: np.array, a: float, b: float):
        return a * x + b


class FitterPool:
    def fit_leastsq(data: Data, fn: ModelFn) -> Prediction:
        xs, ys = np.array(list(data.keys())), np.array(list(data.values()))
        popt, _ = optimization.curve_fit(ModelFnPool.linear, xs, ys)
        return lambda x: fn(x, *popt)
