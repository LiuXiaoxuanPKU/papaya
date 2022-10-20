import json
from unittest import result
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import sys
sys.path.append('.')
from plot_case_study.plot_util import ALG_COLOR, ALG_MAP, ALG_MARKER
from util import Util

with open('text_classification_fp16/v100/results/speed_results_hidden_size.json', 'r') as f:
    lines = f.readlines()

suffix = "pdf"
results = {} 
for l in lines:
    if len(l) == 0:
        continue
    r = json.loads(l)
    if 'hidden_size' not in r:
        continue
    alg, hz, ips = r['algorithm'], int(r['hidden_size']), r['ips']
    if alg is None:
        print(r)
    if hz < 320 or hz > 1900:
        continue
    if alg is None and hz == 1600:
        print(alg, hz, ips)
    if alg not in results:
        results[alg] = {}
    if hz not in results[alg]:
        results[alg][hz] = 0
    if ips > results[alg][hz]:
        if alg is None and hz == 1600:
            print(alg, hz, ips, results[alg][hz])
        results[alg][hz] = ips

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(6, 6)
xs = list(results[None].keys())
ys = list(results[None].values())
polynomial_features = PolynomialFeatures(degree=5, include_bias=False)
linear_regression = LinearRegression()
pipeline = Pipeline(
    [
        ("polynomial_features", polynomial_features),
        ("linear_regression", linear_regression),
    ]
)
pipeline.fit(np.array(xs).reshape(-1, 1), ys)
# ax.plot(xs, pipeline.predict(np.array(xs).reshape(-1, 1)), label="exact_predict")
alg = 'exact'
sample_cnt = 10
xs, ys = Util.sample_data(xs, sample_cnt), Util.sample_data(ys, sample_cnt)
ax.plot(xs, ys, label=ALG_MAP[alg], marker=ALG_MARKER[alg], color=ALG_COLOR[alg], ms=12, lw=3)

xs = list(results['L1'].keys())
ys = list(results['L1'].values())
pipeline.fit(np.array(xs).reshape(-1, 1), ys)
# ax.plot(xs, pipeline.predict(np.array(xs).reshape(-1, 1)), label="quantize_predict")
alg = 'L1'
sample_cnt = 10
xs, ys = Util.sample_data(xs, sample_cnt), Util.sample_data(ys, sample_cnt)
ax.plot(xs, ys, label=ALG_MAP[alg], marker=ALG_MARKER[alg],color=ALG_COLOR[alg], ms=12, lw=3)

xs = list(results['ckpt'].keys())
ys = list(results['ckpt'].values())
xs = []
ys = []
for k in sorted(results['ckpt'].keys()):
    xs.append(k)
    ys.append(results['ckpt'][k])
    
pipeline.fit(np.array(xs).reshape(-1, 1), ys)
# ax.plot(xs, pipeline.predict(np.array(xs).reshape(-1, 1)), label="ckpt_predict")
alg = 'ckpt'
sample_cnt = 10
xs, ys = Util.sample_data(xs, sample_cnt), Util.sample_data(ys, sample_cnt)
ax.plot(xs, ys, label=ALG_MAP[alg], marker=ALG_MARKER[alg],color=ALG_COLOR[alg], ms=12, lw=3)

plt.grid()
plt.xlabel('Model width (hidden size)')
# plt.ylabel('max throughput (records/s)', size=15)
Util.set_tick_label_size([ax])
plt.legend(fontsize=20)
plt.tight_layout()
plt.savefig(f'graphs/case_study/large_model_width.{suffix}')
    
    

    
