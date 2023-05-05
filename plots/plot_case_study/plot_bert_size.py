import json
from unittest import result
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import sys
sys.path.append('.')
# from plots.plot_util import ALG_COLOR, ALG_MAP, ALG_MARKER
# from util import Util

with open('benchmarks/text_classification_fp16/v100/results/speed_results.json', 'r') as f:
    lines = f.readlines()

suffix = "pdf"
results = {} 
for l in lines:
    if len(l) == 0:
        continue
    r = json.loads(l)
    if 'hidden_size' not in r:
        continue
    batch_size = r["batch_size"]
    if batch_size != 24:
        continue
    alg, layer_num, ips = r['algorithm'], int(r['layer_num']), r['ips']
    if ips == -1:
        continue
    alg = "exact" if alg is None else alg
    if alg not in results:
        results[alg] = {}
    assert layer_num not in results[alg]
    results[alg][layer_num] = ips

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(6, 6)
for alg in results:
    xs = list(results[alg].keys())
    ys = list(results[alg].values())
    ax.plot(xs, ys, label=alg)

plt.grid()
# plt.xlabel('Model width (hidden size)')
# plt.ylabel('max throughput (records/s)', size=15)
# Util.set_tick_label_size([ax])
plt.legend(fontsize=20)
plt.tight_layout()
plt.savefig(f'graphs/case_study/bert_size.{suffix}')
    
    

    
