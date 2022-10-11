import json
from unittest import result
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

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
fig.set_size_inches(4, 4)
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
print(len(xs), xs)
print(len(ys), ys)
pipeline.fit(np.array(xs).reshape(-1, 1), ys)
# ax.plot(xs, pipeline.predict(np.array(xs).reshape(-1, 1)), label="exact_predict")
ax.plot(xs, ys, label='exact', marker='o', color='#2596be')
# plt.scatter([1600], [77.51], color='blue')

xs = list(results['L1'].keys())
ys = list(results['L1'].values())
pipeline.fit(np.array(xs).reshape(-1, 1), ys)
# ax.plot(xs, pipeline.predict(np.array(xs).reshape(-1, 1)), label="quantize_predict")
ax.plot(xs, ys, label='quantize', marker='o',color='orange')

xs = list(results['ckpt'].keys())
ys = list(results['ckpt'].values())
xs = []
ys = []
for k in sorted(results['ckpt'].keys()):
    xs.append(k)
    ys.append(results['ckpt'][k])
    
pipeline.fit(np.array(xs).reshape(-1, 1), ys)
# ax.plot(xs, pipeline.predict(np.array(xs).reshape(-1, 1)), label="ckpt_predict")
ax.plot(xs, ys, label='checkpoint', marker='o', color='green')

plt.xlabel('model width (hidden size)', size=15)
plt.ylabel('max throughput (records/s)', size=15)
plt.legend(fontsize='large')
ax.tick_params(axis='both', which='major', labelsize=15)
plt.tight_layout()
plt.savefig(f'graphs/implications/large_model.{suffix}')
    
    

    
