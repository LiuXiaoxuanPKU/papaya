import json
import matplotlib.pyplot as plt

with open('speed_results.json', 'r') as f:
    lines = f.readlines()
   
results = {} 
for l in lines:
    l = l.strip()
    if len(l) == 0:
        continue
    r = json.loads(l)
    alg, hz, ips = r['algorithm'], r['hidden_size'], r['ips']
    if alg not in results:
        results[alg] = {}
    if hz not in results[alg]:
        results[alg][hz] = 0
    if ips > results[alg][hz]:
        results[alg][hz] = ips

xs = list(results[None].keys())
ys = list(results[None].values())
plt.scatter(xs, ys)

xs = list(results['L1'].keys())
ys = list(results['L1'].values())
plt.scatter(xs, ys)

plt.savefig('model')
    
    

    
