import json
import matplotlib.pyplot as plt

MB = 1000000
obj = json.load(open("tensor_size", "r"))
values = obj.values()
plt.hist([v / MB for v in values])
plt.savefig("tensor")