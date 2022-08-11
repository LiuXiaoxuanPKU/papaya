import numpy as np
from stype import PlotStyle

result_path = "./gpt_result"
graph_path = "./graph/gpt"

def plot_average(method):
    batch_size = [1,2,3,4,5,6,8,10]
    avgs = []
    for bz in batch_size:
        with open("%s/%s/%d.log"%(result_path,method,bz),"r") as f:
            ut = [int(line.split(":")[1]) for line in f.readlines()]
        ut = ut[1:]
        avgs.append(np.average(ut))
    PlotStyle.plot(batch_size,avgs,result_path,graph_path,method)
   
if __name__=="__main__":
    for method in ["none","ckpt"]: plot_average(method)