import numpy as np
from stype import PlotStyle

result_path = "./bert_result"
graph_path = "./graph/bert"

def plot_average(method):
    batch_size = list(range(4,58,2))
    avgs = []
    for bz in batch_size:
        with open("%s/%s/%d.log"%(result_path,method,bz),"r") as f:
            ut = [int(line.split(":")[1]) for line in f.readlines()]
        ut = ut[1:]
        avgs.append(np.average(ut))
    PlotStyle.plot(batch_size,avgs,result_path,graph_path,method)
   
if __name__=="__main__":
    for method in ["none","ckpt"]: plot_average(method)