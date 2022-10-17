import numpy as np
from .style import PlotStyle

result_path = "./data"
graph_path = "./graph"

def plot_average(method,result_path,graph_path,algo, model, btime):
    if algo=="bert": batch_size = list(range(4,58,2))
    else: batch_size = [1,2,3,4,5,6,8,10]
    avgs = []
    for bz in batch_size:
        with open("%s/%s_%s/%d.log"%(result_path,algo,method,bz),"r") as f:
            ut = [int(line.split(":")[1]) for line in f.readlines()]
        ut = ut[1:]
        avgs.append(np.average(ut))
    PlotStyle.plot(batch_size,avgs,result_path,graph_path,method,algo, model, btime)
   
if __name__=="__main__":
    for method in ["none","ckpt"]: 
        plot_average(method,result_path,graph_path,"bert")
        plot_average(method,result_path,graph_path,"gpt")