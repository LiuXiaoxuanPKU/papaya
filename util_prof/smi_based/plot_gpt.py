import matplotlib.pyplot as plt
import numpy as np
colors = {1:"black",2:"red",3:"yellow",4:"blue",5:"green",6:"cyan",8:"brown"}
result_path = "./gpt_result"
graph_path = "./graph/gpt"
def plot_multiple(method):
    batch_size = [2,3,4,5]
    # f, plots = plt.subplots(len(batch_size), sharex=True, sharey = True)
    fig, ax = plt.subplots(sharey=True)
    plt.yticks([0,20,40,60,80,100])
    for i,bz in enumerate(batch_size):
        ut = []
        ts = []
        fn = "%s/%s/%d.log"%(result_path,method,bz)
        # ax=fig.add_subplot(label="bz=%d"%bz,frame_on = (i==0),sharey=True)
        with open(fn,"r") as f:
            lines = f.readlines()
        for j,line in enumerate(lines):
            if j<1: continue
            dp = line.split(":")
            ts.append(j-1)
            ut.append(dp[1])
        ax.plot(ts, ut, colors[bz], label="bz=%d"%bz)
    #plt.yticks([0,10,20,30,40,50,75,100])
    plt.legend()
    plt.savefig('multiple.png', bbox_inches='tight')
    fig = plt.figure()

def plot_average(method):
    batch_size = [1,2,3,4,5,6,8,10]
    avgs = []
    for bz in batch_size:
        with open("%s/%s/%d.log"%(result_path,method,bz),"r") as f:
            ut = [int(line.split(":")[1]) for line in f.readlines()]
        ut = ut[1:]
        avgs.append(np.average(ut))
    plt.plot(batch_size,avgs, label="average utilizations")
    # plt.legend(prop={'size': 14})
    plt.tick_params(axis='x', labelsize=14)
    plt.tick_params(axis='y', labelsize=14) 
    plt.xlabel("Batch size", size=16)
    fig = plt.gcf()
    fig.subplots_adjust(left=0.15)
    fig.text(0, 0.45, 'GPU Utilization(%)', va='center', rotation='vertical', size=16)
    fig.set_size_inches(4, 4) 
    plt.savefig('%s/%s_avg.pdf'%(graph_path,method), bbox_inches='tight')
    plt.close()
   
if __name__=="__main__":
    for method in ["none","ckpt"]:
        plot_average(method)