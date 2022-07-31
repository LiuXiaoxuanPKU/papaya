import matplotlib.pyplot as plt
import numpy as np
colors = {10:"black",16:"red",20:"yellow",30:"blue",32:"green",40:"cyan",36:"brown"}
def plot():
    batch_size = [10,20,30,40]
    # f, plots = plt.subplots(len(batch_size), sharex=True, sharey = True)
    fig, ax = plt.subplots(sharey=True)
    plt.yticks([0,20,40,60,80,100])
    for i,bz in enumerate(batch_size):
        ut = []
        ts = []
        fn = "utilization_%d.log"%bz
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

def get_average():
    batch_size = list(range(1,19))
    avgs = []
    for bz in batch_size:
        with open("None_%d_util.log"%bz,"r") as f:
            ut = [int(line.split(":")[1]) for line in f.readlines()]
        ut = ut[5:]
        print(ut)
        avgs.append(np.average(ut))
    plt.plot(batch_size,avgs, label="average utilizations")
    # plt.legend()
    plt.tick_params(axis='x', labelsize=14)
    plt.tick_params(axis='y', labelsize=14) 
    plt.xlabel("Batch size", size=16)
    fig = plt.gcf()
    fig.subplots_adjust(left=0.15)
    fig.text(0, 0.45, 'GPU Utilization(%)', va='center', rotation='vertical', size=16)
    fig.set_size_inches(4, 4) 
    plt.savefig('avg.pdf', bbox_inches='tight')
   
if __name__=="__main__":
    get_average()