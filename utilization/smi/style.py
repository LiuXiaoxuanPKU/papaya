import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('axes',edgecolor='silver')
suffix = "pdf"
class PlotStyle:
    def plot(batch_size,avgs,result_path,graph_path,method,algo):
        plt.style.use(['science','ieee'])
        plt.grid(c="paleturquoise")
        plt.plot(batch_size,avgs, label="average utilizations",color="seagreen",linewidth=2.5)
        plt.xlabel("Batch Size", size=22)
        plt.ylabel("GPU Utilization(\%)", size=22)
        plt.tick_params(axis='x', labelsize=18)
        plt.tick_params(axis='y', labelsize=18)  
        fig = plt.gcf()
        fig.set_size_inches(4, 4) 
        plt.savefig('%s/%s_%s_smi.%s'%(graph_path,algo,method,suffix), bbox_inches='tight')
        plt.close()