import matplotlib.pyplot as plt
from util import Util
import numpy as np

suffix = "pdf"
class PlotStyle:
    def plot(batch_size,avgs,result_path,graph_path,method,algo, model, btime):
        fig, ax1 = plt.subplots(1, 1)
        fig.set_size_inches(4, 4)

        bs_limit = min(max(btime.keys()), max(batch_size))
        xs, ys = [], []
        for i, bt in enumerate(btime.keys()):
            if bt <= bs_limit:
                xs.append(bt)
                ys.append(list(btime.values())[i])

        label_size = 20
        tick_size = 18

        ax2 = ax1.twinx()
        sample_cnt = 10
        ax2.scatter(Util.sample_data(xs, sample_cnt), Util.sample_data(ys, sample_cnt), label='latency')
        xs = np.array(xs)
        ax2.plot(xs, model(xs))
        ax2.set_ylabel("Batch Latency (s)", size=label_size)
        
        ax1.plot(batch_size,avgs, label="utilizations",color="seagreen",linewidth=2.5)
        ax1.set_xlabel("Batch Size", size=label_size)
        ax1.set_ylabel("GPU Utilization(%)", size=label_size)
        
        for ax in [ax1, ax2]:
            ax.tick_params(axis='x', labelsize=tick_size)
            ax.tick_params(axis='y', labelsize=tick_size)  
        fig.savefig('%s/%s_%s_smi.%s'%(graph_path,algo,method,suffix), bbox_inches='tight')