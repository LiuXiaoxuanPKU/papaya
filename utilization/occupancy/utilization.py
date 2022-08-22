from . import occupacy
import json
import matplotlib
matplotlib.use('Agg')
matplotlib.rc('axes',edgecolor='silver')
import matplotlib.pylab as plt

def parseUtil(raw_file, parsed_file, granularity = 0.1):
    bucket = 1000000000*granularity
    utilizations = {}
    sm_utilizations = {}
    ut_weighted = {}
    total_duration = 0
    with open(raw_file,"r") as f:
        data = json.load(f)
    for d in data:
        threads_per_block = d['blockX']*d['blockY']*d['blockZ']
        sm_util = min(d['gridX']*d['gridY']*d['gridZ']/80,1)
        registers_per_thread = d['registersPerThread']
        shared_memory_per_block = d['staticSharedMemory']+d['dynamicSharedMemory']
        duration =  float(d['end']) - float(d['start'])
        timestamp = float(d['start']) + (duration)/2
        input = {
        "version": "7.0",
        "threadsPerBlock": threads_per_block,
        "registersPerThread": registers_per_thread,
        "cudaVersion": "11.1",
        "sharedMemoryPerBlock" : shared_memory_per_block
        }
        oq = occupacy.calculateOccupancy(input)*100
        utilizations[timestamp] = oq
        sm_utilizations[timestamp] = sm_util
        
    min_time = min(utilizations.keys())
    new_util,new_sm_util = {},{}
    for k in utilizations:
        if (k-min_time)//bucket not in new_util: 
            new_util[(k-min_time)//bucket] = []
        if (k-min_time)//bucket not in new_sm_util:
            new_sm_util[(k-min_time)//bucket] = []
        new_util[(k-min_time)//bucket].append(utilizations[k])
        new_sm_util[(k-min_time)//bucket].append(sm_utilizations[k])
    for k in new_util:
        new_util[k] = sum(new_util[k])/len(new_util[k])
        new_sm_util[k] = sum(new_sm_util[k])/len(new_sm_util[k])
    _, ocs = zip(*sorted(new_util.items()))
    # print("avg = ",sum(ocs[2:-2])/len(ocs[2:-2]))
    # _, ocs = zip(*sorted(new_sm_util.items()))
    # print("sm avg time_aggregated= ",sum(ocs[2:-2])/len(ocs[2:-2]))
    # print("sm avg = ", sum(sm_utilizations.values())/len(utilizations.values()))
    # with open('result_sm_bz%d.json'%bz, 'w') as fp:
    #     json.dump(new_sm_util, fp)
    with open(parsed_file, 'w') as fp:
        json.dump(new_util, fp)

def plotUtil(parsed_file, graph_path, granularity = 0.1):
    with open(parsed_file,"r") as f:
        d = json.load(f)
    old_d = d
    d = dict()
    for k in old_d: d[float(k)] = float(old_d[k])
    lists = sorted(d.items())
    x, y = zip(*lists)
    x = [v*granularity for v in x]
    plt.style.use(['science','ieee'])
    plt.plot(x[2:-2],y[2:-2],color="seagreen")
    plt.xlabel("Time(s)", size=22)
    plt.ylabel("Kernel Occupancy(\%)", size=22)
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18) 
    fig = plt.gcf()
    fig.set_size_inches(4, 4) 
    plt.savefig(graph_path)
    plt.close()

if __name__ == "__main__": 
    plotUtil("./data/occupancy_bz_4.json","occupancy_bz_4.pdf",0.1)
    plotUtil("./data/occupancy_bz_64.json","occupancy_bz_64.pdf",0.1)