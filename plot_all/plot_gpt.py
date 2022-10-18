import matplotlib.pyplot as plt
import json

filename = 'GPT/v100/results/speed_results.json'

NET_TO_NAME = {
    "transformer_lm_gpt3_small" : "GPT3-Small",
    "transformer_lm_gpt3_medium" : "GPT3-Medium",
    "transformer_lm_gpt3_large" : "GPT3-Large"
}

def plot(network):
    results = {}
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("/"):
                continue
            obj = json.loads(line)
            if network != obj['network']:
                continue
            if 'alg' not in obj:
                continue
            alg, bz, ips = obj['alg'], obj['batch_size'], obj['ips']
            if ips == -1:
                continue
            if alg is None:
                alg = 'exact'
            if alg not in results:
                results[alg] = {}
            results[alg][bz] = ips

    alg_map = {
        "exact" : "exact",
        "swap" : "swap",
        "L4bit-swap" : "quantize+swap",
        "swap-lz4" : "lz4+swap",
        "ckpt" : "checkpoint",
        "L1" : "quantize",
        "L1_ckpt" : "quantize+checkpoint",
        "swap_ckpt" : "swap+checkpoint"
    }

    fig, ax = plt.subplots()
    for alg in results:
        ax.plot(results[alg].keys(), results[alg].values(), label=alg_map[alg], marker='o')
        ax.legend(prop={"size":13})
        # ax.set_yscale("log")
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    ax.set_title(f"{NET_TO_NAME[network]}", size=22)
    ax.set_xlabel("batch size", size=22)
    ax.set_ylabel("throughput (record/s)", size=22)
    fig.savefig(f'graphs/case_study/{network}.pdf', bbox_inches='tight')
    plt.close()
    
    
plot("transformer_lm_gpt3_small")
plot("transformer_lm_gpt3_medium")
plot("transformer_lm_gpt3_large")