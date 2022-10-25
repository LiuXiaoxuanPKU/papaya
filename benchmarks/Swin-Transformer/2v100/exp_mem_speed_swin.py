import argparse
import json
import os
import time


def run_cmd(cmd):
    print(cmd)
    return os.system(cmd)


def network_to_command(network):
    cmd = "python -m torch.distributed.launch --nproc_per_node 2 --master_port 12345  main.py \
        --cfg configs/NET_patch4_window7_224.yaml --data-path ~/dataset/imagenet --batch-size BS "
    return cmd


def run_benchmark(network, batch_size, ckpt, fp16, actnn_level, get_mem):
    cmd = network_to_command(network)
    cmd = cmd.replace("BS", f"{batch_size}")
    cmd = cmd.replace("NET", f"{network}")

    if ckpt:
        cmd += " --use-checkpoint"
    if fp16 is not None:
        cmd += " --amp-opt-level " + fp16
    if actnn_level is not None:
        cmd += " --level " + actnn_level 
    if get_mem:
        cmd += " --get-mem"
        out_file = "results/mem_results.json"
    else:
        cmd += " --get-speed"
        out_file = "results/speed_results.json"
        
    ret_code = run_cmd(cmd)

    if ret_code != 0:
        with open(out_file, "a") as fout:
            val_dict = {
                "network": network,
                "algorithm": alg,
                "batch_size": batch_size,
                "ips": -1,
            }
            fout.write(json.dumps(val_dict) + "\n")
        print(f"save results to {out_file}")

    time.sleep(2)
    run_cmd("nvidia-smi > /dev/null")
    time.sleep(1)
    return ret_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retry", type=int, default=1)
    parser.add_argument("--get_mem", action='store_true')
    args = parser.parse_args()

    networks = ["swin_large"]
    # algs = ["fp32", "fp32-ckpt", "fp16O1", "fp16O1-ckpt", "fp16O2", "fp16O2-ckpt"]
    # algs = ["fp16O1", "fp16O1-ckpt", "fp16O1-L1", "fp16O1-swap"]
    algs = ["fp16O1", "fp16O1-ckpt", "fp16O1-L1"]
    for net in networks:
        for alg in algs:
            try_cnt = 0
            for batch_size in range(20, 800, 4):
                actnn_level = None
                if "ckpt" in alg:
                    ckpt = True
                else:
                    ckpt = False
                if "L1" in alg or "swap" in alg:
                    actnn_level = alg.split("-")[-1]
                if "fp16" in alg:
                    fp16 = alg.split("-")[0][-2:]
                else:
                    fp16 = None
                ret_code = run_benchmark(net, batch_size, ckpt, fp16, actnn_level, args.get_mem)
                if ret_code != 0:
                    try_cnt += 1
                    if try_cnt == 3:
                        break