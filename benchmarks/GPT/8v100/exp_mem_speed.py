import argparse
import json
import os
import time

from gact.utils import exp_recorder

def run_cmd(cmd):
    print(cmd)
    return os.system(cmd)


def network_to_command(network, bz, max_exp = 50):
    token_per_sample = 512
    cmd = """fairseq-train --task language_modeling \
    ~/dataset/data-bin/wikitext-103 \
    --save-dir checkpoints/transformer_wikitext-103 \
    --arch %s --share-decoder-input-output-embed \
    --dropout 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 \
    --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
    --tokens-per-sample %d --sample-break-mode none  --distributed-world-size 8 \
    --max-tokens %d --update-freq 1 \
    --fp16 \
    --max-update 50000 \
    --required-batch-size-multiple 1 --exp %d""" % (network, token_per_sample,
                                                    bz * token_per_sample,max_exp)#,network+"_"+str(bz)+"_util.log")
    return cmd


def run_benchmark(network, batch_size, max_exp, alg, get_mem = False, large_bucket = False, get_util = False):
    print("get_util = ",get_util)
    cmd = network_to_command(network, batch_size, max_exp)
    exp_recorder.record("network", network)
    exp_recorder.record("alg", alg)
    #cmd += " --utpath %s_%d_util.log"%(alg,batch_size)
    if large_bucket:
        cmd += " --bucket-cap-mb 250 "
    if get_util:
        cmd += " --ut True "
    if alg == "ckpt":
        cmd += " --checkpoint-activations"
        cmd += " --alg ckpt"
    if alg == "cpu-off":
        cmd += " --offload-activations"
        cmd += " --alg cpu-off"
    if alg == "ckpt-cpu-off":
        cmd += " --checkpoint-activations --offload-activations"
        cmd += " --alg ckpt-cpu-off"
    # if alg == "swap":
    #     cmt += " ----ddp-backend fully_sharded --cpu-offload "
    #     cmt.replace("--optimizer adam","--optimizer cpu_adam")
    if alg in ["L1", "L4bit-swap"]:
       cmd += " --alg %s"% alg 
       
    if get_mem:
        out_file = "results/mem_results.json"
    else:
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
    parser.add_argument("--get_util", action='store_true')
    parser.add_argument("--large_bucket", action='store_true')
    parser.add_argument('--network', nargs='*', type=str)
    args = parser.parse_args()
    max_exp_init,max_exp = 30,30
    networks = args.network if args.network else ["transformer_lm_gpt3_small"] #["transformer_lm_gpt3_medium","transformer_lm_gpt3_large"]#
    algs = [None, "ckpt", "L1", "cpu-off", "L4bit-swap"]
    # algs = [None]
    # algs = ["cpu-off"]
    actnn_level = None
    
    for net in networks:
        for alg in algs:
            max_exp = max_exp_init//3 if alg == "swap" else max_exp_init
            try_cnt = 0
            for batch_size in range(1, 800, 1):
                import shutil
                try:
                    shutil.rmtree("checkpoints")
                except: pass
                ret_code = run_benchmark(net, batch_size, max_exp, alg, large_bucket = args.large_bucket, get_util = args.get_util)
                if ret_code != 0:
                    try_cnt += 1
                    if try_cnt == 3:
                        break