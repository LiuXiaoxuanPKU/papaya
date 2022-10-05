import argparse
from datetime import datetime
import os

import suite_gpt

def run_cmd(cmd):
    print(cmd)
    return os.system(cmd)

benchmark_suites = {
    "gpt.tmp": suite_gpt.tmp_suite,
    #"gpt.grid_search_manual": suite_gpt.grid_search_manual,
}

def benchmark_all(args):
    num_gpus = args.nproc_per_node * args.nnodes

    try:
        _ = benchmark_suites[args.suite][num_gpus]
    except KeyError:
        print(f"No available benchmark suite for {args.suite} with {num_gpus} GPUs.")
        exit()
    output_name = args.exp_name + "-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    model = args.suite.split(".")[0]

    for remat in [False, True]:
        for depth in range(24, 80):
            fail = 0
            for bsz in list(range(1, 4)) + list(range(4, 40, 4)):
                case = suite_gpt.BenchmarkCase(bsz, suite_gpt.GPTModelConfig(1024, 1024, depth, 16, 51200), 1, "uniform",
                            suite_gpt.UniformParallelArgs(True, remat, 1, 1, 1, True))
    # for case in benchmark_suites[args.suite][num_gpus]:
                case = tuple(tuple(x) if isinstance(x, tuple) else x for x in case)
                case_str = str((model,) + case)

                if args.nnodes == 1:
                    # Single node
                    ret = run_cmd('python3 -m torch.distributed.launch '
                                f'--nproc_per_node {args.nproc_per_node} '
                                'benchmark_gpt_bert_one_case.py '
                                f'"{case_str}" '
                                f'{output_name}')
                    if ret > 0:
                        fail += 1
                    if fail >= 2:
                        break
                else:
                    # Multiple nodes
                    ret = run_cmd('python3 -m torch.distributed.launch '
                                f'--nproc_per_node {args.nproc_per_node} '
                                f'--nnodes {args.nnodes} '
                                f'--node_rank {args.node_rank} '
                                f'--master_addr {args.master_addr} '
                                f'--master_port {args.master_port} '
                                'benchmark_gpt_bert_one_case.py '
                                f'"{case_str}" '
                                f'{output_name}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nproc_per_node", type=int, required=True)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node_rank", type=int)
    parser.add_argument("--master_addr", type=str)
    parser.add_argument("--master_port", type=str)
    parser.add_argument("--suite", type=str, default="gpt.tmp")
    parser.add_argument("--exp_name", type=str, default="")
    args = parser.parse_args()

    benchmark_all(args)