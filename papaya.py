import exp_bert, exp_gpt, exp_swin, argparse, utilizations
algo_dict = {
    "swin": exp_swin.Experiment,
    "bert": exp_bert.Experiment,
    "gpt": exp_gpt.Experiment
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--machine-tag",
        nargs='*',
        type=str,
        default=["v100"],
        help="tag for machine configuration, e.g. v100/t4/4v100",
    )
    parser.add_argument(
        "--run-new", help="run experiment from scratch,\
            otherwise using existing data", action='store_true'
    )
    parser.add_argument(
        "--plot-graph", help="plot graph for experiment data", action='store_true'
    )
    parser.add_argument('--algos', nargs='*', type=str)
    args = parser.parse_args()

    if args.algos and len(args.algos): algos = [a.lower() for a in args.algos]
    else: algos = list(algo_dict.keys())
    if not all(m in algo_dict for m in algos):
        print("Framework not covered in experiments.")
        return
    
    if args.run_new:
        # run experiment code to be filled
        if len(args.machine_tag)!=1:
            print("[ERROR] Please specify a single tag for current machine configurations.")
            return
        else:
            for m in algos: algo_dict[m].run_experiment(*args.machine_tag)
            
    if "all" in args.machine_tag: args.machine_tag = ["t4","v100","4v100"]
    for tag in args.machine_tag:
        for m in algos: algo_dict[m].do_plot(tag,args.plot_graph)



if __name__=="__main__":
    main()
