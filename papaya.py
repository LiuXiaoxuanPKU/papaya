import exp_bert, exp_gpt, exp_swin, argparse
algo_dict = {
    "swin": exp_swin.Experiment,
    "bert": exp_bert.Experiment,
    "gpt": exp_gpt.Experiment
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--machine-tag",
        required=True,
        type=str,
        default="v100",
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
    if args.run_new:
        # run experiment code to be filled
        raise NotImplementedError

    if args.algos and len(args.algos): algos = [a.lower() for a in args.algos]
    else: algos = list(algo_dict.keys())

    if not all(m in algo_dict for m in algos):
        print("Framework not covered in experiments.")
        return

    for m in algos: algo_dict[m].do_plot(args.machine_tag,args.plot_graph)

if __name__=="__main__":
    main()
