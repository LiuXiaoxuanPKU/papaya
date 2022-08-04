import exp_bert, exp_gpt, exp_swin, argparse
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--machine-tag",
        required=True,
        type=str,
        default="v100",
        help="tag for machine configuration, e.g. v100/t4/4v100",
    )
    parser.add_argument(
        "--run-new", help="true to run experiment from scratch,\
            otherwise using existing data", action=argparse.BooleanOptionalAction
    )
    args = parser.parse_args()
    if args.run_new:
        # run experiment code to be filled
        raise NotImplementedError
    exp_swin.Experiment.do_plot(args.machine_tag)
    exp_bert.Experiment.do_plot(args.machine_tag)
    exp_gpt.Experiment.do_plot(args.machine_tag)