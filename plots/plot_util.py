ALG_MAP = {
        "exact" : "Exact",
        "swap" : "Swap",
        "L4bit-swap" : "Quantize+Swap",
        "swap-lz4" : "LZ4+Swap",
        "ckpt" : "Checkpoint",
        "L1" : "Quantize",
        "L1_ckpt" : "Quantize+Checkpoint",
        "swap_ckpt" : "Swap+Checkpoint",
        "cpu-off": "Swap+Checkpoint",
        "dtr" : "CKPT(DTR)"
    }

ALG_MARKER = {
    "exact" : "o",
    "swap" : "<",
    "L4bit-swap" : "P",
    "swap-lz4" : "d",
    "ckpt" : "X",
    "L1" : "^",
    "L1_ckpt" : ">",
    "swap_ckpt" : "x",
    "cpu-off": "x",
    "dtr" : "X"
    # "dtr" : "s"
}

ALG_COLOR = {
    "exact" : "royalblue",
    "swap" : "forestgreen",
    "L4bit-swap" : "chocolate",
    "swap-lz4" : "gold",
    "ckpt" : "olive",
    "L1" : "mediumvioletred",
    "L1_ckpt" : "lightseagreen",
    "swap_ckpt" : "dodgerblue",
    # "dtr" : "mediumpurple",
    "dtr" : "olive",
    "cpu-off" : "dodgerblue"
}

NET_TO_FOLDER = {
    "resnet50" : "resnet",
    "wide_resnet50_2" : "resnet",
    "resnet152" : "resnet",
    "bert-large-cased" : "text_classification_fp16",
    "swin_large" : "Swin-Transformer",
    "transformer_lm_gpt3_small" : "GPT"
}

NET_TO_ALGS = {
    "resnet50" : [None, "L1", "swap", "L4bit-swap", "dtr"],
    "wide_resnet50_2" : [None, "L1", "L4bit-swap", "swap", "dtr"],
    "bert-large-cased" : [None, "L1", "ckpt", "swap_ckpt", "L4bit-swap"],
    # "bert-large-cased" : [None, "ckpt"],
    # "swin_large" : [None, "ckpt"],
    "swin_large" : [None, "L1", "swap", "ckpt", "L4bit-swap"],
    # gpt should have [None, "L1", "swap", "L4bit-swap"]
    "transformer_lm_gpt3_small" : [None, "L1", "ckpt", "L4bit-swap"]
}