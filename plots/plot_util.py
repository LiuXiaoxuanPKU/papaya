ALG_MAP = {
        "exact" : "Exact",
        "swap" : "Swap",
        "L4bit-swap" : "Quantize+Swap",
        "swap-lz4" : "LZ4+Swap",
        "ckpt" : "Checkpoint",
        "L1" : "Quantize",
        "L1_ckpt" : "Quantize+Checkpoint",
        "swap_ckpt" : "Swap+Checkpoint",
        "dtr" : "CKPT(DTR)"
    }

ALG_MARKER = {
    "exact" : "o",
    "swap" : "^",
    "L4bit-swap" : "P",
    "swap-lz4" : "d",
    "ckpt" : "X",
    "L1" : "^",
    "L1_ckpt" : "o",
    "swap_ckpt" : "o",
    "dtr" : "s"
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
    "dtr" : "mediumpurple"
}

NET_TO_FOLER = {
    "resnet50" : "resnet",
    "wide_resnet50_2" : "resnet",
    "resnet152" : "resnet",
    "bert-large-cased" : "text_classification_fp16",
    "swin_large" : "Swin-Transformer",
    "transformer_lm_gpt3_small" : "GPT"
}