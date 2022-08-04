fairseq-train --task language_modeling \
  ~/dataset/data-bin/wikitext-103 \
  --save-dir checkpoints/transformer_wikitext-103 \
  --arch transformer_lm_gpt3_small --share-decoder-input-output-embed \
  --dropout 0.1 \
  --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
  --tokens-per-sample 512 --sample-break-mode none \
  --max-tokens 4608 --update-freq 16\
  --fp16 \
  --max-update 50000 \
  --required-batch-size-multiple 1
  # --checkpoint-activations \
  # --alg ckpt
# python exp_mem_speed.py
