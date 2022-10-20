# python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345  main.py \
#	--cfg configs/swin_small_patch4_window7_224.yaml --data-path ~/imagenet --batch-size 32 --get-mem --level L1

#python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345  main.py \
#  --cfg configs/swin_large_patch4_window7_224.yaml --data-path ~/dataset/imagenet --batch-size 8 \
#  --get-speed --amp-opt-level O2 --use-checkpoint

echo "====================Swin Transformer=============="
python exp_mem_speed_swin.py
python exp_mem_speed_swin.py --get_mem