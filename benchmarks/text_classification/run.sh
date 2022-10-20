# python exp_mem_speed.py --mode binary_search_max_layer
# python exp_mem_speed.py --mode binary_search_max_hidden_size
# python exp_mem_speed.py --mode linear_scan --layer_num 24 --get_mem
# python exp_mem_speed.py --mode linear_scan --layer_num 24


python exp_mem_speed.py --mode linear_scan --layer_num 24 --hidden_size 256
python exp_mem_speed.py --mode linear_scan --layer_num 24 --hidden_size 512
# python exp_mem_speed.py --mode linear_scan --layer_num 24 --hidden_size 768
# python exp_mem_speed.py --mode linear_scan --layer_num 24 --hidden_size 1024
# python exp_mem_speed.py --mode linear_scan --layer_num 24 --hidden_size 1280
# python exp_mem_speed.py --mode linear_scan --layer_num 24 --hidden_size 1536
# python exp_mem_speed.py --mode linear_scan --layer_num 24 --hidden_size 1792
# python exp_mem_speed.py --mode linear_scan --layer_num 24 --hidden_size 2048
# python exp_mem_speed.py --mode linear_scan --layer_num 24 --hidden_size 2304
# python exp_mem_speed.py --mode linear_scan --layer_num 24 --hidden_size 2560
# python exp_mem_speed.py --mode linear_scan --layer_num 24 --hidden_size 2816
# python exp_mem_speed.py --mode linear_scan --layer_num 24 --hidden_size 3072
# python exp_mem_speed.py --mode linear_scan --layer_num 24 --hidden_size 3328