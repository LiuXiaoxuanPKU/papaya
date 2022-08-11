# python exp_mem_speed.py --mode binary_search_max_layer
# python exp_mem_speed.py --mode binary_search_max_hidden_size
echo "======================Bert FP16====================="
# python exp_mem_speed.py --mode linear_scan --layer_num 24
python exp_mem_speed.py --mode linear_scan --layer_num 24 --get_mem
