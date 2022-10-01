# python exp_mem_speed.py --mode binary_search_max_layer
# python exp_mem_speed.py --mode binary_search_max_hidden_size
echo "======================Bert FP16====================="
# python exp_mem_speed.py --mode linear_scan --layer_num 24
# python exp_mem_speed.py --mode linear_scan --layer_num 24 --get_mem

for ((i = 2 ; i <= 100 ; i+=1)); do
  python exp_mem_speed.py --mode linear_scan --layer_num $i
done