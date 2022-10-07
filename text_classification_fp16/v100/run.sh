# python exp_mem_speed.py --mode binary_search_max_layer
# python exp_mem_speed.py --mode binary_search_max_hidden_size
echo "======================Bert FP16====================="
# python exp_mem_speed.py --mode linear_scan --layer_num 24

for i in {18..80..4}
do
    python exp_mem_speed.py --mode binary_search_max_batch --layer_num $i
done

