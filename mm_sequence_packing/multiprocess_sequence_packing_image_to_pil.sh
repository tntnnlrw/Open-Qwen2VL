#!/bin/bash
start_id=$1
num_proc=$2 # on our server we use 4; you can enlarge this based on you cpu cores
tars_per_gpu=$3 # the number of tar files processed by each process


gpu_name=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader)

start=`date +"%s"`

i=0
while [ $i -lt $num_proc ]; do
  {
    echo $i
    if [ "$4" == "obelics" ]; then
      mkdir Open-Qwen2VL-Data/obelics/obelics_single_pkl_pil
      python mm_sequence_packing/sequence_packing_image_to_pil.py --tar-file-path Open-Qwen2VL-Data/obelics_webdataset --save-path Open-Qwen2VL-Data/obelics_single_pkl_pil --gpu-id $(($i + $start_id)) --tars-per-gpu ${tars_per_gpu}
    elif [ "$4" == "synthdog" ]; then
      mkdir Open-Qwen2VL-Data/synthdog-en/synthdog_single_pkl_pil 
      python mm_sequence_packing/sequence_packing_image_to_pil.py --tar-file-path Open-Qwen2VL-Data/synthdog_webdataset --save-path Open-Qwen2VL-Data/synthdog_single_pkl_pil --gpu-id $(($i + $start_id)) --tars-per-gpu ${tars_per_gpu}
    elif [ "$4" == "ccs" ]; then
      python mm_sequence_packing/sequence_packing_image_to_pil.py --tar-file-path Open-Qwen2VL-Data/ccs_webdataset --save-path Open-Qwen2VL-Data/ccs_single_pkl_pil --gpu-id $(($i + $start_id)) --tars-per-gpu ${tars_per_gpu}
    elif [ "$4" == "laion" ]; then
      python mm_sequence_packing/sequence_packing_image_to_pil.py --tar-file-path Open-Qwen2VL-Data/laion_webdataset --save-path Open-Qwen2VL-Data/laion_single_pkl_pil --gpu-id $(($i + $start_id)) --tars-per-gpu ${tars_per_gpu}
    elif [ "$4" == "datacomp-dfn" ]; then
      python mm_sequence_packing/sequence_packing_image_to_pil.py --tar-file-path Open-Qwen2VL-Data/datacomp_medium_dfn_webdataset --save-path Open-Qwen2VL-Data/datacomp_dfn_single_pkl_pil --gpu-id $(($i + $start_id)) --tars-per-gpu ${tars_per_gpu}
    else
      python mm_sequence_packing/sequence_packing_image_to_pil.py --tar-file-path Open-Qwen2VL-Data/datacomp_medium_mlm_filter_su_85_union_dfn_webdataset/ --save-path Open-Qwen2VL-Data/datacomp_hq_single_pkl_pil --gpu-id $(($i + $start_id)) --tars-per-gpu ${tars_per_gpu}
    fi
  } &
  i=$(($i + 1))
done

wait
end=`date +"%s"`
echo "time: " `expr $end - $start`
