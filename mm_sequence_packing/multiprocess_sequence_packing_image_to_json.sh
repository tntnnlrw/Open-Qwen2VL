#!/bin/bash
num_gpu=$2
gpu_start_id=$1

gpu_name=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader)

start=`date +"%s"`

mkdir $5

i=0
while [ $i -lt $num_gpu ]; do
  {
    echo $i

    python mm_sequence_packing/sequence_packing_image_to_json.py --tar-file-path $4 --save-path $5 --gpu-id $(($i + $gpu_start_id)) --tars-per-gpu $3
  } &
  i=$(($i + 1))
done

wait
end=`date +"%s"`
echo "time: " `expr $end - $start`
