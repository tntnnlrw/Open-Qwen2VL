CKPTID=$1
DATAID=$2
STAGE=$3
TRAIN_NUM_SAMPLES=$4
BSZ=$5
PER_GPU_BSZ=$6

torchrun --nproc_per_node 8 scripts/pretrain.py \
  --stage ${STAGE} \
  --model.type "one-stage+7b" \
  --model.model_id qwen2.5-1.5b-instruct-continue-training-${CKPTID} \
  --model.arch_specifier "no-align+avgpool" \
  --model.vision_backbone_id "siglip-vit-so400m-384px" \
  --model.image_resize_strategy "resize-naive" \
  --model.llm_backbone_id "qwen2.5-1.5b-instruct" \
  --model.pretrain_global_batch_size ${BSZ} \
  --model.pretrain_per_device_batch_size ${PER_GPU_BSZ} \
  --model.pretrain_epochs 1 \
  --mount_path Qwen \
  --run_root_dir checkpoints/ \
  --dataset.type "caption" \
  --trackers=["jsonl",] \
  --dataset.train_num_samples ${TRAIN_NUM_SAMPLES} \
  --dataset.dataset_root_dir data/datacomp/datacomp_hq_single_pkl_pil:data/ccs/ccs_single_pkl_pil/:data/laion/laion_single_pkl_pil/