# Data Filtering for Open-Qwen2VL Pre-Training Data
## Install
We develop our data filtering code based on DataComp repo. Please firstly install the required packages:
```shell
cd data_filtering
pip install -r requirements.txt
```

## Data Downloading
### CC3M-CC12M-SBU
```shell
wget https://storage.googleapis.com/sfr-vision-language-research/BLIP/datasets/ccs_filtered.json
img2dataset --url_list ccs_filtered.json --input_format "json" \
        --url_col "url" --caption_col "caption" --output_format webdataset \
        --output_folder ccs_webdataset --processes_count 32 --thread_count 128 --image_size 512 \
        --resize_only_if_bigger=True --resize_mode="keep_ratio" --skip_reencode=True \
        --enable_wandb False
```

### DataComp-Medium-128M
Please follow the [offical scripts](https://github.com/mlfoundations/datacomp) to download it into webdataset format.


## Data Quality Score Generation
### DFN-CLIP
Since we do not have the access to DFN model checkpoint, we directly use [the released uids of selected high-quality subset from DFN](https://huggingface.co/datasets/apf1/datafilteringnetworks_2b/resolve/main/datacomp_medium_dfn_20m_inds.npy).


### MLM-Filter
MLM-Filter adopts an efficient MLLM to generate four distince and comprehensive metrics to assess the quality of each image-text caption data sample. Please follow the official repo to perform the large scale quality score generation using [mlm-filter-qwen2.5-1.5b-gpt4o](https://huggingface.co/weizhiwang/mlm-filter-qwen2.5-1.5b-gpt4o).

## Data Filtering for MLM-Filter
```shell
INPUT_DATADIR="path/to/datacomp/shards"
python baselines.py --metadata_dir $INPUT_DATADIR --save_path medium_filter_results/medium_semantic_understanding_th_85.npy --name llava_semantic_understanding_score --threshold 85
mkdir "path/to/datacomp/medium_semantic_understanding_th_85"
python resharder.py -i $DOWNLOAD_DIR -o "path/to/datacomp/medium_semantic_understanding_th_85" -s medium_filter_results/medium_semantic_understanding_th_85.npy
```