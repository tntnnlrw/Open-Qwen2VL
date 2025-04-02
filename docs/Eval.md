# MLLM Evaluation

## Install
Please continue the installation within the previously installed [prismatic-vlms](../README.md##Install):
```Shell
conda activate mllm
pip install -e vlm-evaluation
```

## Benchmark Evaluation for Pre-Trained or Fine-Tuned MLLMs
Please run
```Shell
bash eval.sh ${model_dir}
```
- `model_dir`: the directory of the pre-trained or fine-tuned MLLM