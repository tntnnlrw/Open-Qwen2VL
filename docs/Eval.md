# MLLM Evaluation

## Install
Please continue the installation within the previously installed [openqwen2vl](../README.md##Install) environment and ensure the modified `prismatic-vlms` repo has been installed:
```Shell
conda activate openqwen2vl
pip install -e vlm-evaluation
```

## Benchmark Evaluation for Pre-Trained or Fine-Tuned MLLMs
Please run
```Shell
bash eval.sh ${model_dir}
```
- `model_dir`: the directory of the pre-trained or fine-tuned MLLM