"""
registry.py

Exhaustive list of pretrained VLMs (with full descriptions / links to corresponding names and sections of paper).
"""


# === Pretrained Model Registry ===
# fmt: off
MODEL_REGISTRY = {
    # === LLaVa v1.5 Reproductions ===
    "Open-Qwen2VL": {
        "model_id": "Open-Qwen2VL",
        "names": ["Open-Qwen2VL-2B-Instruct"],
        "description": {
            "name": "Open-Qwen2VL-2B-Instruct",
            "optimization_procedure": "2-stage",
            "visual_representation": "SigLIP ViT-SO/14 @ 384p",
            "image_processing": "Naive",
            "language_model": "Qwen2.5 1.5B Instruct",
            "train_epochs": 1,
        }
    },
    "Open-Qwen2VL-base": {
        "model_id": "Open-Qwen2VL-base",
        "names": ["Open-Qwen2VL-2B-base"],
        "description": {
            "name": "Open-Qwen2VL-2B-Instruct",
            "optimization_procedure": "2-stage",
            "visual_representation": "SigLIP ViT-SO/14 @ 384p",
            "image_processing": "Naive",
            "language_model": "Qwen2.5 1.5B Instruct",
            "train_epochs": 1,
        }
    }
}

# Build Global Registry (Model ID, Name) -> Metadata
GLOBAL_REGISTRY = {name: v for k, v in MODEL_REGISTRY.items() for name in [k] + v["names"]}

# fmt: on
