"""
mmlu.py

Task Runner, Dataset Definitions, Builder Functions, and Evaluation Logic for the MMLU visual question answering
dataset. Only loads and processes the MMLU Test-Dev Balanced Split (`testdev_balanced_questions.json`) -- the default
MMLU held-out evaluation split.

"""
import json
import os
import re
import string
import subprocess
from pathlib import Path
from random import Random
from typing import Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision.transforms import Compose
from tqdm import tqdm
import pandas as pd
from numpy import *

from vlm_eval.overwatch import initialize_overwatch
from vlm_eval.tasks.registry import DATASET_REGISTRY
from vlm_eval.util.interfaces import VLM, ImageProcessor

# Initialize Overwatch =>> Wraps `logging.Logger` and `accelerate.PartialState`
overwatch = initialize_overwatch(__name__)

choices = ["A", "B", "C", "D"]

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def format_example(df, idx, include_answer=True):
    # prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(df.loc[idx, "subject"]))
    prompt = df.loc[idx, "question"]
    n_choices = len(df.loc[idx, "choices"])
    for j in range(n_choices):
        prompt += "\n{}. {}".format(choices[j], df.loc[idx, "choices"][j])
    prompt += "\nAnswer with the option's letter from the given choices directly.\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(choices[df.loc[idx, "answer"]])
    return prompt


def get_subcategories():
    return {
        "abstract_algebra": ["math"],
        "anatomy": ["health"],
        "astronomy": ["physics"],
        "business_ethics": ["business"],
        "clinical_knowledge": ["health"],
        "college_biology": ["biology"],
        "college_chemistry": ["chemistry"],
        "college_computer_science": ["computer science"],
        "college_mathematics": ["math"],
        "college_medicine": ["health"],
        "college_physics": ["physics"],
        "computer_security": ["computer science"],
        "conceptual_physics": ["physics"],
        "econometrics": ["economics"],
        "electrical_engineering": ["engineering"],
        "elementary_mathematics": ["math"],
        "formal_logic": ["philosophy"],
        "global_facts": ["other"],
        "high_school_biology": ["biology"],
        "high_school_chemistry": ["chemistry"],
        "high_school_computer_science": ["computer science"],
        "high_school_european_history": ["history"],
        "high_school_geography": ["geography"],
        "high_school_government_and_politics": ["politics"],
        "high_school_macroeconomics": ["economics"],
        "high_school_mathematics": ["math"],
        "high_school_microeconomics": ["economics"],
        "high_school_physics": ["physics"],
        "high_school_psychology": ["psychology"],
        "high_school_statistics": ["math"],
        "high_school_us_history": ["history"],
        "high_school_world_history": ["history"],
        "human_aging": ["health"],
        "human_sexuality": ["culture"],
        "international_law": ["law"],
        "jurisprudence": ["law"],
        "logical_fallacies": ["philosophy"],
        "machine_learning": ["computer science"],
        "management": ["business"],
        "marketing": ["business"],
        "medical_genetics": ["health"],
        "miscellaneous": ["other"],
        "moral_disputes": ["philosophy"],
        "moral_scenarios": ["philosophy"],
        "nutrition": ["health"],
        "philosophy": ["philosophy"],
        "prehistory": ["history"],
        "professional_accounting": ["other"],
        "professional_law": ["law"],
        "professional_medicine": ["health"],
        "professional_psychology": ["psychology"],
        "public_relations": ["politics"],
        "security_studies": ["politics"],
        "sociology": ["culture"],
        "us_foreign_policy": ["politics"],
        "virology": ["health"],
        "world_religions": ["philosophy"],
    }


# === Dataset Indexing / Building Utilities ===
def build_mmlu_indices(root_dir: Path, slim_dataset_sizes: Optional[Tuple[int, ...]], seed: int = 21, shots: int = 5) -> List[Path]:
    """Parse MMLU --> build & write index files w/ necessary VQA keys + additional dataset-specific data."""
    paths = DATASET_REGISTRY["mmlu"]["paths"]
    os.makedirs(dataset_dir := root_dir / paths["dataset_dir"], exist_ok=True)

    # Short-Circuit (if index files have already been built)
    index_files = [dataset_dir / "metadata-full.json"] + (
        []
        if slim_dataset_sizes is None
        else [dataset_dir / f"metadata-slim-{n_slim}.json" for n_slim in slim_dataset_sizes]
    )
    if all([index_file.exists() for index_file in index_files]):
        return index_files

    # Otherwise, load the raw annotations (questions & answers) from the downloaded MMLU raw data
    dev_questions = pd.read_parquet((root_dir / paths["dev"]))
    subject_to_demonstrations = {}
    for i in range(0, dev_questions.shape[0], 5):
        subject = dev_questions.loc[i, "subject"]
        demonstration_prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
        for j in range(i, i+5):
            assert dev_questions.loc[j, "subject"] == subject
            demonstration_prompt += format_example(dev_questions, j)
        subject_to_demonstrations[subject] = demonstration_prompt
    
    test_questions = pd.read_parquet((root_dir / paths["test"]))

    # Build Full Metadata Structure
    index = {}
    for question_id, example in tqdm(test_questions.iterrows(), desc="=> Processing MMLU Raw Dataset:", leave=False):
        qid: int = int(question_id)

        # Build Metadata Entry
        # fmt: off
        index[qid] = {
            # [Required] VQA Task Keys
            "demonstrations": subject_to_demonstrations[example["subject"]],
            "question_id": qid,
            "question": format_example(test_questions, question_id, include_answer=False),
            "answer": choices[example["answer"]],    # MMLU only provides a single ground-truth answer + long answer explanation

            # [Dataset-Specific] Additional Keys
            "subject": example["subject"],
        }
        # fmt: on

    # Assertions on known quantities from MMLU Test-Dev Set
    assert len(index) == 14042, "Expected 14,042 unique questions/answers for MMLU Test-Dev-Balanced!"

    # IMPORTANT =>> Shuffle Question ID order *once* then slice into when building slim datasets
    #               This allows us to 1) have balanced images / shards for the full-scale validation dataset and
    #                                 2) have slim datasets that build off each other (enables caching / testing)
    all_qids = list(index.keys())
    Random(seed).shuffle(all_qids)  # Python `random.shuffle` is an in-place operation for... reasons...

    # Write `metadata.json` (for the complete evaluation set)
    for index_file in index_files:
        if index_file.name == "metadata-full.json":
            with open(index_file, "w") as f:
                json.dump({k: index[k] for k in all_qids}, f)

            # Dump All Questions/Answers to `dataset_dir` in the exact same format as `paths["questions_answers"]`
            # gt_qid2question = {str(qid): qid2question[str(qid)] for qid in all_qids}
            # with open(dataset_dir / "annotations-mmlu-full.json", "w") as f:
            #     json.dump(gt_qid2question, f)

        elif index_file.name.startswith("metadata-slim-"):
            n_slim = int(re.search("-slim-(.+?).json", index_file.name).group(1))
            with open(index_file, "w") as f:
                json.dump({k: index[k] for k in all_qids[:n_slim]}, f)

            # Dump Sampled Questions/Answers to `dataset_dir` in the exact same format as `paths["questions_answers"]`
            # slim_qid2question = {str(qid): qid2question[str(qid)] for qid in all_qids[:n_slim]}
            # with open(dataset_dir / f"annotations-mmlu-slim-{n_slim}.json", "w") as f:
            #     json.dump(slim_qid2question, f)

        else:
            raise ValueError(f"Received unexpected index file `{index_file}`")

    return index_files


# === Index (Metadata-Only) Dataset Declarations ===
class MMLUIndexDataset(Dataset):
    def __init__(self, root_dir: Path, index_file: Path) -> None:
        """Constructs a lightweight PyTorch Dataset that loads from an index file and just returns metadata."""
        self.root_dir, self.index_file = root_dir, index_file

        # Load from `index_file` --> Dict :: qid -> { question / answer / image data } --> flatten
        with open(self.root_dir / self.index_file, "r") as f:
            self.examples = list(json.load(f).values())

    def __getitem__(self, idx: int) -> Tuple[int, str, Path, str]:
        """Return (question_id: int, question: str, img_path: Path, answer: str) for an example."""
        ex = self.examples[idx]
        return ex["question_id"], ex["question"], ex["answer"]

    def __len__(self) -> int:
        return len(self.examples)


# === Map/Iterable Dataset Declarations ===
class MMLUMapDataset(Dataset):
    def __init__(
        self, root_dir: Path, index_file: Path, prompt_fn: Callable[[str], str], image_processor: ImageProcessor = None, prompt_builder = None,
    ) -> None:
        """
        Constructs a fully-fledged PyTorch Map-Style Dataset for evaluating on splits of the MMLU Test-Dev Set. In
        addition to the path to the dataset `index_file` to load from, requires a `prompt_fn` for formatting individual
        questions (model-specific), and an `image_processor` for applying any required image transforms.

        :param root_dir: Absolute path to the project's default root directory with downloads/task data
        :param prompt_fn: Callable that maps a question with the expected prompt template (model-specific)
        :param image_processor: Callable that applies the expected image transforms before yielding (model-specific)
        """
        self.prompt_fn, self.image_processor = prompt_fn, image_processor
        self.root_dir, self.index_file = root_dir, index_file
        self.prompt_builder = prompt_builder

        # Load from `index_file` --> Dict :: qid -> { question / answer / image data } --> flatten
        with open(self.root_dir / self.index_file, "r") as f:
            self.examples = list(json.load(f).values())

    def __getitem__(self, idx: int) -> Tuple[int, str, torch.Tensor, str, str]:
        """Return (question_id: int, question_prompt: str, pixel_values: torch.Tensor, question: str, answer: str)."""
        ex = self.examples[idx]
        question_prompt = ""

        prompt_builder = self.prompt_builder()
        
        prompt_builder.add_turn(role="human", message=ex["question"])
        
        question_prompt = ex["demonstrations"] + ex["question"]

        return ex["question_id"], question_prompt, ex["question"], ex["answer"]

    def __len__(self) -> int:
        return len(self.examples)


# === MMLU Task Runner ===
class MMLUTaskRunner:
    def __init__(
        self,
        root_dir: Path,
        index_file: Path,
        task_results_dir: Path,
        model_id: str,
        prompt_fn: Callable[[str], str],
        image_processor: ImageProcessor,
        prompt_builder,
    ) -> None:
        """Task Runner for the MMLU Dataset; loads data, then runs (distributed) VLM evaluation and writes results."""
        self.root_dir, self.index_file, self.task_results_dir = root_dir, index_file, task_results_dir
        self.model_id, self.prompt_fn, self.image_processor = model_id, prompt_fn, image_processor
        self.prompt_builder = prompt_builder

        # === Unfortunate Pattern =>> Accelerate injects a lot of additional stuff into env; minimize collateral ===
        from accelerate import PartialState

        self.distributed_state = PartialState()

        # Short-Circuit (if results/metrics already exist)
        os.makedirs(self.task_results_dir, exist_ok=True)
        if (self.task_results_dir / "metrics.json").exists():
            overwatch.info(f"MMLU Metrics for Model `{self.model_id}` already exist =>> Exiting!", ctx_level=1)
            return

        # Build (Map/Iterable) Dataset, using Model-Specific Prompt & Image Processor
        overwatch.info(f"Assembling MMLU Map-Style Dataset from {self.root_dir / self.index_file}", ctx_level=1)
        self.dataset = MMLUMapDataset(self.root_dir, self.index_file, self.prompt_fn, self.image_processor, self.prompt_builder)

    def evaluate(self, vlm: VLM, device_batch_size: int, num_workers: int) -> None:
        """Initialize Dataloader & partition data across ranks, writing metrics to disk on termination."""
        sampler = DistributedSampler(
            self.dataset,
            num_replicas=self.distributed_state.num_processes,
            rank=self.distributed_state.process_index,
            shuffle=False,
            drop_last=False,
        )
        dataloader = DataLoader(self.dataset, batch_size=device_batch_size, sampler=sampler, num_workers=num_workers)

        # Start Evaluation
        result_qa_pairs = {}
        try:
            overwatch.info(f"Distributing Evaluation across {self.distributed_state.num_processes} GPUs", ctx_level=1)
            for question_ids, question_prompts, questions, answers in tqdm(
                dataloader,
                desc="=>> Evaluating",
                disable=not self.distributed_state.is_main_process,
            ):
                # if isinstance(pixel_values, torch.Tensor):
                #     pixel_values = pixel_values.to(self.distributed_state.device)
                # elif isinstance(pixel_values, dict):
                #     pixel_values = {k: v.to(self.distributed_state.device) for k, v in pixel_values.items()}
                # else:
                #     raise ValueError(f"Unexpected `pixel_values` type = {type(pixel_values)}")

                gen_answers = vlm.generate_text_answer(question_prompts)
                # print(question_prompts[0], gen_answers)
                for question_id, gen_answer, question, answer in zip(
                    question_ids, gen_answers, questions, answers, strict=True
                ):
                    qid = int(question_id.item())
                    result_qa_pairs[qid] = {
                        "question_id": qid,
                        "question": question,
                        "model_output": gen_answer,#.replace("\n", "").replace(".", "").replace(",", "").lower().split("<|endofchunk|>")[0],
                        "ground_truth_answer": answer,
                    }

        finally:
            with open(self.task_results_dir / f"results+rank-{self.distributed_state.process_index}.json", "w") as f:
                json.dump(result_qa_pairs, f, indent=2)

        # Block on all processes before returning!
        self.distributed_state.wait_for_everyone()
        overwatch.info("Done Evaluating =>> Exiting!", ctx_level=1)


# === Official Score Function (Calls the lightly modified official MMLU evaluation script in `util/evaluation/mmlu` ===
class MMLUScorer:
    def __init__(
        self,
        dataset_id: str,
        task_results_dir: Path,
        full_result_qa_pairs: Dict[str, Dict],
        annotations_file: Path,
        questions_file: Optional[Path] = None,
        split: str = "testdev_balanced",
    ) -> None:
        """Wrapper around the official MMLU evaluation script; handles converting results to/from MMLU format."""
        self.dataset_id, self.task_results_dir = dataset_id, task_results_dir
        self.annotations_file, self.questions_file, self.split = annotations_file, questions_file, split
        self.full_result_qa_pairs = full_result_qa_pairs

        # Convert Results to Official MMLU Format
        self.convert_results()

    def convert_results(self) -> None:
        """MMLU Evaluation Script expects List[{"questionId": str, "prediction": str (lower case, no punkt)}]."""

        # Dump Full Results to JSON (for later inspection)
        with open(self.task_results_dir / "full-results.json", "w") as f:
            json.dump(self.full_result_qa_pairs, f, indent=2)

        # Convert to MMLU Expected Format --> with answer formatting (strip punctuation & lowercase)
        predictions = []
        for example in self.full_result_qa_pairs.values():
            clean_prediction = example["model_output"].translate(str.maketrans("", "", string.punctuation)).lower()
            predictions.append({"questionId": example["question_id"], "prediction": clean_prediction})

        # Write Predictions to Disk
        with open(self.task_results_dir / "mmlu-formatted-predictions.json", "w") as f:
            json.dump(predictions, f, indent=2)

    def score(self, model_id: str) -> Dict[str, float]:
        """Invoke `vlm_eval.util.evaluation.mmlu --> eval.py`; capture output to parse for accuracy/metrics."""
        qfile = self.annotations_file
        pfile = self.task_results_dir / "full-results.json"
        
        correct = []
        for qid, qa in json.load(open(pfile, "r")).items():
            correct.append(qa["ground_truth_answer"] in qa["model_output"])
        accuracy = mean(correct)

        # Run Official Evaluation Script
        # output = subprocess.run(
        #     f"python vlm_eval/util/evaluation/mmlu/eval.py --tier {self.split} --questions {qfile} --predictions {pfile}",
        #     capture_output=True,
        #     check=True,
        #     shell=True,
        # )

        # MMLU Evaluation Script outputs a *ton* of metrics --> for now, just parse out the aggregated accuracy!
        # eval_blob = output.stdout.decode("utf-8")
        # with open(self.task_results_dir / "mmlu-eval-output.txt", "w") as f:
        #     f.write(eval_blob)

        # Create Metrics Dictionary & Log
        # accuracy = float(re.search("Accuracy: (.*)%", eval_blob).group(1))
        overwatch.info(
            f"Results for Model `{model_id}` on {self.dataset_id} (Split = {self.split})\n"
            f"          => Accuracy (Official): {accuracy:.3f}"
        )

        return {"accuracy": accuracy}
