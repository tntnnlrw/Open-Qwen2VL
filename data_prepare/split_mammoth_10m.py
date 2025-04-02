import json
from tqdm import tqdm
from transformers import AutoTokenizer
import jsonlines
import os

n_patches = 729

data = json.load(open("data/MAmmoTH-VL-Instruct-12M/mammoth_si_10M.json"))

with jsonlines.open("data/MAmmoTH-VL-Instruct-12M/mammoth_si_10M_simple.jsonl", 'w') as jsonlw:
    for sample in tqdm(data):
        simple_sample = {}
        if "image" in sample:
            path = sample["image"].replace("/", "_").replace(".", "_")[:200] + ".json"
        else:
            path = (str(sample["source"]) + str(sample["id"])).replace(".", "_") + ".json"
            if os.path.exists(f"data/MAmmoTH-VL-Instruct-12M/instruction_si_split/{path}"):
                # print(path)
                continue
        if not path.endswith(".json"):
            print(path)
            continue
        n_words = n_patches + sum([len(turn["value"].replace("<image>", "").split()) for turn in sample["conversations"]])

        simple_sample["path"] = path
        simple_sample["length"] = n_words

        jsonlw.write(simple_sample)

        with open(f"data/MAmmoTH-VL-Instruct-12M/instruction_si_split/{path}", "w") as f:
            f.write(json.dumps(sample))