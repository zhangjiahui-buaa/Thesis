import numpy as np
import os
from PIL import Image
import PIL
from preprocess.Hateful.utils import *
from typing import List, DefaultDict, Dict
import jsonlines
from tqdm import tqdm


class Hateful_Example:
    def __init__(self, ex_id: int,
                 text: str,
                 image_np: np.ndarray,
                 label=None):
        self.id = ex_id
        self.text = text
        self.tokens = self.process_text(self.text)
        self.image_np = image_np
        self.label = label

    def process_text(self, raw_text: str) -> List[str]:
        return raw_text.lower().strip().split()  # TODO: process raw text data
        raise NotImplementedError()


class _Hateful_Dataset:
    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir
        self.examples: Dict[str] = DefaultDict(list)
        self.load_raw_all_datas()
        self.statistic = self.get_label_category()
        print(self.statistic)

    def get_label_category(self):
        statistic = DefaultDict(int)
        for example in self.examples['train_examples']:
            if example.label == hateful:
                statistic['train_hateful'] += 1
            else:
                statistic['train_non_hateful'] += 1
        for example in self.examples['dev_examples']:
            if example.label == hateful:
                statistic['dev_hateful'] += 1
            else:
                statistic['dev_non_hateful'] += 1
        for example in self.examples['test_examples']:
            if example.label == hateful:
                statistic['test_hateful'] += 1
            else:
                statistic['test_non_hateful'] += 1

        return statistic

    def load_one_split(self, split_type):
        if split_type == "train":
            label_file = jsonlines.open("datasets/Hateful/train.jsonl")
        elif split_type == "dev":
            label_file = jsonlines.open("datasets/Hateful/dev_seen.jsonl")
        else:
            label_file = jsonlines.open("datasets/Hateful/test_seen.jsonl")
        for label in tqdm(label_file):
            _id = label["id"]
            image = Image.open(os.path.join(self.data_dir, label["img"])).convert('RGB')
            image_np = np.array(image)
            if "label" in label:
                example = Hateful_Example(_id, label["text"], image_np, label["label"])
            else:
                example = Hateful_Example(_id, label["text"], image_np, None)
            self.examples[split_type + "_examples"].append(example)

    def load_raw_all_datas(self) -> None:
        self.load_one_split("train")
        self.load_one_split("dev")
        self.load_one_split("test")
        return


if __name__ == '__main__':
    test_dataset = _Hateful_Dataset('datasets/Hateful')
