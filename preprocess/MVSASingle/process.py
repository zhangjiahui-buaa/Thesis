import numpy as np
import os
from PIL import Image
import PIL
import cv2
from preprocess.MVSASingle.utils import *
import re
from typing import List, DefaultDict

class MVSA_Example:
    def __init__(self, ex_id: int,
                 text: str,
                 image: PIL.Image.Image,
                 image_np: np.ndarray,
                 text_label: int,
                 image_label: int):
        self.id = ex_id
        self.text = text
        self.tokens = self.process_text(self.text)
        self.image = image
        self.image_np = image_np
        self.text_label = text_label
        self.image_label = image_label
        self.combined_label = self.process_label(ex_id, text_label,image_label)

    def process_text(self, raw_text: str) -> List[str]:
        return raw_text.lower().strip().split()  # TODO: process raw text data
        raise NotImplementedError()

    def process_label(self, ex_id, text_label, image_label):
        if text_label == Positive:
            if not image_label == Negative:
                return Positive
            else:
                print(ex_id)
                return None  # TODO: label it manually
        elif text_label == Negative:
            if not image_label == Positive:
                return Negative
            else:
                print(ex_id)
                return None  # TODO: label it manually
        else:
            return image_label
        raise NotImplementedError()


class _MVSA_Dataset:
    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir
        self.examples: List[MVSA_Example] = []
        self.conflict = 0
        self.load_raw_all_datas(data_dir=self.data_dir)
        self.statistic = self.get_label_category()
        print(self.statistic)

    def get_label_category(self):
        statistic = DefaultDict(int)
        for example in self.examples:
            if example.image_label == Positive:
                statistic['image_positive'] += 1
            elif example.image_label == Negative:
                statistic['image_negative'] += 1
            else:
                statistic['image_neutural'] += 1

            if example.text_label == Positive:
                statistic['text_positive'] += 1
            elif example.text_label == Negative:
                statistic['text_negative'] += 1
            else:
                statistic['text_neutral'] += 1
        return statistic

    def load_raw_all_datas(self, data_dir: str) -> None:
        raw_labels = open(os.path.join(data_dir, 'labelResultAll.txt'), 'r')
        lines = raw_labels.readlines()[1:]
        split_lines = [re.split('[\t,]', line.strip()) for line in lines]

        for item in split_lines:
            _id = int(item[0])
            raw_text_label = item[1]
            raw_image_label = item[2]
            try:
                example = self.load_raw_single_data(data_dir, _id, raw_text_label, raw_image_label)
            except:
                print("error in processing data {}".format(_id))
            else:
                if not example.combined_label is None:
                    self.examples.append(example)
        return

    def load_raw_single_data(self, data_dir: str, _id: int, raw_text_label: str, raw_image_label: str) -> MVSA_Example:
        text_file = open(os.path.join(data_dir, 'data/{}.txt'.format(_id)), 'r', encoding='utf-8', errors='ignore')
        text = ' '.join(text_file.readlines())
        image = Image.open(os.path.join(data_dir, 'data/{}.jpg'.format(_id))).convert('RGB')
        #image = cv2.imread(os.path.join(data_dir, 'data/{}.jpg'.format(_id)))

        image_np = np.array(image)
        text_label, image_label = convert_raw_label[raw_text_label], convert_raw_label[raw_image_label]
        example = MVSA_Example(_id, text, image, image_np, text_label, image_label)
        return example

    def split_train_and_dev(self, examples):
        raise NotImplementedError()


if __name__ == '__main__':
    test_dataset = _MVSA_Dataset('datasets/MVSA_single')
