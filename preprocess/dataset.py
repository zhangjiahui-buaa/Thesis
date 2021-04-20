import torch
from torch import nn
from torch.utils.data.dataset import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader
from typing import List, Dict, Tuple
from preprocess.MVSASingle.process import _MVSA_Dataset, MVSA_Example
from preprocess.Hateful.process import _Hateful_Dataset, Hateful_Example
from torch.utils.data.sampler import Sampler, SequentialSampler, BatchSampler, SubsetRandomSampler
from transformers import BertTokenizer
from tqdm import tqdm
from logging import info, warning
import torchvision
import math
from PIL import Image
import copy


class SortedSampler(Sampler):
    """ Samples elements sequentially, always in the same order.
    Args:
        data (iterable): Iterable data.
    Example:
        >>> list(SortedSampler(range(10)))
        [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    """

    def __init__(self, data):
        super().__init__(data)
        self.data = data
        self.sort_key = lambda x: x
        zip_ = [(i, self.sort_key(row)) for i, row in enumerate(self.data)]
        zip_ = sorted(zip_, key=lambda r: r[1])
        self.sorted_indexes = [item[0] for item in zip_]

    def __iter__(self):
        return iter(self.sorted_indexes)

    def __len__(self):
        return len(self.data)


class BucketBatchSampler(BatchSampler):
    """ `BucketBatchSampler` toggles between `sampler` batches and sorted batches.

    Typically, the `sampler` will be a `RandomSampler` allowing the user to toggle between
    random batches and sorted batches. A larger `bucket_size_multiplier` is more sorted and vice
    versa.
    """

    def __init__(self, sampler, batch_size, drop_last, bucket_size_multiplier=100) -> None:
        super().__init__(sampler, batch_size, drop_last)
        self.bucket_sampler = BatchSampler(sampler,
                                           min(batch_size * bucket_size_multiplier, len(sampler)),
                                           False)

    def __iter__(self):
        for bucket in self.bucket_sampler:
            sorted_sampler = SortedSampler(bucket)
            for batch in SubsetRandomSampler(list(BatchSampler(sorted_sampler, self.batch_size, self.drop_last))):
                yield [bucket[i] for i in batch]

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return math.ceil(len(self.sampler) / self.batch_size)


class MVSA_Dataset(Dataset):
    def __init__(self, examples: List[MVSA_Example], tokenizer: BertTokenizer, transforms, device: torch.device,
                 max_encode_length: int, sort_by_length: bool) -> None:
        super(MVSA_Dataset, self).__init__()
        self.tokenizer = tokenizer
        self.device = device
        self.transforms = transforms
        self.max_encode_length = max_encode_length
        self.examples = self.encode_examples(examples, sort_by_length)

    def __getitem__(self, index: int) -> Dict:
        result = copy.deepcopy(self.examples[index])
        result['input_image'] = self.transforms(result['original_image']).to(self.device)
        return result

    def __len__(self) -> int:
        return len(self.examples)

    def encode_examples(self, examples: List[MVSA_Example], sort_by_length: bool) -> List[Dict]:
        new_examples = []
        for example in tqdm(examples):
            new_example = self.encode_example(example)
            if new_example is None:
                continue
            new_examples.append(new_example)

        if len(new_examples) < len(examples):
            warning("ignoring {} examples that are longer than {}".format(len(examples) - len(new_examples), self))

        if not sort_by_length:
            return new_examples

        sorted_examples = sorted(new_examples, key=lambda x: x['input_token_ids'].size(0))
        return list(sorted_examples)

    def encode_example(self, example: MVSA_Example) -> Dict:
        original_image, tokens = example.image, example.tokens

        input_tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
        input_tokens_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
        return {
            'input_token_ids': torch.tensor(input_tokens_ids, dtype=torch.long, device=self.device),
            'input_tokens': input_tokens,
            'original_image': original_image,
            'input_image': None,
            'combined_label': torch.tensor([example.combined_label], dtype=torch.long, device=self.device),
            'text_label': torch.tensor([example.text_label], dtype=torch.long, device=self.device),
            'image_label': torch.tensor([example.image_label], dtype=torch.long, device=self.device)
        }


class Hate_Dataset(MVSA_Dataset):
    def __init__(self, examples: List[Hateful_Example], tokenizer: BertTokenizer, transforms, device: torch.device,
                 max_encode_length: int, sort_by_length: bool) -> None:
        super(Hate_Dataset, self).__init__(examples, tokenizer, transforms, device,
                                           max_encode_length, sort_by_length)

    def __getitem__(self, index):
        result = copy.deepcopy(self.examples[index])
        Image_PIL = Image.fromarray(self.examples[index]['original_image_np'])
        result['input_image'] = self.transforms(Image_PIL).to(self.device)
        return result

    def encode_example(self, example: Hateful_Example) -> Dict:
        original_image_np, tokens = example.image_np, example.tokens

        input_tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
        input_tokens_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
        return {
            'input_token_ids': torch.tensor(input_tokens_ids, dtype=torch.long, device=self.device),
            'input_tokens': input_tokens,
            'original_image_np': original_image_np,
            'input_image': None,
            'combined_label': torch.tensor([example.label], dtype=torch.long, device=self.device),
        }


def tensor_collate_fn(inputs: List[Dict], is_training: bool) -> Dict:
    assert len(inputs) > 0
    collated = {}
    for key in inputs[0]:
        values = [x[key] for x in inputs]
        if key == 'input_token_ids':
            collated[key] = pad_sequence(values, batch_first=True, padding_value=0)
        elif key in ['input_image', 'text_label', 'image_label', 'combined_label']:
            collated[key] = torch.stack(values)
        else:
            collated[key] = values
    collated['is_training'] = is_training
    return collated


def load_MVSA_data_iterator(path: str,
                            tokenizer: BertTokenizer,
                            transform: torchvision.transforms,
                            batch_size: int,
                            device: torch.device,
                            bucket: bool,
                            max_encode_length: int,
                            ):
    all_examples = _MVSA_Dataset(path).examples
    train_examples = all_examples[400:]
    dev_examples = all_examples[:400]

    del all_examples  # save memory cost
    dev_dataset = MVSA_Dataset(dev_examples, tokenizer, transform, device, max_encode_length, False)
    dev_data_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: tensor_collate_fn(x, False))

    if bucket:
        train_dataset = MVSA_Dataset(train_examples, tokenizer, transform, device, max_encode_length, True)
        train_data_loader = DataLoader(
            train_dataset,
            batch_sampler=BucketBatchSampler(SequentialSampler(list(range(len(train_dataset)))), batch_size=batch_size,
                                             drop_last=False),
            collate_fn=lambda x: tensor_collate_fn(x, True))

        return train_data_loader, dev_data_loader

    else:
        train_dataset = MVSA_Dataset(train_examples, tokenizer, transform, device, max_encode_length, False)
        train_data_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda x: tensor_collate_fn(x, True))
        return train_data_loader, dev_data_loader


def load_Hate_data_iterator(path: str,
                            tokenizer: BertTokenizer,
                            transform: torchvision.transforms,
                            batch_size: int,
                            device: torch.device,
                            bucket: bool,
                            max_encode_length: int,
                            ):
    all_examples = _Hateful_Dataset(path).examples
    train_examples, dev_examples = all_examples['train_examples'], \
                                   all_examples['dev_examples']

    del all_examples  # save memory cost
    dev_dataset = Hate_Dataset(dev_examples, tokenizer, transform, device, max_encode_length, False)
    dev_data_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: tensor_collate_fn(x, False))
    if bucket:
        train_dataset = Hate_Dataset(train_examples, tokenizer, transform, device, max_encode_length, True)
        train_data_loader = DataLoader(
            train_dataset,
            batch_sampler=BucketBatchSampler(SequentialSampler(list(range(len(train_dataset)))), batch_size=batch_size,
                                             drop_last=False),
            collate_fn=lambda x: tensor_collate_fn(x, True))

        return train_data_loader, dev_data_loader

    else:
        train_dataset = Hate_Dataset(train_examples, tokenizer, transform, device, max_encode_length, False)
        train_data_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda x: tensor_collate_fn(x, True))
        return train_data_loader, dev_data_loader


if __name__ == '__main__':
    # 5GB memory cost on MVSA
    MVSA_train_data_loader, MVSA_dev_data_loader = load_MVSA_data_iterator('datasets/MVSA_single',
                                                                 BertTokenizer.from_pretrained("bert-base-uncased"),
                                                                 torchvision.transforms.Compose(
                                                                     [torchvision.transforms.Resize(256),
                                                                      torchvision.transforms.CenterCrop(224),
                                                                      torchvision.transforms.ToTensor(),
                                                                      torchvision.transforms.Normalize(
                                                                          [0.485, 0.456, 0.406],
                                                                          [0.229, 0.224, 0.225])
                                                                      ]),
                                                                 4,
                                                                 "cpu",
                                                                 True,
                                                                 512)
    # 8GB memory cost on Hate
    '''Hate_train_data_loader, Hate_dev_data_loader = load_Hate_data_iterator('datasets/Hateful',
                                                                           BertTokenizer.from_pretrained(
                                                                               "bert-base-uncased"),
                                                                           torchvision.transforms.Compose(
                                                                               [torchvision.transforms.Resize(256),
                                                                                torchvision.transforms.CenterCrop(224),
                                                                                torchvision.transforms.ToTensor(),
                                                                                torchvision.transforms.Normalize(
                                                                                    [0.485, 0.456, 0.406],
                                                                                    [0.229, 0.224, 0.225])
                                                                                ]),
                                                                           4,
                                                                           "cpu",
                                                                           True,
                                                                           512)'''
