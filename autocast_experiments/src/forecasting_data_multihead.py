# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import random
import json
import numpy as np
import pickle
from torch._C import TensorType

class FiDDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 n_context=None,
                 question_prefix='question:',
                 title_prefix='title:',
                 passage_prefix='context:',
                 choices_prefix='choices:',
                 bound_prefix='bounds:',
                 max_choice_len=12,
                 cat=None):
        self.data = data
        self.n_context = n_context
        self.question_prefix = question_prefix
        self.title_prefix = title_prefix
        self.passage_prefix = passage_prefix
        self.choices_prefix=choices_prefix
        self.bound_prefix = bound_prefix
        self.max_choice_len = max_choice_len
        self.cat = cat
        # self.sort_data()

    def __len__(self):
        return len(self.data)

    def get_target(self, example):
        STR2BOOL = {'yes': 1, 'Yes': 1, 'no': 0, 'No': 0}

        if self.cat == 2:
            return self.max_choice_len + 2 + 1  # structure: mc + (invalid choices) + t/f + re for integer labeling
        elif self.cat == 1:
            if ord(example['answers'][0]) - ord('A') + 1 > self.max_choice_len:
                return self.max_choice_len
            else:
                return ord(example['answers'][0]) - ord('A')
        elif self.cat == 0:
            return STR2BOOL[example['answers'][0]] + self.max_choice_len + 1

    def __getitem__(self, index):
        example = self.data[index]
        question = self.question_prefix + " " + example['question']
        choices = example['choices']
        target = self.get_target(example)

        # Append available choices for MC questions
        if self.cat == 1:
            choices = [str(i) + ': ' + choices[i] for i in range(len(choices))]
            question = question + ' ' + self.choices_prefix + ' ' + ' | '.join(choices) + '.'
        elif self.cat == 2:
            min, max = choices['min'], choices['max']
            question = question + ' ' + self.bound_prefix + ' min: ' + min + ' | max: ' + max + '.'

        if 'ctxs' in example and len(example['ctxs']) > 0 and self.n_context is not None:
            f = self.title_prefix + " {} " + self.passage_prefix + " {}"
            if len(example['ctxs']) < self.n_context: # if we don't have enough articles
                add_on = self.n_context - len(example['ctxs'])
                example['ctxs'].extend([example['ctxs'][0]] * add_on)
            contexts = example['ctxs'][:self.n_context]
            passages = [f.format(c['title'], c['text']) for c in contexts]
            scores = [float(c['score']) for c in contexts]
            scores = torch.tensor(scores)
            # TODO(egrave): do we want to keep this?
            if len(contexts) == 0:
                contexts = [question]
        else:
            passages, scores = None, None

        return {
            'index' : index,
            'question' : question,
            'target' : target,
            'choices': choices,
            'passages' : passages,
            'scores' : scores
        }

    def sort_data(self):
        if self.n_context is None or not 'score' in self.data[0]['ctxs'][0]:
            return
        for ex in self.data:
            ex['ctxs'].sort(key=lambda x: float(x['score']), reverse=True)

    def get_example(self, index):
        return self.data[index]

def encode_passages(batch_text_passages, tokenizer, max_length):
    passage_ids, passage_masks = [], []
    for k, text_passages in enumerate(batch_text_passages):
        text_passages = [tp + ' </s>' for tp in text_passages]
        p = tokenizer.batch_encode_plus(
            text_passages,
            max_length=max_length,
            pad_to_max_length=True,
            return_tensors='pt',
            truncation=True
        )
        passage_ids.append(p['input_ids'][None])
        passage_masks.append(p['attention_mask'][None])

    passage_ids = torch.cat(passage_ids, dim=0)
    passage_masks = torch.cat(passage_masks, dim=0)
    return passage_ids, passage_masks.bool()

class Collator(object):
    def __init__(self, text_maxlength, tokenizer, answer_maxlength=20):
        self.tokenizer = tokenizer
        self.text_maxlength = text_maxlength
        self.answer_maxlength = answer_maxlength

    def __call__(self, batch):
        assert(batch[0]['target'] != None)
        index = torch.tensor([ex['index'] for ex in batch])
        targets = [ex['target'] for ex in batch]

        
        ########## Not usuful for GPT-2 based forecasting ##########
        # tfmc_indices, re_indices, tf_indices, mc_indices = [], [], [], []
        # for i in range(len(targets)):
        #     if not isinstance(choices[i], dict):
        #         tfmc_indices.append(i)
        #         if len(choices) > 2:
        #             mc_indices.append(i)
        #         else:
        #             tf_indices.append(i)
        #     else:
        #         re_indices.append(i)

        # tfmc_len, re_len, tf_len, mc_len = len(tfmc_indices), len(re_indices), len(tf_indices), len(mc_indices)
        # length = max(tfmc_len, re_len, tf_len, mc_len)
        # if tfmc_len < length:
        #     tfmc_indices = tfmc_indices + [-1] * (length - tfmc_len)
        # if re_len < length:
        #     re_indices = re_indices + [-1] * (length - re_len)
        # if tf_len < length:
        #     tf_indices = tf_indices + [-1] * (length - tf_len)
        # if mc_len < length:
        #     mc_indices = mc_indices + [-1] * (length - mc_len)
        # lengths = torch.tensor([tfmc_len, re_len, tf_len, mc_len])
        # indices = torch.tensor([tfmc_indices] + [re_indices] + [tf_indices] + [mc_indices])
        indices, lengths = None, None
        labels = torch.tensor(targets).view(-1, 1)
        
        def append_question(example):
            if example['passages'] is None:
                return [example['question']]
            return [example['question'] + " " + t for t in example['passages']]
        text_passages = [append_question(example) for example in batch]
        passage_ids, passage_masks = encode_passages(text_passages,
                                                    self.tokenizer,
                                                    self.text_maxlength)

        return (index, labels, indices, lengths, passage_ids, passage_masks)

def load_data(data_path=None, topn=1, global_rank=-1, world_size=-1):
    assert data_path
    if data_path.endswith('.jsonl'):
        data = open(data_path, 'r')
    elif data_path.endswith('.json'):
        with open(data_path, 'r') as fin:
            data = json.load(fin)

    examples = []
    for k, example in enumerate(data):
        if global_rank > -1 and not k%world_size==global_rank:
            continue
        if data_path is not None and data_path.endswith('.jsonl'):
            example = json.loads(example)
            
        examples.append(example)
        
    ## egrave: is this needed?
    if data_path is not None and data_path.endswith('.jsonl'):
        data.close()

    return examples
