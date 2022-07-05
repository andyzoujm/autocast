# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import random
import json
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 n_context=None,
                 over_sample=False,
                 question_prefix='question:',
                 title_prefix='title:',
                 passage_prefix='context:',
                 choices_prefix='choices:',
                 bound_prefix='bounds:'):
        self.data = data
        self.n_context = n_context
        self.question_prefix = question_prefix
        self.title_prefix = title_prefix
        self.passage_prefix = passage_prefix
        self.choices_prefix=choices_prefix
        self.bound_prefix = bound_prefix
        # self.sort_data()
        print('WARNING: NOT DOING RERANKING')

        self.data_by_class_displayed = False

        self.pre_filter(over_sample)
    
    def pre_filter(self, over_sample):
        
        self.data_by_class = {}
        valid_data = []
        for example in self.data:
            label = example['answers'][0]
            if label not in self.data_by_class:
                self.data_by_class[label] = []
            self.data_by_class[label].append(example)
            valid_data.append(example)
        self.data = valid_data
        
        if over_sample:
            self.over_sample()
    
    def over_sample(self):
        max_count = 0
        for label in self.data_by_class:
            max_count = max(max_count, len(self.data_by_class[label]))
        data = []
        for label in self.data_by_class:
            class_data = self.data_by_class[label]
            data.extend(class_data)
            class_count = len(class_data)
            over_samples = np.random.choice(class_data, max_count - class_count, replace=True)
            data.extend(over_samples)
        
        self.data = data

    def __len__(self):
        if not self.data_by_class_displayed:
            output_str = ''
            for label in self.data_by_class:
                output_str += f"{len(self.data_by_class[label])} {label} "
            # print("# samples by class:", output_str)
            self.data_by_class_displayed = True
        return len(self.data)

    def get_target(self, example):
        if 'target' in example:
            target = example['target']
            return target + ' </s>'
        elif 'answers' in example:
            if isinstance(example['choices'], dict):
                return example['answers'][0] #+ ' </s>'
            return random.choice(example['answers']) + ' </s>'
        else:
            return None

    def __getitem__(self, index):
        example = self.data[index]
        question = self.question_prefix + " " + example['question']
        choices = example['choices']
        target = self.get_target(example)
        field = example['field']

        # Append available choices for MC questions
        if (not target[:-5].lower().strip() in ['yes', 'no']) and not isinstance(choices, dict):
            choices = [chr(i + ord('A')) + ': ' + choices[i] for i in range(len(choices))]
            question = question + ' ' + self.choices_prefix + ' ' + ' | '.join(choices) + '.'
        if isinstance(choices, dict):
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
            'scores' : scores,
            'field' : field
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
            padding='max_length',
            max_length=max_length,
            return_tensors='pt',
            truncation=True
        )
        passage_ids.append(p['input_ids'][None])
        passage_masks.append(p['attention_mask'][None])

    passage_ids = torch.cat(passage_ids, dim=0)
    passage_masks = torch.cat(passage_masks, dim=0)
    return passage_ids, passage_masks.bool()

class Collator(object):
    def __init__(self, text_maxlength, tokenizer, answer_maxlength=20, n_context=0):
        self.tokenizer = tokenizer
        self.text_maxlength = text_maxlength
        self.answer_maxlength = answer_maxlength
        self.n_context = n_context

    def __call__(self, batch):
        assert(batch[0]['target'] != None)
        index = torch.tensor([ex['index'] for ex in batch])
        targets = [ex['target'] for ex in batch]
        choices = [ex['choices'] for ex in batch]
        fields = [ex['field'] for ex in batch]

        tfmc_indices, re_indices, tf_indices, mc_indices = [], [], [], []
        for i in range(len(targets)):
            if not isinstance(choices[i], dict):
                tfmc_indices.append(i)
                if targets[i][:-5].lower().strip() in ['yes', 'no']:
                    tf_indices.append(i)
                else:
                    mc_indices.append(i)
                    
            else:
                re_indices.append(i)

        tfmc_len, re_len, tf_len, mc_len = len(tfmc_indices), len(re_indices), len(tf_indices), len(mc_indices)
        length = max(tfmc_len, re_len, tf_len, mc_len)
        if tfmc_len < length:
            tfmc_indices = tfmc_indices + [-1] * (length - tfmc_len)
        if re_len < length:
            re_indices = re_indices + [-1] * (length - re_len)
        if tf_len < length:
            tf_indices = tf_indices + [-1] * (length - tf_len)
        if mc_len < length:
            mc_indices = mc_indices + [-1] * (length - mc_len)
        lengths = torch.tensor([tfmc_len, re_len, tf_len, mc_len])
        indices = torch.tensor([tfmc_indices] + [re_indices] + [tf_indices] + [mc_indices])

        targets_tokenized = self.tokenizer.batch_encode_plus(
            targets,
            padding='max_length',
            max_length=self.answer_maxlength if self.answer_maxlength > 0 else None,
            return_tensors='pt',
            truncation=True if self.answer_maxlength > 0 else False,
        )
        targets_ids = targets_tokenized["input_ids"]
        targets_mask = targets_tokenized["attention_mask"].bool()  # normally it's not used; we will not add regression results in
        targets_ids = targets_ids.masked_fill(~targets_mask, -100)
        
        labels = []
        feature_len = len(targets_ids[-1])
        for i in range(len(index)):
            if not isinstance(choices[i], dict):
                labels.append(targets_ids[i])
            else:
                labels.append(torch.full((feature_len,), float(targets[i])))
        labels = torch.stack(labels).to(torch.float32)

        def append_question(example, n):
            if example['passages'] is None:
                if n is not None:
                    return [example['question']] * n
                return [example['question']]
            return [example['question'] + " " + t for t in example['passages']]
        text_passages = [append_question(example, self.n_context) for example in batch]
        passage_ids, passage_masks = encode_passages(text_passages,
                                                     self.tokenizer,
                                                     self.text_maxlength)

        return (index, labels, indices, lengths, passage_ids, passage_masks, fields)

def load_data(data_path=None, global_rank=-1, world_size=-1):
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
        if not 'id' in example:
            example['id'] = k
        for c in example['ctxs']:
            if not 'score' in c:
                c['score'] = 1.0 / (k + 1)
        examples.append(example)
        
    ## egrave: is this needed?
    if data_path is not None and data_path.endswith('.jsonl'):
        data.close()

    return examples
