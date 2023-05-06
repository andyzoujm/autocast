# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd
import torch
import json


class FiDDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        question,
        research_schedule,
        research_material,
        n_context=None,
        question_prefix="question:",
        title_prefix="title:",
        passage_prefix="context:",
        choices_prefix="choices:",
        bound_prefix="bounds:",
        max_choice_len=12,
        cat=None,
    ):
        self.date_index = pd.DatetimeIndex([date for date in research_material]).sort()
        self.answer = question["answer"]
        self.choices = choices

        # Format the question.
        self.question = question_prefix + " " + question["question"]
        if question["qtype"] == "mc":
            choices = question["choices"]
            formatted_choices = [f"{i}: {choice}" for i, choice in enumerate(choices)]
            choice_string = " | ".join(formatted_choices)
            self.question = f"{self.question} {choices_prefix} {choice_string}."
        
        self.research_schedule = research_schedule
        self.research_material = research_material
        self.n_context = n_context
        self.question_prefix = question_prefix
        self.title_prefix = title_prefix
        self.passage_prefix = passage_prefix
        self.choices_prefix = choices_prefix
        self.bound_prefix = bound_prefix
        self.max_choice_len = max_choice_len

    def __len__(self):
        return len(self.research_schedule)

    def __getitem__(self, index):
        date = self.date_index[index].strftime()
        example = self.research_schedule[date]
        scores = pd.DataFrame({"score": example}).rename_index("doc_ids")
        docs = scores.join(self.research_material)
        docs = docs.sample(n=self.n_context, replace=True)
        scores = torch.tensor(docs["scores"].to_numpy())
        passages = (
            f"{self.title_prefix} "
            + docs["title"]
            + f" {self.passage_prefix} "
            + docs["text"]
        )
        return {
            "index": index,
            "question": self.question,
            "target": self.answer,
            "choices": self.choices,
            "passages": passages,
            "scores": scores,
        }

    def sort_data(self):
        if self.n_context is None or not "score" in self.data[0]["ctxs"][0]:
            return
        for ex in self.data:
            ex["ctxs"].sort(key=lambda x: float(x["score"]), reverse=True)

    def get_example(self, index):
        return self.data[index]


def encode_passages(batch_text_passages, tokenizer, max_length):
    passage_ids, passage_masks = [], []
    for k, text_passages in enumerate(batch_text_passages):
        text_passages = [tp + " </s>" for tp in text_passages]
        p = tokenizer.batch_encode_plus(
            text_passages,
            max_length=max_length,
            pad_to_max_length=True,
            return_tensors="pt",
            truncation=True,
        )
        passage_ids.append(p["input_ids"][None])
        passage_masks.append(p["attention_mask"][None])

    passage_ids = torch.cat(passage_ids, dim=0)
    passage_masks = torch.cat(passage_masks, dim=0)
    return passage_ids, passage_masks.bool()


class Collator(object):
    def __init__(self, text_maxlength, tokenizer, answer_maxlength=20):
        self.tokenizer = tokenizer
        self.text_maxlength = text_maxlength
        self.answer_maxlength = answer_maxlength

    def __call__(self, batch):
        assert batch[0]["target"] != None
        index = torch.tensor([ex["index"] for ex in batch])
        targets = [ex["target"] for ex in batch]

        indices, lengths = None, None
        labels = torch.tensor(targets).view(-1, 1)

        def append_question(example):
            if example["passages"] is None:
                return [example["question"]]
            return [example["question"] + " " + t for t in example["passages"]]

        text_passages = [append_question(example) for example in batch]
        passage_ids, passage_masks = encode_passages(
            text_passages, self.tokenizer, self.text_maxlength
        )

        return (index, labels, indices, lengths, passage_ids, passage_masks)


def load_data(data_path=None, topn=1, global_rank=-1, world_size=-1):
    assert data_path
    if data_path.endswith(".jsonl"):
        data = open(data_path, "r")
    elif data_path.endswith(".json"):
        with open(data_path, "r") as fin:
            data = json.load(fin)

    examples = []
    for k, example in enumerate(data):
        if global_rank > -1 and not k % world_size == global_rank:
            continue
        if data_path is not None and data_path.endswith(".jsonl"):
            example = json.loads(example)

        examples.append(example)

    ## egrave: is this needed?
    if data_path is not None and data_path.endswith(".jsonl"):
        data.close()

    return examples
