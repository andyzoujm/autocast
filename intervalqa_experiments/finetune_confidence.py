# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
import argparse
import logging
import math
import os
import random
from pathlib import Path
import numpy as np
import csv

import datasets
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.utils.data.sampler import Sampler

import transformers
from accelerate import Accelerator
from huggingface_hub import Repository
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)
from transformers.file_utils import get_full_repo_name
from transformers.utils.versions import require_version

import utils


logger = logging.getLogger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    args = parser.parse_args()

    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.

    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset("glue", args.task_name)
    else:
        # Loading the dataset from local csv or json file.
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = (args.train_file if args.train_file is not None else args.valid_file).split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.


    # confidence intervals we want
    confidence_levels = list(range(50,100,5))

    # Labels
    if args.task_name is not None:
        is_regression = args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = len(confidence_levels) * 2 + 1 # plus one point estimate
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    # Preprocessing the datasets
    if args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            logger.info(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        return result

    # with accelerator.main_process_first():
    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        desc="Running tokenizer on dataset",
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

    # class AdaptiveBinningSampler(Sampler):
    #     def __init__(self, dataset):
    #         self.labels = [obj['labels'] for obj in dataset]
    #         self.sorted_idx = np.argsort(self.labels)
    #         self.switch = 0
            
    #     def __iter__(self):
    #         final_indices = []
    #         n = args.per_device_train_batch_size * 2
    #         for i in range(0, len(self.sorted_idx), n):
    #             subset = np.random.permutation(self.sorted_idx[i:i + n])
    #             for j in range(0, len(subset), args.per_device_train_batch_size):
    #                 final_indices.append(subset[j:j + args.per_device_train_batch_size])
    #         final_indices[:-1] = np.random.permutation(final_indices[:-1])
    #         idx = [int(i) for indices in final_indices for i in indices]
    #         if self.switch % 2:
    #             random.shuffle(idx)
    #         self.switch += 1
    #         return iter(idx)
        
    #     def __len__(self):
    #         return len(self.labels)

    # train_dataloader = DataLoader(
    #     train_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.per_device_train_batch_size, sampler=AdaptiveBinningSampler(train_dataset)
    # )

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Get the metric function
    if args.task_name is not None:
        metric = load_metric("glue", args.task_name)
    else:
        metric = load_metric("accuracy")

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=True)
    completed_steps = 0


    ###############################
    ## confidence interval ##
    ###############################


    num_intervals = len(confidence_levels)
    # interval_criterion = nn.HuberLoss(reduction='none')
    # pe_criterion = nn.HuberLoss(reduction='none')

    interval_criterion = nn.MSELoss(reduction='none')
    pe_criterion = nn.MSELoss(reduction='none')

    softplus = nn.Softplus()
    lam = 0.1 / num_intervals


    def get_confidence_intervals(logits):

        deltas, point_estimates = softplus(logits[:, :-1]), logits[:, -1:]
        lower_deltas, higher_deltas = deltas[:, :num_intervals], deltas[:, num_intervals:]
        interval_lengths = lower_deltas + higher_deltas
        # lower_deltas, higher_deltas = torch.cumsum(lower_deltas, dim=1), torch.cumsum(higher_deltas, dim=1)
        lower_deltas, higher_deltas = utils.cumsum(lower_deltas), utils.cumsum(higher_deltas)
        lowers, uppers = point_estimates - lower_deltas, point_estimates + higher_deltas

        return lowers, uppers, point_estimates, interval_lengths

    # Saving the predictions
    model_size = args.model_name_or_path.split('-')[-1]
    bs = args.per_device_train_batch_size
    n_e = args.num_train_epochs
    max_l = args.max_length
    filename = f'{model_size}_bs{bs}_ep{n_e}_ml{max_l}'
    fields = ['Point Estimates', 'Lowers', 'Uppers', 'Interval Lengths', 'Labels']
    pred_results = []

    for epoch in range(args.num_train_epochs):
        
        model.train()
        train_containment_tensors = []
        train_pe_dist = []
        train_interval_length = []
        train_labels = []

        for step, batch in enumerate(train_dataloader):

            labels = batch['labels'][:, None].repeat(1, num_intervals)
            train_labels.append(labels[:,0])
            del batch['labels']
            
            outputs = model(**batch)
            logits = outputs.logits

            lowers, uppers, point_estimates, interval_lengths = get_confidence_intervals(logits)
            containment, ci_error = utils.evaluate(lowers, uppers, labels, confidence_levels)
            low_containment_mask = ci_error < 0
            train_containment_tensors.append(containment)

            point_estimate_loss = pe_criterion(point_estimates, labels[:,0:1]) / (1+torch.abs(labels))
            train_pe_dist.append(torch.abs(torch.exp(point_estimates) / torch.exp(labels[:,0:1]) - 1).median().item())

            miscalibration_loss = ((pe_criterion(lowers, labels) * (lowers > labels) + pe_criterion(uppers, labels) * (uppers < labels)) / (1+torch.abs(labels))).mean(dim=0)
            interval_length_loss = interval_criterion(interval_lengths, torch.zeros_like(interval_lengths, device=interval_lengths.device)) / (1+torch.abs(labels))
            train_interval_length.append(((torch.exp(uppers) - torch.exp(lowers)) / torch.exp(labels[:,0:1])).median(dim=0).values.detach().cpu())
            loss = point_estimate_loss.mean() + (low_containment_mask * miscalibration_loss).mean() + lam * (~low_containment_mask * interval_length_loss.mean(dim=0)).mean()

            # if epoch < 5:
            #     loss = point_estimate_loss.mean()
            # else:
            #     loss = (low_containment_mask * miscalibration_loss).mean() + lam * (~low_containment_mask * interval_length_loss.mean(dim=0)).mean()


            # logger.info(f"{(miscalibration_loss.mean(), interval_length_loss.mean())}")
            # logger.info(f"{(lowers[0], uppers[0], labels[0, 0])}")

            # print('LOSS', loss.item())

            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                # nn.utils.clip_grad_norm(model.parameters(), 2)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= args.max_train_steps:
                break
        
        train_metric = torch.cat(train_containment_tensors).mean(dim=0) * 100
        train_adaptive_rms = utils.adaptive_binning_rms(train_containment_tensors, train_labels, confidence_levels)
        logger.info(f"epoch {epoch} | train rms: {utils.round_tensor(train_metric, 1)}")
        logger.info(f"epoch {epoch} | train rms: {utils.rms(train_metric, confidence_levels):.2f} | pe dist: {np.median(train_pe_dist):.2f} | interval len: {np.mean(np.median(torch.stack(train_interval_length), axis=0)):.2f} | lengths: {np.around(np.median(torch.stack(train_interval_length), axis=0), 2)}")
        logger.info(f"epoch {epoch} | train binning rms: {utils.round_tensor(train_adaptive_rms, 1)}")
        logger.info(f"epoch {epoch} | train binning rms: {np.mean(train_adaptive_rms.numpy()):.2f}")

        model.eval()
        eval_containment_tensors = []
        eval_pe_dist = []
        eval_interval_length = []
        extra_eval_interval_length = []
        eval_labels = []

        # only save last epoch
        pred_results = []

        with torch.no_grad():
            for step, batch in enumerate(eval_dataloader):

                labels = batch['labels'][:, None].repeat(1, num_intervals)
                eval_labels.append(labels[:,0])
                del batch['labels']
                
                outputs = model(**batch)
                logits = outputs.logits

                lowers, uppers, point_estimates, interval_lengths = get_confidence_intervals(logits)
                containment, _ = utils.evaluate(lowers, uppers, labels, confidence_levels)
                eval_containment_tensors.append(containment)

                # Saving the results
                pred_results.extend([list(row) for row in zip(point_estimates.detach().cpu().tolist(), 
                                                              lowers.detach().cpu().tolist(), 
                                                              uppers.detach().cpu().tolist(), 
                                                              interval_lengths.detach().cpu().tolist(), 
                                                              labels[:, 0:1].detach().cpu().tolist())])

                # point_estimate_loss = criterion(point_estimates, labels[:,0:1])
                eval_pe_dist.append(torch.abs(torch.exp(point_estimates) / torch.exp(labels[:,0:1]) - 1).median().item())
                # interval_length_loss = criterion(interval_lengths, torch.zeros_like(interval_lengths, device=interval_lengths.device))
                eval_interval_length.append(((torch.exp(uppers) - torch.exp(lowers)) / torch.exp(labels[:,0:1])).median(dim=0).values.detach().cpu())

                extra_eval_interval_length.append(((torch.exp(uppers) - torch.exp(lowers)) / torch.exp(labels[:,0:1])))

                # if step == 0:
                #     logger.info(f"{(torch.exp(lowers[:5]), torch.exp(uppers[:5]))}")
                #     logger.info(f"{torch.exp(labels[:5])}")

        eval_metric = torch.cat(eval_containment_tensors).mean(dim=0) * 100
        eval_adaptive_rms = utils.adaptive_binning_rms(eval_containment_tensors, eval_labels, confidence_levels)
        logger.info(f"epoch {epoch} | eval rms: {utils.round_tensor(eval_metric, 1)}")
        logger.info(f"epoch {epoch} | eval rms: {utils.rms(eval_metric, confidence_levels):.2f} | pe dist: {np.median(eval_pe_dist):.2f} | interval len: {np.mean(np.median(torch.stack(eval_interval_length), axis=0)):.2f} | lengths: {np.around(np.median(torch.stack(eval_interval_length), axis=0), 2)}")
        logger.info(f"epoch {epoch} | eval binning rms: {utils.round_tensor(eval_adaptive_rms, 1)}")
        logger.info(f"epoch {epoch} | eval binning rms: {np.mean(eval_adaptive_rms.numpy()):.2f}\n")


        # labels = torch.cat(eval_labels)
        # lengths = torch.cat(extra_eval_interval_length)[labels.argsort()].detach().cpu().numpy()
        # bin_size = 100
        # for i in range(0, lengths.shape[0], bin_size):
        #     print(i, np.median(lengths[i:i+bin_size]))



        if args.push_to_hub and epoch < args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
                repo.push_to_hub(
                    commit_message=f"Training in progress epoch {epoch}", blocking=False, auto_lfs_prune=True
                )

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)

    # import pickle
    # with open(filename, 'w') as csvfile:
    #     csvwriter = csv.writer(csvfile)
    #     csvwriter.writerow(fields)
    #     csvwriter.writerows(pred_results)

    # with open(f'{filename}.pkl', 'wb') as handle:
    #     pickle.dump(pred_results, handle)


if __name__ == "__main__":
    main()