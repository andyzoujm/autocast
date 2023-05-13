# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import time
import sys
import copy
import pickle
import pandas as pd
import torch
from torch._C import _LegacyVariableBase, _create_function_from_graph
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import transformers
import numpy as np
from pathlib import Path
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    DistributedSampler,
    SequentialSampler,
    dataloader,
    MapDataPipe,
)
from src.options import Options

import src.slurm
import src.util
import src.evaluation
import src.forecasting_data_multihead
import src.model

# from transformers import TransformerWrapper, Decoder
from transformers import GPT2Model
import pandas as pd


def identity_collate_fn(data):
    return data


def train(
    model,
    fid_model,
    optimizer,
    scheduler,
    all_params,
    step,
    train_dataset,
    eval_dataset,
    opt,
    fid_collator,
    forecaster_collator,
    best_dev_em,
    checkpoint_path,
):
    if opt.is_main:
        try:
            tb_logger = torch.utils.tensorboard.SummaryWriter(
                Path(opt.checkpoint_dir) / opt.name
            )
        except:
            tb_logger = None
            logger.warning("Tensorboard is not available.")

    torch.manual_seed(
        opt.global_rank + opt.seed
    )  # different seed for different sampling depending on global_rank
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=False,
        num_workers=2,
        collate_fn=identity_collate_fn,
    )

    loss_fn_tf = nn.BCELoss(reduction="none")
    loss_fn_LSM = nn.LogSoftmax(dim=-1)
    loss_fn_re = nn.MSELoss(reduction="none")

    model.train()
    for epoch in range(opt.epochs):
        epoch += 1
        # train_dataloader.dataset.over_sample()  # in multihead we disable oversampling

        curr_loss, curr_loss_tf, curr_loss_mc, curr_loss_re = 0.0, 0.0, 0.0, 0.0
        em_tf, em_mc, em_re = [], [], []
        exactmatch = []
        crowd_em_tf, crowd_em_mc, crowd_em_re = [], [], []
        crowd_exactmatch = []
        my_preds_tf, my_preds_mc, my_preds_re = [], [], []
        my_predictions = []
        time0 = time.time()
        for i, batch in enumerate(train_dataloader):
            step += 1

            # logger.info(f"top of loop {int(time.time() - time0)} sec")
            # time0 = time.time()

            fid_outputs_batch, targets_batch, true_labels_batch, cats_batch = (
                [],
                [],
                [],
                [],
            )
            for fid_dataset, targets, true_label, cat in batch:
                fid_outputs = get_fid_outputs(
                    fid_model, fid_dataset, opt, fid_collator, mode="train"
                )
                fid_outputs_batch.append(fid_outputs)
                targets = targets.to(device=fid_outputs.device)
                targets_batch.append(targets)
                true_labels_batch.append(true_label)
                cats_batch.append(cat)

            # logger.info(f"get_fid_outputs {int(time.time() - time0)} sec")
            # time0 = time.time()

            forecaster_outputs = forecaster_collator(
                fid_outputs_batch, targets_batch, true_labels_batch, cats_batch
            )
            X, mask, labels, true_labels, categories, seq_ends = forecaster_outputs

            # logger.info(f"collate {int(time.time() - time0)} sec")
            # time0 = time.time()

            hidden_state = model(X, mask=mask)  # (B, SEQ, FiD_H)

            # logger.info(f"gpt forward {int(time.time() - time0)} sec")
            # time0 = time.time()

            tf_logits = tf_classifier(hidden_state)[categories == 0, ...]
            tf_probs = F.softmax(tf_logits, dim=-1)[..., 0]
            mc_logits = mc_classifier(hidden_state)[categories == 1, ...]
            re_results = regressor(hidden_state).squeeze(-1)[categories == 2, ...]
            tf_labels = labels[categories == 0, ...][..., 0:1]
            mc_labels = labels[categories == 1, ...]
            re_labels = labels[categories == 2, ...][..., 0]
            tf_mask = mask[categories == 0, ...]
            mc_mask = mask[categories == 1, ...]
            mc_mask_indi = mc_labels >= 0.0
            re_mask = mask[categories == 2, ...]

            loss_tf, loss_mc, loss_re = (
                torch.tensor(0.0).cuda(),
                torch.tensor(0.0).cuda(),
                torch.tensor(0.0).cuda(),
            )
            size_tf, size_mc, size_re = (
                tf_mask.sum().item(),
                mc_mask.sum().item(),
                re_mask.sum().item(),
            )
            size_tf_seq, size_mc_seq, size_re_seq = (
                tf_mask.sum(dim=1),
                mc_mask.sum(dim=1),
                re_mask.sum(dim=1),
            )
            if len(tf_labels) > 0:
                loss_tf_logprobs = loss_fn_LSM(tf_logits)
                loss_tf = -loss_tf_logprobs * torch.cat(
                    (tf_labels, 1 - tf_labels), dim=-1
                )
                loss_tf = (
                    (loss_tf.sum(dim=2) * tf_mask).sum(dim=1) / size_tf_seq
                ).mean()
            if len(mc_labels) > 0:
                loss_mc_logprobs = loss_fn_LSM(mc_logits)
                loss_mc = -loss_mc_logprobs * mc_labels
                loss_mc = (
                    (loss_mc * mc_mask_indi).sum(dim=(1, 2)) / size_mc_seq
                ).mean() / 1.74
            if len(re_labels) > 0:
                loss_re = loss_fn_re(re_results, re_labels)
                loss_re = ((loss_re * re_mask).sum(dim=1) / size_re_seq).mean() / 0.18

            train_loss = loss_tf + loss_mc + loss_re  # TODO: re-weigh?

            # logger.info(f"compute loss {int(time.time() - time0)} sec")
            # time0 = time.time()

            train_loss.backward()

            # logger.info(f"loss backward {int(time.time() - time0)} sec")
            # time0 = time.time()

            seq_ends_indices_tf = seq_ends[categories == 0].unsqueeze(-1)
            seq_ends_indices_mc = seq_ends[categories == 1].unsqueeze(-1)
            seq_ends_expand = seq_ends_indices_mc.expand(
                -1, mc_labels.size()[-1]
            ).unsqueeze(1)
            seq_ends_indices_re = seq_ends[categories == 2].unsqueeze(-1)

            true_labels_tf = true_labels[categories == 0]
            true_labels_mc = true_labels[categories == 1]
            true_labels_re = true_labels[categories == 2]

            if len(true_labels_tf) > 0:
                crowd_preds_tf = (
                    torch.gather(tf_labels.squeeze(-1), -1, seq_ends_indices_tf).view(
                        -1
                    )
                    > 0.5
                )
                preds_tf = (
                    torch.gather(tf_probs, -1, seq_ends_indices_tf).view(-1) > 0.5
                )
            else:
                crowd_preds_tf = torch.tensor([], device=true_labels_tf.device)
                preds_tf = torch.tensor([], device=true_labels_tf.device)

            if len(true_labels_mc) > 0:
                crowd_preds_mc = torch.argmax(
                    torch.gather(mc_labels, 1, seq_ends_expand).squeeze(1), dim=-1
                )
                preds_mc = torch.argmax(
                    torch.gather(mc_logits, 1, seq_ends_expand).squeeze(1), dim=-1
                )
            else:
                crowd_preds_mc = torch.tensor([], device=true_labels_mc.device)
                preds_mc = torch.tensor([], device=true_labels_mc.device)

            if len(true_labels_re) > 0:
                crowd_preds_re = torch.gather(re_labels, -1, seq_ends_indices_re).view(
                    -1
                )
                preds_re = torch.gather(re_results, -1, seq_ends_indices_re).view(-1)
            else:
                crowd_preds_re = torch.tensor([], device=true_labels_re.device)
                preds_re = torch.tensor([], device=true_labels_re.device)

            crowd_em_tf.extend(
                (true_labels_tf == crowd_preds_tf).detach().cpu().numpy()
            )
            em_tf.extend((true_labels_tf == preds_tf).detach().cpu().numpy())
            crowd_em_mc.extend(
                (true_labels_mc == crowd_preds_mc).detach().cpu().numpy()
            )
            em_mc.extend((true_labels_mc == preds_mc).detach().cpu().numpy())
            crowd_em_re.extend(
                -torch.abs(true_labels_re - crowd_preds_re).detach().cpu().numpy()
            )
            em_re.extend(-torch.abs(true_labels_re - preds_re).detach().cpu().numpy())

            my_preds_tf.extend(preds_tf.detach().cpu().numpy())
            my_preds_mc.extend(preds_mc.detach().cpu().numpy())
            my_preds_re.extend(preds_re.detach().cpu().numpy())

            # logger.info(f"compute metrics {int(time.time() - time0)} sec")
            # time0 = time.time()

            if step % opt.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(all_params, opt.clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # logger.info(f"optimizer step {int(time.time() - time0)} sec")
            # time0 = time.time()

            train_loss = src.util.average_main(train_loss, opt)
            curr_loss += train_loss.item()

            if size_tf > 0:
                curr_loss_tf += src.util.average_main(loss_tf, opt).item()
            if size_mc > 0:
                curr_loss_mc += src.util.average_main(loss_mc, opt).item()
            if size_re > 0:
                curr_loss_re += src.util.average_main(loss_re, opt).item()

        # out-of-order overall stats
        crowd_exactmatch = crowd_em_tf + crowd_em_mc + crowd_em_re
        exactmatch = em_tf + em_mc + em_re
        # my_predictions = my_preds_tf + my_preds_mc + my_preds_re

        logger.info(
            f"Epoch {epoch} finished | {50*np.mean(exactmatch):.2f} EM | {50*np.mean(crowd_exactmatch):.2f} crowd EM"
        )
        if len(em_tf) == 0:
            logger.info(f"TRAIN: For T/F: Predicted N/A")
        else:
            logger.info(
                f"TRAIN: For T/F: Predicted {em_tf.count(1)} Match {em_tf.count(0)} Wrong \
            ({my_preds_tf.count(1)} YES {my_preds_tf.count(0)} NO) | EM: {round(em_tf.count(1) / len(em_tf) * 100, 2)}"
            )
        if len(em_mc) == 0:
            logger.info(f"       For MC:  Predicted N/A")
        else:
            logger.info(
                f"       For MC:  Predicted {em_mc.count(1)} Match {em_mc.count(0)} Wrong | \
            EM: {round(em_mc.count(1) / len(em_mc) * 100, 2)}"
            )
        if len(em_re) == 0:
            logger.info(f"       For Reg: Predicted N/A")
        else:
            logger.info(f"       For Reg: Predicted Dist {-np.mean(em_re)}")
        logger.info(f"{int(time.time() - time0)} sec")

        dev_em, test_loss, crowd_em = evaluate(
            model,
            fid_model,
            eval_dataset,
            fid_collator,
            forecaster_collator,
            opt,
            epoch,
        )
        model.train()
        if opt.is_main:
            if dev_em > best_dev_em:
                best_dev_em = dev_em
                # src.util.save(fid_model, optimizer, scheduler, step, best_dev_em,
                #             opt, checkpoint_path, 'best_dev')
            log = f"{step} / {opt.total_steps} | "
            log += f"train: {curr_loss / len(train_dataloader):.3f}; {curr_loss_tf / len(train_dataloader):.3f} / \
            {curr_loss_mc / len(train_dataloader):.3f} / {curr_loss_re / len(train_dataloader):.3f} | "
            log += f"test: {test_loss:.3f} | "
            log += f"evaluation: {100*dev_em:.2f} EM (crowd: {100*crowd_em:.2f} EM) | "
            log += f"lr: {scheduler.get_last_lr()[0]:.5f}"
            logger.info(log)
            curr_loss = 0.0
            curr_loss_tf = 0.0
            curr_loss_mc = 0.0
            curr_loss_re = 0.0
            if tb_logger is not None:
                tb_logger.add_scalar("Evaluation", dev_em, step)
                tb_logger.add_scalar("Training", curr_loss, step)

        if not opt.epochs and step > opt.total_steps:
            return

    if opt.is_main:
        src.util.save(
            fid_model,
            optimizer,
            scheduler,
            step,
            best_dev_em,
            opt,
            checkpoint_path,
            f"epoch-{epoch}-fidmodel",
        )
        src.util.save(
            model,
            optimizer,
            scheduler,
            step,
            best_dev_em,
            opt,
            checkpoint_path,
            f"epoch-{epoch}-gptmodel",
        )


def evaluate(model, fid_model, dataset, fid_collator, forecaster_collator, opt, epoch):
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=opt.per_gpu_batch_size * 4,
        drop_last=False,
        num_workers=2,
        collate_fn=identity_collate_fn,
    )
    model.eval()

    loss_fn_tf = nn.BCELoss(reduction="none")
    loss_fn_LSM = nn.LogSoftmax(dim=-1)
    loss_fn_re = nn.MSELoss(reduction="none")

    total_loss = 0.0
    em_tf, em_mc, em_re = [], [], []
    exactmatch = []
    crowd_em_tf, crowd_em_mc, crowd_em_re = [], [], []
    crowd_exactmatch = []
    my_preds_tf, my_preds_mc, my_preds_re = [], [], []
    my_predictions = []
    time0 = time.time()
    device = torch.device("cpu")
    raw_logits = []
    # model = model.module if hasattr(model, "module") else model
    with torch.no_grad():
        for _, batch in enumerate(dataloader):
            fid_outputs_batch, targets_batch, true_labels_batch, cats_batch = (
                [],
                [],
                [],
                [],
            )
            for fid_dataset, targets, true_label, cat in batch:
                fid_outputs = get_fid_outputs(
                    fid_model, fid_dataset, opt, fid_collator, mode="eval"
                )
                fid_outputs_batch.append(fid_outputs)
                targets = targets.to(device=fid_outputs.device)
                targets_batch.append(targets)
                true_labels_batch.append(true_label)
                cats_batch.append(cat)

            forecaster_outputs = forecaster_collator(
                fid_outputs_batch, targets_batch, true_labels_batch, cats_batch
            )
            X, mask, labels, true_labels, categories, seq_ends = forecaster_outputs

            hidden_state = model(X, mask=mask)  # (B, SEQ, FiD_H)
            tf_logits = tf_classifier(hidden_state)[categories == 0, ...]
            tf_probs = F.softmax(tf_logits, dim=-1)[..., 0]
            mc_logits = mc_classifier(hidden_state)[categories == 1, ...]
            re_results = regressor(hidden_state).squeeze(-1)[categories == 2, ...]
            tf_labels = labels[categories == 0, ...][..., 0:1]
            mc_labels = labels[categories == 1, ...]
            re_labels = labels[categories == 2, ...][..., 0]
            tf_mask = mask[categories == 0, ...]
            mc_mask = mask[categories == 1, ...]
            mc_mask_indi = mc_labels >= 0.0
            re_mask = mask[categories == 2, ...]

            loss_tf, loss_mc, loss_re = (
                torch.tensor(0.0).cuda(),
                torch.tensor(0.0).cuda(),
                torch.tensor(0.0).cuda(),
            )
            size_tf_seq, size_mc_seq, size_re_seq = (
                tf_mask.sum(dim=1),
                mc_mask.sum(dim=1),
                re_mask.sum(dim=1),
            )
            if len(tf_labels) > 0:
                loss_tf_logprobs = loss_fn_LSM(tf_logits)
                loss_tf = -loss_tf_logprobs * torch.cat(
                    (tf_labels, 1 - tf_labels), dim=-1
                )
                loss_tf = (
                    (loss_tf.sum(dim=2) * tf_mask).sum(dim=1) / size_tf_seq
                ).mean()
            if len(mc_labels) > 0:
                loss_mc_logprobs = loss_fn_LSM(mc_logits)
                loss_mc = -loss_mc_logprobs * mc_labels
                loss_mc = (
                    (loss_mc * mc_mask_indi).sum(dim=(1, 2)) / size_mc_seq
                ).mean() / 1.74
            if len(re_labels) > 0:
                loss_re = loss_fn_re(re_results, re_labels)
                loss_re = ((loss_re * re_mask).sum(dim=1) / size_re_seq).mean() / 0.18

            train_loss = loss_tf + loss_mc + loss_re  # TODO: re-weigh?
            total_loss += train_loss.item()

            seq_ends_indices_tf = seq_ends[categories == 0].unsqueeze(-1)
            seq_ends_indices_mc = seq_ends[categories == 1].unsqueeze(-1)
            seq_ends_expand = seq_ends_indices_mc.expand(
                -1, mc_labels.size()[-1]
            ).unsqueeze(1)
            seq_ends_indices_re = seq_ends[categories == 2].unsqueeze(-1)

            true_labels_tf = true_labels[categories == 0]
            true_labels_mc = true_labels[categories == 1]
            true_labels_re = true_labels[categories == 2]

            if len(true_labels_tf) > 0:
                crowd_preds_tf = (
                    torch.gather(tf_labels.squeeze(-1), -1, seq_ends_indices_tf).view(
                        -1
                    )
                    > 0.5
                )
                preds_tf = (
                    torch.gather(tf_probs, -1, seq_ends_indices_tf).view(-1) > 0.5
                )
            else:
                crowd_preds_tf = torch.tensor([], device=true_labels_tf.device)
                preds_tf = torch.tensor([], device=true_labels_tf.device)

            if len(true_labels_mc) > 0:
                crowd_preds_mc = torch.argmax(
                    torch.gather(mc_labels, 1, seq_ends_expand).squeeze(1), dim=-1
                )
                preds_mc = torch.argmax(
                    torch.gather(mc_logits, 1, seq_ends_expand).squeeze(1), dim=-1
                )
            else:
                crowd_preds_mc = torch.tensor([], device=true_labels_mc.device)
                preds_mc = torch.tensor([], device=true_labels_mc.device)

            if len(true_labels_re) > 0:
                crowd_preds_re = torch.gather(re_labels, -1, seq_ends_indices_re).view(
                    -1
                )
                preds_re = torch.gather(re_results, -1, seq_ends_indices_re).view(-1)
            else:
                crowd_preds_re = torch.tensor([], device=true_labels_re.device)
                preds_re = torch.tensor([], device=true_labels_re.device)

            crowd_em_tf.extend(
                (true_labels_tf == crowd_preds_tf).detach().cpu().numpy()
            )
            em_tf.extend((true_labels_tf == preds_tf).detach().cpu().numpy())
            crowd_em_mc.extend(
                (true_labels_mc == crowd_preds_mc).detach().cpu().numpy()
            )
            em_mc.extend((true_labels_mc == preds_mc).detach().cpu().numpy())
            crowd_em_re.extend(
                -torch.abs(true_labels_re - crowd_preds_re).detach().cpu().numpy()
            )
            em_re.extend(-torch.abs(true_labels_re - preds_re).detach().cpu().numpy())

            my_preds_tf.extend(preds_tf.detach().cpu().numpy())
            my_preds_mc.extend(preds_mc.detach().cpu().numpy())
            my_preds_re.extend(preds_re.detach().cpu().numpy())

            tf_count, mc_count, re_count = 0, 0, 0
            seq_ends = seq_ends.detach().to(device).numpy() + 1
            tf_logits = tf_logits.detach().to(device).numpy()
            mc_logits = mc_logits.detach().to(device).numpy()
            re_results = re_results.detach().to(device).numpy()
            for i, cat in enumerate(categories):
                if cat == 0:
                    raw_logits.append(tf_logits[tf_count][: seq_ends[i]])
                    tf_count += 1
                elif cat == 1:
                    raw_logits.append(mc_logits[mc_count][: seq_ends[i]])
                    mc_count += 1
                elif cat == 2:
                    raw_logits.append(re_results[re_count][: seq_ends[i]])
                    re_count += 1

    # out-of-order overall stats
    crowd_exactmatch = crowd_em_tf + crowd_em_mc + crowd_em_re
    exactmatch = em_tf + em_mc + em_re
    my_predictions = my_preds_tf + my_preds_mc + my_preds_re

    if not checkpoint_path.exists():
        checkpoint_path.mkdir()
    outpath = checkpoint_path / f"results_epoch{epoch}.obj"
    with open(outpath, "wb") as f:
        pickle.dump(raw_logits, f)

    if len(em_tf) == 0:
        logger.info(f"EVAL:  For T/F: Predicted N/A")
    else:
        logger.info(
            f"EVAL:  For T/F: Predicted {em_tf.count(1)} Match {em_tf.count(0)} Wrong \
        ({my_preds_tf.count(1)} YES {my_preds_tf.count(0)} NO) | EM: {round(em_tf.count(1) / len(em_tf) * 100, 2)}"
        )
    if len(em_mc) == 0:
        logger.info(f"       For MC:  Predicted N/A")
    else:
        logger.info(
            f"       For MC:  Predicted {em_mc.count(1)} Match {em_mc.count(0)} Wrong | \
        EM: {round(em_mc.count(1) / len(em_mc) * 100, 2)}"
        )
    if len(em_re) == 0:
        logger.info(f"       For Reg: Predicted N/A")
    else:
        logger.info(f"       For Reg: Predicted Dist {np.mean(em_re)}")
    logger.info(f"{int(time.time() - time0)} sec")

    exactmatch, test_loss = src.util.weighted_average(
        np.mean(exactmatch) / 2, total_loss / len(dataloader), opt
    )
    return exactmatch, test_loss, np.mean(crowd_exactmatch) / 2


STR2BOOL = {"yes": 1, "Yes": 1, "no": 0, "No": 0}


class ForecastingDataset:
    """
    Iterative predictions as sequence modeling
    where each token embeddings is replaced by
    hidden-state representation of daily news articles
    """

    def __init__(
        self,
        questions: pd.DataFrame,
        crowd,
        schedule,
        corpus,
        opt,
        max_seq_len=128,
        n_context=None,
        question_prefix="question:",
        title_prefix="title:",
        passage_prefix="context:",
        choices_prefix="choices:",
        bound_prefix="bounds:",
    ):
        self.corpus = corpus.rename_axis("doc_id")
        self.schedule = schedule
        self.questions = questions.loc[schedule.keys(),:]
        self.crowd = crowd
        self.corpus = corpus
        self.opt = opt
        # adjust crowd to true targets
        self.max_seq_len = max_seq_len
        self.n_context = n_context
        self.question_prefix = question_prefix
        self.title_prefix = title_prefix
        self.passage_prefix = passage_prefix
        self.choices_prefix = choices_prefix
        self.bound_prefix = bound_prefix
        self.data_by_class_displayed = False
        self.data_by_class = {}
        self.int_keys = [key for key in self.schedule]

        # self.pre_filter(over_sample)

    def __len__(self):
        return len(self.schedule)

    def __getitem__(self, index):
        question_data = self.questions.iloc[index, :]
        question_id = question_data.name
        q_schedule = self.schedule.get(question_id)
        q_crowd = self.crowd.get(question_id)
        q_schedule = pd.DataFrame(q_schedule)
        if "date" not in q_schedule.columns:
            print("stop here")
        q_schedule = q_schedule.set_index("date")
        q_crowd = pd.DataFrame.from_dict(q_crowd, orient="index")
        q_schedule.index = pd.to_datetime(q_schedule.index)
        q_crowd.index = pd.to_datetime(q_crowd.index)
        truncated_common_dates = q_crowd.index.intersection(
            q_schedule.index
        ).sort_values(ascending=False)[: self.max_seq_len]
        q_schedule = q_schedule.loc[truncated_common_dates, :]
        q_crowd = q_crowd.loc[truncated_common_dates, :]
        research_materal = self.corpus.loc[q_schedule["doc_id"], :]
        answer = question_data.get("answer")
        qtype = question_data.get("qtype")
        if qtype == "t/f":
            code = 0
        elif qtype == "mc":
            code = 1
        elif qtype == "num":
            code = 2

        targets = torch.tensor(q_crowd.to_numpy())

        # Pad targets to be n x max_choice_len
        targets_padded = torch.full((targets.size(0), max_choice_len), -1.0)
        targets_padded[: targets.size(0), : targets.size(1)] = targets

        fid_dataset = src.forecasting_data_multihead.FiDDataset(
            question_data,
            q_schedule,
            research_materal,
            self.n_context,
            self.question_prefix,
            self.title_prefix,
            self.passage_prefix,
            self.choices_prefix,
            self.bound_prefix,
            max_choice_len,
        )
        return fid_dataset, targets_padded, answer, code


def get_fid_outputs(model, dataset, opt, collator, mode):
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(
        dataset, sampler=sampler, batch_size=16, drop_last=False, collate_fn=collator
    )

    # logger.info(f"get fid dataloader {float(time.time() - time0)} sec")
    # time0 = time.time()

    outputs = []

    ### NO GRADIENTS ####
    if not opt.finetune_encoder:
        mode = "eval"
    ### NO GRADIENTS ####

    model.train(mode == "train")
    for i, batch in enumerate(dataloader):
        (_, labels, _, _, context_ids, context_mask) = batch

        # logger.info(f"fid forward started {float(time.time() - time0)} sec")
        # time0 = time.time()

        # TODO: we could pass in labels here too for additional training signal
        with torch.set_grad_enabled(mode == "train"):
            model_output = model(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                labels=labels.cuda(),  # we use true labels for FiD hidden states
                output_hidden_states=True,
            )
        hidden_state = model_output[3][-1]
        outputs.append(hidden_state)

        # logger.info(f"fid forward finished {float(time.time() - time0)} sec")
        # time0 = time.time()

    outputs = torch.cat(outputs, dim=0)  # (n_examples, 1, hidden_size)
    outputs = outputs.view(outputs.shape[0], -1)  # (n_examples, hidden_size)

    # TODO: we could add a linear layer here

    # logger.info(f"get hidden from fid {float(time.time() - time0)} sec")

    if mode == "eval":
        outputs = outputs.detach()

    return outputs


def forecaster_collate_fn(examples, labels, true_labels, cats):
    seq_lengths = [len(label) for label in labels]
    examples = pad_sequence(examples, batch_first=True, padding_value=0.0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-2.0)
    mask = torch.ones_like(labels[:, :, 0]).bool()
    for i, seq_len in enumerate(seq_lengths):
        mask[i][seq_len:] = 0
    seq_ends = torch.tensor(seq_lengths, device=mask.device) - 1  # last day to predict
    true_labels = torch.tensor(true_labels, device=mask.device)
    cats = torch.tensor(cats, device=mask.device)
    assert examples.shape[:2] == mask.shape, (examples.shape[:2], mask.shape)

    return examples, mask, labels, true_labels, cats, seq_ends


def get_gpt(fid_hidden_size, gpt_hidden_size, opt, model_name="gpt2"):
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name)

    # model = GPT2Model.from_pretrained(model_name)
    model = GPT2Model(config)

    # directly use hidden state features from FiD
    # model.set_input_embeddings(nn.Linear(fid_hidden_size, gpt_hidden_size))
    # model.lm_head = nn.Identity(gpt_hidden_size)  # nn.Identity()
    model = model.cuda()

    input_embeddings = nn.Linear(fid_hidden_size, gpt_hidden_size).cuda()

    def gpt2_forward(self, X, mask):
        X = input_embeddings(X)
        # get last hidden state
        return model._call_impl(inputs_embeds=X, attention_mask=mask)[
            0
        ]  # last hidden state, (presents), (all hidden_states), (attentions)

    GPT2Model.__call__ = gpt2_forward
    # forecaster.parameters = lambda: model.parameters()
    # forecaster.train = lambda: model.train()
    # forecaster.eval = lambda: model.eval()

    tf_head = nn.Linear(gpt_hidden_size, 2)
    mc_head = nn.Linear(gpt_hidden_size, max_choice_len)
    regressor_head = nn.Sequential(nn.Linear(gpt_hidden_size, 1), nn.Sigmoid())

    tf_head = tf_head.cuda()
    mc_head = mc_head.cuda()
    regressor_head = regressor_head.cuda()

    return model, input_embeddings, tf_head, mc_head, regressor_head


tf_classifier, mc_classifier, regressor = None, None, None
max_choice_len = 12

if __name__ == "__main__":
    options = Options()
    options.add_reader_options()
    options.add_forecaster_options()
    options.add_optim_options()
    opt = options.parse()
    # opt = options.get_options(use_reader=True, use_optim=True)

    torch.manual_seed(opt.seed)
    src.slurm.init_distributed_mode(opt)
    src.slurm.init_signal_handler()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(script_dir, "run.log")

    logger = src.util.init_logger(opt.is_main, opt.is_distributed, log_path)

    model_name = "t5-" + opt.model_size
    model_class = src.model.FiDT5

    # load data
    tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)
    fid_collator = src.forecasting_data_multihead.Collator(
        opt.text_maxlength, tokenizer, answer_maxlength=opt.answer_maxlength
    )

    
    if opt.model_path:
        checkpoint_path = Path(script_dir) / "checkpoint" / opt.model_path
    else:
        checkpoint_path = Path(script_dir) / "checkpoint" / "latest"

    if checkpoint_path.exists():
        model, optimizer, scheduler, opt_checkpoint, step, best_dev_em = src.util.load(
            model_class, checkpoint_path, opt, reset_params=True
        )
        logger.info(f"Model loaded from {checkpoint_path}")
    else:
        t5 = transformers.T5ForConditionalGeneration.from_pretrained(model_name)
        model = src.model.FiDT5(t5.config)
        model.load_t5(t5.state_dict())
        model = model.to(opt.local_rank)
        optimizer, scheduler = src.util.set_optim(opt, model)
        step, best_dev_em = 0, 0.0
        
    if opt.is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            find_unused_parameters=False,
        )

    model.reset_head_to_identity()  # get hidden state output instead of lm_logits
    model = model.cuda()

    # use golbal rank and world size to split the eval set on multiple gpus
    # train_examples = src.forecasting_data_multihead.load_data(
    #     opt.train_data,
    #     opt.n_context,
    #     global_rank=opt.global_rank,
    #     world_size=opt.world_size,
    # )
    from datasets import Dataset

    ccnews_path = os.path.join(script_dir, "data/cc_news")
    corpus = (
        Dataset.load_from_disk(ccnews_path)
        .to_pandas()
        .set_index("id")
    )

    train_questions_path = os.path.join(script_dir, opt.train_questions)
    train_crowd_path = os.path.join(script_dir, opt.train_crowd)
    train_schedule_path = os.path.join(script_dir, opt.train_schedule)
    train_questions = pd.read_json(train_questions_path, orient="index")
    with open(train_crowd_path, "r") as datafile:
        train_crowd = json.load(datafile)
    with open(train_schedule_path, "r") as datafile:
        train_schedule = json.load(datafile)

    train_dataset = ForecastingDataset(
        train_questions,
        train_crowd,
        train_schedule,
        corpus,
        opt,
        max_seq_len=opt.max_seq_len,
        n_context=opt.n_context,
    )
    # use golbal rank and world size to split the eval set on multiple gpus
    # eval_examples = src.forecasting_data_multihead.load_data(
    #     opt.eval_data,
    #     opt.n_context,
    #     global_rank=opt.global_rank,
    #     world_size=opt.world_size,
    # )
    test_questions_path = os.path.join(script_dir, opt.test_questions)
    test_crowd_path = os.path.join(script_dir, opt.test_crowd)
    test_schedule_path = os.path.join(script_dir, opt.test_schedule)
    test_questions = pd.read_json(test_questions_path, orient="index")
    with open(test_crowd_path, "r") as datafile:
        test_crowd = json.load(datafile)
    with open(test_schedule_path, "r") as datafile:
        test_schedule = json.load(datafile)
    eval_dataset = ForecastingDataset(
        test_questions,
        test_crowd,
        test_schedule,
        corpus,
        opt,
        max_seq_len=opt.max_seq_len,
        n_context=opt.n_context,
    )

    # initialize forecaster here
    # gpt_model_name = 'gpt2'
    # if opt.model_size == 'base':
    #     fid_hidden_size = 768
    # elif opt.model_size == 'large':
    #     fid_hidden_size = 1024
    #     gpt_model_name += '-medium'

    gpt_model_name = "gpt2"
    if opt.model_size == "small":
        fid_hidden_size = 512
        gpt_hidden_size = 768
    elif opt.model_size == "base":
        fid_hidden_size = 768
        gpt_hidden_size = 1024
        gpt_model_name += "-medium"
    elif opt.model_size == "large":
        fid_hidden_size = 1024
        gpt_hidden_size = 1280
        gpt_model_name += "-large"
    elif opt.model_size == "3b":
        fid_hidden_size = 1024
        gpt_hidden_size = 1600
        gpt_model_name += "-xl"

    forecaster, input_embeddings, tf_classifier, mc_classifier, regressor = get_gpt(
        fid_hidden_size, gpt_hidden_size, opt, gpt_model_name
    )
    all_params = (
        list(model.parameters())
        + list(forecaster.parameters())
        + list(input_embeddings.parameters())
        + list(tf_classifier.parameters())
        + list(mc_classifier.parameters())
        + list(regressor.parameters())
    )

    #### NO GRADIENTS ####
    # all_params = list(forecaster.parameters()) + list(input_embeddings.parameters()) + \
    #              list(tf_classifier.parameters()) + list(mc_classifier.parameters()) + list(regressor.parameters())
    #### NO GRADIENTS ####

    optimizer, scheduler = src.util.set_optim(opt, model, all_params)

    logger.info(f"TRAIN EXAMPLE {len(train_dataset)}")
    logger.info(f"EVAL EXAMPLE {len(eval_dataset)}")
    logger.info("Start training")

    train(
        forecaster,
        model,
        optimizer,
        scheduler,
        all_params,
        step,
        train_dataset,
        eval_dataset,
        opt,
        fid_collator,
        forecaster_collate_fn,
        best_dev_em,
        checkpoint_path,
    )

    # logger.info("Start evaluating")
    # evaluate(forecaster, model, eval_dataset, fid_collator, forecaster_collate_fn, opt, 0)
