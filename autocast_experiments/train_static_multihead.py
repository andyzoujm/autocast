# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
import sys
import torch
import pickle
from torch._C import TensorType
import torch.nn.functional as F
import transformers
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from src.options import Options
import torch.distributed as dist

import src.slurm
import src.util
import src.evaluation
import src.data_multihead
import src.model_multihead


def train(model, optimizer, scheduler, step, train_dataset, eval_dataset, opt, collator, best_dev_em, checkpoint_path):

    if opt.is_main:
        try:
            tb_logger = torch.utils.tensorboard.SummaryWriter(Path(opt.checkpoint_dir)/opt.name)
        except:
            tb_logger = None
            logger.warning('Tensorboard is not available.')

    torch.manual_seed(opt.global_rank + opt.seed) #different seed for different sampling depending on global_rank
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=True,
        # num_workers=2,
        collate_fn=collator
    )

    loss, curr_loss, curr_loss_tfmc, curr_loss_re = 0.0, 0.0, 0.0, 0.0
    model.train()
    for epoch in range(opt.epochs):
        
        epoch += 1
        # train_dataloader.dataset.over_sample()

        for i, batch in enumerate(train_dataloader):
            step += 1
            (_, labels, indices, lengths, context_ids, context_mask, _) = batch

            train_loss, _, (loss_tfmc, loss_re) = model(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                indices=indices.cuda(),
                lengths=lengths.cuda(),
                labels=labels.cuda()
            )[2:]

            train_loss.backward()
            
            if step % opt.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            train_loss = src.util.average_main(train_loss, opt)
            curr_loss += train_loss.item()
            curr_loss_tfmc += loss_tfmc.item()
            curr_loss_re += loss_re.item()
        
        logger.info(f"Epoch {epoch} finished")

        train_em = evaluate(model, train_dataset, tokenizer, collator, opt, epoch, 'train')
        dev_em = evaluate(model, eval_dataset, tokenizer, collator, opt, epoch)
        model.train()
        if opt.is_main:
            if dev_em > best_dev_em:
                best_dev_em = dev_em
                # src.util.save(model, optimizer, scheduler, step, best_dev_em,
                #             opt, checkpoint_path, 'best_dev')
            log = f"{step} / {opt.total_steps} | "
            log += f"train: {curr_loss/opt.eval_freq:.3f}; {curr_loss_tfmc/opt.eval_freq: .3f}; {curr_loss_re/opt.eval_freq: .3f} (EM: {100*train_em:.2f}) | "
            log += f"evaluation: {100*dev_em:.2f}EM | "
            log += f"lr: {scheduler.get_last_lr()[0]:.5f}"
            logger.info(log)
            curr_loss = 0.0
            curr_loss_tfmc = 0.0
            curr_loss_re = 0.0
            if tb_logger is not None:
                tb_logger.add_scalar("Evaluation", dev_em, step)
                tb_logger.add_scalar("Training", curr_loss / (opt.eval_freq), step)
        
        if not opt.epochs and step > opt.total_steps:
            return
    
    if opt.is_main:
        src.util.save(model, optimizer, scheduler, step, best_dev_em,
                    opt, checkpoint_path, f"epoch-{epoch}")

def evaluate(model, dataset, tokenizer, collator, opt, epoch, mode='eval'):
    TF_TOKENS = sum(tokenizer(['no','yes'])['input_ids'], [])
    MC_TOKENS = sum(tokenizer([chr(i + ord('A')) for i in range(12)])['input_ids'], [])

    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
        sampler=sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=False,
        # num_workers=2,
        collate_fn=collator
    )
    model.eval()
    total = 0
    tf_em, mc_em, re_em, exactmatch = [], [], [], []
    tf_predictions, mc_predictions, re_predictions, my_predictions = [], [], [], []
    fields_tf, fields_mc, fields_re = [], [], []
    model = model.module if hasattr(model, "module") else model
    device = torch.device('cpu')
    raw_logits = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            (idx, labels, indices, lengths, context_ids, context_mask, fields) = batch

            re_outputs = model.forward(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                indices=indices.cuda(),
                lengths=lengths.cuda(),
                labels=labels.cuda()
            )[3]
            re_outputs = re_outputs.view(-1, re_outputs.size(-1))

            tfmc_outputs, scores = model.generate(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                max_length=10,
                indices=indices.cuda(),
                lengths=lengths.cuda()
            )

            output_logits = torch.stack(scores).swapaxes(0, 1).detach().to(device)

            indices_re = indices[1][:lengths[1]]
            indices_tf = indices[2][:lengths[2]]
            indices_mc = indices[3][:lengths[3]]
            
            fields_re.extend(list(np.take(fields, indices_re.detach().to(device).tolist())))
            fields_tf.extend(list(np.take(fields, indices_tf.detach().to(device).tolist())))
            fields_mc.extend(list(np.take(fields, indices_mc.detach().to(device).tolist())))

            labels_re = torch.index_select(labels, 0, indices_re)[:, 0].view(-1).detach().to(device).tolist()

            tf_scores, mc_scores = [], []
            tf_logits, mc_logits = [], []
            tf_ans, mc_ans = [], []
            for k, (o, lgs) in enumerate(zip(tfmc_outputs, output_logits)):
                
                ans = tokenizer.decode(o, skip_special_tokens=True)

                gold = dataset.get_example(idx[k])['answers']
                score = src.evaluation.ems(ans, gold)
                total += 1

                if k in indices_tf:
                    tf_scores.append(score)
                    tf_em.append(score)
                    tf_ans.append(ans)
                    tf_predictions.append(ans)

                    tf_logits.append(lgs[0, TF_TOKENS])
                    
                elif k in indices_mc:
                    mc_scores.append(score)
                    mc_em.append(score)
                    mc_ans.append(ans)
                    mc_predictions.append(ans)

                    mc_logits.append(lgs[0, MC_TOKENS])

            re_ans = []
            if len(labels_re) > 0:
                re_ans = re_outputs.view(-1).detach().to(device).tolist()
            re_scores = [np.abs(re_ans[i] - labels_re[i]) \
                         for i in range(len(labels_re))]
            total += len(re_scores)
            re_predictions.extend(re_ans)
            re_em.extend(re_scores)

            temp_scores, temp_predictions = [], []
            tf_count, mc_count, re_count = 0, 0, 0
            re_outputs = re_outputs.to(device).tolist()
            for i in range(len(idx)):
                if i in indices_tf:
                    temp_scores.append(tf_scores[tf_count])
                    if mode == 'eval':
                        temp_predictions.append(tf_ans[tf_count])
                        raw_logits.append(tf_logits[tf_count])
                    tf_count += 1
                elif i in indices_mc:
                    temp_scores.append(mc_scores[mc_count])
                    if mode == 'eval':
                        temp_predictions.append(mc_ans[mc_count])
                        raw_logits.append(mc_logits[mc_count])
                    mc_count += 1
                elif i in indices_re:
                    temp_scores.append(-re_scores[re_count])
                    if mode == 'eval':
                        temp_predictions.append(re_ans[re_count])
                        raw_logits.append(re_outputs[re_count])
                    re_count += 1
                    
            exactmatch.extend(temp_scores)
            my_predictions.extend(temp_predictions)
    
    
    if opt.is_distributed:
        objects = [tf_em, mc_em, re_em, tf_predictions, mc_predictions, re_predictions, fields_tf, fields_mc, fields_re, raw_logits]
        all_objects = [None for _ in range(opt.world_size)]
        dist.gather_object(objects, all_objects if dist.get_rank() == 0 else None)
        
        if opt.is_main:
            main_list = [[] for _ in range(len(objects))]
            for rank, obj_list in enumerate(all_objects):
                for i, obj in enumerate(obj_list):
                    main_list[i] += obj # extend list to gather
            tf_em, mc_em, re_em, tf_predictions, mc_predictions, re_predictions, fields_tf, fields_mc, fields_re, raw_logits = main_list
            fields_tf = np.array(fields_tf)
            fields_mc = np.array(fields_mc)
            fields_re = np.array(fields_re)
    else:
        fields_tf = np.array(fields_tf)
        fields_mc = np.array(fields_mc)
        fields_re = np.array(fields_re)
    
    if mode == 'eval' and (not opt.is_distributed or opt.is_main):
        if len(tf_em) == 0:
            logger.info(f"EVAL: For T/F: Predicted N/A")
        else:
            logger.info(f"EVAL: For T/F: Predicted {tf_em.count(1)} Match {tf_em.count(0)} Wrong \
            ({tf_predictions.count('yes')} YES {tf_predictions.count('no')} NO) | EM: {round(tf_em.count(1) / len(tf_em) * 100, 2)}")
        if len(mc_em) == 0:
            logger.info(f"       For MC:  Predicted N/A")
        else:
            logger.info(f"       For MC:  Predicted {mc_em.count(1)} Match {mc_em.count(0)} Wrong | \
            EM: {round(mc_em.count(1) / len(mc_em) * 100, 2)}")
        if len(re_em) == 0:
            logger.info(f"       For Reg: Predicted N/A")
        else:
            logger.info(f"       For Reg: Dist {np.mean(re_em)}")

    if mode == 'train' and (not opt.is_distributed or opt.is_main):
        if len(tf_em) == 0:
            logger.info(f"TRAIN: For T/F: Predicted N/A")
        else:
            logger.info(f"TRAIN: For T/F: Predicted {tf_em.count(1)} Match {tf_em.count(0)} Wrong \
            ({tf_predictions.count('yes')} YES {tf_predictions.count('no')} NO) | EM: {round(tf_em.count(1) / len(tf_em) * 100, 2)}")
        if len(mc_em) == 0:
            logger.info(f"       For MC:  Predicted N/A")
        else:
            logger.info(f"       For MC:  Predicted {mc_em.count(1)} Match {mc_em.count(0)} Wrong | \
            EM: {round(mc_em.count(1) / len(mc_em) * 100, 2)}")
        if len(re_em) == 0:
            logger.info(f"       For Reg: Predicted N/A")
        else:
            logger.info(f"       For Reg: Dist {np.mean(re_em)}")
    

    if mode == 'eval' and (not opt.is_distributed or opt.is_main):
        with open(checkpoint_path / f'results_epoch{epoch}.obj', 'wb') as f:
            pickle.dump(raw_logits, f)

    # For now we count the regression error rate within 4% as an "exact match"
    exactmatch, total = src.util.weighted_average(np.mean(exactmatch)/2, total, opt)
    return exactmatch

if __name__ == "__main__":
    options = Options()
    options.add_reader_options()
    options.add_optim_options()
    opt = options.parse()
    #opt = options.get_options(use_reader=True, use_optim=True)

    torch.manual_seed(opt.seed)
    src.slurm.init_distributed_mode(opt)
    src.slurm.init_signal_handler()

    checkpoint_path = Path(opt.checkpoint_dir)/opt.name
    checkpoint_exists = checkpoint_path.exists()
    if opt.is_distributed:
        torch.distributed.barrier()
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    #if not checkpoint_exists and opt.is_main:
    #    options.print_options(opt)
    #checkpoint_path, checkpoint_exists = util.get_checkpoint_path(opt)

    logger = src.util.init_logger(
        opt.is_main,
        opt.is_distributed,
        checkpoint_path / 'run.log'
    )

    model_name = 't5-' + opt.model_size
    model_class = src.model_multihead.FiDT5

    #load data
    opt.n_context = opt.n_context or None
    tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)
    collator = src.data_multihead.Collator(opt.text_maxlength, tokenizer,
                                           answer_maxlength=opt.answer_maxlength, n_context=opt.n_context)

    # use golbal rank and world size to split the eval set on multiple gpus
    train_examples = src.data_multihead.load_data(
        opt.train_data, 
        global_rank=opt.global_rank, 
        world_size=opt.world_size,
    )
    train_dataset = src.data_multihead.Dataset(train_examples, opt.n_context, over_sample=False)
    # use golbal rank and world size to split the eval set on multiple gpus
    eval_examples = src.data_multihead.load_data(
        opt.eval_data,
        global_rank=opt.global_rank,
        world_size=opt.world_size,
    )
    eval_dataset = src.data_multihead.Dataset(eval_examples, opt.n_context, over_sample=False)

    if not checkpoint_exists and opt.model_path == "none":
        t5 = transformers.T5ForConditionalGeneration.from_pretrained(model_name)
        model = src.model_multihead.FiDT5(t5.config)
        model.load_t5_multihead(t5.state_dict())
        model = model.to(opt.local_rank)
        optimizer, scheduler = src.util.set_optim(opt, model)
        step, best_dev_em = 0, 0.0
    elif opt.model_path == "none":
        load_path = checkpoint_path / 'checkpoint' / 'latest'
        model, optimizer, scheduler, opt_checkpoint, step, best_dev_em = \
            src.util.load(model_class, load_path, opt, reset_params=False)
        logger.info(f"Model loaded from {load_path}")
    else:
        model, optimizer, scheduler, opt_checkpoint, step, best_dev_em = \
            src.util.load(model_class, opt.model_path, opt, reset_params=True)
        logger.info(f"Model loaded from {opt.model_path}")

    model.set_checkpoint(opt.use_checkpoint)
    logger.info("Setting up Distributed Training")

    if opt.is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            find_unused_parameters=False,
        )

    logger.info(f"NUM EXAMPLE {len(train_dataset)}")
    logger.info("Start training")
    train(
        model,
        optimizer,
        scheduler,
        step,
        train_dataset,
        eval_dataset,
        opt,
        collator,
        best_dev_em,
        checkpoint_path
    )

    # logger.info("Start evaluating")
    # evaluate(model, eval_dataset, tokenizer, collator, opt, 100, 'eval')
