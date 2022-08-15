#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Command line tool to get dense results and validate them
"""

import json
import time
import datetime
from datetime import date, timedelta
import argparse
import numpy as np
import pandas as pd

from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25
from beir.reranking.models import CrossEncoder
from beir.reranking import Rerank

def save_results(
    questions,
    question_answers,
    question_choices,
    question_targets,
    question_ids,
    question_expiries,
    out_file
):
    merged_data = []
    for i, q in enumerate(questions):
        q_id = question_ids[i]
        q_answers = question_answers[i]
        q_choices = question_choices[i]
        q_targets = question_targets[i]
        expiry = question_expiries[i]

        merged_data.append(
            {
                "question_id": q_id,
                "question": q,
                "answers": str(q_answers),
                "choices": q_choices,
                "question_expiry": expiry,
                "targets": [
                    {
                        "date": index,
                        "target": str(row["target"]) if "target" in row else
                                [str(val) for val in row.values.tolist() if str(val).replace('.','',1).isdigit()],
                        "ctxs": row["ctxs"]
                    }
                    for index, row in q_targets.iterrows()
                ],
                "field": None
            }
        )

    with open(out_file, "w", encoding='utf-8') as writer:
        writer.write(json.dumps(merged_data, indent=4, ensure_ascii=False) + "\n")
    print("Saved results * scores  to %s", out_file)


def main():
    parser = argparse.ArgumentParser(description='Arguments for BM25+CE retriever.')
    parser.add_argument('--beginning', type=str, required=True, help='startg retrieving on this date')
    parser.add_argument('--expiry', type=str, required=True, help='finish retrieving on this date')
    parser.add_argument('--n_docs', type=int, required=True, help='retrieve n daily articles for each question')
    parser.add_argument('--out_file', type=str, required=True, help='output file')
    cfg = parser.parse_args()

    # get questions & answers
    questions = []
    question_choices = []
    question_answers = []
    question_targets = []
    question_ids = []
    question_expiries = []

    ds_key = 'autocast'

    assert cfg.beginning and cfg.expiry
    start_date = datetime.datetime.strptime(cfg.beginning, "%Y-%m-%d")
    end_date = datetime.datetime.strptime(cfg.expiry, "%Y-%m-%d")

    def daterange(start_date, end_date):
        for n in range(int((end_date - start_date).days)):
            yield start_date + timedelta(n)
    
    dates = [str(date.date()) for date in daterange(start_date, end_date)]
    date_to_question_idx = [[] for _ in dates]

    autocast_questions = json.load(open('autocast_questions.json'))
    autocast_questions = [q for q in autocast_questions if q['status'] == 'Resolved']
    for question_idx, ds_item in enumerate(autocast_questions):
        question = ds_item['question']
        background = ds_item['background']
        answers = [ds_item['answer']]
        choices = ds_item['choices']
        qid = ds_item['id']
        expiry = ds_item['close_time']

        if ds_item['qtype'] != 'mc':
            df = pd.DataFrame(ds_item['crowd'])
            df['date'] = df['timestamp'].map(lambda x: x[:10])
            crowd = df.groupby('date').mean().rename(columns={df.columns[1]: 'target'})
            crowd_preds = crowd
        else:
            df = pd.DataFrame(ds_item['crowd'])
            df['date'] = df['timestamp'].map(lambda x: x[:10])
            fs = np.array(df['forecast'].values.tolist())
            for i in range(fs.shape[1]):
                df[f'{i}'] = fs[:,i]
            crowd = df.groupby('date').mean()
            crowd_preds = crowd
        
        crowd_preds.drop(crowd_preds.tail(1).index, inplace=True) # avoid leakage
        crowd_preds['ctxs'] = None
        questions.append(question)
        question_choices.append(choices)
        question_answers.append(answers)
        question_targets.append(crowd_preds)
        question_ids.append(qid)
        question_expiries.append(expiry)

        for date_idx, date in enumerate(dates):
            if date in crowd_preds.index:
                date_to_question_idx[date_idx].append(question_idx)

    time0 = time.time()
    print("Reading questions took %f sec.", time.time() - time0)

    from datasets import Dataset
    from datasets.utils.logging import set_verbosity_error
    set_verbosity_error()

    cc_news_dataset = Dataset.load_from_disk('cc_news')
    cc_news_df = cc_news_dataset.to_pandas() # load all data in memory
    cc_news_df["id"] = cc_news_df.index

    for date_idx, date in enumerate(dates):
        cc_news_df_daily = cc_news_df[cc_news_df['date'] == date]
        if len(cc_news_df_daily) == 0: continue

        k = min(cfg.n_docs, len(cc_news_df_daily))

        ids = cc_news_df_daily['id'].values.tolist()
        titles = cc_news_df_daily['title'].values.tolist()
        texts = cc_news_df_daily['text'].values.tolist()

        daily_corpus = {}
        for i in range(len(ids)):
            json_obj = {}
            json_obj["title"] = titles[i]
            json_obj["text"] = texts[i]
            daily_corpus[str(ids[i])] = json_obj

        question_indices = date_to_question_idx[date_idx]
        daily_queries = {}
        for question_idx in question_indices:
            question = questions[question_idx]
            id = str(ds_key) + "_" + str(question_idx)
            daily_queries[id] = question

        if len(daily_queries) == 0:
            print("no queries for: " + str(date))
            continue

        model = BM25(hostname='http://localhost:9200', index_name=ds_key+"_rainbowquartz_bm25_ce", initialize=True)
        retriever = EvaluateRetrieval(model)
        try:
            scores = retriever.retrieve(daily_corpus, daily_queries)
            print("retrieval done") 
        except Exception as e:
            print("retrieval exception: " + str(e))
            continue

        try:
            # CE reranking
            cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-electra-base')
            reranker = Rerank(cross_encoder_model, batch_size=256)
            rerank_scores = reranker.rerank(daily_corpus, daily_queries, scores, top_k=min(100, k))
            print("reranking done")
        except Exception as e:
            print("reranking exception: " + str(e))
            continue

        for score_idx in rerank_scores:
            top_k = list(rerank_scores[score_idx].items())[:k]

            ctxs = [
                {
                    "id": doc_id,
                    "title": daily_corpus[doc_id]["title"],
                    "text": daily_corpus[doc_id]["text"],
                    "score": score
                } 
                for doc_id, score in top_k
            ]
            # print(ctxs)

            question_idx = int(score_idx.split('_')[-1])
            question_targets[question_idx].at[date, 'ctxs'] = ctxs
        
        if date_idx % 100 == 0:
            print(f"\n{'='*20}\nDone retrieval for 100 days, now at {date}\n{'='*20}\n")
            print("time: %f sec.", time.time() - time0)
            time0 = time.time()

    save_results(
        questions,
        question_answers,
        question_choices,
        question_targets,
        question_ids,
        question_expiries,
        cfg.out_file,
    )

if __name__ == "__main__":
    main()
