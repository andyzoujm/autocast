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
import os
import time

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
    out_file,
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
                        "target": str(row["target"])
                        if "target" in row
                        else [
                            str(val)
                            for val in row.values.tolist()
                            if str(val).replace(".", "", 1).isdigit()
                        ],
                        "ctxs": row["ctxs"],
                    }
                    for index, row in q_targets.iterrows()
                ],
                "field": None,
            }
        )

    with open(out_file, "w", encoding="utf-8") as writer:
        writer.write(json.dumps(merged_data, indent=4, ensure_ascii=False) + "\n")
    print("Saved results * scores  to %s", out_file)


def main():
    parser = argparse.ArgumentParser(description="Arguments for BM25+CE retriever.")
    parser.add_argument(
        "--beginning", type=str, required=True, help="startg retrieving on this date"
    )
    parser.add_argument(
        "--expiry", type=str, required=True, help="finish retrieving on this date"
    )
    parser.add_argument(
        "--n_docs",
        type=int,
        required=True,
        help="retrieve n daily articles for each question",
    )
    parser.add_argument("--out_file", type=str, required=True, help="output file")
    cfg = parser.parse_args()


    ds_key = "autocast"

    assert cfg.beginning and cfg.expiry
    dates = pd.date_range(cfg.beginning, cfg.expiry)

    # Get the absolute path of the current script
    script_path = os.path.abspath(__file__)

    # Get the directory containing the script
    script_dir = os.path.dirname(script_path)

    autocast_questions_path = os.path.join(script_dir, "autocast_questions.json")

    autocast_questions = pd.json_normalize(
        json.load(open(autocast_questions_path)),
        record_path="crowd",
        meta=["question", "answer", "choices", "id", "close_time"],
    ).set_index("id")

    autocast_questions["date"] = pd.to_datetime(autocast_questions["timestamp"]).dt.date
    question_targets = autocast_questions.groupby(["id", "date"])["forecast"].apply(
        lambda series: np.array(series.to_list()).mean(axis=0)
    )
    time0 = time.time()
    print("Reading questions took %f sec.", time.time() - time0)

    from datasets import Dataset
    from datasets.utils.logging import set_verbosity_error

    set_verbosity_error()

    cc_news_path = os.path.join(script_dir, "cc_news")

    cc_news_dataset = Dataset.load_from_disk(cc_news_path)
    cc_news_df = cc_news_dataset.to_pandas()  # load all data in memory

    for date_idx, date in enumerate(dates):
        cc_news_df_daily = cc_news_df[cc_news_df["date"] == date]
        k = min(cfg.n_docs, cc_news_df_daily.shape[0])
        if k == 0:
            continue
        daily_corpus = cc_news_df_daily.to_dict(orient="index")
        daily_queries = autocast_questions.loc[autocast_questions["date"]==date, "question"].to_dict()
        if len(daily_queries) == 0:
            print("no queries for: " + str(date))
            continue

        model = BM25(
            hostname="http://localhost:9200",
            index_name="autocast",
            initialize=True,
        )
        retriever = EvaluateRetrieval(model)
        try:
            scores = retriever.retrieve(daily_corpus, daily_queries)
            print("retrieval done")
        except Exception as e:
            print("retrieval exception: " + str(e))
            continue

        try:
            # CE reranking
            cross_encoder_model = CrossEncoder("cross-encoder/ms-marco-electra-base")
            reranker = Rerank(cross_encoder_model, batch_size=256)
            rerank_scores = reranker.rerank(
                daily_corpus, daily_queries, scores, top_k=min(100, k)
            )
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
                    "score": score,
                }
                for doc_id, score in top_k
            ]
            # print(ctxs)

            question_idx = int(score_idx.split("_")[-1])
            question_targets[question_idx].at[date, "ctxs"] = ctxs

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
