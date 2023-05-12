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
import argparse
import pandas as pd
import traceback
from collections import defaultdict

from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25
from beir.reranking.models import CrossEncoder
from beir.reranking import Rerank


def main():
    parser = argparse.ArgumentParser(description="Arguments for BM25+CE retriever.")
    parser.add_argument(
        "--n_docs",
        type=int,
        required=True,
        help="retrieve n daily articles for each question",
    )
    parser.add_argument("--in_file", type=str, required=True, help="In file")
    parser.add_argument("--out_file", type=str, required=True, help="output file")
    cfg = parser.parse_args()

    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    question_path = os.path.join(script_dir, cfg.in_file)
    questions = pd.read_json(question_path, orient="index")
    questions["publish_time"] = pd.to_datetime(questions["publish_time"])
    questions["close_time"] = pd.to_datetime(questions["close_time"])

    from datasets import Dataset
    from datasets.utils.logging import set_verbosity_error

    set_verbosity_error()

    cc_news_path = os.path.join(script_dir, "cc_news")
    cc_news_df = Dataset.load_from_disk(cc_news_path).to_pandas()
    cc_news_df = cc_news_df.set_index("id")
    cc_news_df.index = cc_news_df.index.map(str)

    # cross_encoder_model = CrossEncoder("cross-encoder/ms-marco-electra-base")
    # reranker = Rerank(cross_encoder_model, batch_size=256)

    # Initialize an empty dictionary with the desired structure
    training_schedule = defaultdict(list)

    start_date = questions["publish_time"].min()
    end_date = questions["close_time"].max()
    for date in pd.date_range(start_date, end_date):
        daily_corpus = cc_news_df[cc_news_df["date"].dt.date == date.date()].to_dict(
            orient="index"
        )
        daily_queries = questions.loc[
            (questions["publish_time"] < date) & (date < questions["close_time"]),
            "question",
        ].to_dict()
        if not daily_corpus or not daily_queries:
            continue
        model = BM25(
            hostname="http://localhost:9200",
            index_name="autocast",
            initialize=True,
        )
        retriever = EvaluateRetrieval(model, k_values=[cfg.n_docs])
        try:
            scores = retriever.retrieve(daily_corpus, daily_queries)
            print("retrieval done")
        except Exception as e:
            print("retrieval exception: " + str(e))
            continue
        # try:
        #     # CE reranking
        #     rerank_scores = reranker.rerank(
        #         daily_corpus,
        #         daily_queries,
        #         scores,
        #         top_k=min(100, cfg.n_docs, len(daily_corpus)),
        #     )
        #     print("reranking done")
        # except Exception as e:
        #     print("retrieval exception:")
        #     print(traceback.format_exc())
        #     continue
        # Update the training_schedule dictionary with the new structure
        date_str = date.strftime("%Y-%m-%d")
        for question_id, doc_scores in scores.items():
            for doc_id, score in doc_scores.items():
                training_schedule[question_id].append(
                    {"date": date_str, "doc_id": int(doc_id), "score": score}
                )
    out_path = os.path.join(script_dir, cfg.out_file)
    with open(out_path, "w") as outfile:
        json.dump(training_schedule, outfile, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
