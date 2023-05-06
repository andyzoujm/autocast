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

    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    autocast_questions = pd.read_json(
        os.path.join(script_dir, "autocast_questions.json"),
    )[["id", "question", "publish_time", "close_time"]].set_index("id")
    autocast_questions["publish_time"] = pd.to_datetime(
        autocast_questions["publish_time"]
    ).dt.tz_localize(None)
    autocast_questions["close_time"] = (
        pd.to_datetime(autocast_questions["close_time"], errors="coerce")
        .dt.tz_localize(None)
        .fillna(pd.Timestamp.now())
    )  # replaces distant dates not supported by pd.Timestamp with current timestamp.

    from datasets import Dataset
    from datasets.utils.logging import set_verbosity_error

    set_verbosity_error()

    cc_news_path = os.path.join(script_dir, "cc_news")
    cc_news_df = Dataset.load_from_disk(cc_news_path).to_pandas()
    cc_news_df.index = cc_news_df.index.map(str)
    training_schedule = {}

    cross_encoder_model = CrossEncoder("cross-encoder/ms-marco-electra-base")
    reranker = Rerank(cross_encoder_model, batch_size=256)

    # Initialize an empty dictionary with the desired structure
    training_schedule = defaultdict(lambda: defaultdict(dict))

    for date in pd.date_range(cfg.beginning, cfg.expiry):
        daily_corpus = cc_news_df[cc_news_df["date"] == date].to_dict(orient="index")
        daily_queries = autocast_questions.loc[
            (autocast_questions["publish_time"] < date)
            & (date < autocast_questions["close_time"]),
            "question",
        ].to_dict()
        if not daily_corpus or not daily_queries:
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
            rerank_scores = reranker.rerank(
                daily_corpus,
                daily_queries,
                scores,
                top_k=min(100, cfg.n_docs, len(daily_corpus)),
            )
            print("reranking done")
        except Exception as e:
            print("retrieval exception:")
            print(traceback.format_exc())
            continue

        # Update the training_schedule dictionary with the new structure
        date_str = date.strftime("%Y-%m-%d")
        for question_id, doc_scores in rerank_scores.items():
            training_schedule[question_id][date_str] = doc_scores

    with open(cfg.out_file, "w", encoding="utf-8") as writer:
        writer.write(json.dumps(training_schedule, indent=4, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
