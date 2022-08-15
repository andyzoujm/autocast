#!/bin/sh

export B='2016-01-01'
export E='2022-04-12'

python retrieve_cc_news_bm25+ce.py \
  out_file=autocast_cc_news_retrieved.json \
  n_docs=10 \
  beginning=$B \
  expiry=$E
