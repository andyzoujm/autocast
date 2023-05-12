#!/bin/sh

# Download the original autocast dataset.
wget https://people.eecs.berkeley.edu/~hendrycks/autocast.tar.gz
tar -xf autocast.tar.gz

# Process the questions.
python preprocess_autocast_questions.py

# Download and preproccess the docs.
python preprocess_cc_news.py

# Clean up.
rm autocast.tar.gz
rm -r autocast
