# Political Leaning Classification

Determine if the most popular replies to tweets are in agreement with the author. Part of a small experiment for COMP 301 to discuss social media bubbles and echo chambers. 

## Description

The code uses three Hugging Face transformers to classify the political leaning of tweets and their replies. 

The models used are:

- [MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli](https://huggingface.co/MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli) for NLI entailment
- [cardiffnlp/twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) for sentiment analysis
- [cardiffnlp/twitter-roberta-base-irony](https://huggingface.co/cardiffnlp/twitter-roberta-base-irony) for irony detection

`data/input` contains the input tweets, which were hand selected and labeled for classification. `data/output/features.csv` is a CSV file with predicted entailment, sentiment, and irony for each reply to a tweet.

## Getting Started

The initial run will download the models. Expect it to be a little slower than subsequent runs.

```bash
pip install transformers torch pandas
python main.py
```

For training the models, run:

```bash
pip install scikit-learn joblib
python train.py
```

## Acknowledgments

Inspired by [Nasuto A, Rowe F (2024), Understanding anti-immigration sentiment spreading on Twitter](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0307917), PLOS ONE. Parts of the code are adapted from the [code accompanying the paper](https://github.com/fcorowe/stigma_covid/tree/main).