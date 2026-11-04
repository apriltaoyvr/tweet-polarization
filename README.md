# Political Leaning Classification

A mini-experiment for COMP 301. This code uses Hugging Face transformers ([facebook/bart-large-mnli](https://huggingface.co/facebook/bart-large-mnli), [cardiffnlp/twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)) to classify the political leanings of replies and authors to selected tweets. The goal is to check if the majority of replies to a tweet are aligned with the political leanings of the tweet's author.

## Getting Started

```bash
pip install transformers torch
python main.py
```

The initial run will download the models. Expect it to be a little slower than subsequent runs.

## Acknowledgments

Inspired by [Nasuto A, Rowe F (2024), Understanding anti-immigration sentiment spreading on Twitter](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0307917), PLOS ONE. Parts of the code are adapted from the [code accompanying the paper](https://github.com/fcorowe/stigma_covid/tree/main).