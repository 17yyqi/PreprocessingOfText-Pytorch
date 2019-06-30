# -*- coding: utf-8 -*-
"""
Spyder Editor

读取原始的斯坦福数据集，处理成.csv文件。
在开始写模型之前跑一边这个程序就可以，.csv文件被存储下来之后就不用再跑了
"""
import os

import pandas

dictPath = 'F:\stanfordSentimentTreebank\stanfordSentimentTreebank\dictionary.txt'
labelPath = 'F:\stanfordSentimentTreebank\stanfordSentimentTreebank\sentiment_labels.txt'
sentPath = 'F:\stanfordSentimentTreebank\stanfordSentimentTreebank\datasetSentences.txt'
splitPath = 'F:\stanfordSentimentTreebank\stanfordSentimentTreebank\datasetSplit.txt'


def get_phrase_sentiments(base_directory):
    def group_labels(label):
        if label in ["very negative", "negative"]:
            return "negative"
        elif label in ["positive", "very positive"]:
            return "positive"
        else:
            return "neutral"

    dictionary = pandas.read_csv(dictPath, sep="|")
    dictionary.columns = ["phrase", "id"]
    dictionary = dictionary.set_index("id")

    sentiment_labels = pandas.read_csv(labelPath, sep="|")
    sentiment_labels.columns = ["id", "sentiment"]
    sentiment_labels = sentiment_labels.set_index("id")

    phrase_sentiments = dictionary.join(sentiment_labels)

    phrase_sentiments["fine"] = pandas.cut(phrase_sentiments.sentiment, [0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                           include_lowest=True,
                                           labels=["very negative", "negative", "neutral", "positive", "very positive"])
    phrase_sentiments["coarse"] = phrase_sentiments.fine.apply(group_labels)
    return phrase_sentiments


def get_sentence_partitions(base_directory):
    sentences = pandas.read_csv(sentPath, index_col="sentence_index",
                                sep="\t")
    splits = pandas.read_csv(splitPath, index_col="sentence_index")
    return sentences.join(splits).set_index("sentence")


def partition(base_directory):
    phrase_sentiments = get_phrase_sentiments(base_directory)
    sentence_partitions = get_sentence_partitions(base_directory)
    # noinspection PyUnresolvedReferences
    data = phrase_sentiments.join(sentence_partitions, on="phrase")
    data["splitset_label"] = data["splitset_label"].fillna(1).astype(int)
    data["phrase"] = data["phrase"].str.replace(r"\s('s|'d|'re|'ll|'m|'ve|n't)\b", lambda m: m.group(1))
    return data.groupby("splitset_label")


base_directory = 'G:'
output_directory = 'G:'
os.makedirs(output_directory, exist_ok=True)
for splitset, partition in partition(base_directory):
    split_name = {1: "train", 2: "test", 3: "dev"}[splitset]
    filename = os.path.join(output_directory, "stanford-sentiment-treebank.%s.csv" % split_name)
    del partition["splitset_label"]
    partition.to_csv(filename)
