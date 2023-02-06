import numpy as np
import json
import os

file = open("rawData\oracle-cards-2023_01_17.json", 'r', encoding='utf-8')
raw_cards = json.load(file)
cards = []
for raw_card in raw_cards["cardlist"]:
    try:
        card = {}
        card["name"] = raw_card["name"]
        card["rules"] = raw_card["oracle_text"]
        card["cost"] = raw_card["mana_cost"]
        card["type"] = raw_card["type_line"]
        card["rank"] = float(raw_card["edhrec_rank"])
        card["usd"] = float(raw_card["prices"]["usd"])
        cards.append(card)
    except:
        continue


print(cards[0])
file.close()
################################################################
################################################################
################################################################


# Tensorflow Stuff
import tensorflow
from tensorflow import keras


def extractTokenized(source, field, tokenizer):
    res = [item[field] for item in source]
    res = np.array(res)
    tokenizer.fit_on_texts(res)
    res = tokenizer.texts_to_sequences(res)
    res = keras.preprocessing.sequence.pad_sequences(res)
    return res


def extractValue(source, field):
    res = [item[field] for item in source]
    res = np.array(res)
    return res

def normalizeValues(x):
    x -= x.mean(axis=0)
    x /= x.std(axis=0)
    return x

def maxLengthString(array):
    return len(max(array, key=len))


# Extract arrays
vocab_size = 10000
tokenizer = keras.preprocessing.text.Tokenizer(num_words=vocab_size)
# Tokenize the text
names = extractTokenized(cards, "name", tokenizer)
costs = extractTokenized(cards, "cost", tokenizer)
types = extractTokenized(cards, "type", tokenizer)
rules = extractTokenized(cards, "rules", tokenizer)
print(rules[0])

print(names.shape, costs.shape, types.shape, rules.shape)
input_tensor = np.dstack((names, costs, types, rules))
print(input_tensor.shape)

# Normalize training values
# ranks = normalizeValues(extractValue(cards, "rank"))
usd = extractValue(cards, "usd")
# Get the max string length
max_length = maxLengthString(input_tensor)
