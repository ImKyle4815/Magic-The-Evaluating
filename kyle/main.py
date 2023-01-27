import json

import numpy as np

file = open("../dataset/oracle-cards.json")
raw_cards = json.load(file)
cards = []
for raw_card in raw_cards:
    if not ("name" in raw_card and "edhrec_rank" in raw_card and "oracle_text" in raw_card and "mana_cost" in raw_card and "type_line" in raw_card):
        continue
    else:
        card = {}
        card["name"] = raw_card["name"]
        card["rules"] = raw_card["oracle_text"]
        card["cost"] = raw_card["mana_cost"]
        card["type"] = raw_card["type_line"]
        card["rank"] = raw_card["edhrec_rank"]
        cards.append(card)

cards = cards[:10000]

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


def maxLengthString(array):
    return len(max(array, key=len))


# Extract arrays
vocab_size = 10000
tokenizer = keras.preprocessing.text.Tokenizer(num_words=vocab_size)
names = extractTokenized(cards, "name", tokenizer)
rules = extractTokenized(cards, "rules", tokenizer)
costs = extractTokenized(cards, "cost", tokenizer)
types = extractValue(cards, "type")
input_tensor = keras.layers.concatenate([names,costs,types,rules], axis=-1)
max_length = maxLengthString(input_tensor)

# Building the model
model = keras.Sequential([
    keras.layers.Embedding(input_dim=vocab_size, output_dim=64, input_length=max_length),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compiling the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the model
model.fit(input_tensor, ranks, epochs=10)
