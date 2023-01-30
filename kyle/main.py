import json

import numpy as np

file = open("../dataset/oracle-cards.json")
raw_cards = json.load(file)
cards = []
for raw_card in raw_cards:
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
print(names.shape, costs.shape, types.shape, rules.shape)
input_tensor = np.concatenate([names, costs, types, rules], axis=-1)
# Normalize training values
# ranks = normalizeValues(extractValue(cards, "rank"))
usd = extractValue(cards, "usd")
# Get the max string length
max_length = maxLengthString(input_tensor)

# Building the model
model = keras.Sequential([
    keras.layers.Embedding(input_dim=vocab_size, output_dim=64, input_length=max_length),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='linear')
])

# Compiling the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Training the model
model.fit(input_tensor, usd, epochs=10)
