import json
import math

import numpy as np

def transformPrice(price):
    # return float(math.ceil(price))
    return min(math.log10(price + 1) / 2, 1)

file = open("../dataset/oracle-cards.json")
raw_cards = json.load(file)
cards = []
for raw_card in raw_cards:
    try:
        # Create a blank card
        card = {}
        # Required fields (strings)
        card["name"] = raw_card["name"]
        card["rules"] = raw_card["oracle_text"].replace(card["name"], "CARDNAME")
        card["cost"] = raw_card["mana_cost"].replace("}{", " ").replace("/", "")[1:-2]
        card["type"] = raw_card["type_line"]
        # Required fields (nums)
        card["cmc"] = int(raw_card["cmc"])
        # Optional fields
        card["power"] = 0 if "power" not in raw_card else int(raw_card["power"])
        card["toughness"] = 0 if "toughness" not in raw_card else int(raw_card["toughness"])
        card["loyalty"] = 0 if "loyalty" not in raw_card else int(raw_card["loyalty"])
        card["cmc"] = int(raw_card["cmc"])
        # Evaluation labels
        card["rank"] = float(raw_card["edhrec_rank"])
        card["usd"] = transformPrice(float(raw_card["prices"]["usd"]))
        # print(card["usd"])
        # Add the card to the list
        cards.append(card)
    except:
        continue
# exit(0)


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
print("Beginning to tokenize the text")
rules = extractTokenized(cards, "rules", tokenizer)
# names = extractTokenized(cards, "name", tokenizer)
# names = np.pad(names, (0, rules.shape[1] - names.shape[1]))
costs = extractTokenized(cards, "cost", tokenizer)
# costs = np.pad(costs, (0, rules.shape[1] - names.shape[1]))
types = extractTokenized(cards, "type", tokenizer)
# types = np.pad(types, (0, rules.shape[1] - names.shape[1]))
print("Finished tokenizing the text")
# print("Shapes are: ", rules.shape, names.shape, costs.shape, types.shape)
input_tensor = np.concatenate([costs, types, rules], axis=-1)
# input_tensor = np.stack((names, costs, types, rules))
print("Final Shape: ", input_tensor.shape)
print(input_tensor[0])
# Normalize training values
# ranks = normalizeValues(extractValue(cards, "rank"))
usd = extractValue(cards, "usd")
# usd = normalizeValues(extractValue(cards, "usd"))
# Get the max string length
max_length = maxLengthString(input_tensor)

# Building the model
model = keras.Sequential([
    keras.layers.Embedding(input_dim=vocab_size, output_dim=64, input_length=max_length),
    keras.layers.Conv1D(128, 32, 8, activation='relu'),
    # keras.layers.MaxPooling1D(2),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compiling the model
model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])

# Training the model
model.fit(input_tensor, usd, epochs=10)
