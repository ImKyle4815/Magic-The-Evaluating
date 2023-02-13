import json
import math

import numpy as np

def transformPrice(price):
    if price < 0.17:
        return 0
    else:
        return 1

def categorizeType(type):
    if "Creature" in type:
        return 0
    elif "Instant" in type:
        return 0.2
    elif "Sorcery" in type:
        return 0.4
    elif "Enchantment" in type:
        return 0.6
    elif "Artifact" in type:
        return 0.8
    elif "Land" in type:
        return 1
    else:
        raise Exception("Ignoring type: " + type)

def categorizeCMC(cmc):
    if cmc < 10:
        return cmc / 10
    else:
        return 10

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
        card["type"] = categorizeType(raw_card["type_line"])
        # Required fields (nums)
        card["cmc"] = categorizeCMC(int(raw_card["cmc"]))
        # Optional fields
        card["power"] = 0 if "power" not in raw_card else int(raw_card["power"])
        card["toughness"] = 0 if "toughness" not in raw_card else int(raw_card["toughness"])
        card["loyalty"] = 0 if "loyalty" not in raw_card else int(raw_card["loyalty"])
        # Evaluation labels
        card["rank"] = float(raw_card["edhrec_rank"])
        card["usd"] = transformPrice(float(raw_card["prices"]["usd"]))
        # print(card["usd"])
        # Add the card to the list
        if int(raw_card["released_at"][:4]) >= 2020:
            continue
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
vocab_size = 1048
tokenizer = keras.preprocessing.text.Tokenizer(num_words=vocab_size)
# Tokenize the text
print("Beginning to tokenize the text")
rules = extractTokenized(cards, "rules", tokenizer)
# costs = extractTokenized(cards, "cost", tokenizer)
types = extractValue(cards, "type")
cmc = extractValue(cards, "cmc")
power = (extractValue(cards, "power"))
toughness = (extractValue(cards, "toughness"))
metadata = np.stack((cmc, power, toughness, types), axis=-1)
print("Finished tokenizing the text.")
# print(input_tensor[0])
# Prepare training values (labels)
usd = extractValue(cards, "usd")
# Get the input shapes
textShape = maxLengthString(rules)
print(metadata.shape)
metadataShape = metadata.shape[1]


# Building the model

# text inputs
textInputs = keras.Input(shape=(textShape,))
# text processing layers
embeddedText = keras.layers.Embedding(input_dim=vocab_size, output_dim=128, input_length=textShape)(textInputs)
convLayer1 = keras.layers.Conv1D(128, 8, activation='relu')(embeddedText)
maxPoolingLayer1 = keras.layers.MaxPooling1D(2)(convLayer1)
convLayer2 = keras.layers.Conv1D(64, 8, activation='relu')(maxPoolingLayer1)
maxPoolingLayer2 = keras.layers.MaxPooling1D(2)(convLayer2)
flattenedText = keras.layers.Flatten()(maxPoolingLayer2)
# metadata inputs
metadataInputs = keras.Input(shape=(metadataShape,))
# concatenate text with numeric metadata
concatenatedLayers = keras.layers.concatenate([flattenedText, metadataInputs])
# final processing
denseLayer1 = keras.layers.Dense(16, activation='relu')(concatenatedLayers)
# output layer
outputLayer = keras.layers.Dense(1, activation='sigmoid')(denseLayer1)
# construct the model
model = keras.models.Model(inputs=[textInputs, metadataInputs], outputs=outputLayer)

# Compiling the model
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# Training the model
model.fit([rules, metadata], usd, epochs=10)
