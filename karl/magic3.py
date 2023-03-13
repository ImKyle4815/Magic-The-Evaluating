import json
from random import random

import numpy as np
import tensorflow
from tensorflow import keras
from keras import backend as K
import matplotlib.pyplot as plt
import pandas as pd



def get_rules_text(card):
    cardName = card["name"]
    rules_text = card["oracle_text"]
    rules_text = rules_text.replace(cardName, "cardname")
    rules_text = rules_text.replace(".", " . ")
    rules_text = rules_text.replace("\n", " \n ")
    # rules_text = rules_text.sub(r'\s*\([^\(\)]*\)\s*', ' ', rules_text)
    return rules_text.lower()


def categorize_price(price, cutoffs):
    num_categories = len(cutoffs)
    categorization = [0] * num_categories
    for i in range(num_categories):
        if price < cutoffs[i]:
            categorization[i] = 1
            return categorization
    raise Exception("Unable to categorize price.")


def get_type(card):
    type_line = card["type_line"]
    if "Creature" in type_line:
        return 0
    elif "Instant" in type_line:
        return 0.2
    elif "Sorcery" in type_line:
        return 0.4
    elif "Enchantment" in type_line:
        return 0.6
    elif "Artifact" in type_line:
        return 0.8
    elif "Land" in type_line:
        return 1
    else:
        raise Exception("Ignoring type: " + type_line)


def get_colors(card):
    c = [0] * 5
    if "W" in card["colors"]:
        c[0] = 1
    if "U" in card["colors"]:
        c[1] = 1
    if "B" in card["colors"]:
        c[2] = 1
    if "R" in card["colors"]:
        c[3] = 1
    if "G" in card["colors"]:
        c[4] = 1
    return c


def categorize_cmc(cmc):
    if cmc < 10:
        return cmc / 10
    else:
        return 1


def parse_raw_cards(filepath):
    file = open(filepath)
    raw_cards = json.load(file)
    all_rules_texts = []
    all_metadata = []
    all_prices = []
    for raw_card in raw_cards:
        try:
            # Apply Filters
            if int(raw_card["released_at"][0:4]) < 2004:
                continue
            # Strings to be tokenized
            card_rules_text = get_rules_text(raw_card)
            # Concatenated Metadata
            card_type = get_type(raw_card)
            cmc = categorize_cmc(int(raw_card["cmc"]))
            colors = get_colors(raw_card)
            power = 0 if "power" not in raw_card else int(raw_card["power"])
            toughness = 0 if "toughness" not in raw_card else int(raw_card["toughness"])
            num_abilities = card_rules_text.count('/n')
            card_metadata = [card_type, cmc, num_abilities] + colors + [power, toughness]
            # Evaluation labels
            card_price = float(raw_card["prices"]["usd"])
            # Add the values to the lists
            all_rules_texts.append(card_rules_text)
            all_metadata.append(card_metadata)
            all_prices.append(card_price)
        except:
            continue
    return np.array(all_rules_texts), np.array(all_metadata), np.array(all_prices)


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def normalizeValues(x, std, mean):
    x = x.astype("float64")
    x -= mean.astype("float64")
    x /= std.astype("float64")
    return x


def unnormalizeValues(x, std, mean):
    return x * std + mean


def extract_tokenized(source, tokenizer):
    tokenizer.fit_on_texts(source)
    res = tokenizer.texts_to_sequences(source)
    res = keras.preprocessing.sequence.pad_sequences(res)
    return res


def train(text, vocab_size, metadata, prices, num_epochs = 10):
    textShape = len(max(text, key=len))
    metadataShape = metadata.shape[1]
    textInputs = keras.Input(shape=(textShape,))

    embeddedText = keras.layers.Embedding(input_dim=vocab_size, output_dim=128, input_length=textShape)(textInputs)
    convLayer1 = keras.layers.Conv1D(32, 8, activation='relu')(embeddedText)
    maxPoolingLayer1 = keras.layers.MaxPooling1D(2)(convLayer1)
    convLayer2 = keras.layers.Conv1D(16, 8, activation='relu')(maxPoolingLayer1)
    maxPoolingLayer2 = keras.layers.MaxPooling1D(2)(convLayer2)

    flattenedText = keras.layers.Flatten()(maxPoolingLayer2)

    metadataInputs = keras.Input(shape=(metadataShape,))
    # concatenate text with numeric metadata
    concatenatedLayers = keras.layers.concatenate([flattenedText, metadataInputs])
    # final processing
    denseLayer1 = keras.layers.Dense(64, activation='relu')(concatenatedLayers)

    denseLayer2 = keras.layers.Dense(32, activation='relu')(denseLayer1)

    # output layer
    outputLayer = keras.layers.Dense(1)(denseLayer2)
    # construct the model
    model = keras.models.Model(inputs=[textInputs, metadataInputs], outputs=outputLayer)
    # COMPILING THE MODEL
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=[rmse])
    # TRAINING THE MODEL


    text_train, text_test = text[:20000], text[:20000]
    metadata_train, metadata_test = metadata[:20000], metadata[:20000]
    prices_train, prices_test = prices[:20000], prices[:20000]

    model.fit([text_train, metadata_train], prices_train, batch_size=32, epochs=num_epochs, validation_split=0.2)

    test_loss = model.evaluate([text_test, metadata_test], prices_test, batch_size=32)
    print(test_loss)


(rules_texts, metadata, prices) = parse_raw_cards("../dataset/oracle-cards.json")
print(prices)
prices_std = prices.std(axis=0)
prices_mean = prices.mean(axis=0)
prices = normalizeValues(prices, prices_std, prices_mean)
print(prices)

vocab_size = 2048
tokenizer = keras.preprocessing.text.Tokenizer(num_words=vocab_size)
tokenized_rules_texts = extract_tokenized(rules_texts, tokenizer)

train(tokenized_rules_texts, vocab_size, metadata, prices, 15)

