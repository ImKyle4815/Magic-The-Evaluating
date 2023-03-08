import json
import random
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras import backend as K
import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
        card["rarity"] = raw_card["rarity"]
        card["rank"] = float(raw_card["edhrec_rank"])
        card["usd"] = float(raw_card["prices"]["usd"])
        cards.append(card)
    except:
        continue
random.shuffle(cards)


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def tokenize(data, field, tokenizer):
    try:
        res = [c[field] for c in data]
    except:
        res = [data[field]]
    res = np.array(res)
    tokenizer.fit_on_texts(res)
    res = tokenizer.texts_to_matrix(res)
    return res


def extractValue(source, field):
    try:
        res = [item[field] for item in source]
    except:
        res = [source[field]]
    res = np.array(res)
    return res


def maxLengthString(array):
    return len(max(array, key=len))


def normalizeValues(x, std, mean):
    x = x.astype("float64")
    x -= mean.astype("float64")
    x /= std.astype("float64")
    return x


def unnormalizeValues(x, std, mean):
    return x * std + mean


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['rmse'], label='emse')
    plt.ylim([0, 1])
    plt.xlabel('Epoch')
    plt.ylabel('Error (Price)')
    plt.legend()
    plt.grid(True)
    plt.show()
    print("PLOTTING")


# PROCESS DATA
t = Tokenizer(num_words=100, lower=1, oov_token="<OOV>")
names = tokenize(cards, "name", t)
rules = tokenize(cards, "rules", t)
costs = tokenize(cards, "cost", t)
types = tokenize(cards, "type", t)
# rarity = tokenize(cards, "rarity", t)

# rank = extractValue(cards, "rank")
# rank_std = rank.std(axis=0)
# rank_mean = rank.mean(axis=0)
# rank = normalizeValues(rank, rank_std, rank_mean)
# rank.resize([22482, 100])

prices = extractValue(cards, "usd")
prices_std = prices.std(axis=0)
prices_mean = prices.mean(axis=0)
prices = normalizeValues(prices, prices_std, prices_mean)

input_tensor = tf.concat([names, rules, costs, types], axis=1)
max_length = maxLengthString(input_tensor)

## TRAIN MODEL

model = keras.Sequential([
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=[rmse])

x_train, x_test = input_tensor[:20000], input_tensor[20000:]
y_train, y_test = prices[:20000], prices[20000:]

history = model.fit(x_train, y_train, epochs=25, batch_size=32, validation_split=0.1)
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()
plot_loss(history)

test_loss = model.evaluate(x_test, y_test, batch_size=32)
print(test_loss)

# INDIVIDUAL CARD TESTING


new_card = {"name": "Sinew Sliver", "rules": "All Sliver creatures get +1/+1.", "cost": "{1}{W}",
            "type": "Creature — Sliver", "rarity": "common", "rank": 4367}

name = tokenize(new_card, "name", t)
rule = tokenize(new_card, "rules", t)
cost = tokenize(new_card, "cost", t)
type = tokenize(new_card, "type", t)
# rarity = tokenize(new_card, "rarity", t)
#
# rank = normalizeValues(extractValue(new_card, "rank"), rank_std, rank_mean)
# rank.resize([1, 100])
#
new_input_data = tf.concat([name, rule, cost, type], axis=1)

new_price = model.predict(new_input_data)[0][0]

print("Predicted price:", unnormalizeValues(new_price, prices_std, prices_mean))
