import json
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
import keras
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


def tokenize(data, field, tokenizer):
    try:
        res = [c[field] for c in data]
    except:
        res = [data[field]]
    res = np.array(res)
    tokenizer.fit_on_texts(res)
    res = tokenizer.texts_to_matrix(res)
    return res

#
# def vectorize_sequences(sequences, dimension=10000):
#     results = np.zeros((len(sequences), dimension))
#     for i, sequence in enumerate(sequences):
#         for j in sequence:
#             results[i, j] = 1
#     return results


def extractValue(source, field):
    res = [item[field] for item in source]
    res = np.array(res)
    return res


def maxLengthString(array):
    return len(max(array, key=len))


def normalizeValues(x):
    x -= x.mean(axis=0)
    x /= x.std(axis=0)
    return x


def unnormalizeValues(x, std, mean):
    return x * std + mean


t = Tokenizer(num_words=100, lower=1, oov_token="<OOV>")
names = tokenize(cards, "name", t)
rules = tokenize(cards, "rules", t)
costs = tokenize(cards, "cost", t)
types = tokenize(cards, "type", t)
prices = extractValue(cards, "usd")
std = prices.std(axis=0)
mean = prices.mean(axis=0)

prices = normalizeValues(prices)
input_tensor = tf.concat([names, rules, costs, types], axis=1)
max_length = maxLengthString(input_tensor)

model = keras.Sequential([
    keras.layers.Dense(units=64, activation='relu'),
    keras.layers.Dense(units=32, activation='relu'),
    keras.layers.Dense(units=1, activation="linear")
])

model.compile(optimizer="adam", loss="mse")

x_train, x_test = input_tensor[:20000], input_tensor[20000:]
y_train, y_test = prices[:20000], prices[20000:]

model.fit(x_train, y_train, epochs=15, batch_size=32)

test_loss = model.evaluate(x_test, y_test, batch_size=32)
print(test_loss)

new_card = {"name": "Tocasia's Welcome", "rules": "Whenever one or more creatures with mana value 3 or less enter the battlefield under your control, draw a card. This ability triggers only once each turn.", "cost": "{2}{W}", "type": "Enchantment",
            "rank": 8.0}

name = tokenize(new_card, "name", t)
rule = tokenize(new_card, "rules", t)
cost = tokenize(new_card, "cost", t)
type = tokenize(new_card, "type", t)
#
new_input_data = tf.concat([name, rule, cost, type], axis=1)

new_price = model.predict(new_input_data)[0][0]

print("Predicted price:", unnormalizeValues(new_price, std, mean), "\nunnormalized:", new_price)
