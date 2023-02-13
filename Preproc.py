import numpy as np
import json
import os
import re
import unicodedata

file = open("rawData\oracle-cards-2023_01_17.json", 'r', encoding='utf-8')
raw_cards = json.load(file)


def replace_accented_letters(rule_text):
    rule_text = unicodedata.normalize('NFKD', rule_text).encode('ascii', 'ignore').decode('utf-8')
    rule_text = re.sub(r'[^\x00-\x7F]+', '', rule_text)
    return rule_text

def remove_parenthesized_text(rules_text):
    return re.sub(r'\s*\([^\(\)]*\)\s*', ' ', rules_text)

def replace_cardname(card_name, rules_text):
    if card_name in rules_text:
        rules_text.replace(card_name, "cardname")
    return rules_text

def remove_punctuation(rules_text):
    return re.sub(r'[^\w\s]', '', rules_text)
            
            
def cleanRulesText(card_name, rules_text):
    rules_text = replace_cardname(card_name, rules_text)
    rules_text = remove_punctuation(rules_text)
    rules_text = remove_parenthesized_text(rules_text)
    rules_text = replace_accented_letters(rules_text)
    return rules_text.lower()


cards = []
for raw_card in raw_cards:
    try:
        # Create a blank card
        card = {}
        # Required fields (strings)
        card["name"] = raw_card["name"]
        card["rules"] = cleanRulesText(raw_card["oracle_text"], card["name"])
        
        
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
