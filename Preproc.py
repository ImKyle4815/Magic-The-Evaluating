import json
import numpy as np
import tensorflow
from tensorflow import keras
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
        card["rules"] = cleanRulesText(raw_card["oracle_text"])
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


########################################################################################################################
########################################################################################################################
########################################################################################################################




########################################################################################################################
########################################################################################################################
########################################################################################################################


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


# Extract arrays
vocab_size = 1048
tokenizer = keras.preprocessing.text.Tokenizer(num_words=vocab_size)
# Tokenize the text
print("Beginning to tokenize the text")
rules = extractTokenized(cards, "rules", tokenizer)
types = extractValue(cards, "type")
cmc = extractValue(cards, "cmc")
power = (extractValue(cards, "power"))
toughness = (extractValue(cards, "toughness"))
metadata = np.stack((cmc, power, toughness, types), axis=-1)
print("Finished tokenizing the text.")
# Prepare training values (labels)
usd = extractValue(cards, "usd")

train(rules, vocab_size, metadata, usd, 10)