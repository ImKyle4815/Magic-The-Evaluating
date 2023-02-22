import json
import re
import unicodedata
import numpy as np
import tensorflow
from tensorflow import keras



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
    return rules_text.replace(".", " . ")


def getRulesText(card):
    card_name = card["name"]
    rules_text = card["oracle_text"]
    rules_text = replace_cardname(card_name, rules_text)
    rules_text = remove_punctuation(rules_text)
    rules_text = remove_parenthesized_text(rules_text)
    rules_text = replace_accented_letters(rules_text)
    return rules_text.lower()

def categorizePrice(price, cutoffs):
    numCategories = len(cutoffs)
    categorization = [0] * numCategories
    for i in range(numCategories):
        if price < cutoffs[i]:
            categorization[i] = 1
            return categorization
    raise Exception("Unable to categorize price.")

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


priceCutoffs = [0.1, 0.25, 0.5, 1, 10, 100]
file = open("../dataset/oracle-cards.json")
raw_cards = json.load(file)
cards = []
for raw_card in raw_cards:
    try:
        # Create a blank card
        card = {}
        # Required fields (strings)
        card["rules"] = getRulesText(raw_card)
        card["type"] = categorizeType(raw_card["type_line"])
        # Required fields (nums)
        card["cmc"] = categorizeCMC(int(raw_card["cmc"]))
        # Optional fields
        card["power"] = 0 if "power" not in raw_card else int(raw_card["power"])
        card["toughness"] = 0 if "toughness" not in raw_card else int(raw_card["toughness"])
        # Evaluation labels
        card["usd"] = categorizePrice(float(raw_card["prices"]["usd"]), priceCutoffs)
        # Add the card to the list
        cards.append(card)
    except:
        continue


########################################################################################################################
########################################################################################################################
########################################################################################################################


def printPriceCategorizationStats(cardList, cutoffs):
    numCards = len(cardList)
    print("Num Cards:", numCards)
    for i in range(len(cutoffs)):
        print("Under ${:2.2f}: {:2.2%}".format(cutoffs[i], sum(c["usd"][i] == 1 for c in cardList) / numCards))


printPriceCategorizationStats(cards, priceCutoffs)


########################################################################################################################
########################################################################################################################
########################################################################################################################


def train(text, vocabSize, metadata, prices, numOutputCategories, numEpochs = 10):
    # PREPARING VALUES
    textShape = len(max(text, key=len))
    metadataShape = metadata.shape[1]

    # BUILDING THE MODEL
    # text inputs
    textInputs = keras.Input(shape=(textShape,))
    # text processing layers
    embeddedText = keras.layers.Embedding(input_dim=vocabSize, output_dim=128, input_length=textShape)(textInputs)
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
    outputLayer = keras.layers.Dense(numOutputCategories, activation='softmax')(denseLayer1)
    # construct the model
    model = keras.models.Model(inputs=[textInputs, metadataInputs], outputs=outputLayer)

    # COMPILING THE MODEL
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    # TRAINING THE MODEL
    model.fit([rules, metadata], prices, epochs=numEpochs, validation_split=0.2)


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
vocab_size = 2048
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

train(rules, vocab_size, metadata, usd, len(priceCutoffs), 10)