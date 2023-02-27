import json
import numpy as np
import re
import unicodedata
import os

CARDS_FILE_PATH = "rawData\oracle-cards-20230215100208.json"
MODEL_WEIGHTS_FILE_NAME = "mtg_price_predictor_weights.h5"

def replace_cardname(card_name, rules_text):
    if card_name in rules_text:
        rules_text.replace(card_name, "cardname")
    return rules_text

def remove_parenthesized_text(rules_text):
    return re.sub(r'\s*\([^\(\)]*\)\s*', '', rules_text)

def remove_punctuation(rules_text):
    rules_text = re.sub(r'(?<!\s):', ' :', rules_text)
    rules_text = re.sub(r'[^\w\s/+—:-]', '', rules_text)
    rules_text = re.sub(r' {2,}', ' ', rules_text)
    return rules_text
            
def replace_accented_letters(rule_text):
    rule_text = unicodedata.normalize('NFKD', rule_text).encode('ascii', 'ignore').decode('utf-8')
    rule_text = re.sub(r'[^\x00-\x7F]+', '', rule_text)
    return rule_text

def space_newlines(rules_text):
    return re.sub(r'\n', ' \n ', rules_text)

            
def cleanRulesText(card_name, rules_text):
    rules_text = replace_cardname(card_name, rules_text)
    rules_text = remove_parenthesized_text(rules_text)
    rules_text = remove_punctuation(rules_text)
    rules_text = replace_accented_letters(rules_text)
    rules_text = space_newlines(rules_text)
    return rules_text.lower()

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
        return 1

def load_cards(path):
    file = open(path, 'r', encoding='utf-8')
    raw_cards = json.load(file)

    cards = []
    for raw_card in raw_cards:
        try:
            # Create a blank card
            card = {}
            # Required fields (strings)
            card["name"] = raw_card["name"]
            card["rules"] = cleanRulesText(card["name"], raw_card["oracle_text"])
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
            card["usd"] = float(raw_card["prices"]["usd"])
            # print(card["usd"])
            cards.append(card)
        except Exception as e:
            #print(e)
            continue
    file.close()
    return cards
    
########################################################################################################################
########################################################################################################################
########################################################################################################################
from tensorflow import keras
from keras.layers import Input, LSTM, concatenate, Dense
from keras.models import Model
from keras.metrics import RootMeanSquaredError


def create_model(vocab_size, text_shape):
    # Input Layer
    text_input = keras.Input(shape=(text_shape,))
    # Embedding Layer
    text_embed = keras.layers.Embedding(input_dim=vocab_size, output_dim=100, input_length=text_shape)(text_input)
    
    text_lstm = LSTM(64, dropout=0.2)(text_embed)

    # Define the linear regression architecture for metadata
    metadata_input = Input(shape=(4,))
    metadata_dense = Dense(4, activation='relu')(metadata_input)

    # Concatenate the LSTM layer and metadata layer 
    concat = concatenate([text_lstm, metadata_dense])
    
    # Add a final output layer for regression
    output = Dense(1, activation='linear')(concat)

    # Define the model
    model = Model(inputs=[text_input, metadata_input], outputs=output)

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse', 'mae', 'mape', RootMeanSquaredError()])
    # TRAINING THE MODEL
    
    return model

########################################################################################################################
########################################################################################################################
########################################################################################################################
# Use RMS to measure accuracy


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

def normalizeValues(num_set):
    ns_mean = num_set.mean(axis=0)
    ns_std = num_set.std(axis=0) 
    num_set = (num_set - ns_mean)/ns_std
    return (num_set, ns_mean, ns_std)

def deNormalizeValues(num_set, num_set_mean, num_set_std):
    return (num_set * num_set_std) + num_set_mean


def remove_outliers(labelset, dataset,range_max, range_min):
    print("Removing normalized outliers...")
    count = 0
    for i in range(len(labelset)-1, 0, -1):
        if labelset[i] > range_max or labelset[i] < range_min:
            labelset = np.delete(labelset, i)
            del dataset[i]
            count+=1
    print("Deleted: " + str(count))
    print("New Size: " + str(len(cards)))
    return (labelset, dataset)


def tokenize_input_data(cards, tokenizer):
    print("Beginning to shape text and metadata.")

    # Normalize and Remove any cards that are outliers.
    usd = normalizeValues(extractValue(cards, "usd"))[0]
    (usd, cards) = remove_outliers(usd, cards,  1, -1)    
    
    rules = extractTokenized(cards, "rules", tokenizer)
    
    # Create Meta Data
    types = extractValue(cards, "type")
    cmc = extractValue(cards, "cmc")
    power = (extractValue(cards, "power"))
    toughness = (extractValue(cards, "toughness"))
    metadata = np.stack((cmc, power, toughness, types), axis=-1)
    
    print("Text data shape: " + str(rules.shape))
    print("Text data shape: " + str(metadata.shape))
    
    print("Finished shaping text and metadata.")
    # (Text Data, Meta Data, Labels)
    return (rules, metadata, usd)

###################################################################################################################

# KNOWN VOCAB MAX SIZE (21,077)
vocab_size = 15000
tokenizer = keras.preprocessing.text.Tokenizer(num_words=vocab_size)

print("Loading and cleaning cards...")
cards = load_cards(CARDS_FILE_PATH)


# (Text Data, Meta Data, Labels)
print("Tokenizing input data...")
input_data = tokenize_input_data(cards.copy(), tokenizer)

# Test split 80%
split_index = (int(len(input_data[0])*0.8))

(train_text, test_text) = input_data[0][:split_index], input_data[0][split_index:]
(train_metadata, test_metadata) = input_data[1][:split_index], input_data[1][split_index:]
(train_labels, test_labels) = input_data[2][:split_index], input_data[2][split_index:]

#Create Model
print("Creating model...")
model = create_model(vocab_size, len(max(train_text, key=len)))

if os.path.exists(MODEL_WEIGHTS_FILE_NAME):
    print("Loading presaved weights...")
    model = model.load_weights(MODEL_WEIGHTS_FILE_NAME)
else:
    model_history = model.fit([train_text, train_metadata], train_labels, epochs=10, validation_split=0.2, batch_size=32)
    model.save_weights(MODEL_WEIGHTS_FILE_NAME)


#Normalize the labels for comparison
(usd_norm, usd_mean, usd_std) = normalizeValues(extractValue(cards, "usd"))

#Run predictions
predictions = model.predict([test_text, test_metadata])

#Show Results
print("Before denormalization")
print(predictions)

deNormalized_values = deNormalizeValues(predictions, usd_mean, usd_std)

print("After denormalization")
print(deNormalized_values)
print(deNormalized_values[0])
print((cards[split_index:])[0])