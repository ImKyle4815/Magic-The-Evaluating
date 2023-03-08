import json
import numpy as np
import re
import unicodedata
import os
import math
import matplotlib.pyplot as plt

CARDS_FILE_PATH = "rawData\oracle-cards-20230215100208.json"
MODEL_WEIGHTS_FILE_NAME = "mtg_price_predictor_weights.keras"

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

def get_type_as_int(type):
    if "Creature" in type:
        return 0
    elif "Instant" in type:
        return 1
    elif "Sorcery" in type:
        return 2
    elif "Enchantment" in type:
        return 3
    elif "Artifact" in type:
        return 4
    elif "Land" in type:
        return 5
    else:
        raise Exception("Ignoring type: " + type)

def encode_at_limit(value, value_limit, number_of_value_categories):
    encoding_array = np.zeros((number_of_value_categories,), dtype=int)
    if value:
        value = value if value < value_limit else 9 
        encoding_array[value] = 1
    return encoding_array

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
            card["type"] = encode_at_limit(get_type_as_int(raw_card["type_line"]), 10, 10)
            # Required fields (nums)
            card["cmc"] = encode_at_limit(int(raw_card["cmc"]), 10, 10)
            # Optional fields
            card["power"] = encode_at_limit(int(raw_card["power"]), 10, 10)
            card["toughness"] = encode_at_limit(int(raw_card["power"]), 10, 10)
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
from keras import callbacks

def create_model(vocab_size, text_shape, metadata_shape):
    # Input Layers
    text_input = Input(shape=text_shape)
    metadata_input = Input(shape=metadata_shape)

    # Embedding Layer
    text_embed = keras.layers.Embedding(input_dim=vocab_size, output_dim=100, input_length=text_shape)(text_input)
    
    text_lstm = LSTM(64, dropout=0.2)(text_embed)

    metadata_flatten = keras.layers.Flatten()(metadata_input)
    # Concatenate the LSTM layer and metadata layer 
    concat = concatenate([text_lstm, metadata_flatten])
    
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


#************************************************#
#                                                #
#                Helper Functions                #
#                                                #
#************************************************#

def extract_tokenized(source, field, tokenizer):
    res = [item[field] for item in source]
    res = np.array(res)
    tokenizer.fit_on_texts(res)
    res = tokenizer.texts_to_sequences(res)
    res = keras.preprocessing.sequence.pad_sequences(res)
    return res


def extract_value(source, field):
    res = [item[field] for item in source]
    res = np.array(res)
    return res

def normalize_values(num_set):
    ns_mean = num_set.mean(axis=0)
    ns_std = num_set.std(axis=0) 
    num_set = (num_set - ns_mean)/ns_std
    return (num_set, ns_mean, ns_std)

def deNormalize_values(num_set, num_set_mean, num_set_std):
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
    usd = normalize_values(extract_value(cards, "usd"))[0]
    (usd, cards) = remove_outliers(usd, cards,  1, -1)    
    
    rules = extract_tokenized(cards, "rules", tokenizer)
    
    # Create Meta Data
    types = extract_value(cards, "type")
    cmc = extract_value(cards, "cmc")
    power = extract_value(cards, "power")
    toughness = extract_value(cards, "toughness")
    metadata = np.stack((cmc, power, toughness, types), axis=-1)
    
    print("Text data shape: " + str(rules.shape))
    print("Meta data shape: " + str(metadata.shape))
    
    print("Finished shaping text and metadata.")
    # (Text Data, Meta Data, Labels)
    return (rules, metadata, usd)


def rmse(actual, predicted):
    return math.sqrt(np.square(np.subtract(actual, predicted)).mean())

# def plot_results(actual, predicted):
#     _, ax = plt.subplots()

#     ax.scatter(x = range(0, len(actual)), y=actual, c = 'blue', label = 'Actual', alpha = 0.3)
#     ax.scatter(x = range(0, len(predicted)), y=predicted, c = 'red', label = 'Predicted', alpha = 0.3)

#     plt.title('Actual and predicted values')
#     plt.xlabel('')
#     plt.ylabel('Price')
#     plt.legend()
#     plt.show()
    

###################################################################################################################


#************************************************#
#                                                #
#                Execution Code                  #
#                                                #
#************************************************#

# KNOWN VOCAB MAX SIZE (21,077)
vocab_size = 15000
tokenizer = keras.preprocessing.text.Tokenizer(num_words=vocab_size)

print("Loading and cleaning cards...")
cards = load_cards(CARDS_FILE_PATH)

# (Text Data, Meta Data, Labels)
print("Tokenizing input data...")
model_data = tokenize_input_data(cards.copy(), tokenizer)

# Test split 80%
split_index = (int(len(model_data[0])*0.8))

(train_text, test_text) = model_data[0][:split_index], model_data[0][split_index:]
(train_metadata, test_metadata) = model_data[1][:split_index], model_data[1][split_index:]
(train_labels, test_labels) = model_data[2][:split_index], model_data[2][split_index:]

#Create Model
print("Creating model...")
model = create_model(vocab_size, max(train_text, key=len).shape, train_metadata[0].shape)

if os.path.exists(MODEL_WEIGHTS_FILE_NAME):
    print("Loading presaved weights...")
    model = model.load_weights(MODEL_WEIGHTS_FILE_NAME)
else:
    call_backs = [callbacks.ModelCheckpoint(filepath=MODEL_WEIGHTS_FILE_NAME, save_best_only=True)]
    model_history = model.fit([train_text, train_metadata], train_labels, epochs=10, callbacks=call_backs, validation_split=0.2, batch_size=32)
    model.save_weights(MODEL_WEIGHTS_FILE_NAME)


#Normalize the labels for comparison
(_, usd_mean, usd_std) = normalize_values(extract_value(cards, "usd"))

#Run predictions
predictions = model.predict([test_text, test_metadata])

#Show Results
print("Before denormalization")
print(predictions)

deNormalized_predictions = deNormalize_values(predictions, usd_mean, usd_std)
deNormalized_labels = deNormalize_values(test_labels, usd_mean, usd_std)

print("After denormalization")
print(deNormalized_predictions)
test_card_split = cards[split_index:]
print(test_card_split[0]["name"] + ": " + "Predicted: " + str(deNormalized_predictions[0]) +  " Actual: " + str(deNormalized_labels[0]))
print(test_card_split[1]["name"] + ": " + "Predicted: " + str(deNormalized_predictions[1]) +  " Actual: " + str(deNormalized_labels[1]))

print("RMSE")
print(rmse(deNormalized_labels, deNormalized_predictions))