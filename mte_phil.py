import json
import numpy as np
import re
import unicodedata
import os
import math
import matplotlib.pyplot as plt
import random
from random import shuffle


CARDS_FILE_PATH = "rawData\oracle-cards-20230215100208.json"
MODEL_WEIGHTS_FILE_NAME = "mtg_price_predictor_weights.keras"


#************************************************#
#                                                #
#                Cleaning Functions              #
#                                                #
#************************************************#


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
    # Initialize a 0 array
    encoding_array = np.zeros((number_of_value_categories,), dtype=int)
    if value is not None:
        # If the value is under the specified limit, use that, otherwise use the maximum value.
        value = value if value < value_limit else number_of_value_categories-1 
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
            card["cmc"] = encode_at_limit(int(raw_card["cmc"]), 10, 10)
            card_power = raw_card.get("power", None)
            card["power"] = encode_at_limit(int(card_power) if card_power is not None else None, 10, 10)
            card_toughness = raw_card.get("toughness", None)
            card["toughness"] = encode_at_limit(int(card_toughness) if card_toughness is not None else None, 10, 10)
            card["usd"] = float(raw_card["prices"]["usd"])
            cards.append(card)
        except Exception as e:
            continue
        
    file.close()
    return cards
    

#************************************************#
#                                                #
#                Model Function                  #
#                                                #
#************************************************#

from tensorflow import keras
from keras.layers import Input, LSTM, concatenate, Dense, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Reshape
from keras.models import Model
from keras.metrics import RootMeanSquaredError
from keras import callbacks

def create_model(vocab_size, text_shape, metadata_shape):
    #--------------------#
    ## Text Data Layers ##
    #--------------------#
    text_input = Input(shape=text_shape)

    # Embedding Layer
    text_embed = keras.layers.Embedding(input_dim=vocab_size, output_dim=100, input_length=text_shape)(text_input)
        
    # 1D Convolutional Layer for Text
    text_conv = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(text_embed)
    text_pool = MaxPooling1D(pool_size=2)(text_conv)
    
    # LSTM For text 
    text_lstm = LSTM(64, dropout=0.2)(text_pool)

    #--------------------#
    ## Meta Data Layers ##
    #--------------------#
    metadata_input = Input(shape=metadata_shape)

    # Reshape the metadata input to have a shape of (batch_size, height, width, channels)
    metadata_reshape = Reshape(target_shape=(4, 10, 1))(metadata_input)

    # 2D Convolutional Layer for Metadata
    metadata_conv = Conv2D(filters=64, kernel_size=(2,2), padding='same', activation='relu')(metadata_reshape)

    # Meta Data Pool
    metadata_pool = MaxPooling2D(pool_size=(2, 2))(metadata_conv)

    metadata_flatten = keras.layers.Flatten()(metadata_pool)
    
    #--------------------#
    ##  Concat Layers   ##
    #--------------------#
    # Concatenate the LSTM layer and metadata layer 
    concat = concatenate([text_lstm, metadata_flatten])
    
    # Add a final output layer for regression
    output = Dense(1, activation='tanh')(concat)

    # Define the model
    model = Model(inputs=[text_input, metadata_input], outputs=output)

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse', 'mae', 'mape', RootMeanSquaredError()])
    
    return model


#************************************************#
#                                                #
#                Helper Functions                #
#                                                #
#************************************************#

# Expects a numpy array of each rules text as source.
def tokenize(source, tokenizer):
    source_copy = source.copy()
    tokenizer.fit_on_texts(source_copy)
    res = tokenizer.texts_to_sequences(source_copy)
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

def remove_field_outliers_from_dataset(labelset, dataset, range_min, range_max):
    print("Removing normalized outliers...")
    count = 0
    for i in range(len(labelset)-1, 0, -1):
        if labelset[i] > range_max or labelset[i] < range_min:
            labelset = np.delete(labelset, i)
            del dataset[i]
            count+=1
    print("Deleted: " + str(count))
    print("New Size: " + str(len(dataset)))
    return (labelset, dataset)


def extract_multiple_values(source, fields):
    print("Beginning to shape text and metadata.")
    values = []
    for field in fields:
        values.append(extract_value(source, field))
    return values 
    
 
def rmse(actual, predicted):
    return math.sqrt(np.square(np.subtract(actual, predicted)).mean())

def mae(actual, predicted):
    return np.mean(np.abs(actual - predicted))

#************************************************#
#                                                #
#                Execution Function              #
#                                                #
#************************************************#

def run_mte():
    print("Loading and cleaning cards...")
    cards = load_cards(CARDS_FILE_PATH)
    random.seed(2828)
    shuffle(cards)
    print(cards[0])
    # Prepare Data
    print("Preparing input data...")
    ## Prepare Labels
    (normalized_labels, usd_mean, usd_std) = normalize_values(extract_value(cards, "usd"))
    (normalized_labels, reduced_cards) = remove_field_outliers_from_dataset(normalized_labels, cards, -1, 1)

    ## Prepare Rules
    # KNOWN VOCAB MAX SIZE (21,077)
    vocab_size = 10000
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=vocab_size)
    tokenized_rules = tokenize(extract_value(reduced_cards, "rules"), tokenizer)

    ## Prepare Meta Data
    meta_data_values = extract_multiple_values(reduced_cards, ["type", "cmc", "power", "toughness"])
    tokenized_metadata = np.stack((meta_data_values[0], meta_data_values[1], meta_data_values[2], meta_data_values[3]), axis=1)

    # Generate Test Split
    split_index = (int(len(reduced_cards)*0.8))
    
    ## Split Data
    (reduced_cards_train, reduced_cards_test) = reduced_cards[:split_index], reduced_cards[split_index+1:]
    (train_text, test_text) = tokenized_rules[:split_index], tokenized_rules[split_index+1:]
    (train_metadata, test_metadata) = tokenized_metadata[:split_index], tokenized_metadata[split_index+1:]
    (train_labels, test_labels) = normalized_labels[:split_index], normalized_labels[split_index+1:]

    # Create Model
    print("Creating model...")
    model = create_model(vocab_size, max(train_text, key=len).shape, train_metadata[0].shape)

    # Load Model if it exists
    if os.path.exists(MODEL_WEIGHTS_FILE_NAME):
        print("Loading presaved weights...")
        model.load_weights(MODEL_WEIGHTS_FILE_NAME)
    else:
        call_backs = [callbacks.ModelCheckpoint(filepath=MODEL_WEIGHTS_FILE_NAME, save_best_only=True)]
        model_history = model.fit([train_text, train_metadata], train_labels, epochs=20, callbacks=call_backs, validation_split=0.2, batch_size=32)

    #Run predictions
    predictions = model.predict([test_text, test_metadata])


    # Display Results
    deNormalized_labels = deNormalize_values(test_labels, usd_mean, usd_std)
    deNormalized_predictions = deNormalize_values(predictions, usd_mean, usd_std)

    print("After denormalization")
    print("Test Labels")
    print(deNormalized_labels)
    print("Predictions")
    print(deNormalized_predictions)

    for i in range(0, 10):
        print(("Card Name: " + reduced_cards_test[i]["name"]).ljust(45) + (" Predicted: " + str(deNormalized_predictions[i])).ljust(30) + (" Actual: " + str(reduced_cards_test[i]["usd"])).ljust(20))

    print("RMSE")
    print(rmse(deNormalized_labels, deNormalized_predictions))
    print("MAE")
    print(mae(deNormalized_labels, deNormalized_predictions))
    
run_mte()