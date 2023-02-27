import json
import numpy as np
import tensorflow
from tensorflow import keras


########################################################################################################################
########################################################################################################################
########################################################################################################################


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


def parse_raw_cards(filepath, price_cutoff_categories):
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
            type = get_type(raw_card)
            cmc = categorize_cmc(int(raw_card["cmc"]))
            colors = get_colors(raw_card)
            power = 0 if "power" not in raw_card else int(raw_card["power"])
            toughness = 0 if "toughness" not in raw_card else int(raw_card["toughness"])
            card_metadata = [type, cmc, power, toughness] + colors
            # Evaluation labels
            card_price = categorize_price(float(raw_card["prices"]["usd"]), price_cutoff_categories)
            # Add the values to the lists
            all_rules_texts.append(card_rules_text)
            all_metadata.append(card_metadata)
            all_prices.append(card_price)
        except:
            continue
    return np.array(all_rules_texts), np.array(all_metadata), np.array(all_prices)


########################################################################################################################
########################################################################################################################
########################################################################################################################


def train(text, vocab_size, metadata, prices, num_output_categories, num_epochs=10):
    # PREPARING VALUES
    textShape = len(max(text, key=len))
    metadataShape = metadata.shape[1]
    # BUILDING THE MODEL
    # text inputs
    textInputs = keras.Input(shape=(textShape,))
    # text processing layers
    embeddedText = keras.layers.Embedding(input_dim=vocab_size, output_dim=128, input_length=textShape)(textInputs)
    convLayer1 = keras.layers.Conv1D(256, 8, activation='relu')(embeddedText)
    maxPoolingLayer1 = keras.layers.MaxPooling1D(2)(convLayer1)
    convLayer2 = keras.layers.Conv1D(128, 8, activation='relu')(maxPoolingLayer1)
    maxPoolingLayer2 = keras.layers.MaxPooling1D(2)(convLayer2)
    convLayer3 = keras.layers.Conv1D(64, 8, activation='relu')(maxPoolingLayer2)
    maxPoolingLayer3 = keras.layers.MaxPooling1D(2)(convLayer3)
    flattenedText = keras.layers.Flatten()(maxPoolingLayer3)
    # metadata inputs
    metadataInputs = keras.Input(shape=(metadataShape,))
    # concatenate text with numeric metadata
    concatenatedLayers = keras.layers.concatenate([flattenedText, metadataInputs])
    # final processing
    denseLayer1 = keras.layers.Dense(16, activation='relu')(concatenatedLayers)
    # output layer
    outputLayer = keras.layers.Dense(num_output_categories, activation='softmax')(denseLayer1)
    # construct the model
    model = keras.models.Model(inputs=[textInputs, metadataInputs], outputs=outputLayer)
    # COMPILING THE MODEL
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    # TRAINING THE MODEL
    model.fit([text, metadata], prices, epochs=num_epochs, validation_split=0.2)


########################################################################################################################
########################################################################################################################
########################################################################################################################


def print_price_categorization_stats(card_prices, cutoffs):
    num_cards = len(card_prices)
    print("Num Cards:", num_cards)
    print("Num Categories:", len(cutoffs))
    for i in range(len(cutoffs)):
        print("Under ${:2.2f}: {:2.2%}".format(cutoffs[i], sum(p[i] == 1 for p in card_prices) / num_cards))


def extract_tokenized(source, tokenizer):
    tokenizer.fit_on_texts(source)
    res = tokenizer.texts_to_sequences(source)
    res = keras.preprocessing.sequence.pad_sequences(res)
    return res


########################################################################################################################
########################################################################################################################
########################################################################################################################

# Declare price categories
price_cutoffs = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 1, 5, 10, 50, 100]

# Parse the raw cards and extract arrays of interest
(rules_texts, metadata, prices) = parse_raw_cards("../dataset/oracle-cards.json", price_cutoffs)

# Print out the price categorization breakdowns
print_price_categorization_stats(prices, price_cutoffs)

# Tokenize the rules text
print("Beginning to tokenize the text")
vocab_size = 2048
tokenizer = keras.preprocessing.text.Tokenizer(num_words=vocab_size)
tokenized_rules_texts = extract_tokenized(rules_texts, tokenizer)
print("Finished tokenizing the text.")

# Train the network
train(tokenized_rules_texts, vocab_size, metadata, prices, len(price_cutoffs), 10)
