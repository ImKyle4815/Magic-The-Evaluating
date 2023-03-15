import json
import numpy as np
import os
from tensorflow import keras


# File structure constants
CARDS_FILE_PATH = "./input/oracle-cards-dataset.json"
MODEL_WEIGHTS_FILE_NAME = "output/categorized_model/weights.pb"
# Dataset and training features
MAX_CARD_TEXT_LENGTH = 256
METADATA_LENGTH = 15
VOCAB_SIZE = 2048
PRICE_CATEGORIES = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.50,
    1.00,
    10.00,
    1000.00
]


########################################################################################################################
########################################################################################################################
########################################################################################################################


def index_to_one_hot(index, size):
    a = [0] * size
    a[index] = 1
    return a


def one_hot_categories(sample, categories):
    num_categories = len(categories)
    a = [0] * num_categories
    for index in range(num_categories):
        if categories[index] in sample:
            a[index] = 1
    return a


def get_text(card):
    cardName = card["name"]
    text = card["oracle_text"]
    text = text.replace(cardName, "cardname")
    text = text.replace(".", " . ")
    text = text.replace("\n", " \n ")
    return text.lower()


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
    categories = ["Creature", "Instant", "Sorcery", "Enchantment", "Artifact", "Land"]
    return one_hot_categories(type_line, categories)


def get_colors(card):
    colors = card["colors"]
    categories = ["w", "u", "b", "r", "g"]
    return one_hot_categories(colors, categories)


def categorize_cmc(cmc):
    if cmc < 10:
        return cmc / 10
    else:
        return 1


def parse_raw_card(raw_card, price_cutoff_categories):
    # Apply Filters
    # if int(raw_card["released_at"][0:4]) < 2004:
    #     raise Exception("Failed to parse raw card")
    # Strings to be tokenized
    card_text = get_text(raw_card)
    # Concatenated Metadata
    card_type = get_type(raw_card)
    cmc = categorize_cmc(int(raw_card["cmc"]))
    colors = get_colors(raw_card)
    power = 0 if "power" not in raw_card else int(raw_card["power"]) / 10
    toughness = 0 if "toughness" not in raw_card else int(raw_card["toughness"]) / 10
    num_abilities = card_text.count('/n') / 10
    card_metadata = card_type + colors + [cmc, num_abilities, power, toughness]
    # Evaluation labels
    card_price = float(raw_card["prices"]["usd"])
    card_price_category = categorize_price(card_price, price_cutoff_categories)
    # Add the values to the lists
    return {
        "name": raw_card["name"],
        "text": card_text,
        "metadata": card_metadata,
        "price": card_price,
        "price_category": card_price_category
    }


def parse_raw_cards(filepath, price_cutoff_categories):
    file = open(filepath)
    raw_cards = json.load(file)
    res = []
    for raw_card in raw_cards:
        try:
            res.append(parse_raw_card(raw_card, price_cutoff_categories))
        except:
            continue
    return np.array(res)


def split_test_data(data=np.array([]), split=0.8):
    np.random.shuffle(data)

    split_index = (int(len(data) * split))

    return data[:split_index], data[split_index + 1:]


########################################################################################################################
########################################################################################################################
########################################################################################################################


def get_categorization_model(card_texts_shape, card_texts_vocab_size, card_metadata_shape, num_output_categories):
    # BUILDING THE MODEL
    # TEXT PATH
    # text inputs
    text_path_inputs = keras.Input(shape=(card_texts_shape,))
    # text processing layers
    text_path_layer = keras.layers.Embedding(input_dim=card_texts_vocab_size, output_dim=128,
                                             input_length=card_texts_shape)(text_path_inputs)
    text_path_layer = keras.layers.Conv1D(128, 8, activation='relu')(text_path_layer)
    text_path_layer = keras.layers.MaxPooling1D(2)(text_path_layer)
    text_path_layer = keras.layers.Conv1D(64, 8, activation='relu')(text_path_layer)
    text_path_layer = keras.layers.MaxPooling1D(2)(text_path_layer)
    text_path_layer = keras.layers.Conv1D(32, 8, activation='relu')(text_path_layer)
    text_path_layer = keras.layers.MaxPooling1D(2)(text_path_layer)
    text_path_layer = keras.layers.Conv1D(32, 8, activation='relu')(text_path_layer)
    text_path_layer = keras.layers.MaxPooling1D(2)(text_path_layer)
    text_path_layer = keras.layers.Flatten()(text_path_layer)
    # METADATA PATH
    # metadata inputs
    metadata_path_inputs = keras.Input(shape=(card_metadata_shape,))
    # metadata processing layers
    metadata_path_layer = keras.layers.Dense(64, activation='relu')(metadata_path_inputs)
    # CONCAT PATH
    # concatenate text with numeric metadata
    concat_path_layer = keras.layers.concatenate([text_path_layer, metadata_path_layer])
    # final processing
    concat_path_layer = keras.layers.Dense(32, activation='relu')(concat_path_layer)
    concat_path_layer = keras.layers.Dense(16, activation='relu')(concat_path_layer)
    # output layer
    concat_path_layer = keras.layers.Dense(num_output_categories, activation='softmax')(concat_path_layer)
    # construct the model
    model = keras.models.Model(inputs=[text_path_inputs, metadata_path_inputs], outputs=concat_path_layer)
    # COMPILING THE MODEL
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    # TRAINING THE MODEL
    return model


def get_callbacks():
    return [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3),
        keras.callbacks.ModelCheckpoint(filepath=MODEL_WEIGHTS_FILE_NAME, save_best_only=True),
        keras.callbacks.TensorBoard(log_dir="output/logs")
    ]

########################################################################################################################
########################################################################################################################
########################################################################################################################


def tokenize_texts(texts):
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=VOCAB_SIZE)
    tokenizer.fit_on_texts(texts)
    res = tokenizer.texts_to_sequences(texts)
    res = keras.preprocessing.sequence.pad_sequences(res, maxlen=MAX_CARD_TEXT_LENGTH)
    return res


def print_price_categorization_stats(card_prices, cutoffs):
    num_cards = len(card_prices)
    print("Num Cards:", num_cards)
    print("Num Categories:", len(cutoffs))
    for i in range(len(cutoffs)):
        print("Under ${:2.2f}: {:2.2%}".format(cutoffs[i], sum(p[i] == 1 for p in card_prices) / num_cards))


def extract_feature(array, key):
    res = []
    for item in array:
        res.append(item[key])
    return np.array(res)


########################################################################################################################
########################################################################################################################
########################################################################################################################


if __name__ == "__main__":
    # Parse the raw cards and extract arrays of interest
    parsed_cards = parse_raw_cards(CARDS_FILE_PATH, PRICE_CATEGORIES)
    # Split out the test and train data
    train_cards, test_cards = split_test_data(parsed_cards)
    # Extract all the desired features
    train_categorized_prices = extract_feature(train_cards, "price_category")
    train_texts = tokenize_texts(extract_feature(train_cards, "text"))
    train_metadata = extract_feature(train_cards, "metadata")
    test_categorized_prices = extract_feature(test_cards, "price_category")
    test_texts = tokenize_texts(extract_feature(test_cards, "text"))
    test_metadata = extract_feature(test_cards, "metadata")

    # Print out the price categorization breakdowns
    print_price_categorization_stats(train_categorized_prices, PRICE_CATEGORIES)

    # Create the model
    model = get_categorization_model(MAX_CARD_TEXT_LENGTH, VOCAB_SIZE, METADATA_LENGTH, len(PRICE_CATEGORIES))

    # Either load the model, or train it
    if os.path.exists(MODEL_WEIGHTS_FILE_NAME):
        print("[ Loading pre-saved weights ]")
        model.load_weights(MODEL_WEIGHTS_FILE_NAME)
    else:
        model.fit([train_texts, train_metadata], train_categorized_prices, epochs=1, validation_split=0.2,
                  callbacks=get_callbacks())

    # Evaluate the model against the test set
    model.evaluate([test_texts, test_metadata], test_categorized_prices)

    # Run predictions against random test cards
    for test_card in test_cards[:10]:
        print(test_card["name"], "is worth", test_card["price"], "under category", test_card["price_category"],
              "and predicts as category",
              model.predict([np.array(tokenize_texts([test_card["text"]])), np.array([test_card["metadata"]])]))

    # TO RUN THE TENSORBOARD WEB SERVER:
    # tensorboard --logdir ./output/logs
