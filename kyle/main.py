import json

file = open("../dataset/oracle-cards.json")
cards = json.load(file)
training_data = []
training_labels = []
for card in cards:
    if not ("name" in card and "edhrec_rank" in card):
        continue
    else:
        training_data.append(card["name"])
        training_labels.append(card["edhrec_rank"])

