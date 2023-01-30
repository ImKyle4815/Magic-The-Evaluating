import json

file = open("oracle-cards-20230120100202.json")

j = json.load(file)

for i in j["cards"]:
    print(i['name'], " + ", i['oracle_text'], "\n\n")

print(len(j['cards']))
