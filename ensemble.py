import json
import math

files = [json.load(open("model1.json", 'r')),
         json.load(open("model2.json", 'r')),
         json.load(open("model3.json", 'r'))]

ensemble = {}
result = {}
label = json.load(open("label.json", 'r'))
for file in files:
    for _id in file.keys():
        if _id in ensemble:
            ensemble[_id].append(file[_id])
        else:
            ensemble[_id] = [file[_id]]
total = 0
correct = 0
for _id in ensemble.keys():
    votes = ensemble[_id]
    if math.fsum(votes) < 3:
        result[_id] = 0
    else:
        result[_id] = 1
    if result[_id] == label[_id]:
        correct += 1
    total += 1

print("ensemble accuracy".format(total / correct))
