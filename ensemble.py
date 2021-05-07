import json
import math
from torchmetrics import AUROC
import torch
pred_files = [json.load(open("pred_tnt.json", 'r')),
              json.load(open("pred_vit.json", 'r')),
              json.load(open("pred_cnn.json", 'r')),
              json.load(open("pred_pit.json", 'r')),
              json.load(open("pred_swint.json", 'r')),
              ]
prob_files = [json.load(open("prob_tnt.json", 'r')),
              json.load(open("prob_vit.json", 'r')),
              json.load(open("prob_cnn.json", 'r')),
              json.load(open("prob_pit.json", 'r')),
              json.load(open("prob_swint.json", 'r')), ]
ensemble_pred = {}
ensemble_prob = {}
result = {}
all_prob = []
all_label = []
label = json.load(open("label.json", 'r'))

for i, file in enumerate(pred_files):
    for _id in file.keys():
        if _id in ensemble_pred:
            ensemble_pred[_id].append(file[_id])
            ensemble_prob[_id].append(prob_files[i][_id])
        else:
            ensemble_pred[_id] = [file[_id]]
            ensemble_prob[_id] = [prob_files[i][_id]]
total = 0
correct = 0
for _id in ensemble_pred.keys():
    votes = ensemble_pred[_id]
    if math.fsum(votes) < 3:
        result[_id] = 0
    else:
        result[_id] = 1
    if result[_id] == label[_id]:
        correct += 1

    gap = 0
    res = [0.5, 0.5]
    for item in ensemble_prob[_id]:
        if item[int(result[_id])] > 0.5 and math.fabs(item[0] - item[1]) > gap:
            gap = math.fabs(item[0] - item[1])
            res = item
    all_prob.append(res)
    all_label.append(label[_id])

    total += 1
auroc = AUROC(num_classes=2, pos_label=1)
auroc_res = auroc(torch.tensor(all_prob), torch.tensor(all_label))
print("ensemble accuracy: {}".format(correct / total))
print("ensemble auroc: {}".format(auroc_res))
