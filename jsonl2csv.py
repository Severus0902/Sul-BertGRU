import json
import pandas as pd
import numpy as np
import sys
import math

def entropy(data):
    """
    计算信息熵
    """
    data=np.array(data)
    length = len(data)
    counter = {}
    data=(data*100).astype(int)
    for item in data:
        counter[item] = counter.get(item, 0) + 1
    ent = 0.0
    for _, cnt in counter.items():
        p = float(cnt) / length
        ent -= p * math.log2(p)
    return ent


input_file = sys.argv[1]
output_file = sys.argv[2]


# input_file = '1.jsonl'
# output_file = 'output.csv'

with open(input_file, 'r') as json_file:
    json_list = list(json_file)

fout = open(output_file,'w')
for json_str in json_list:
    tokens = json.loads(json_str)["features"]
    for token in tokens:
        if token['token'] in ['[CLS]','[SEP]']:
            continue
        else:
            entropies=[]
            for i in range(12):
                entropies.append(entropy(token['layers'][-i-1]['values']))
            weights=[]
            for i in range(12):
                weights.append(entropies[i]/sum(entropies))
            last_layers = np.sum([
                weights[0]*np.array(token['layers'][-1]['values']),
                weights[1]*np.array(token['layers'][-2]['values']),
                weights[2]*np.array(token['layers'][-3]['values']),
                weights[3]*np.array(token['layers'][-4]['values']),
                weights[4]*np.array(token['layers'][-5]['values']),
                weights[5]*np.array(token['layers'][-6]['values']),
                weights[6]*np.array(token['layers'][-7]['values']),
                weights[7]*np.array(token['layers'][-8]['values']),
                weights[8]*np.array(token['layers'][-9]['values']),
                weights[9]*np.array(token['layers'][-10]['values']),
                weights[10]*np.array(token['layers'][-11]['values']),
                weights[11]*np.array(token['layers'][-12]['values']),
            ], axis=0)
            fout.write(f'{",".join(["{:f}".format(i) for i in last_layers])}\n')
    