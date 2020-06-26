import zlib
import numpy as np
import pandas as pd
import hashlib
from collections import defaultdict
HEADER = "timestamp;label;C1;C2;C3;C4;C5;C6;C7;C8;C9;C10;CG1;CG2;CG3;l1;l2;C11;C12".split(';')
COUNT_BASIC = 2
COUNT_CATEG_FIRST_GROUP = 10
COUNT_GROUP = 3
COUNT_CATEG_SECOND_GROUP = 2


OUTPUT_HEADER = "timestamp;Label;I1;I2;C1;C2;C3;C4;C5;C6;C7;C8;C9;C10;C11;C12;C13;C14;C15;C16;C17;C18;C19;C20;C21;C22;C23;C24;C25;C26;C27".split(';')

CG1_GF = {  "C13" : [96, 438, 52, 154, 53, 422, 419, 279, 335, 150, 385, 139], 
            "C14" : [331, 99, 49, 399, 330, 416, 276, 205, 130, 412, 332, 268],
            "C15" : [123, 449, 151, 54, 273, 76, 435, 437, 334, 74, 341, 122],
            "C16" : [357, 18, 124, 336, 57, 155, 333, 277, 59, 343, 58, 382],
            "C17": []}

CG2_GF = {  "C18" : [20755, 28695, 207, 19883, 7636, 29768, 16676, 8395, 29432, 16810, 7326, 8833],
            "C19" : [29347, 29463, 15650, 14322, 9793, 3746, 30009, 3864, 7923, 31328, 20892, 823],
            "C20" : [10328, 2254, 25731, 16444, 6746, 30749, 3326, 18705, 4378, 2293, 30235, 516],
            "C21" : [24338, 25900, 8426, 2390, 6619, 19225, 24843, 18714, 944, 17179, 2253, 22582],
            "C22": []}

CG3_GF = {  "C23" : [49272, 15529, 46839, 33701, 21144, 43845, 15769, 44289, 3311, 23339, 40461, 56445],
            "C24" : [7592, 38444, 5178, 54846, 45902, 49517, 38346, 47396, 20076, 45501, 57895, 49784],
            "C25" : [46594, 45509, 18336, 19563, 17419, 49513, 46401, 55412, 49962, 37588, 27636, 2340],
            "C26" : [11599, 56597, 43844, 48114, 49518, 24887, 75, 28676, 49966, 8894, 6214, 43666],
            "C27": []}

COUNT_CATEGORIAL = COUNT_CATEG_FIRST_GROUP + COUNT_CATEG_SECOND_GROUP

def GetFeatures(input_str, global_statistics):
    
    features = {}
    for feature_id, feature_str in enumerate(input_str.split(';')):

        feature_name = HEADER[feature_id]
        if feature_name == 'label':
            feature_name = 'Label'
        
        elif feature_name.startswith('CG'):
            features.update(ProcessGroup(feature_name, global_stat, feature_str))
        
        elif feature_name.startswith('l'):
            feature_name = feature_name.replace('l', 'I')

        if not feature_name.startswith('CG'):
            features.update({feature_name: feature_str})
        
        if feature_name.startswith('C') and not feature_name.startswith('CG'):
            global_statistics[feature_name+"-"+feature_str] += 1

    return features

def CreateOutFeatures(input_str, statistics):

    features = GetFeatures(input_str, statistics)
    out = ""

    for feature_name in OUTPUT_HEADER:
        feature_val = features.get(feature_name)
        if feature_val is None:
            feature_val = ''
        out += feature_val + ","
    return out[:-1]

def ProcessGroup(group_name, stat, group_feature):
    
    vals_array = None
    if group_name == 'CG1':
        vals_array = CG1_GF
    elif group_name == 'CG2':
        vals_array = CG2_GF
    elif group_name == 'CG3':
        vals_array = CG3_GF
    else:
        raise Exception("Unknown group")

    result = defaultdict(list)
    for val in group_feature.split(','):
        if val == '':
            continue
        else:
            val = int(val)

        for key, feature_vals in vals_array.items():
            if len(feature_vals) == 0: 
                result[key].append(str(val))
                break
            if val in feature_vals:
                result[key].append(str(val))
                break
    for key in result:
        if len(result[key]) == 0:
            result[key] = ''
        else:
            result[key] = str(zlib.adler32((",".join(sorted(result[key]))).encode()))
        global_stat[key+"-"+result[key]] += 1
    return result


def ProcessFile(filename_in, filename_out, global_statistics):
    with open(filename_in, 'r', encoding='utf-8') as input, \
            open(filename_out, 'w', encoding='utf-8') as output:
        output.write(','.join(OUTPUT_HEADER)+'\n')
        for counter, line in enumerate(input):
            if line.startswith('timestamp'):
                continue

            line = line[:-1]
            out_line = CreateOutFeatures(line, global_statistics)
            output.write(out_line+'\n')
            if counter % 100000 == 0:
                print(filename_in, counter)

global_stat = defaultdict(int)

ProcessFile('train.csv', 'train_p.csv', global_stat)
ProcessFile('test.csv', 'test_p.csv', global_stat)

sorted_stat = sorted(global_stat.items(), key=lambda kv: -kv[1])
with open('stat.txt', 'w', encoding='utf-8') as stat_file:
    for stat_name, stat_count in sorted_stat:
        stat_file.write(stat_name+"\t"+str(stat_count)+"\n")
