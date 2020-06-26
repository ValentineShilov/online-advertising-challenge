#!/usr/bin/env python3

import argparse, csv, sys

from common import *

if len(sys.argv) == 1:
    sys.argv.append('-h')

parser = argparse.ArgumentParser()
parser.add_argument('csv_path', type=str)
parser.add_argument('dense_path', type=str)
parser.add_argument('sparse_path', type=str)
args = vars(parser.parse_args())

#ValentineS:
target_cat_feats = ['C14-1064159205', 'C12-106', 'C16-337426089', 'C13-1757768508', 'C3-2252', 'C11-0', 'C14-3781537013', 'C17-2058081684', 'C6-995', 'C8-419', 'C10-821', 'C9-15', 'C12-105', 'C7-2', 'C8-452', 'C12-103', 'C15-3071806138', 'C11-1', 'C12-104', 'C8-293', 'C5-88', 'C5-10', 'C8-177', 'C10-802', 'C5-30', 'C7-0']



with open(args['dense_path'], 'w') as f_d, open(args['sparse_path'], 'w') as f_s:
    for row in csv.DictReader(open(args['csv_path'])):
        feats = []
        for j in range(1, 3):
            val = row['I{0}'.format(j)]
            if val == '':
                val = -10 
            feats.append('{0}'.format(val))
        f_d.write(row['Label'] + ' ' + ' '.join(feats) + '\n')
        
        cat_feats = set()
        for j in range(1, 28): #ValentineS
            field = 'C{0}'.format(j)
            key = field + '-' + row[field]
            cat_feats.add(key)

        feats = []
        for j, feat in enumerate(target_cat_feats, start=1):
            if feat in cat_feats:
                feats.append(str(j))
        f_s.write(row['Label'] + ' ' + ' '.join(feats) + '\n')
