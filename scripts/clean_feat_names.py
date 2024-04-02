# feature name clean
# e.g., EducationField -> education field, SENDER_ACCOUNT_ID -> sender account id
# standardize the feature texts for correct tokenization by Tokenizer
import os
import sys
sys.path.append(os.getcwd())
import json
import pandas as pd

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, choices=['pretrain', 'finetune'], help='Which training task is to conduct', required=True)
parser.add_argument("--task", type=str, choices=['binclass', 'regression', 'multiclass'], help='Which part of datasets to standardize', required=True)
parser.add_argument("--overwrite", action='store_true', help='Overwrite existing clean feature name map')

args = parser.parse_args()

if args.mode == 'pretrain':
    assert args.task != 'multiclass'
    if args.task == 'binclass':
        from lib import PRETRAIN_BIN_DATA as DATA
    elif args.task == 'regression':
        from lib import PRETRAIN_REG_DATA as DATA
else:
    if args.task == 'binclass':
        from lib import FINETUNE_BIN_DATA as DATA
    elif args.task == 'regression':
        from lib import FINETUNE_REG_DATA as DATA
    else:
        from lib import FINETUNE_MUL_DATA as DATA

if os.path.exists(DATA / 'feature_names.json'):
    print('clean feature name map already exisits')
    if not args.overwrite:
        print('exit')
        sys.exit(0)
    else:
        print('overwrite, parsing new feature name map')
        os.remove(DATA / 'feature_names.json')

# Check the upper case
def check_upper_num(x: str):
    x1, x2 = list(x), list(x.lower())
    num_upper = sum([c1 != c2 for c1, c2 in zip(x1, x2)])
    if num_upper > 1:
        return True
    if num_upper == 1 and x[0].lower() == x[0]:
        return True
    return False

# Check the Camel Case
def fix_camel_case(x: str):
    words = []
    cur_char = x[0].lower()
    min_chars_len = 10000
    for c in x[1:]:
        if c.isupper():
            min_chars_len = min(min_chars_len, len(cur_char))
            words.append(cur_char)
            cur_char = c.lower()
        else:
            cur_char += c
    min_chars_len = min(min_chars_len, len(cur_char))
    words.append(cur_char)
    return ' '.join(words), min_chars_len


used_datasets = [file for file in os.listdir(DATA) if file.endswith('.csv')]
feature_to_check = []
dropped_datasets = []
feature_name_dict = {} # standardized feature name
for file in used_datasets:
    df = pd.read_csv(DATA / file)
    if any(dd in file for dd in dropped_datasets):
        print(f'skip [dataset] {file}')
        continue
    for feature in df.columns[:-1]:
        if feature in feature_name_dict:
            continue
        temp = feature
        if '_' in temp:
            temp = ' '.join(temp.lower().split('_'))
        if '.' in feature:
            temp = ' '.join(temp.lower().split('.'))
        if '-' in feature:
            temp = ' '.join(temp.lower().split('-'))
        
        if check_upper_num(temp):
            std_feature, min_char_len = fix_camel_case(feature)
            if min_char_len == 1:
                if any(dd in file for dd in dropped_datasets):
                    print(f'Strange [dataset] {file} -> [feature] {feature}')
                else:
                    feature_name_dict[feature] = feature # keep special terms
                continue
            feature_name_dict[feature] = std_feature
        else:
            feature_name_dict[feature] = temp.lower()

# store the clean feature name map
with open(DATA / 'feature_names.json', 'w') as f:
    json.dump(feature_name_dict, f, indent=4)
print('clean feature name map has been generated')