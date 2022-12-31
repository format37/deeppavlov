###
# This script:
# 1. read df
# 2. get each row text value
# 3. split text by '.'
# 4. evaluate classes for each phrase
# 5. mid for each row which add to class column
###

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json


def predict_zero_shot(text, label_texts, model, tokenizer, label='entailment', normalize=True):
    label_texts
    tokens = tokenizer([text] * len(label_texts), label_texts, truncation=True, return_tensors='pt', padding=True)
    with torch.inference_mode():
        result = torch.softmax(model(**tokens.to(model.device)).logits, -1)
    proba = result[:, model.config.label2id[label]].cpu().numpy()
    if normalize:
        proba /= sum(proba)
    return proba

print('loading model')
model_checkpoint = 'cointegrated/rubert-base-cased-nli-threeway'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
if torch.cuda.is_available():
    model.cuda()

print('reading params')
# Parameters
with open('nli_column.json', 'r') as f:
    config = json.load(f)
classes = config['classes']

print('reading df')
df = pd.read_csv(config['csv_name'])
# add columns from classes
for class_name in classes:
    df[class_name] = 0

print('evaluating classes')
# evaluate a middle value for each phrase and each class
for idx, row in df.iterrows():
    # define a structure to store the
    # classes and their probabilities
    classes_prob = {}
    for class_name in classes:
        classes_prob[class_name] = []

    text = row['text']
    # split text by '.'
    phrases = text.split('.')
    # evaluate classes for each phrase
    for phrase in phrases:
        prediction = predict_zero_shot(phrase, classes, model, tokenizer)
        # get for each row
        for i, class_name in enumerate(classes):
            # add to the structure
            classes_prob[class_name].append(prediction[i])
    # calculate middle value for each classes_prob
    for class_name in classes:
        df.at[idx, class_name] = sum(classes_prob[class_name]) / len(classes_prob[class_name])

### Load btc price
btc = pd.read_csv('BTC_USD Bitfinex Historical Data_2018_07_24-2022_07_24.csv')
# Date	Price	Open	High	Low	Vol.	Change %
 
# append one day to the df date column
# df['date'] = pd.to_datetime(df['date']) + pd.Timedelta(days=1)
df['date'] = df['date'] + pd.Timedelta(days=1)

# btc Date format: 'Jul 24, 2022'
# convert to datetime
btc['Date'] = pd.to_datetime(btc['Date'])

# df date format: 2018-07-25 09:47:06
# it is datetime
# convert remove the time part
df['date'] = df['date'].dt.date

# concatenate df and btc by date
df = pd.concat([df, btc], axis=1)

# drop nan from btc
df = df.dropna()

# last column 'change %' format: '+0.00%'
# remove % and convert to float
df['change %'] = df['change %'].str.replace('%', '')
df['change %'] = df['change %'].astype(float)

# order by date
df = df.sort_values('date')

# group df by date and change. avg bull and bear
df = df.groupby('date').mean()

# reset index
df = df.reset_index()

print('saving df')
# save to csv
df.to_csv('temp.csv')

print('done!')

# df.date format: 2018-07-24T09:47:06	
# remove time part
# df.date format: 2018-07-24
df['date'] = df['date'].dt.date