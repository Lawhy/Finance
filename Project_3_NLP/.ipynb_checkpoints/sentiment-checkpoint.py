# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Sentiment Analysis
# ### (using Baidu Brain's NlP tools)

from aip import AipNlp
import numpy as np
import pandas as pd

# +
# Important setup (收钱授权)
APP_ID = "16323168"
API_KEY = "azMXa7UuqFH8qsXTsumh1XoF"
SECRET_KEY = "CfYjxNl4SMAkorMpXIhAclyIQ3nAzwE9"

client = AipNlp(APP_ID, API_KEY, SECRET_KEY)
# -

# Testing excel files accessibility
df = pd.read_excel('data/record1.xls')
df = df.append(pd.read_excel('data/record2.xls'))
df.head(10)

# Prepare Data
all_data = pd.DataFrame()
for i in range(17):
    i += 1
    path = "data/record" + str(i) + ".xls"
    all_data = all_data.append(pd.read_excel(path), sort=False)
all_data.info()

data = all_data[['文档号码','投资者关系活动主要内容介绍']]
print('Before cleaning:')
print(data.shape)
data = data.dropna()
print('After cleaning:')
print(data.shape)
print(data.head(10))

# Test a single document
sample = data.iloc[0][1]
sample = sample.replace('\n','')
sample = sample.replace(' ', '')
print(sample[0:500])
sample_result = client.sentimentClassify(sample[0:1000])
sample_result

# manipulate json items
sample_result['items'][0]


# generate results for one document after cleaning
def analyse(doc, doc_num):
    # dividing
    set_of_docs = []
    num_of_portions = int((len(doc) / 1000)) + 1
    for i in range(num_of_portions):
        sub_doc = doc[i*1000:(i+1)*1000]
        if not i == num_of_portions - 1:
            assert len(sub_doc) == 1000
        else:
            assert len(sub_doc) <= 1000
        set_of_docs.append(sub_doc)
    # sentiment analysis for each sub_doc
    results = []
    for sub_doc in set_of_docs:
        if not sub_doc:
            break
        result_dict = client.sentimentClassify(sub_doc)['items'][0]
        result_dict['weight'] = len(sub_doc) / len(doc)
        results.append(result_dict)
    # transfer to the wanted scores
    scores = pd.DataFrame()
    scores['文档号码'] = [doc_num]
    scores['positive_prob'] = sum([ (r_dict['positive_prob'] * r_dict['weight']) for r_dict in results ])
    scores['negative_prob'] = sum([ (r_dict['negative_prob'] * r_dict['weight']) for r_dict in results ])
    scores['sentiment'] = round(sum([ (r_dict['sentiment'] * r_dict['weight']) for r_dict in results ]))
    scores['positive_prob_c'] = sum([ (r_dict['positive_prob'] * r_dict['weight'] * r_dict['confidence']) for r_dict in results ])
    scores['negative_prob_c'] = sum([ (r_dict['negative_prob'] * r_dict['weight'] * r_dict['confidence']) for r_dict in results ])
    scores['sentiment_c'] = round(sum([ (r_dict['sentiment'] * r_dict['weight'] * r_dict['confidence']) for r_dict in results ]))
    return scores
# analyse(sample, '1200573239')


all_scores = pd.DataFrame()
error_docs = []
empty_docs = []
count = 0
for index, d in data.iterrows():
    print('[' + str(count) + '] Processing document ' + str(d['文档号码']) + '...')
    count += 1
    doc = d['投资者关系活动主要内容介绍']
    doc_num = d['文档号码']
    # cleaning
    doc = doc.replace('\n', '')
    doc = doc.replace(' ', '')
    doc = doc.replace('\xa0', '')
    doc = doc.replace('\t', '')
    doc = doc.replace('\uf0a1', '')
    if not doc:
        print('The document is empty!')
        empty_docs.append(str(doc_num))
        continue 
    try:
        scores = analyse(doc, str(doc_num))
        all_scores = all_scores.append(scores)
    except:
        error_docs.append(str(doc_num))
        print('An error happened when processing document ' + str(doc_num))

# Error document testing
error = data.iloc[102][1]
e_num = data.iloc[102][0]
# cleaning
error = error.replace('\n', '')
error = error.replace(' ', '')
error = error.replace('\xa0', '')
error = error.replace('\t', '')
error = error.replace('\uf0a1', '')
# analyse(error, str(e_num))
# all_scores

# Get the final results
all_scores

df_error_docs = pd.DataFrame(error_docs, columns=['文档编号'])
df_error_docs.to_excel('error_docs.xlsx', index=False)

all_scores.to_excel('sentiment/sentiment.xlsx', index=False)

pd.DataFrame(empty_docs, columns=['文档编号']).to_excel('empty_docs.xlsx', index=False)

# Handling the error docs
client.sentimentClassify('急速发展的现代化城市各种奢华酒店和奢华享受很不错值得一去斋月的时候去虽然热但是体验了不一样的当地文化很受用')

error_data = data[data['文档号码'].isin(error_docs)]
error_test = error_data.iloc[0]
error_text = error_test[1]
error_num = error_test[0]
error_text = error_text.replace('\n', '')
error_text = error_text.replace(' ', '')
error_text = error_text.replace('\xa0', '')
error_text = error_text.replace('\t', '')
error_text = error_text.replace('\uf0a1', '')
#     client.sentimentClassify(sub_doc)['items'][0]
# analyse(error_text, error_num)
# client.sentimentClassify(error_text)['items']
analyse(error_text, error_num)

import re


# process the error docs after second-level cleaning
def clean_error_doc(error_doc):
    chi = r'([\u4E00-\u9FA5，。：！《》？——、])'
    pa = re.compile(chi)
    return "".join(re.findall(pa, error_doc))


error_doc_nums = pd.read_excel("sentiment/error_docs.xlsx")
error_data = data[data['文档号码'].isin(error_doc_nums['文档编号'].tolist())]
error_data

rest_scores = pd.DataFrame()
rest_error_docs = []
rest_empty_docs = []
rest_count = 0
for index, d in error_data.iterrows():
    print('[' + str(rest_count) + '] Processing document ' + str(d['文档号码']) + '...')
    rest_count += 1
    doc = d['投资者关系活动主要内容介绍']
    doc_num = d['文档号码']
    # cleaning
    doc = clean_error_doc(doc)
    if not doc:
        print('The document is empty!')
        rest_empty_docs.append(str(doc_num))
        continue 
    try:
        scores = analyse(doc, str(doc_num))
        rest_scores = rest_scores.append(scores)
    except:
        rest_error_docs.append(str(doc_num))
        print('An error happened when processing document ' + str(doc_num))

rest_scores.to_excel('sentiment/sentiment2.xlsx', index=False)
