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

import re
import jieba
import pandas as pd
import numpy as np

# Prepare Data
all_data = pd.DataFrame()
for i in range(17):
    i += 1
    path = "data/record" + str(i) + ".xls"
    all_data = all_data.append(pd.read_excel(path), sort=False)
data = all_data[['文档号码','投资者关系活动主要内容介绍']]
print('Before cleaning:')
print(data.shape)
data = data.dropna()
print('After cleaning:')
print(data.shape)
print(data.head(10))


# process the error docs after second-level cleaning
def clean(doc):
    chi = r'([\u4E00-\u9FA5]|[0-9]|[“”、。《》！，：；？\.%])'
    pa = re.compile(chi)
    return "".join(re.findall(pa, doc))


print(clean(data.iloc[10][1]))
print("--------------")
print(data.iloc[10][1])

seg_list = jieba.cut(clean(data.iloc[10][1][0:500]), cut_all=False, HMM=True)
print("Default Mode: " + "/ ".join(seg_list))  # 默认模式


# +
def sent_seg(cleaned_doc):
    sent_pa = re.compile(r'.+?[？。！]')
    return re.findall(sent_pa, cleaned_doc)

def pure_sent(sent):
    cleaned_sent_pa = re.compile(r'([\u4E00-\u9FA5])')
    return ''.join(re.findall(cleaned_sent_pa, sent))
        
# Size of a doc is defined as the total number of valid Chinese characters
def raw_process(doc):
    cleaned_doc = clean(doc)
    sents = sent_seg(cleaned_doc)
    total_length = sum([len(pure_sent(sent)) for sent in sents])
    avg_sent_length = total_length / len(sents)
    return {
        'sents': sents,
        'size' : total_length,
        'avg_size' : avg_sent_length 
    }
    
raw_process(data.iloc[0][1])
# -

with open('stop_words/中文停用词表.txt', 'r', encoding='UTF-8-sig') as f:
    stop_words = [ word.strip().replace('\n', '') for word in f.readlines()]
symbols = stop_words[0:26]
stop_words

''.join(symbols)


# +
def gen_freq_dist(doc):
    stat = raw_process(doc)
    sents = stat['sents']
    freq_dist = dict()
    pa = re.compile(r'([$0123456789?_“”、。《》！，：；？\.%])')
    for sent in sents:
        # calculate sent length after
        words = jieba.cut(sent, cut_all=False, HMM=True)
        for word in words:
            # ignore all the stop words
            if (not word in stop_words) and (not re.findall(pa, word)):
                if not word in freq_dist.keys():
                    freq_dist[word] = 1
                else:
                    freq_dist[word] += 1
    return { 'freq_dist' : freq_dist, 
             'size' : stat['size'],
             'avg_size' : stat['avg_size']
           }

gen_freq_dist(data.iloc[0][1])
# -

ch_pa = re.compile(r'([\u4E00-\u9FA5])')
with open('常用词.txt', 'r', encoding='UTF-8-sig') as f:
    common_words = [ ''.join(re.findall(ch_pa, line)) for line in f.readlines()]
print(common_words[500:600])
print(len(common_words))
print('为了' in common_words)
