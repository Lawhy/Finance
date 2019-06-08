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

# # Similarity
#
# ### 对于每个公司的文档，计算其与之前一段时间文档的相似度

import re
import jieba
import pandas as pd
import numpy as np
import json
import datetime
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ### Stop words list

# global variables initialisation
with open('../stop_words/中文停用词表.txt', 'r', encoding='UTF-8-sig') as f:
    stop_words = [ word.strip().replace('\n', '') for word in f.readlines()]
symbols = stop_words[0:26]
print('e.g.', stop_words[23:33])

# ### Data
# 1. Read the original data; 
# 2. Merge the existing data with the date information (here step 1 & 2 are specific to the current project, for future use, please make sure the input data contains all the wanted information);

# #### Read the original data

# Single input
data_path = '请输入数据路径'  # e.g. data/record1.xls
all_data = pd.read_excel(data_path)
all_data.info()

# Multiple inputs
# 如果能把所有的record数据都集成一个excel文件，就可以用上面的代码
all_data = pd.DataFrame()
for i in range(17):
    i += 1
    path = "../data/record" + str(i) + ".xls"
    all_data = all_data.append(pd.read_excel(path), sort=False)
all_data.info()

# 取出想要的数据并且去除空行
wanted_columns = ['文档号码','投资者关系活动主要内容介绍'] 
data = all_data[wanted_columns]
print('Before cleaning:')
print(data.shape)
data = data.dropna() # drop rows with null values
print('After cleaning:')
print(data.shape)

# #### Read the extra date information and merge it with the original data

# 读取日期数据
date_info = pd.read_stata('../data/date.dta')
# 根据文档号码将两份数据合并,并去掉空行
data = pd.merge(data, date_info, on='文档号码')
data = data.dropna()
data.shape

data.head(10)

# 把公司代号变成category,而不是保留数值意义
data = data.astype({'stkcd' : str})

# Test data, take stkcd 2635
test_data = data[data['stkcd'] == '2635']
test_data.head(5)

# 拿到所有unique的公司代号
company_ids = list(pd.unique(data['stkcd']))
print('There are', len(company_ids), 'companies.')


# ### Preprocessing

# +
# 数据预处理所需要的所有方法

# clean the document, only Chinese characters, Numbers and Punctuations are left.
def clean(doc):
    chi = r'([\u4E00-\u9FA5]|[0-9]|[“”、。《》！，：；？\.%])'
    pa = re.compile(chi)
    return "".join(re.findall(pa, doc))

# sentence segmentation
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
    if not cleaned_doc or not len(sents):
        return {
            'sents': [],
            'size': 0,
            'avg_sent_len' : 0
        }
    else:
        total_length = sum([len(pure_sent(sent)) for sent in sents])
        avg_sent_length = total_length / len(sents)
        return {
            'sents': sents,
            'size' : total_length,
            'avg_sent_len' : avg_sent_length 
        }

# generate frequency distribution for each document, vital step for bag_of_words representation
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
                freq_dist.setdefault(word, 0)
                freq_dist[word] += 1
    return { 'freq_dist' : freq_dist, 
             'size' : stat['size'],
             'avg_sent_len' : stat['avg_sent_len'],
             'n_sents' : len(sents)
           }


# -

# ### Full frequencey distribution
#
# 使用之前已经生成过的all_freq_dist.json文件

with open('all_freq_dist.json', 'r', encoding='UTF-8-sig') as f:
    all_freq_dist = json.load(f)
all_freq_dist_df = pd.DataFrame.from_dict(all_freq_dist, orient='index', columns=['freq'])
print('Most frequent word is: ' + str(np.argmax(all_freq_dist_df['freq'])))
all_freq_dist_df.describe()

# ### Bag of words construction 
# #### (For Cosine Similarity)
#
# 有了完整的freq_dist以后就能把每个document都转换成bag_of_words形式。

# 先把frequency低于3的全部去掉
low_freq_words = [word for word in all_freq_dist.keys() if all_freq_dist[word] <= 3]
for lw in low_freq_words:
    del all_freq_dist[lw]
print('Remaining number of words:', len(all_freq_dist.keys()))

vec = DictVectorizer()
all_bow = vec.fit_transform(all_freq_dist).toarray()
print('e.g.', vec.get_feature_names()[12000:12010])


# generate bag_of_words for each document
def gen_bag_of_words(doc):
    global vec
    return vec.transform(gen_freq_dist(doc)['freq_dist']).toarray()


test_doc1 = test_data.iloc[0][1]
test_doc2 = test_data.iloc[1][1]
bow_t1 = gen_bag_of_words(test_doc1)
bow_t2 = gen_bag_of_words(test_doc2)
bow_t3 = gen_bag_of_words(data.iloc[10][1])

cosine_similarity(np.vstack((bow_t1, bow_t2)))

cosine_similarity(bow_t1, bow_t2)[0][0]

cosine_similarity(bow_t1, bow_t3)
