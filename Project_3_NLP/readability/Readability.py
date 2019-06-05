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

# # Readability

# ## Part 0. Packages and Data

# ### 0.1: Import necessary packages

import re
import jieba
import pandas as pd
import numpy as np
import json
from nltk.probability import *
import math

# ### 0.2: Read data
#
# #### （代码效果与sentiment analysis里的一样） 

# Single input
data_path = '请输入数据路径'  # e.g. data/record1.xls
all_data = pd.read_excel(data_path)
all_data.info()

# Multiple inputs
all_data = pd.DataFrame()
for i in range(17):
    i += 1
    path = "../data/record" + str(i) + ".xls"
    all_data = all_data.append(pd.read_excel(path), sort=False)
all_data.info()

data = all_data[['文档号码','投资者关系活动主要内容介绍']]
print('Before cleaning:')
print(data.shape)
data = data.dropna()
print('After cleaning:')
print(data.shape)


# ## Part 1. Preprocessing
# ### 1.1 Cleaning

def clean(doc):
    chi = r'([\u4E00-\u9FA5]|[0-9]|[“”、。《》！，：；？\.%])'
    pa = re.compile(chi)
    return "".join(re.findall(pa, doc))


# #### 以下两个cells只是可视化cleaning和分词的效果，可以忽略

print(clean(data.iloc[0][1]))
print("--------------")
print(data.iloc[0][1])

seg_list = jieba.cut(clean(data.iloc[0][1][0:500]), cut_all=False, HMM=True)
print("Default Mode: " + "/ ".join(seg_list))  # 默认模式


# ### 1.2 Split into sentences

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
    
raw_process(data.iloc[0][1])
# -

# ### 1.3 Stop word list

with open('../stop_words/中文停用词表.txt', 'r', encoding='UTF-8-sig') as f:
    stop_words = [ word.strip().replace('\n', '') for word in f.readlines()]
symbols = stop_words[0:26]
stop_words


# ### 1.4 Generating frequency distribution 
# (using stop word list)

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
             'avg_sent_len' : stat['avg_sent_len'],
             'n_sents' : len(sents)
           }

gen_freq_dist(data.iloc[0][1])
# -

# ## Part 2 Compute the readability in different ways
#
# ### 2.1 Grade and Semester 
# (using common word list)

ch_pa = re.compile(r'([\u4E00-\u9FA5])')
with open('../common_words/常用词.txt', 'r', encoding='UTF-8-sig') as f:
    common_words = [ ''.join(re.findall(ch_pa, line)) for line in f.readlines()]
print(len(common_words))
print('为了' in common_words)


# $\begin{aligned}
# 年级 &= 17.52547988 + 0.00242523 \times 课文长度 \\
#       &+ 0.04414527 \times 平均句长 - 18.33435443 * 常用字比率
# \end{aligned}$
#
# $\begin{aligned}
# 学期 &= 34.53858379 + 0.00491625 \times 课文长度 \\
#       &+ 0.08996394 \times 平均句长 - 36.73710603 * 常用字比率
# \end{aligned}$

# +
def grade_n_semester(doc):
    stat = gen_freq_dist(doc)
    freq_dist = stat['freq_dist']
    n_words = sum(freq_dist.values())
    
    # compute the percentage of common words
    n_common_words = 0
    for word in freq_dist.keys():
        if word in common_words:
            n_common_words += freq_dist[word]
    
    # all the required statistics
    article_length = stat['size']
    avg_sent_len = stat['avg_sent_len']
    
    # for empty doc
    if not article_length or not avg_sent_len:
        return {
            'grade' : None,
            'semester' : None,
            'common_words_percentage' : None
        }
    
    percent_common_words = n_common_words / n_words
    
    # compute grade & semester
    grade = 17.52547988 + 0.00242523 * article_length \
            + 0.04414527 * avg_sent_len - 18.33435443 * percent_common_words
    semester = 34.53858379 + 0.00491625 * article_length \
             + 0.08996394 * avg_sent_len - 36.73710603 * percent_common_words
    
    return {
        'grade' : grade,
        'semester' : semester,
        'common_words_percentage' : percent_common_words
    }

grade_n_semester(data.iloc[0][1])


# -

# ### 2.2 Fog value and its modified version 
# (using the full frequency distribution (all_freq_dist) and the document frequency distribution (df_dist))

# +
# One-time block 
# 建立一个完整的 frequency distribution，推荐只跑一次将数据储存以复用
def init_all_freq_dist():
    all_freq_dist = dict()
    count = 0
    for index, d in data.iterrows():
        print('[' + str(count) + '] Processing document ' + str(d['文档号码']) + '...')
        fd = gen_freq_dist(d[1])['freq_dist']
        for k in fd.keys():
            if k not in all_freq_dist.keys():
                all_freq_dist[k] = fd[k]
            else:
                all_freq_dist[k] += fd[k]
        count += 1
    return all_freq_dist

all_freq_dist = init_all_freq_dist()
with open('all_freq_dist.json', 'w+', encoding='UTF-8-sig') as f:
    json.dump(all_freq_dist, f)

# +
# 如果前一个cell已经完整跑完一次，只需要跑这个cell就能拿到完整的 frequency distribution
with open('all_freq_dist.json', 'r', encoding='UTF-8-sig') as f:
    all_freq_dist = json.load(f)

all_freq_dist_df = pd.DataFrame.from_dict(all_freq_dist, orient='index', columns=['freq'])
print('Most frequent word is: ' + str(np.argmax(all_freq_dist_df['freq'])))
all_freq_dist_df.describe()


# -

# Given the summary here, apparently defining 5% or 10% as rare/complex words is not appropriate, so instead, define complex words to be frequency less than 1 or 3.

def is_complex(word, threshold=1):
    global all_freq_dist
    return all_freq_dist[word] <= threshold 


# +
# One-time block
# 建立一个完整的 document frequency distribution，推荐只跑一次将数据储存以复用
# df的定义在 10-K readability 那篇paper里
def init_df_dist():
    df_dist = dict()
    count = 0
    for index, d in data.iterrows():
        print('[' + str(count) + '] Processing document ' + str(d['文档号码']) + '...')
        fd = gen_freq_dist(d[1])['freq_dist']
        for k in fd.keys():
            df_dist.setdefault(k, 0)
            df_dist[k] += 1
        count += 1
    return df_dist

df_dist = init_df_dist()
with open('df_dist.json', 'w+', encoding='UTF-8-sig') as f:
    json.dump(df_dist, f)

# +
with open('df_dist.json', 'r', encoding='UTF-8-sig') as f:
    df_dist = json.load(f)

df_dist_df = pd.DataFrame.from_dict(df_dist, orient='index', columns=['df'])
print('Most diverse word is: ' + str(np.argmax(df_dist_df['df'])))
df_dist_df.describe()


# -

# 10-K readability paper 里定义的词比重
def weight_of_word(word):
    global data, df_dist
    N = data.shape[0]
    df = df_dist[word]
    return np.log(N / df) / np.log(N)


# +
def fog(word_per_sent, percent_cw):
    return 0.4 * (word_per_sent + 100 * percent_cw)

def fog_all(doc):
    global all_freq_dist
    stat = gen_freq_dist(doc)
    freq_dist = stat['freq_dist']
    n_words = sum(freq_dist.values())
    n_sents = stat['n_sents']
    
    # for empty doc
    if not n_sents or not n_words:
        return {
            'original_fog_t1' : None,
            'original_fog_t3' : None,
            'weighted_fog_t1' : None,
            'weighted_fog_t3' : None
        }
    
    # complex words percentage
    percent_cw_dict = {
        'o_t1' : (sum([freq_dist[word] for word in freq_dist.keys() if is_complex(word, threshold=1)]) / n_words),
        'o_t3' : (sum([freq_dist[word] for word in freq_dist.keys() if is_complex(word, threshold=3)]) / n_words),
        'w_t1' : (sum([(weight_of_word(word) * freq_dist[word]) \
                               for word in freq_dist.keys() if is_complex(word, threshold=1)]) / n_words),
        'w_t3' : (sum([(weight_of_word(word) * freq_dist[word]) \
                               for word in freq_dist.keys() if is_complex(word, threshold=3)]) / n_words)
    }
    
    word_per_sent = n_words / n_sents
    
    return {
            'original_fog_t1' : fog(word_per_sent, percent_cw_dict['o_t1']),
            'original_fog_t3' : fog(word_per_sent, percent_cw_dict['o_t3']),
            'weighted_fog_t1' : fog(word_per_sent, percent_cw_dict['w_t1']),
            'weighted_fog_t3' : fog(word_per_sent, percent_cw_dict['w_t3'])
    }


# -

# ### 2.3 Computation

all_scores = pd.DataFrame(columns=['文档号码', '总长度', '平均句长', '常用词比例', '年级', '学期', 'original_fog_t1', 'original_fog_t3', 'weighted_fog_t1', 'weighted_fog_t3'])
count = 0
error_docs = []
for index, d in data.iterrows():
    print('[' + str(count) + '] Processing document ' + str(d['文档号码']) + '...')
    count += 1
    doc = d['投资者关系活动主要内容介绍']
    doc_num = d['文档号码']
    scores = pd.DataFrame(columns=['文档号码', '总长度', '平均句长', '常用词比例', '年级', '学期', 'original_fog_t1', 'original_fog_t3', 'weighted_fog_t1', 'weighted_fog_t3'])
    scores['文档号码'] = [doc_num]
    try:
        raw_stat = raw_process(doc)
        scores['总长度'] = [raw_stat['size']]
        scores['平均句长'] = [raw_stat['avg_sent_len']]
        # 学期,年级
        result_1 = grade_n_semester(doc)
        scores['常用词比例'] = [result_1['common_words_percentage']]
        scores['年级'] = [result_1['grade']]
        scores['学期'] = [result_1['semester']]
        # fog
        result_2 = fog_all(doc)
        scores['original_fog_t1'] = [result_2['original_fog_t1']]
        scores['original_fog_t3'] = [result_2['original_fog_t3']]
        scores['weighted_fog_t1'] = [result_2['weighted_fog_t1']]
        scores['weighted_fog_t3'] = [result_2['weighted_fog_t3']]
        # appending
        all_scores = all_scores.append(scores)
    except:
        error_docs.append(doc_num)
        print('An error happened when processing document ' + str(doc_num))

data[data['文档号码'].isin(error_docs)] #看还有没有无法处理的文件

all_scores.to_excel('readability.xlsx', index=False) #储存最终结果

# ### Note: 注意如果数据更换了 之前生成的all_freq_dist.json和df_freq_dist.json 要随着之前的结果一起被移除，以免产生误解！
