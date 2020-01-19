# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ## Linear Regression Model
#
# (Based on bag_of_words representation)

import re
import jieba
import pandas as pd
import numpy as np
import json
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

# ### Stop words list

# global variables initialisation
with open('../stop_words/中文停用词表.txt', 'r', encoding='UTF-8-sig') as f:
    stop_words = [ word.strip().replace('\n', '') for word in f.readlines()]
symbols = stop_words[0:26]
print('e.g.', stop_words[23:33])

# ### Data

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
wanted_columns = ['文档号码','投资者关系活动主要内容介绍', '证券代码（请务必使用text格式 以保留代码中的0）', '日期（格式统一为xx/xx/xxxx（日月年））']
data = all_data[wanted_columns]
print('Before cleaning:')
print(data.shape)
data = data.dropna() # drop rows with null values
print('After cleaning:')
print(data.shape)

# ### Merge Data by Cusip and Date

data = data.rename(columns={"文档号码": "ID", "投资者关系活动主要内容介绍": "content", 
                     "证券代码（请务必使用text格式 以保留代码中的0）": "cusip",
                     "日期（格式统一为xx/xx/xxxx（日月年））": "date"})
data.head(10)

ref_data = pd.read_excel('ref_data.xlsx')
ref_data.head(10)

from datetime import datetime

data = data.reset_index(drop=True)

# 有部分的date格式不对，进行提前清理

for i, dp in data.iterrows():
    try:
        str(dp['date'].strftime("%Y%m"))
    except:
        data.loc[i] = None

data = data.dropna().reset_index(drop=True)

data['date'] = data['date'].apply(lambda x: x.strftime("%Y%m"))

data['date'] = data['date'].apply(lambda x: str(x))

ref_data['cdt'] = ref_data['cdt'].apply(lambda x: str(x))

np.sort(data['date'].unique())

np.sort(ref_data['cdt'].unique())


# 选取2012第一季度到2018第一季度的data并转换日期格式

# +
# year-month to year-season
def yyyymm_to_yyyyss(yyyymm):
    
    assert len(yyyymm) == 6
    yyyy = yyyymm[:4]
    mm = yyyymm[-2:]
    if mm == '01' or mm == '02' or mm == '03':
        return yyyy + '01'
    elif mm == '04' or mm == '05' or mm == '06':
        return yyyy + '02'
    elif mm == '07' or mm == '08' or mm == '09':
        return yyyy + '03'
    elif mm == '10' or mm == '11' or mm == '12':
        return yyyy + '04'
    else:
        print('wrong month: {}'.format(mm))

yyyymm_to_yyyyss('201513')    
# -

data['date'] = data['date'].apply(lambda x: yyyymm_to_yyyyss(x))

np.sort(data['date'].unique())


# +
def keep_date(date):
    yyyy = int(date[:4])
    if 2012 <= yyyy and yyyy <= 2018:
        return date
    else:
        return None
    
np.sort(data['date'].apply(lambda x: keep_date(x)).dropna().unique())
# -

data['date'] = data['date'].apply(lambda x: keep_date(x))

data = data.dropna()

# 至此data里仅保留2012第一季度到2018第一季度(2018最高只到二月份）的数据

len(np.unique(ref_data['cusip']))


def check(cusip):
    if len(cusip) == 6:
        return cusip
    else:
        return None


len(data['cusip'].apply(lambda x: check(x)).dropna().unique())

data = data.rename(columns={"date": "yyyyss"})

data.to_excel('all_records_2012_2018.xlsx', index=False)


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
# 1. 如果已经对当前的data完整运行过readability，请将readability文件夹里面的all_freq_dist.json复制到regression文件夹里，从而复用数据，并使用下数第二个cell进行读取。
# 2. 如果尚未生成当前data的完整freq_dist，请跑一次下数第一个cell（One-time-block）进行生成。

# +
# One-time block 
# 建立一个完整的 frequency distribution，推荐只跑一次将数据储存以复用
def init_all_freq_dist():
    count = 0
    all_freq_dist = dict()
    for index, d in data.iterrows():
        print('[' + str(count) + '] Processing document ' + str(d['文档号码']) + '...')
        fd = gen_freq_dist(d[1])['freq_dist']
        for k in fd.keys():
            all_freq_dist.setdefault(k, 0)
            all_freq_dist[k] += fd[k]
        count += 1
    return all_freq_dist

all_freq_dist = init_all_freq_dist()
with open('all_freq_dist.json', 'w+', encoding='UTF-8-sig') as f:
    json.dump(all_freq_dist, f)
# -

# 如果前一个cell已经完整跑完一次，只需要跑这个cell就能拿到完整的 frequency distribution
with open('all_freq_dist.json', 'r', encoding='UTF-8-sig') as f:
    all_freq_dist = json.load(f)
all_freq_dist_df = pd.DataFrame.from_dict(all_freq_dist, orient='index', columns=['freq'])
print('Most frequent word is: ' + str(np.argmax(all_freq_dist_df['freq'])))
all_freq_dist_df.describe()

# ### Bag of words construction
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


# +
def gen_bag_of_words(doc):
    global vec
    return vec.transform(gen_freq_dist(doc)['freq_dist']).toarray()

def all_bag_of_words(col_name_for_docs, limited=False):
    global vec, data
    dimension = len(vec.get_feature_names())
    count = 0
    init = True
    X = []
    for index, data_point in data.iterrows():
        # 文档号码如果不存在将以下print注释，或者替换成另外指明数据的列
        print('[' + str(count) + '] Transforming document ' + str(data_point['文档号码']) + '...')
        # Initialise X with first document
        if init:
            X = gen_bag_of_words(data_point[col_name_for_docs])
            init = False
        else:
            X = np.vstack((X, gen_bag_of_words(data_point[col_name_for_docs])))
        count += 1
        
        if limited and count == 1000:
            # For test use, just use the first 1000 rows
            break
    
    return X

# returning the coeffcients of the linear regression model after fitting X and y
def lr_coeffs(X, y):
    global vec
    features = list(vec.get_feature_names())
    reg = LinearRegression().fit(X, y)
    coeffs = list(reg.coef_)
    result = pd.DataFrame(columns=['Feature', 'Coefficients'])
    result['Feature'] = features
    result['Coefficients'] = coeffs
    return result.sort_values(by=['Coefficients'], ascending=False)


# -

# Simply for testing
test_X = all_bag_of_words('投资者关系活动主要内容介绍', limited=True)
test_y = np.dot(test_X, np.array([1, 2] * int(59536/2))) + 3
dummy = pd.DataFrame()
dummy['Y'] = list(test_y)
dummy['文档号码'] = list(data['文档号码'][:1000])
dummy.to_excel('dummy_Y.xlsx')
test_result = lr_coeffs(test_X, test_y)

X = all_bag_of_words('投资者关系活动主要内容介绍', limited=False)


# ### Finally...

# 这里我们需要读取真正的Y值，格式为excel文件，且仅有两列，一列是ID（比如文档号码），一列是Y值。
#
# $\textbf{注意！}$ 读取Y值的文件里，ID的对应顺序要和提供训练数据的文档ID一致！一个简单的办法就是把Y值先按ID添加到原数据中，再进行分割即可。

def load_Y(file_path, col_name_ID, col_name_Y):
    global data
    df = pd.read_excel(file_path)
    if list(df[col_name_ID]) == list(data[col_name_ID]):
        print('训练数据与Y值的文档ID成功匹配！')
        return np.array(list(df[col_name_Y]))
    else:
        print('警告！训练数据与Y值的文档ID不匹配，请检查！')
        return None


y_path = '请替换成储存Y值文件的路径' # e.g. dummy_Y.xlsx
doc_ID_name = '请替换文档ID的名称' # e.g. 文档号码
y_name = '请替换Y值的名称' 
y = load_Y(Y_path, doc_ID_name, Y_name)

result = lr_coeffs(X, y)
# 储存结果
result.to_excel('word_ranking.xlsx')
