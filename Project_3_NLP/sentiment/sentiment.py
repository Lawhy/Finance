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
# ### (using Baidu Brain's NLP tools)

# ### Step 1: Import necessary pacakages

from aip import AipNlp
import numpy as np
import pandas as pd
import re

# ### Step 2: 百度大脑使用授权 (如果失效可自行注册一个账号，按照官网的指示拿到以下三个值）

# +
# Important setup for Baidu Brain's NLP
APP_ID = "16323168"
API_KEY = "azMXa7UuqFH8qsXTsumh1XoF"
SECRET_KEY = "CfYjxNl4SMAkorMpXIhAclyIQ3nAzwE9"

client = AipNlp(APP_ID, API_KEY, SECRET_KEY)
# -

# ### Step 3.1: 单一输入文件请用以下代码 （输入excel文件，尾缀是xls或者xlsx）

# Single input
data_path = '请输入数据路径'  # e.g. data/record1.xls
all_data = pd.read_excel(data_path)
all_data.info()

# ### Step 3.2: 多文件输入（由于这次给的数据命名为 recordN.xls 的形式， N为1到17） 所以用一个for-loop 把所有数据读取）

# Multiple inputs
all_data = pd.DataFrame()
for i in range(17):
    i += 1
    path = "data/record" + str(i) + ".xls"
    all_data = all_data.append(pd.read_excel(path), sort=False)
all_data.info()

# ### Step 4: 只取数据中有用的列，并且去掉所有数据残缺的行, dropna()的功能就是去掉具有空值的行

data = all_data[['文档号码','投资者关系活动主要内容介绍']] # 如果需要不同名字的列，请修改中括号里面的内容，以逗号分隔
print('Before cleaning:')
print(data.shape)
data = data.dropna()
print('After cleaning:')
print(data.shape)


# ### Step 5: Sentiment Analysis Method 封装

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


# ### Step 6: Sentiment Analysis 执行 （运行时间根据数据多少而定）

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

# ### Step 7: 结果储存
# #### （Outputs有三种，其中sentiment.xlsx是算好的结果，error_docs.xlsx 和 empty_docs.xlsx 分别记录了无法处理的文件和空文件的文档号码）

all_scores.to_excel('sentiment.xlsx', index=False)
pd.DataFrame(error_docs, columns=['文档号码']).to_excel('error_docs.xlsx', index=False)
pd.DataFrame(empty_docs, columns=['文档号码']).to_excel('empty_docs.xlsx', index=False)


# ### Step 8: 定义clean_error_doc方法来将无法完整分析的error_docs二次处理，只留下中文，数字和标点符号

# process the error docs after second-level cleaning
def clean_error_doc(error_doc):
    chi = r'([\u4E00-\u9FA5]|[0-9]|[“”、。《》！，：；？\.%])'
    pa = re.compile(chi)
    return "".join(re.findall(pa, error_doc))


# ### Step 9: 读取error_docs数据，再次执行情感分析，并且最终保存结果 
# #### （不排除一小部分文件在cleaning之后还是无法处理，但通常情况是该文档不包含以上定义的有效字符，即：中文，数字和标点符号）

error_doc_nums = pd.read_excel("error_docs.xlsx")
error_data = data[data['文档号码'].isin(error_doc_nums['文档编号'].tolist())]
error_data.head(10)

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

rest_scores.to_excel('sentiment_rest.xlsx', index=False)
