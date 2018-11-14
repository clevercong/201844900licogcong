import os
import nltk
import string
import imp
from collections import Counter
from nltk.stem.porter import *
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize,word_tokenize

f_path = "./20news-bydate-train"
#数据导入
def handle_data(base_path='data/20news-18828',  out1='out/datas.csv', out2='out/labels.csv'):
  print("数据导入并预处理中")
  datas = []
  labels=[]
  if not os.path.exists(out1):
    i=0
    j=-1
    for folder in os.listdir(f_path):
       path = os.path.join(f_path,folder)
       j+=1
       for filename in os.listdir(path):
           filepath = os.path.join(path,filename)
           labels.append(j)
           with open(filepath,'r',encoding="ISO-8859_1") as f:
               data=f.read()
               datas.append(prohandle_data(data))
           i+=1
           print(i)
           f.close()
    das=[str(da)for da in datas]
    pd.DataFrame(das).to_csv(out1,sep=" ",header=None,index=None)
    pd.DataFrame(labels).to_csv(out2, sep=" ", header=None, index=None)
  else:
      datas=np.array(pd.read_csv(out1,sep=" ",header=None))
      datas = datas[0].replace("\n", "").replace("\r", "").replace("\t", "")
      labels=np.array(pd.read_csv(out2, sep=" ", header=None))

  return datas,labels


#预处理数据
def prohandle_data(datas):
    prodatas = []
    # 预处理：小写转换，去除特殊字符，去除停用词，词干提取
    lowers = str(datas).lower()
    remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
    no_punctuation = lowers.translate(remove_punctuation_map)
    tokens = nltk.word_tokenize(no_punctuation)
    list_stopWords = list(set(stopwords.words('english')))
    # print(list_stopWords)
    filtered_words = [w for w in tokens if not w in list_stopWords]
    stemmer = PorterStemmer()
    stemmers = []
    for item in filtered_words:
        stemmers.append(stemmer.stem(item))
        prodatas.append(stemmers)
    return prodatas

# 生成词典
def createdict(datas,out1='out/dict.csv'):
    dict = []
    print('创建字典中')
    if not os.path.exists('out/dict.csv'):
        count= []
        for data in datas:
            count += data
        count = Counter(count)
        for token in count:
            # 过滤词频
            if count[token] >= 4 and token not in dict:
                dict.append(str(token))
        pd.DataFrame(dict).to_csv(out1, sep=" ", header=None, index=None)
    else:
        dict = np.array(pd.read_csv(out1, sep=" ", header=None)).reshape(1, -1)[0]
    return dict


# 生成向量
def vsm(datas, dict):
    print('创建0-1向量')
    if not os.path.exists('out/vsm.csv'):
      vectors = []
      i = 0
      for data in datas:
          vector = []
          for item in dict:
              if item in datas:
                  vector.append(1)
              else:
                vector.append(0)
          vectors.append(vector)
          i += 1
          print(i)
      pd.DataFrame(vectors).to_csv('out/vsm.csv', sep=" ", header=None, index=None)
    else:
      print('向量模型已存在')



if  __name__=='__main__':
    datas,labels=handle_data()
    dict=createdict(datas)
    vsm(datas,dict)
