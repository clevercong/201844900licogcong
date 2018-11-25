import vector
from sklearn.metrics import accuracy_score
from collections import Counter
import math
from sklearn.model_selection import train_test_split

#获取到预处理的数据
def handle_data():
    data, train_Y = vector.handle_data(base_path='data/20news-18828', out1='out/train_X.csv',
                                      out2='out/train_Y.csv')

    dictionary = vector.createdict(data, 'out/dictionary.csv')
    vsm_train = vector.vsm(data, dictionary)
    return dictionary,vsm_train, train_Y

def count(documents, labels, dictionary):
    # 每类中单词的数量
    keyName = []
    #训练数据集中所有的单词数量
    all = []
    # 每类数量
    kind = []
    for i in range(len(documents)):
        # 只保留词典中出现的token
        document = list(filter(lambda token: token in dictionary, documents[i]))
        all += document
        if labels[i] < len(keyName):
            keyName[int(labels[i])] += document
            kind[int(labels[i])] += 1
        else:
            keyName.append(document)
            kind.append(1)
        print(i)
    return keyName, kind, all

#计算测试文档属于某个类的概率p(cate|doc)=p(w1,w2,...|cate)*p(cate)
#多项式模型+平滑技术：
#p(word|cate)=(类cate下单词word出现的次数+1)/(类cate下单词总数+训练数据中不重复的单词总数)
#p(cate)=类cate下单词总数/训练数据中单词总数
def NBprocess(test_X, test_Y, keyName, kind, all, dictionary):
    print('NBprocess')
    counter0 = []
    counter1 = Counter(all)
    for kind in range(20):
        counter0.append(Counter(keyName[kind]))
    prediction = []
    for i in range(len(test_X)):
        pre = []
        for kind in range(20):
            # P(具有某特征|属于某类)
            P1 = 0
            # P(具有某特征)
            P2 = 0
            features = counter0[kind]
            for token in test_X[i]:
                if token in dictionary:
                    P0 = math.log((features[token] + 1) / (len(keyName[kind]) + len(dictionary)))
                    P1 += P0
                    P0 = math.log((counter1[token] + 1) / (len(all[kind]) + len(dictionary)))
                    P2 += P0
            # P('属于某类')
            Pk = math.log(kind[kind] / 18828.0)
            # P('属于某类'|'具有某特征')
            Pki = P1 + Pk - P2
            pre.append([kind, Pki])
        pre = sorted(pre, key=lambda item: -item[1])
        print(i, pre[0][0])
        prediction.append(pre[0][0])
    # 输出准确率
    print("the accuracy is:\t", accuracy_score(test_Y, prediction))


if __name__ == '__main__':
    dictionary, train_samples, train_labels = handle_data()
    train_x, test_x, train_y, test_y = train_test_split(train_samples, train_labels, test_size=0.2, random_state=1)
    # 对train set预处理
    keyName, kind, all = count(train_x, train_y, dictionary)
    # 分类
    NBprocess(test_x, test_y, keyName, kind, all, dictionary)

