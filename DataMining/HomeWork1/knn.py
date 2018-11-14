import imp
import vector
import numpy as np
from sklearn.model_selection import train_test_split
import heapq
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
#from sklearn.metrics.pairwise import euclidean_distances
#from sklearn.metrics.pairwise import cosine_distances
import test1


f_path='data/20news-18828'
def handle_data():
    data, train_Y = vector.handle_data(base_path='data/20news-18828', out1='out/train_X.csv',
                                      out2='out/train_Y.csv')

    dictionary = vector.createdict(data, 'out/dictionary.csv')
    vsm_train = vector.vsm(data, dictionary)
    return vsm_train, train_Y

#计算欧式距离
#def classify(trainSet, testSet, labels, k):
    #dataSize = testSet.shape[0]
    #diffMat = np.tile(trainSet, (dataSize,1)) -testSet
    #sqdiffMat = diffMat ** 2
    #sqDistance = sqdiffMat.sum(axis = 1)
    #distances = sqDistance ** 0.5
    #sortedDist = distances.argsort()
    #classCount ={}
    #for i in range(k):
        #voteIlable = labels[sortedDist[i]]
        #classCount[voteIlable] = classCount.get(voteIlable, 0) + 1
    #sortedClassCount = sorted(classCount.items(),
     #key=operator.itemgetter(1), reverse=True)
    #return sortedClassCount[0][0]
# cosine distance
#def cal_cosine(train_item, test_item):
    #vec_train = np.array(train_item)
    #vec_test = np.array(test_item)
    #distance = np.dot(vec_train, vec_test) / (np.linalg.norm(vec_train) * (np.linalg.norm(vec_test)))
    #return distance


def KNN(path, K):
    print('hello')
    train_samples, train_labels = handle_data()
    train_x, test_x, train_y, test_y = train_test_split(train_samples, train_labels, test_size=0.2, random_state=1)
    print(len(train_x), len(train_y), len(test_x), len(test_y))
    labels_true = []
    labels_pre = []
    print('计算余弦相似度中')
    dis = cosine_similarity(test_x, train_x)
    print(len(dis))
    for index in range(len(dis)):
        labels_true.append(test_y[index])
        dis_array = dis[index]
        dis_array = dis_array.tolist()
        min_distances = map(dis_array.index, heapq.nsmallest(K, dis_array))
        labels = []
        for item in list(min_distances):
            pre_test_label = train_y[item]
            labels.append(pre_test_label)
        label_counts = Counter(labels)
        top_one = label_counts.most_common(1)
        labels_pre.append(top_one[0][0])
        # print(top_one[0][0])
    # 计算分类准确率
    right = 0
    for i in range(len(labels_pre)):
        if labels_pre[i] == labels_true[i]:
            right += 1
    acc = right / len(labels_pre)
    print("When k is " + str(K) + " ,the succession of knn is :" + str(acc))
# 计算succesion
if __name__ == "__main__":
    for i in range(1, 51):
        KNN(f_path, i)






