from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans,AffinityPropagation,MeanShift,SpectralClustering,DBSCAN,AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.feature_extraction.text import TfidfVectorizer  # 将单词进行向量化
from nltk.tokenize import word_tokenize
import warnings
import collections
import json
#def read_data(path='Tweets.txt'):
 #   documents = []
  #  labels = []
   # with open(path) as file:
    #    lines = file.read().split('\n')[:-1]
     #   for line in lines:
      #      line = json.loads(str(line))
       #     documents.append(line['text'].split())
        #    labels.append(line['cluster'])
    #return documents, labels
 # K-means算法
def t_kmeans(x,y,k):
    km = KMeans(n_clusters=k)
    result_kmeans = km.fit_predict(x)
    print('K-means的准确率:', normalized_mutual_info_score(result_kmeans, y))

    # AffinityPropagation算法
def t_AffinityPropagation(x,y,k):
    ap = AffinityPropagation(damping=0.55, max_iter=575, convergence_iter=575, copy=True, preference=None,
                             affinity='euclidean', verbose=False)
    result_ap = ap.fit_predict(x)
    print('AffinityPropagation算法的准确率:', normalized_mutual_info_score(result_ap, y))

    # meanshift算法
def t_MeanShift(x, y, k):
    ms = MeanShift(bandwidth=0.65, bin_seeding=True)
    result_ms = ms.fit_predict(x)
    print('meanshift算法的准确率:', normalized_mutual_info_score(result_ms, y))

    # SpectralClustering算法
def t_SpectralClustering(x, y, k):
    sc = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', n_neighbors=4, eigen_solver='arpack', n_jobs=1)
    result_sc = sc.fit_predict(x)
    print('SpectralClustering算法的准确率:', normalized_mutual_info_score(result_sc, y))

    # DBSCAN算法
def t_DBSCAN(x, y, k):
    db = DBSCAN(eps=0.7, min_samples=1)
    result_db = db.fit_predict(x)
    print('DBSCAN算法的准确率:', normalized_mutual_info_score(result_db, y))

    # AgglomerativeClustering算法
def t_AgglomerativeClustering(x, y, k):
    ac = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='ward')
    result_ac = ac.fit_predict(x)
    print('AgglomerativeClustering算法的准确率:', normalized_mutual_info_score(result_ac, y))

    # GaussianMixture算法
def t_GaussianMixture(x, y, k):
    gm = GaussianMixture(n_components=k, covariance_type='diag', max_iter=20, random_state=0)
    gm.fit(x)
    result_gm = gm.predict(x)
    print('GaussianMixture算法的准确率:', normalized_mutual_info_score(result_gm, y))

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    word_list=[]
    label_list=[]
    Data=[]
    for line in open('Tweets.txt', 'r').readlines():
        dic = eval(line)
        word_list.append(dic["text"])
        label_list.append(dic["cluster"])
    print(label_list)
    tfidfvectorizer = TfidfVectorizer(tokenizer=word_tokenize, stop_words='english')  # 初始化分词器
    # tfidfvectorizer = TfidfVectorizer(stop_words='english')  # 初始化分词器
    Data = tfidfvectorizer.fit_transform(word_list).toarray() # 将文本转化为向量
    vectorizer=CountVectorizer()
    #该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j],表示j词在i类文本下的词频
    transformer=TfidfTransformer()
    #该类会统计每个词语的tf-idf权值
    #tfidf=transformer.fit_transform(vectorizer.fit_transform(Data))
    #第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
   # weight=tfidf.toarray()
    k=len(collections.Counter(label_list))
    print(k)
    t_kmeans(Data,label_list,k)
    t_AffinityPropagation(Data, label_list, k)
    t_MeanShift(Data, label_list, k)
    t_SpectralClustering(Data, label_list, k)
    t_DBSCAN(Data, label_list, k)
    t_AgglomerativeClustering(Data,label_list,k)
    t_GaussianMixture(Data,label_list,k)