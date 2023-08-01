from gensim.models.ldamodel import LdaModel
from gensim.corpora import Dictionary
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.cluster import KMeans

# 给定的文本数据
data = ['如何看待李群、李代数在机器人控制中的应用？',
        '如何在一个月内入门李群？',
        '介绍一些知识图谱的实际应用类项目',
        'NLP、KG相关学习资源汇总',
        '聚类 · 机器学习数学基础',
        '文本分类算法综述',
        'Transformer模型之Encoder结构在文本分类中的应用',
        '线性代数极简史',
        '如何对积分求导？',
        '奇异值的物理意义是什么？',
        '香农的信息论究竟牛在哪里？']

# 使用TF-IDF向量化文本数据
vectorizer = TfidfVectorizer(max_features=500, max_df=0.85, stop_words=None)
X = vectorizer.fit_transform(data)

# 使用KMeans聚类算法将文本进行分组
n_clusters = 6
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X)
# 打印每个类别的标签及其文本
labels = kmeans.labels_

# 将文本数据转换为gensim的字典格式
dictionary = Dictionary([text.split() for text in data])

# 将文本数据转换为gensim的语料格式
corpus = [dictionary.doc2bow(text.split()) for text in data]

# 使用LDA算法从文本数据中自动提取主题
num_topics = len(labels)  # 根据给定标签的数量设置主题的数量
lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=42)

# 打印每个主题作为标签
def extract_topics(lda_model, dictionary, n_words=4):
    topics = []
    for topic_id in range(len(labels)):
        topic_keywords = [word for word, _ in lda_model.show_topic(topic_id, topn=n_words)]
        topics.append(topic_keywords)
    return topics

# 提取每个主题作为标签
extracted_labels = extract_topics(lda_model, dictionary)
print('extracted_labels', extracted_labels)
print('labels', labels)

# 打印每个类别的标签及其文本
# for i, label in enumerate(labels):
#     print(np.argmax(lda_model.get_document_topics(corpus), axis=1))
#     print(i)
#     cluster_texts = np.array(data)[np.argmax(lda_model.get_document_topics(corpus), axis=1)[1] == i]
#     print(f"Cluster {label} (Size: {len(cluster_texts)}):")
#     print(cluster_texts)
#     print(f"Cluster {label} Labels:")
#     print(extracted_labels[i])
#     print()
