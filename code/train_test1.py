import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 准备数据
# 假设我们已经有一个包含问题的DataFrame，列名为'text'
data = pd.DataFrame({
    'text': ['如何看待李群、李代数在机器人控制中的应用？',
        '如何在一个月内入门李群？',
        '介绍一些知识图谱的实际应用类项目',
        'NLP、KG相关学习资源汇总',
        '聚类 · 机器学习数学基础',
        '文本分类算法综述',
        'Transformer模型之Encoder结构在文本分类中的应用',
        '线性代数极简史',
        '如何对积分求导？',
        '奇异值的物理意义是什么？',
        '香农的信息论究竟牛在哪里？'],  # 将上述问题文本填入
})

# 2. 数据预处理
# 去除停用词，标点符号等
stop_words = set(stopwords.words('chinese'))
def preprocess_text(text):
    words = word_tokenize(text)
    return ' '.join([word for word in words if word.isalnum() and word not in stop_words])

data['text'] = data['text'].apply(preprocess_text)

# 3. 特征提取
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(data['text'])

# 4. 聚类
num_clusters = 3  # 假设我们希望聚成3类，可以根据实际需求调整
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X)
data['cluster'] = kmeans.labels_

# 5. 可视化结果（可选，用于观察聚类效果）
sns.countplot(data['cluster'])
plt.title('Cluster Distribution')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.show()

# 6. 输出聚类结果
print(data[['text', 'cluster']])
