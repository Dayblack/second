#!/usr/bin/env python
# coding: utf-8

# In[1]:


#针对信用卡违约率进行分析
"""目标：针对这个数据集构建一个分析信用卡违约率的分类器"""


# In[2]:


"""流程简介
1.加载数据
2.探索数据，采用数据可视化方式对数据建立更直观的了解；数据集需要手动划分为训练集与测试集
3.数据规范化与分类，使用Pipeline管道，将数据规范化设置为第一步，分类作为第二步。待定模型选取
SVM、决策树、随机森林、KNN，并通过GridSearchCV找到最优超参数与最优分数。
"""


# In[3]:


import pandas as pd
from sklearn.model_selection import learning_curve, train_test_split,GridSearchCV
#导入特征缩放模块
from sklearn.preprocessing import StandardScaler
#导入管道
from sklearn.pipeline import Pipeline
#导入评价指标模块
from sklearn.metrics import accuracy_score
#导入四种模型
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
#导入可视化模块
from matplotlib import pyplot as plt
import seaborn as sns
# 数据加载
data = data = pd.read_csv(r'C:\Users\ASUS\Desktop\data\UCI_Credit_Card.csv')


# In[5]:


#数据探索
print(data.shape)
print(data.describe())


# In[9]:


#查看下个月违约情况
next_month = data['default.payment.next.month'].value_counts()
print(next_month)
df = pd.DataFrame({'default.payment.next.month': next_month.index,'values': next_month.values})


# In[10]:


#可视化
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.figure(figsize = (6,6))
plt.title('信用卡违约率客户\n (违约：1，守约：0)')
sns.set_color_codes("pastel")
sns.barplot(x = 'default.payment.next.month', y="values", data=df)
locs, labels = plt.xticks()
plt.show()


# In[11]:


#特征选择：1.去掉ID；2.其他暂无
data.drop(['ID'], inplace=True, axis =1)


# In[12]:


#划分train与test
target = data['default.payment.next.month'].values
columns = data.columns.tolist()
columns.remove('default.payment.next.month')
features = data[columns].values
# 30%作为测试集，其余作为训练集
train_x, test_x, train_y, test_y = train_test_split(features, target, test_size=0.30, stratify = target, random_state = 1)
 


# In[14]:


# 构造各种分类器
classifiers = [
    SVC(random_state = 1, kernel = 'rbf'),    
    DecisionTreeClassifier(random_state = 1, criterion = 'gini'),
    RandomForestClassifier(random_state = 1, criterion = 'gini'),
    KNeighborsClassifier(metric = 'minkowski'),
]
# 分类器名称
classifier_names = [
            'svc', 
            'decisiontreeclassifier',
            'randomforestclassifier',
            'kneighborsclassifier',
]
# 分类器参数
classifier_param_grid = [
            {'svc__C':[1], 'svc__gamma':[0.01]},
            {'decisiontreeclassifier__max_depth':[6,9,11]},
            {'randomforestclassifier__n_estimators':[3,5,6]} ,
            {'kneighborsclassifier__n_neighbors':[4,6,8]},
]


# In[15]:


# 对具体的分类器进行GridSearchCV参数调优
def GridSearchCV_work(pipeline, train_x, train_y, test_x, test_y, param_grid, score = 'accuracy'):
    response = {}
    gridsearch = GridSearchCV(estimator = pipeline, param_grid = param_grid, scoring = score)
    # 寻找最优的参数和最优的准确率分数
    search = gridsearch.fit(train_x, train_y)
    print("GridSearch最优参数：", search.best_params_)
    print("GridSearch最优分数： %0.4lf" %search.best_score_)
    predict_y = gridsearch.predict(test_x)
    print("准确率 %0.4lf" %accuracy_score(test_y, predict_y))
    response['predict_y'] = predict_y
    response['accuracy_score'] = accuracy_score(test_y,predict_y)
    return response


# In[16]:


for model, model_name, model_param_grid in zip(classifiers, classifier_names, classifier_param_grid):
    pipeline = Pipeline([
            ('scaler', StandardScaler()),
            (model_name, model)
    ])
    result = GridSearchCV_work(pipeline, train_x, train_y, test_x, test_y, model_param_grid , score = 'accuracy')


# In[ ]:


#结论
#SVM 分类器的准确率最高，测试准确率为 0.8172

