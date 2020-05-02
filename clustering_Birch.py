import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import Birch
from sklearn import metrics


# data preprocessing
# **********************************************************************************************************************
CSV_FILE_PATH = './data/clustering/Frogs_MFCCs.csv'
df = pd.read_csv(CSV_FILE_PATH)
df = df.sample(frac=1).reset_index(drop=True)
# print(df.head(5))    # MFCCs_1-MFCCs_22: dtype=float64
df_feature = df.iloc[:, 0:22]
# df_label = df.iloc[:, 22]
df_label = df['Family']
# print(df_feature.head(5))
# print(df_label.head(5))   # Name: Family, dtype: object
# print(df_label.shape)   # (7195,)
# print(type(df_label[0]))   # <class 'str'>

feature = df_feature.values
# feature = df_feature.iloc[:, 0:8].values
# feature = df_feature.iloc[:, 8:16].values
# feature = df_feature.iloc[:, 16:].values
# feature = df_feature.iloc[:, 10:].values
# print(feature.shape)   # (7195, 12)
label = df_label.values
# print(type(label))   # <class 'numpy.ndarray'>
# print(label.shape)  # (7195,)
# label = label.astype(np.str)
print(type(label[0]))  # <class 'str'> (if astype(np.str): <class 'numpy.str_'>)
data_num = len(df)  # 7195
feature_num = feature.shape[1]  # 22


# Birch settings and model instantiation
# **********************************************************************************************************************
k = 4
model = Birch(n_clusters=k)
y_pred = model.fit_predict(feature)
print('Congratulations,cluster complete!')
# print(y_pred.shape)   # (7195,)
# print(type(y_pred))   # <class 'numpy.ndarray'>
# print(type(y_pred[0]))   # <class 'numpy.int64'>
print(np.unique(y_pred))  # [0 1 2 3]

# count data in each clusters after Birch
counter = np.zeros(k, dtype=np.int)  # k=len(np.unique(y_pred))
# for i in np.unique(y_pred):
for i in range(4):
    counter[i] = np.sum(y_pred == i)
print(counter)
order = np.argsort(counter)
rank = [0, 0, 0, 0]
for index in range(4):
    rank[order[index]] = index
print(rank)


# visualization definition
# **********************************************************************************************************************
def plot_raw_2D(result, label):
    number = data_num  # global variable, 7195
    Bufo_list_x = []
    Bufo_list_y = []
    Dend_list_x = []
    Dend_list_y = []
    Hyli_list_x = []
    Hyli_list_y = []
    Lept_list_x = []
    Lept_list_y = []

    for i in range(number):  # len(result)=len(label)+k
        if label[i] == 'Bufonidae':
            Bufo_list_x.append(result[i, 0])
            Bufo_list_y.append(result[i, 1])
        elif label[i] == 'Dendrobatidae':
            Dend_list_x.append(result[i, 0])
            Dend_list_y.append(result[i, 1])
        elif label[i] == 'Hylidae':
            Hyli_list_x.append(result[i, 0])
            Hyli_list_y.append(result[i, 1])
        else:
            Lept_list_x.append(result[i, 0])
            Lept_list_y.append(result[i, 1])

    plt.plot(Bufo_list_x, Bufo_list_y, 'r.', label='Bufonidae')
    plt.plot(Dend_list_x, Dend_list_y, 'go', label='Dendrobatidae')
    plt.plot(Hyli_list_x, Hyli_list_y, 'b*', label='Hylidae')
    plt.plot(Lept_list_x, Lept_list_y, 'ys', label='Leptodactylidae')

    plt.legend(loc='upper right')
    plt.title('Raw data in 2D figure')


def plot_cluster_2D(result, pred, rank):
    number = data_num  # global variable, 7195
    type1_list_x = []
    type1_list_y = []
    type2_list_x = []
    type2_list_y = []
    type3_list_x = []
    type3_list_y = []
    type4_list_x = []
    type4_list_y = []

    for i in range(number):  # len(result)=len(label)+k
        if pred[i] == 0:
            type1_list_x.append(result[i, 0])
            type1_list_y.append(result[i, 1])
        elif pred[i] == 1:
            type2_list_x.append(result[i, 0])
            type2_list_y.append(result[i, 1])
        elif pred[i] == 2:
            type3_list_x.append(result[i, 0])
            type3_list_y.append(result[i, 1])
        else:
            type4_list_x.append(result[i, 0])
            type4_list_y.append(result[i, 1])

    plot_list = ['r.', 'go', 'b*', 'ys']

    plt.plot(type1_list_x, type1_list_y, plot_list[rank[0]], label='type1')
    plt.plot(type2_list_x, type2_list_y, plot_list[rank[1]], label='type2')
    plt.plot(type3_list_x, type3_list_y, plot_list[rank[2]], label='type3')
    plt.plot(type4_list_x, type4_list_y, plot_list[rank[3]], label='type4')

    plt.legend(loc='upper right')
    plt.title('Clustered data in 2D figure')


# evaluation and metrics
# **********************************************************************************************************************
ts = TSNE(n_components=2, init='pca', random_state=0)
result = ts.fit_transform(feature)

# figure in 2D
plt.subplot(121)
plot_raw_2D(result, label)
plt.subplot(122)
plot_cluster_2D(result, y_pred, rank)
plt.show()


# numerical metrics
# **********************************************************************************************************************z
def compute_accuracy(y_pred, rank, label):
    m = len(y_pred)
    pred_label = np.zeros(m, dtype=np.str)
    pred_index = np.zeros(m)
    label_index = np.zeros(m)
    type_list = ['Bufonidae', 'Dendrobatidae', 'Hylidae', 'Leptodactylidae']
    type_dict = {'Bufonidae': 0, 'Dendrobatidae': 1, 'Hylidae': 2, 'Leptodactylidae': 3}
    count = 0
    # label = label.astype(np.str)
    # pred_label = np.squeeze(pred_label)
    # label = np.squeeze(label)
    for i in range(m):
        pred_label[i] = type_list[rank[y_pred[i]]]
        pred_index[i] = rank[y_pred[i]]
        label_index[i] = type_dict[label[i]]
        if type_list[rank[y_pred[i]]] == label[i]:
            count += 1
    # return count/m
    # return float(np.sum(pred_label == label) / m)
    return float(np.sum(pred_index == label_index) / m)


acc = compute_accuracy(y_pred, rank, label)
acc *= 100   # convert to percentage
print('clustering accuracy: %.6f ' % acc + '%')
ch_score = metrics.calinski_harabasz_score(feature, y_pred)
print("Calinski_Harabasz Score: "+str(ch_score))
