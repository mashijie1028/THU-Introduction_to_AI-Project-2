import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

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
feature = df_feature.iloc[:, 0:8].values
feature = df_feature.iloc[:, 8:16].values
feature = df_feature.iloc[:, 16:-1].values
feature = df_feature.iloc[:, 10:-1].values
# print(feature.shape)   # (7195, 11)
label = df_label.values
# print(type(label[0]))  # <class 'str'>
data_num = len(df)  # 7195
feature_num = feature.shape[1]  # 22


# K-Means model definition and metrics
# **********************************************************************************************************************
def dist_eclud(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


def init_centroid(data, k):
    """Generate k center within the range of data set."""
    n = data.shape[1]  # features
    centroids = np.zeros((k, n))  # init with (0,0)....
    for i in range(n):
        dmin, dmax = np.min(data[:, i]), np.max(data[:, i])
        centroids[:, i] = dmin + (dmax - dmin) * np.random.rand(k)
    return centroids


def KMeans(data, k):
    m = np.shape(data)[0]  # number of data
    # column1: sample in which cluster
    # column2: square dist between sample and center
    cluster_assessment = np.mat(np.zeros((m, 2)))
    cluster_change = True

    #  Init centroids
    centroids = init_centroid(data, k)
    while cluster_change:
        cluster_change = False

        # step1: for all feature points(num_row)
        for i in range(m):
            minDist = 100000.0
            minIndex = -1

            # for all centers遍
            # step2: find the nearest center
            for j in range(k):
                # calculate dist
                distance = dist_eclud(centroids[j, :], data[i, :])
                if distance < minDist:
                    minDist = distance
                    minIndex = j
            # step3: update each row of cluster
            if cluster_assessment[i, 0] != minIndex:
                cluster_change = True
                cluster_assessment[i, :] = minIndex, minDist ** 2

        # step4: update centroid
        for j in range(k):
            points_in_cluster = data[np.nonzero(cluster_assessment[:, 0].A == j)[0]]  # find all points in cluster
            if len(points_in_cluster) != 0:
                centroids[j, :] = np.mean(points_in_cluster, axis=0)  # mean in each cluster

    print("Congratulations,cluster complete!")
    return centroids, cluster_assessment


# plot 2D figure
def plot_raw_2D(result, label, k):
    number = data_num  # global variable, 7195
    Bufo_list_x = []
    Bufo_list_y = []
    Dend_list_x = []
    Dend_list_y = []
    Hyli_list_x = []
    Hyli_list_y = []
    Lept_list_x = []
    Lept_list_y = []
    centroid_list_x = []
    centroid_list_y = []

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
    for i in range(k):
        centroid_list_x.append(result[i + number, 0])
        centroid_list_y.append(result[i + number, 1])

    plt.plot(Bufo_list_x, Bufo_list_y, 'r.', label='Bufonidae')
    plt.plot(Dend_list_x, Dend_list_y, 'go', label='Dendrobatidae')
    plt.plot(Hyli_list_x, Hyli_list_y, 'b*', label='Hylidae')
    plt.plot(Lept_list_x, Lept_list_y, 'ys', label='Leptodactylidae')
    plt.plot(centroid_list_x, centroid_list_y, 'kx', label='Centroid')

    plt.legend(loc='upper right')
    plt.title('Raw data in 2D figure')


# compute metrics
def compute_SSE(cluster_assessment):
    square_dist = cluster_assessment[:, 1]
    return np.sum(square_dist.A)


def compute_centroid_type(label, cluster_assessment, k):
    type_dict = {'Bufonidae': 0, 'Dendrobatidae': 1, 'Hylidae': 2, 'Leptodactylidae': 3}
    m = cluster_assessment.shape[0]
    counter = np.zeros((k, 4))
    # centroid_type = np.zeros(k)
    for i in range(m):
        j = cluster_assessment[i, 0]
        type = label[i]
        k = type_dict[type]
        j = int(j)
        k = int(k)
        counter[j, k] += 1

    centroid_type = np.argmax(counter, axis=1)   # centroid_type: convert str to int, relationship in type_dict
    return counter, centroid_type


def compute_BSS(centroids, counter):
    k = counter.shape[0]
    global_mean = np.mean(centroids, axis=0)
    bss = 0
    for i in range(k):
        point_num = np.sum(counter[i, :])
        bss += point_num * dist_eclud(global_mean, centroids[i, :])

    return bss


def compute_entropy(counter):
    k = counter.shape[0]
    m = np.sum(counter)
    entropy = np.zeros(k)
    for i in range(k):
        for j in range(4):
            pr = counter[i, j]/np.sum(counter[i, :])
            if pr != 0:
                entropy[i] -= pr*np.log2(pr)

    entropy_total = 0.0
    for i in range(k):
        if entropy[i] != 0:
            entropy_total += float(np.sum(counter[i, :])/m*entropy[i])

    return entropy_total


def compute_purity(counter):
    k = counter.shape[0]
    m = np.sum(counter)
    purity = np.zeros(k)
    for i in range(k):
        max_pr = counter[i, np.argmax(counter[i, :])]/np.sum(counter[i, :])
        purity[i] = max_pr

    purity_total = 0.0
    for i in range(k):
        purity_total += float(np.sum(counter[i, :]) / m * purity[i])

    return purity_total


def compute_accuracy(label, cluster_assessment, centroid_type):
    m = len(label)
    k = len(centroid_type)
    # centroid_type = centroid_type.reshape(k, 1)
    type_dict = {'Bufonidae': 0, 'Dendrobatidae': 1, 'Hylidae': 2, 'Leptodactylidae': 3}
    pred_acc = 0
    for i in range(m):
        j = cluster_assessment[i, 0]
        j = int(j)
        if int(type_dict[label[i]]) == centroid_type[j]:
            pred_acc += 1

    return float(pred_acc/m)


def plot_cluster_2D(result, cluster_assessment, centroid_type):
    m = data_num   # global variable
    type_dict = {'Bufonidae': 0, 'Dendrobatidae': 1, 'Hylidae': 2, 'Leptodactylidae': 3}
    type1_list_x = []
    type1_list_y = []
    type2_list_x = []
    type2_list_y = []
    type3_list_x = []
    type3_list_y = []
    type4_list_x = []
    type4_list_y = []

    for i in range(m):  # len(result)=len(label)+k
        j = cluster_assessment[i, 0]
        j = int(j)
        type_index = centroid_type[j]
        if type_index == 0:
            type1_list_x.append(result[i, 0])
            type1_list_y.append(result[i, 1])
        elif type_index == 1:
            type2_list_x.append(result[i, 0])
            type2_list_y.append(result[i, 1])
        elif type_index == 2:
            type3_list_x.append(result[i, 0])
            type3_list_y.append(result[i, 1])
        else:
            type4_list_x.append(result[i, 0])
            type4_list_y.append(result[i, 1])

    plot_list = ['r.', 'go', 'b*', 'ys']

    plt.plot(type1_list_x, type1_list_y, plot_list[0], label='type1')
    plt.plot(type2_list_x, type2_list_y, plot_list[1], label='type2')
    plt.plot(type3_list_x, type3_list_y, plot_list[2], label='type3')
    plt.plot(type4_list_x, type4_list_y, plot_list[3], label='type4')

    plt.legend(loc='upper right')
    plt.title('Clustered data in 2D figure')


# implement and model instantiation
# **********************************************************************************************************************
k = 4
centroids, cluster_assessment = KMeans(feature, k)
# print(type(cluster_assessment))   # <class 'numpy.matrix'>

whole = np.vstack((feature, centroids))
ts = TSNE(n_components=2, init='pca', random_state=0)
result = ts.fit_transform(whole)
# print(type(centroids))   # <class 'numpy.ndarray'>
# print(type(whole))   # <class 'numpy.ndarray'>
# print(type(result))   # <class 'numpy.ndarray'>
# print(whole.shape)   # <class 'numpy.ndarray'>
# print(result.shape)   # (7199, 2)
# print(type(whole[0,0]))   # <class 'numpy.float64'>
# print(type(result[0,1]))   # <class 'numpy.float32'>
plot_raw_2D(result, label, k)
plt.show()


# evalute and print
# **********************************************************************************************************************
sse = compute_SSE(cluster_assessment)
counter, centroid_type = compute_centroid_type(label, cluster_assessment, k)
bss = compute_BSS(centroids, counter)
entropy = compute_entropy(counter)
purity = compute_purity(counter)
accuracy = compute_accuracy(label, cluster_assessment, centroid_type)
accuracy *= 100   # convert to percentage

plt.subplot(121)
plot_raw_2D(result, label, k)
plt.subplot(122)
plot_cluster_2D(result, cluster_assessment, centroid_type)
plt.show()

print('show the evaluations: ')
print('SSE: '+str(sse))
print('BSS: '+str(bss))
print('total entropy: '+str(entropy))
print('total purity: '+str(purity))
# print('clustering accuracy: '+str(accuracy))
print('clustering accuracy: %.6f ' % accuracy + '%')
