import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix

# data preprocess
# **********************************************************************************************************************
# chosen features:job/marital/education/balance/housing/loan/duration
# one-hot:job/marital/education
CSV_FILE_PATH = './data/classification/train_set.csv'
df = pd.read_csv(CSV_FILE_PATH)
df = df.sample(frac=1).reset_index(drop=True)
one_hot = pd.get_dummies(df[['job', 'marital', 'education']])
df = df.join(one_hot)
# print(df.head(5))
# print(df.shape)
# print(type(df))
# labels = df.iloc[:, -1].values
# print(labels.shape)
# print(type(labels))
# print(type(df.loc[0,'job']))

mean_balance = df['balance'].mean()
mean_duration = df['duration'].mean()


def convert_housing(housing):
    if housing == 'yes':
        return 1.0
    else:
        return 0.0


def convert_loan(loan):
    if loan == 'yes':
        return 1.0
    else:
        return 0.0


def convert_balance(balance):
    return balance / mean_balance


def convert_duration(duration):
    return duration / mean_duration


df['balance'] = df['balance'].astype('float')
df['duration'] = df['duration'].astype('float')
df['housing'] = df['housing'].apply(convert_housing)
df['loan'] = df['loan'].apply(convert_loan)
df['balance'] = df['balance'].apply(convert_balance)
df['duration'] = df['duration'].apply(convert_duration)
# print(df.iloc[0, :])

df_chosen = df[['job_admin.', 'job_blue-collar', 'job_entrepreneur', 'job_housemaid', 'job_management', 'job_retired',
                'job_self-employed', 'job_services', 'job_student', 'job_technician', 'job_unemployed', 'job_unknown',
                'marital_divorced', 'marital_married', 'marital_single', 'education_primary', 'education_secondary',
                'education_tertiary', 'education_unknown', 'balance', 'housing', 'loan', 'duration', 'y'
                ]]
# print(type(df_chosen.loc[0, 'loan']))  # <class 'numpy.float64'>
# print(type(df_chosen.loc[0, 'job_admin.'])) # <class 'numpy.uint8'>
df_chosen = df_chosen.iloc[:, :].astype('float')
# print(df_chosen.head(5))
# print(type(df_chosen.loc[0, 'loan']))  # <class 'numpy.float64'>
# print(type(df_chosen.loc[0, 'job_admin.'])) # <class 'numpy.float64'>

data = df_chosen.values
# print(data.shape) (25317, 24)
num_data, num_feature = data.shape
num_feature -= 1

train_set = data[0:20000, :]
test_set = data[20000:num_data, :]
train_x = train_set[:, 0:num_feature]
train_y = train_set[:, -1].astype(np.int32)
test_x = test_set[:, 0:num_feature]
test_y = test_set[:, -1].astype(np.int32)

# SVM settings and fit
# classifier = svm.SVC(kernel='linear')
classifier = svm.SVC(kernel='rbf')
# classifier = svm.SVC(kernel='poly', degree=8)
# classifier = svm.SVC(kernel='sigmoid')
classifier.fit(train_x, train_y)

train_pred = classifier.predict(train_x)
test_pred = classifier.predict(test_x)
print(len(test_y))   # 5317
print(len(test_pred))   # 5317

# show accuracy
epsilon = 1e-6


def show_acc(pred, label):
    size = len(pred)
    print('%.6f ' % (float(np.sum(pred == label) / size) * 100) + '%')


def show_one(pred):
    print(np.sum((pred == 1)))


print('training set accuracy: ')
show_acc(train_pred, train_y)
print('test set accuracy: ')
show_acc(test_pred, test_y)

# show other metrics
print('training set: ')
print(confusion_matrix(train_y, train_pred))
print(classification_report(train_y, train_pred))
print('test set: ')
print(confusion_matrix(test_y, test_pred))
print(classification_report(test_y, test_pred))

show_one(train_pred) # 538
show_one(test_pred) # 151

# print(test_pred.dtype)   # int32
