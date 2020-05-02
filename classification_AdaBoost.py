import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier

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


# AdaBoost settings and train
classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                                algorithm="SAMME",
                                n_estimators=200)
classifier.fit(train_x, train_y)

# evaluate
test_pred = classifier.predict(test_x)
print("AdaBoost")
print(classification_report(test_y, test_pred))
print("acc: ", accuracy_score(test_y, test_pred))
