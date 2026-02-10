import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'

from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from dataset_script import dataset

if __name__ == '__main__':
    dataset = [[row[0] + row[10]] + [el for ind, el in enumerate(row) if ind not in [0, 10]] for row in dataset]
    data_class_good = [row for row in dataset if row[-1] == 'good']
    data_class_bad = [row for row in dataset if row[-1] == 'bad']
    c = int(input())
    p = int(input())
    if c == 0:
        train_set = data_class_good[:int(len(data_class_good) * p / 100)] + \
                    data_class_bad[:int(len(data_class_bad) * p / 100)]
        test_set = data_class_good[int(len(data_class_good) * p / 100):] + \
                   data_class_bad[int(len(data_class_bad) * p / 100):]
        train_x = [row[:-1] for row in train_set]
        test_x = [row[:-1] for row in test_set]
        train_y = [row[-1] for row in train_set]
        test_y = [row[-1] for row in test_set]
    else:
        train_set = data_class_good[int(len(data_class_good) * (100 - p) / 100):] + \
                    data_class_bad[int(len(data_class_bad) * (100 - p) / 100):]
        test_set = data_class_good[:int(len(data_class_good) * (100 - p) / 100)] + \
                   data_class_bad[:int(len(data_class_bad) * (100 - p) / 100)]
        train_x = [row[:-1] for row in train_set]
        test_x = [row[:-1] for row in test_set]
        train_y = [row[-1] for row in train_set]
        test_y = [row[-1] for row in test_set]

    classifier1 = GaussianNB()
    classifier1.fit(train_x, train_y)

    acc1 = 0

    for x, y in zip(test_x, test_y):
        pred = classifier1.predict([x])[0]
        if pred == y:
            acc1 += 1
    acc1 = acc1 / len(test_x)
    print(f'Accuracy with a sum of columns: {acc1}')

    minMax = MinMaxScaler([-1, 1])
    minMax.fit(train_x)

    classifier2 = GaussianNB()
    classifier2.fit(minMax.transform(train_x), train_y)

    acc2 = 0

    for x, y in zip(test_x, test_y):
        pred = classifier2.predict(minMax.transform([x]))[0]
        if pred == y:
            acc2 += 1
    acc2 = acc2 / len(test_x)
    print(f'Accuracy with a sum of columns and feature scaling: {acc2}')
