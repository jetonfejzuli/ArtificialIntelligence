import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'

from sklearn.tree import DecisionTreeClassifier

from dataset_script import dataset

if __name__ == '__main__':
    p = int(input())
    c = input()
    l = int(input())

    train_set = dataset[:int(len(dataset) * p / 100)]
    test_set = dataset[int(len(dataset) * p / 100):]

    train_x = [row[:-1] for row in train_set]
    train_y = [row[-1] for row in train_set]

    test_x = [row[:-1] for row in test_set]
    test_y = [row[-1] for row in test_set]

    classifier1 = DecisionTreeClassifier(criterion=c, max_leaf_nodes=l, random_state=0)
    classifier1.fit(train_x, train_y)

    acc1 = 0
    for x, y in zip(test_x, test_y):
        pred = classifier1.predict([x])[0]
        if pred == y:
            acc1 += 1
    acc1 = acc1 / len(test_x)

    print(f'Accuracy with the original classifier: {acc1}')

    train_y_perch = [1 if row == 'Perch' else 0 for row in train_y]
    test_y_perch = [1 if row == 'Perch' else 0 for row in test_y]

    train_y_roach = [1 if row == 'Roach' else 0 for row in train_y]
    test_y_roach = [1 if row == 'Roach' else 0 for row in test_y]

    train_y_bream = [1 if row == 'Bream' else 0 for row in train_y]
    test_y_bream = [1 if row == 'Bream' else 0 for row in test_y]

    classifier_perch = DecisionTreeClassifier(criterion=c, max_leaf_nodes=l, random_state=0)
    classifier_perch.fit(train_x, train_y_perch)

    classifier_roach = DecisionTreeClassifier(criterion=c, max_leaf_nodes=l, random_state=0)
    classifier_roach.fit(train_x, train_y_roach)

    classifier_bream = DecisionTreeClassifier(criterion=c, max_leaf_nodes=l, random_state=0)
    classifier_bream.fit(train_x, train_y_bream)

    acc2 = 0

    for i in range(len(test_x)):
        pred_perch = classifier_perch.predict([test_x[i]])[0]
        pred_roach = classifier_roach.predict([test_x[i]])[0]
        pred_bream = classifier_bream.predict([test_x[i]])[0]
        y = test_y[i]
        if y == 'Perch':
            if pred_perch == 1 and pred_roach == 0 and pred_bream == 0:
                acc2 += 1
        elif y == 'Roach':
            if pred_perch == 0 and pred_roach == 1 and pred_bream == 0:
                acc2 += 1
        elif y == 'Bream':
            if pred_perch == 0 and pred_roach == 0 and pred_bream == 1:
                acc2 += 1
    acc2 = acc2 / len(test_x)

    print(f'Accuracy with the ensemble classifier: {acc2}')
