import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
from submission_script import *
from dataset_script import dataset
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OrdinalEncoder

# This is a dataset sample, for training and evaluation use the
# imported dataset variable
dataset_sample = [['1', '35', '12', '5', '1', '100', '0'],
                  ['1', '29', '7', '5', '1', '96', '1'],
                  ['1', '50', '8', '1', '3', '132', '0'],
                  ['1', '32', '11.75', '7', '3', '750', '0'],
                  ['1', '67', '9.25', '1', '1', '42', '0']]

if __name__ == '__main__':
    # Your code here
    input_record = input().strip().split()
    input_record = [[float(x) for x in input_record]]
    split_idx = int(0.85 * len(dataset))
    train_set = dataset[:split_idx]
    test_set = dataset[split_idx:]
    train_x = [[float(v) for v in row[:-1]] for row in train_set]
    train_y = [row[-1] for row in train_set]
    test_x = [[float(v) for v in row[:-1]] for row in test_set]
    test_y = [row[-1] for row in test_set]


    # At the end you are required to submit the dataset,
    # the classifier and the encoder via calling the folllowing functions

    classifier = GaussianNB()
    classifier.fit(train_x, train_y)

    correct = 0
    for x, y_true in zip(test_x, test_y):
        y_pred = classifier.predict([x])[0]
        if y_pred == y_true:
            correct += 1
    accuracy = correct / len(test_set)

    predicted_class = classifier.predict(input_record)[0]

    # Print results
    class_probs = classifier.predict_proba(input_record)[0]
    print(accuracy)
    print(predicted_class)
    import numpy as np

    np.set_printoptions(precision=8, suppress=True)
    print(np.array([class_probs]))

    # Submit results
    # submit_train_data(train_x, train_y)
    # submit_test_data(test_x, test_y)
    # submit_classifier(classifier)
    # submit the train dataset
    # submit_train_data(train_X, train_Y)

    # submit the test dataset
    # submit_test_data(test_X, test_Y)

    # submit the classifier
    # submit_classifier(classifier)

    # re-import at the end / do not remove the following line
