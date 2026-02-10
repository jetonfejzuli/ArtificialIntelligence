import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
from submission_script import *
from dataset_script import dataset
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier

# This is a sample from the dataset, for training/evaluation use the imported variable dataset
dataset_sample = [[180.0, 23.6, 25.2, 27.9, 25.4, 14.0, 'Roach'],
                  [12.2, 11.5, 12.2, 13.4, 15.6, 10.4, 'Smelt'],
                  [135.0, 20.0, 22.0, 23.5, 25.0, 15.0, 'Perch'],
                  [1600.0, 56.0, 60.0, 64.0, 15.0, 9.6, 'Pike'],
                  [120.0, 20.0, 22.0, 23.5, 26.0, 14.5, 'Perch']]

if __name__ == '__main__':
    # Your code here
    split_index = int(0.85 * len(dataset))
    col_index = int(input())
    num_decision_trees = int(input())
    criterion = str(input())
    input_record = input().strip().split()
    input_record = [float(x) for x in input_record]
    input_record = [v for i, v in enumerate(input_record) if i != col_index]

    train_set = dataset[:split_index]
    train_x = [[float(v) for i, v in enumerate(row[:-1]) if i != col_index] for row in train_set]
    train_y = [row[-1] for row in train_set]

    test_set = dataset[split_index:]
    test_x = [[float(v) for i, v in enumerate(row[:-1]) if i != col_index] for row in test_set]
    test_y = [row[-1] for row in test_set]

    classifier = RandomForestClassifier(n_estimators=num_decision_trees, criterion=criterion, random_state=0)
    classifier.fit(train_x, train_y)

    accuracy = 0
    for i in range(len(test_set)):
        predicted = classifier.predict([test_x[i]])[0]
        true = test_y[i]
        if predicted == true:
            accuracy += 1

    accuracy = accuracy / len(test_set)

    print(f'Accuracy: {accuracy}')
    predicted_class = classifier.predict([input_record])[0]
    print(predicted_class)
    class_probs = classifier.predict_proba([input_record])[0]
    print(print(class_probs))
    # At the end you need to submit the dataset, classifier and encoder by calling the following functions

    # submit the training set
    submit_train_data(train_x, train_y)

    # submit the test set
    submit_test_data(test_x, test_y)

    # submit the classifier
    submit_classifier(classifier)
