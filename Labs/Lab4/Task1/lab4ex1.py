import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
from submission_script import *
from dataset_script import dataset
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

# This is a sample from the dataset, for training/evaluation use the imported variable dataset
dataset_sample = [['C', 'S', 'O', '1', '2', '1', '1', '2', '1', '2', '0'],
                  ['D', 'S', 'O', '1', '3', '1', '1', '2', '1', '2', '0'],
                  ['C', 'S', 'O', '1', '3', '1', '1', '2', '1', '1', '0'],
                  ['D', 'S', 'O', '1', '3', '1', '1', '2', '1', '2', '0'],
                  ['D', 'A', 'O', '1', '3', '1', '1', '2', '1', '2', '0']]

if __name__ == '__main__':
    # Your code here
    split_percent = int(input())
    criterion = str(input())
    split_index = int((100 - split_percent) / 100 * len(dataset))

    encoder = OrdinalEncoder()
    encoder.fit([row[:-1] for row in dataset])

    train_set = dataset[split_index:]
    train_x = [row[:-1] for row in train_set]
    train_y = [row[-1] for row in train_set]
    train_x = encoder.transform(train_x)

    test_set = dataset[:split_index]
    test_x = [row[:-1] for row in test_set]
    test_y = [row[-1] for row in test_set]
    test_x = encoder.transform(test_x)

    classifier = DecisionTreeClassifier(criterion=criterion, random_state=0)
    classifier.fit(train_x, train_y)

    print(f'Depth: {classifier.get_depth()}')
    print(f'Number of leaves: {classifier.get_n_leaves()}')

    accuracy = 0
    for i in range(len(test_set)):
        predicted = classifier.predict([test_x[i]])[0]
        true = test_y[i]
        if predicted == true:
            accuracy += 1

    accuracy = accuracy / len(test_set)

    print(f'Accuracy: {accuracy}')
    features_importances = list(classifier.feature_importances_)
    most_important_feature = features_importances.index(max(features_importances))
    print(f'Most important feature: {most_important_feature}')

    least_important_feature = features_importances.index(min(features_importances))
    print(f'Least important feature: {least_important_feature}')
    # At the end you need to submit the dataset, classifier and encoder by calling the following functions

    # submit the training set
    submit_train_data(train_x, train_y)

    # submit the test set
    submit_test_data(test_x, test_y)

    # submit the classifier
    submit_classifier(classifier)

    # submit the encoder
    submit_encoder(encoder)
