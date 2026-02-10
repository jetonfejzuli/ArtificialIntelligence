import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'

from submission_script import *
from dataset_script import dataset
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder

# This is a dataset sample, for training and evaluation use the
# imported dataset variable
dataset_sample = [['C', 'S', 'O', '1', '2', '1', '1', '2', '1', '2', '0'],
                  ['D', 'S', 'O', '1', '3', '1', '1', '2', '1', '2', '0'],
                  ['C', 'S', 'O', '1', '3', '1', '1', '2', '1', '1', '0'],
                  ['D', 'S', 'O', '1', '3', '1', '1', '2', '1', '2', '0'],
                  ['D', 'A', 'O', '1', '3', '1', '1', '2', '1', '2', '0']]

if __name__ == '__main__':
    # Your code here
    input_record = input().strip().split()

    split_idx = int(0.75 * len(dataset))
    train_set = dataset[:split_idx]
    test_set = dataset[split_idx:]

    train_x = [row[:-1] for row in train_set]
    train_y = [row[-1] for row in train_set]
    test_x = [row[:-1] for row in test_set]
    test_y = [row[-1] for row in test_set]

    # Encode categorical features
    encoder = OrdinalEncoder()
    encoder.fit(train_x)
    train_x_enc = encoder.transform(train_x)
    test_x_enc = encoder.transform(test_x)

    # Train Naive Bayes classifier
    classifier = CategoricalNB()
    classifier.fit(train_x_enc, train_y)

    # Evaluate accuracy
    correct = 0
    for x, y_true in zip(test_x_enc, test_y):
        y_pred = classifier.predict([x])[0]
        if y_pred == y_true:
            correct += 1
    accuracy = correct / len(test_set)

    # Encode and predict input record
    input_enc = encoder.transform([input_record])
    predicted_class = classifier.predict(input_enc)[0]
    class_probs = classifier.predict_proba(input_enc)

    # Print results
    print(accuracy)
    print(predicted_class)
    import numpy as np

    np.set_printoptions(suppress=False, precision=8)
    print(class_probs)

    # Submit results
    submit_train_data(train_x_enc, train_y)
    submit_test_data(test_x_enc, test_y)
    submit_classifier(classifier)
    submit_encoder(encoder)
