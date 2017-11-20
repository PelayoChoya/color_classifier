#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

def create_directory():
    '''Creates the directory for storing
        the classifier and returns the
        directory's path'''
    path = os.path.dirname(os.path.abspath(__file__)) + \
            '/classifiers'
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def read_text_file():
    ''' Reads data from the .txt file and
        outputs a list with the data plus
        the class                       '''
    # color plus its class identifier
    colors = {'red':0, 'green':1, 'blue':2}
    data = {}
    for color in colors:
        data[color] = np.loadtxt('../dataset_construction/dataset/' +
                                     color + '.txt', dtype = np.uint8)
    # adds a class identifier to the data
    for key in data:
        data[key] = np.insert(data[key],3,colors[key], axis = 1)
    return data

def data_frame_from_array(array_data):
    ''' Creates a data frame pandas structure
        out of a data array'''
    data_array =[]
    data_array = np.concatenate((array_data['red'], array_data['blue'],array_data['green'] ), axis = 0)
    # creation of the data frame structure
    return pd.DataFrame(data_array, columns = ['L','A','B','COLOR'])

def divide_train_test_dataset(data_frame):
    ''' Divide the dataset into train
        and test subdatasets '''
    # randomly divides the data into training and test data
    df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
    # Create two new dataframes, one with the training rows, one with the test rows
    train_ds, test_ds = df[df['is_train']==True], df[df['is_train']==False]
    # Show the number of observations for the test and training dataframes
    print('Number of observations in the training data:', len(train_ds))
    print('Number of observations in the test data:',len(test_ds))
    return train_ds, test_ds

def create_train_classifier(data_frame, train_ds, features_list):
    ''' Creates a classifier out of the data frame
        and trains it out from the train dataset '''
    # create a random forest classifier
    classifier = RandomForestClassifier(n_jobs=2, random_state=0)
    # train the classifier
    classifier.fit(train_ds[features_list], train_ds['COLOR'])
    return classifier

def test_classifier(classifier, test_ds, train_ds, features_list):
    ''' Outputs the performance of the classifier:
        creates a confusion matrix and displays
        the importance of each variable in the
        predicion '''
    # test data
    preds = classifier.predict(test_ds[features_list])
    print 'Confusion matrix for evaluating the results'
    print 'red = 0, green = 1, blue = 2'
    # display the confusion matrix
    print pd.crosstab(test_ds['COLOR'], preds, rownames=['COLOR'],
                      colnames=['Predicted Color'])
    # display the importance in the prediction of each variable
    print 'Importance in the prediction of each variable, out of 1'
    print list(zip(train_ds[features_list], classifier.feature_importances_))

def save_classifier(classifier):
    ''' Saves the classifier using
        joblib scipy utility'''
    path_to_save = create_directory()
    filename = path_to_save + '/random_forest_classifier.sav'
    joblib.dump(classifier, filename)

if __name__ == '__main__':

    # read data from .txt file
    data = read_text_file()
    # create pandas data frame structure
    df = data_frame_from_array(data)
    # get train and test datasets
    train, test = divide_train_test_dataset(df)
    # create a list containing the features
    features = df.columns[:3]
    # create and train dataset
    clf = create_train_classifier(df, train, features)
    # test the classifier
    test_classifier(clf, test, train, features)
    # save the classifier
    save_classifier(clf)
