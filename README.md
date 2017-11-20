# color_classifier 

## Basics

Changes in illumination make color classification not to be a straightforward process. In order to separate illumination from the color information the LAB space is used. Besides, the LAB space provides a linear-like distribution in the color space.
A Random Forest classifier is used in this package since it provides remarkable results in classification of more than 2 variables.

## Description
This Python package provides two tools:

-Construction of a green, red and blue color dataset out of real time captures

-Create and train a Random Forest Classifier for labeling a pixel color into green, red or blue.

### Dataset construction

The dataset is constructed in real time. For this, after executing the construct_dataset.py script, the user selects a pixel in a window that displays the webcam. 
The user has to specify the colors to construct the dataset from. For example:
python construct_dataset.py red blue green
Results are stored in .txt files.

### Color classifier

The classier is a Random Forest Classifier. It is created executing the random_forest_classifier.py script.
It will output the number of observations in the training and test dataset, the confusion matrix and the importance of each variable in the classification. Besides, the classifier will be stored in a .sav file, accesible using the sklearn joblib utility.

