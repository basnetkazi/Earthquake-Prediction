# Earthquake-Prediction
Earthquake Data Visualization and Prediction

**#Project Team Member: Arun Kumar Basnet**

**# Project Idea:**
Earthquake is the most unpredictable disaster. There is no sign and symptoms and no any measure
to learn about its occurrence in near future. According to the available data planet earth undergoes
nearly 26 earthquakes above 4 M scale daily. But still there is no any concrete system or theory that
is able to predict the earthquake. The pattern of the earthquake is haphazard and it is so
unpredictable but still this project aims to understand the pattern of earthquake in various part of
world. This project is mainly being performed for better understanding of use of data sets and
machine learning algorithm to study pattern and perform prediction through past training data.
However the result from this project might not be reasonable because of the nature of earthquake it
is sure at end of the project it will be able to project some predictions and visualize the pattern of
earthquake occurrence in various part of earth.
In this project the regression model is used for prediction of the earthquake. The product might not
be able to predict and project the output correctly but still the pprediction in terms of the regression
model and the pattern can be recognized by using the datasets.


# **Data Sets:**
I am using the data sets present in the www.kaggle.com
The data is present in the following link (https://www.kaggle.com/usgs/earthquakedatabase/data)
which consist of the earthquake details about various earthquakes from all around
the world from 1965 to 2016 A.D. 

**# Software and Tools:**
This project requires Python 3.5 and the following Python libraries installed:
       IDE: PyCharm
       Tools: Matplotlib, numpy, pandas, sci-kit learn, basemap, pyQt
       Interpreter: Anaconda3.0

**# Task and Experiment Performed with Output:**
Most of the tasks completed in this project are regarding data visualization and also some
prediction models are applied to view how predictive earthquake can be.
1. Data Visualization:
First of all the data are extracted from the data set and summarized. Then the data are
analyzed in terms of how many data are present, average magnitudes of earthquake till
date, count of the data. Then the plotting of the earthquake along with map of world is done
using **basemap**


# **Predicting the earthquake:**
Then for feature selection on the obtained dataset SelectKBest class under sklearn.feature_selection
module was used to select k best scoring features using f_regression as the scoring function. The
training data and testing data were splitted and fetched using train_test_split() function under cross
validation module. Now the LinearRegression() function under linear_model module was used to
predict the magnitude using the linear model. The predicted earthquake was similar to the test data.
And also the error for the prediction was calculated and it was also very less.

