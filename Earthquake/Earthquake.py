# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 18:06:37 2018

@author: Kazi
"""

import pandas as pd
import numpy as np
from PyQt5.QtCore import Qt
import sklearn
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
from mpl_toolkits.basemap import Basemap
# %matplotlib inline
import warnings
import sys
from sklearn import metrics
from PyQt5 import QtGui, QtCore
from PyQt5.QtGui import QIcon, QPixmap, QColor
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QMainWindow, QMessageBox, QLabel, QTableView, QTableWidget, QLineEdit, QTextEdit, QVBoxLayout
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import linear_model
#from sklearn import model_selection
#from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

warnings.filterwarnings('ignore')
disasters = pd.read_csv("database.csv")
disasters["Date"] = pd.to_datetime(disasters["Date"])
selected = disasters[["Date", "Latitude", "Longitude", "Magnitude", "Depth", "Type"]]
earth = selected[selected["Type"] == "Earthquake"]


class MyStream(QtCore.QObject):
    message = QtCore.pyqtSignal(str)
    def __init__(self, parent=None):
        super(MyStream, self).__init__(parent)

    def write(self, message):
        self.message.emit(str(message))

class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.home()

    def home(self):
        self.setGeometry(50, 50, 700, 500)
        self.setWindowTitle("ML Project")
        self.lable = QLabel(self)
        self.lable.setPixmap(QPixmap("Parallax-Background-Dark-08.jpg"))
        self.lable.resize(700, 480)
        self.statusBar().showMessage("Data Visualation and Prediction")
        # self.visuwindow=QMainWindow()
        """self.textEdit = QTextEdit(self)
        self.textEdit.move(50, 50)
        self.textEdit.resize(300, 300)"""
        # self.ui.initUi()
        abc=QIcon("submit-button-clipart-14.png")


        # QMessageBox.question(self, "COntinue...","DO You Want to Continue",QMessageBox.Yes|QMessageBox.No)
        button = QPushButton("Visualiation",self)
        button.setToolTip('Press for Visualation of Earthquake')
        button.move(100, 70)
        button.clicked.connect(self.on_click)
        button2 = QPushButton("Prediction Model", self)
        button2.setToolTip("Press for Predction Module")
        button2.move(100, 170)
        #button.setIcon(abc)
        button2.clicked.connect(self.on_click2)




        """self.lable = QTextEdit("OutputHere", self)
        self.lable.move(150,50)
        self.lable.resize(300,300)
        self.lable.setText(earth.head())"""
        self.show()

    # @pyqtSlot()
    def on_click(self):
        self.close()
        print('Visualation of Data')
        self.ui = visualation()
        myStream = MyStream()
        myStream.message.connect(self.ui.on_myStream_message)

        sys.stdout = myStream
        #myStream.message.connect(self.ui.on_myStream_message)
        # self.visuwindow.show()


    def on_click2(self):
        self.close()
        print("Prediction of Data")
        self.ui = prediction()
        myStream = MyStream()
        myStream.message.connect(self.ui.on_myStream_message)

        sys.stdout = myStream


    def on_myStream_message(self, message):
        self.textEdit.moveCursor(QtGui.QTextCursor.End)
        self.textEdit.insertPlainText(message)


class prediction(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUi()

    def initUi(self):
        #print(earth.head())
        dataset = earth
        dataset["Year"] = dataset['Date'].dt.year
        dataset['Magnitude'] = dataset.Magnitude.astype(int)
        dataset['Magnitude'] = dataset.Magnitude.astype(str)
        self.year = ""

        # dataset['Year'] = dataset.Year.astype(int)

        # print(list(set(dataset["Magnitude"].values)))
        # print(dataset.head())
        self.setGeometry(50, 50, 700, 550)
        self.setWindowTitle("ML Project")
        self.lable = QLabel(self)

        self.lable.setPixmap(QPixmap("Parallax-Background-Dark-08.jpg"))
        self.lable.resize(700, 530)
        #self.statusBar().showMessage("Data Visualation and Prediction")
        self.statusBar().showMessage("Earthquake Prediction Model")

        btnsubmit = QPushButton('Submit', self)
        btnsubmit.setToolTip('Press for Subiting Data')
        btnsubmit.move(100, 65)
        btnsubmit.resize(75, 25)
        btnsubmit.clicked.connect(self.submit)



        btnalgo = QPushButton('Compare Algorithm', self)
        btnalgo.setToolTip('Press for comparing Algorithms')
        btnalgo.move(300, 90)
        btnalgo.resize(100, 30)
        btnalgo.clicked.connect(self.comp_algo)

        btnknn = QPushButton('KNN Algorithm', self)
        btnknn.setToolTip('Press for prediction using KNN Algorithm')
        btnknn.move(20, 200)
        btnknn.resize(100, 30)
        btnknn.clicked.connect(self.knn_algo)

        btnlr = QPushButton('LR Algorithm', self)
        btnlr.setToolTip('Press for prediction using LR Algorithm')
        btnlr.move(580, 200)
        btnlr.resize(100, 30)
        btnlr.clicked.connect(self.lr_algo)

        btnlda = QPushButton('LDA Algorithm', self)
        btnlda.setToolTip('Press for prediction using LDA Algorithm')
        btnlda.move(20, 300)
        btnlda.resize(100, 30)
        btnlda.clicked.connect(self.lda_algo)

        btncart = QPushButton('CART Algorithm', self)
        btncart.setToolTip('Press for prediction using CART Algorithm')
        btncart.move(580, 300)
        btncart.resize(100, 30)
        btncart.clicked.connect(self.cart_algo)

        btnnb = QPushButton('NB Algorithm', self)
        btnnb.setToolTip('Press for prediction using NB Algorithm')
        btnnb.move(20, 400)
        btnnb.resize(100, 30)
        btnnb.clicked.connect(self.nb_algo)

        btnsvm = QPushButton('SVM Algorithm', self)
        btnsvm.setToolTip('Press for prediction using SVM Algorithm')
        btnsvm.move(580, 400)
        btnsvm.resize(100, 30)
        btnsvm.clicked.connect(self.svm_algo)


        self.lable = QLabel("Output Here", self)
        self.lable.move(150, 120)
        self.lable.setStyleSheet('color: White')

        #self.table = QTableView()
        #self.table.move(100, 200)

        self.textEdit = QTextEdit(self)
        self.textEdit.move(150, 150)
        self.textEdit.resize(400, 350)
        self.textEdit.setStyleSheet('color: Grey')


        self.lineedittime = QLineEdit(self)
        self.lineeditlongitude = QLineEdit(self)
        self.lineeditlatitude = QLineEdit(self)

        self.label = QLabel("Enter Year for Prediction", self)
        self.labe2 = QLabel("Enter Longitude for Prediction", self)
        self.labe3 = QLabel("Enter Latitude for Prediction", self)
        self.label.move(100, 10)
        self.lineedittime.move(100, 40)
        self.labe2.move(300, 10)
        self.lineeditlongitude.move(300, 40)
        self.labe3.move(500, 10)
        self.lineeditlatitude.move(500, 40)
        self.lineedittime.resize(100, 20)
        self.lineeditlatitude.resize(100, 20)
        self.lineeditlongitude.resize(100, 20)
        self.label.resize(150, 20)
        self.labe2.resize(150, 20)
        self.labe3.resize(150, 20)
        self.label.setStyleSheet('color: White')
        self.labe2.setStyleSheet('color: White')
        self.labe3.setStyleSheet('color: White')

        btnbk = QPushButton('Back', self)
        btnbk.setToolTip('Press for going Back')
        btnbk.move(580, 490)
        btnbk.resize(100, 30)
        btnbk.clicked.connect(self.back)

        btnbk = QPushButton('Linear Regression', self)
        btnbk.setToolTip('Press for Regression Algorithm')
        btnbk.move(580, 90)
        btnbk.resize(100, 30)
        btnbk.clicked.connect(self.linearregression)

        self.array = dataset.values
        # print(self.array)
        self.X = self.array[:, [6, 1, 2]]
        self.Y = self.array[:, 3]
        self.validation_size = 0.20
        self.seed = 7
        self.X_train, self.X_validation, self.Y_train, self.Y_validation = model_selection.train_test_split(self.X,
                                                                                                            self.Y,
                                                                                                            test_size=self.validation_size,
                                                                                                            random_state=self.seed)
        self.scoring = 'accuracy'
        # print(self.X,self.Y)
        self.show()

    #@pyqtSlot()
    def comp_algo(self):
        models = []
        # print("models")
        models.append(('LR', LogisticRegression()))
        models.append(('LDA', LinearDiscriminantAnalysis()))
        models.append(('KNN', KNeighborsClassifier()))
        models.append(('CART', DecisionTreeClassifier()))
        models.append(('NB', GaussianNB()))
        # models.append(('SVM', SVC()))
        # evaluate each model in turn
        results = []
        names = []
        # print("name")
        for name, model in models:
            # print("loop")

            kfold = model_selection.KFold(n_splits=10, random_state=self.seed)
            # print("kfold")
            cv_results = model_selection.cross_val_score(model, self.X_train, self.Y_train, cv=kfold,
                                                         scoring=self.scoring)

            # print("cv")
            results.append(cv_results)
            names.append(name)
            msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
            print(msg)

            # Compare Algorithms
        fig = plt.figure()
        fig.suptitle('Algorithm Comparison')
        ax = fig.add_subplot(111)
        plt.boxplot(results)
        ax.set_xticklabels(names)
        plt.show()

    def knn_algo(self):

        knn = KNeighborsClassifier()
        knn.fit(self.X_train, self.Y_train)
        self.predictions = knn.predict(self.X_validation)
        self.textEdit.clear()
        print("For Algorithm KNeighborsClassifier")
        print("The Prediction of the Algorithm:",self.predictions)
        # print("done")
        if (self.year == ""):
            self.accuracy()

    def lr_algo(self):

        lr = LogisticRegression()
        lr.fit(self.X_train, self.Y_train)
        self.predictions = lr.predict(self.X_validation)
        self.textEdit.clear()
        print("For Algorithm LogisticRegression")
        print("The Prediction of the Algorithm:",self.predictions)
        if (self.year == ""):
            self.accuracy()

    def lda_algo(self):

        lda = LinearDiscriminantAnalysis()
        lda.fit(self.X_train, self.Y_train)
        self.predictions = lda.predict(self.X_validation)
        self.textEdit.clear()
        print("For Algorithm LinearDiscriminantAnalysis")
        print("The Prediction of the Algorithm:",self.predictions)
        if (self.year == ""):
            self.accuracy()

    def cart_algo(self):
        cart = DecisionTreeClassifier()
        cart.fit(self.X_train, self.Y_train)
        self.predictions = cart.predict(self.X_validation)
        self.textEdit.clear()
        print("For Algorithm DecisionTreeClassifier")
        print("The Prediction of the Algorithm:",self.predictions)
        if (self.year == ""):
            self.accuracy()

    def nb_algo(self):
        nb = GaussianNB()
        nb.fit(self.X_train, self.Y_train)
        self.predictions = nb.predict(self.X_validation)
        self.textEdit.clear()
        print("For Algorithm GaussianNB")
        print("The Prediction of the Algorithm:",self.predictions)
        if (self.year == ""):
            self.accuracy()

    def svm_algo(self):
        svm = SVC()
        svm.fit(self.X_train, self.Y_train)
        self.predictions = svm.predict(self.X_validation)
        self.textEdit.clear()
        print("For Algorithm SVM")
        print("The Prediction of the Algorithm:",self.predictions)
        if (self.year == ""):
            self.accuracy()

    def submit(self):
        #print("self.X_validation")

        self.year = self.lineedittime.text()
        #print("X")
        latitude = self.lineeditlatitude.text()
        longitude = self.lineeditlongitude.text()
        if (self.year != "" and latitude != "" and longitude != ""):
            self.year = int(self.year)
            #print("Y")
            latitude = float(latitude)
            longitude = float(longitude)

            self.X_validation = list([[self.year, latitude, longitude]])
            print("The prediction is being done for the data:",self.X_validation)

    def accuracy(self):
        print("")
        print("Accuracy Score of the Algorithm: ",accuracy_score(self.Y_validation, self.predictions))
        print("")
        print("Confusios Matrix of Prediction:")

        print(confusion_matrix(self.Y_validation, self.predictions))
        print("")
        print("Classification Report:")

        print(classification_report(self.Y_validation, self.predictions))
        print("")
        print("Discrete Valuses of predictions from algorithm:", list(set(self.predictions)))


    def on_myStream_message(self, message):
        self.textEdit.moveCursor(QtGui.QTextCursor.End)
        self.textEdit.insertPlainText(message)

    def back(self):
        self.close()
        self.ui=Window()

    def linearregression(self):
        self.textEdit.clear()
        abc = selected
        #print(earth.head())
        abc["Year"] = abc['Date'].dt.year
        abc['Year'] = abc.Year.astype(int)
        #print("A")
        array = abc.values
        final = array[:, [6]]
        magnitude = abc["Magnitude"]
        # print("abc")
        #print(selected.head())
        X, Y = final, magnitude
        #print(X)
        # print("def")
        X_new = SelectKBest(f_regression, k="all").fit(X, magnitude)
        #print("def")
        X_new1 = X_new.transform(X)

        X_train, X_test, y_train, y_test = model_selection.train_test_split(X_new1, magnitude, random_state=1)
        print("After slicing data into test and train:")
        print("Shape of X_test is:", X_test.shape)
        print("Shape of X_train is:", X_train.shape)
        print("Shape of Y_train is:", y_train.shape)
        print("Shape of Y_test is:", y_test.shape)
        print("")
        linreg = linear_model.LinearRegression()
        linreg.fit(X_train, y_train)

        y_pred = linreg.predict(X_test)
        print("Shape of Y_prediction is:", y_pred.shape)
        print("")
        print("After fitting the regression we have:")
        print("Intercept of the fitting curve", linreg.intercept_)

        print("Cofficients of the fitting curves", linreg.coef_)
        print(" ")

        plt.xlabel('Year')
        plt.ylabel('test value or predicted value')
        plt.plot(X_test, y_test, '.', X_test,  linreg.predict(X_test), 'r')
        plt.show()
        print(" ")

        print("Mean Absolute Error:")
        print(metrics.mean_absolute_error(y_test, y_pred))
        print(" ")
        print("Mean Squared Error:")
        print(metrics.mean_squared_error(y_test, y_pred))
        print(" ")
        print("Sqrt Mean Squared Error:")
        print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
        print(" ")

        if(self.year!=""):
            #print("y")
            self.year=int(self.year)
            #print("x")
            output=linreg.predict([[self.year]])
            print("Prediction For year", self.year,"is : ", output)



class visualation(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUi()

    def initUi(self):
        #print("xy")
        self.setGeometry(50, 50, 700, 500)
        self.setWindowTitle("ML Project")
        self.lable = QLabel(self)
        self.lable.setPixmap(QPixmap("Parallax-Background-Dark-08.jpg"))
        self.lable.resize(700, 480)
        self.statusBar().showMessage("Data Visualation and Prediction")
        self.statusBar().showMessage("Earthquake Visualation")

        self.lable = QLabel("Output Here", self)
        self.lable.move(250, 25)
        self.lable.setStyleSheet('color: White')
        #self.table = QTableView()
        #self.table.move(100, 200)

        #highly_affected = earth[earth["Magnitude"] >= 8]
        self.longitudes = earth["Longitude"].tolist()
        self.latitudes = earth["Latitude"].tolist()

        self.textEdit = QTextEdit(self)
        self.textEdit.move(250, 50)
        self.textEdit.resize(400, 350)
        self.textEdit.setStyleSheet('color: Grey')
        #self.textEdit.setWindowTitle(self,"Outputs")

        btnprint = QPushButton('Dataset', self)
        btnprint.setToolTip('Press for viewing dataset')
        btnprint.move(100, 70)
        btnprint.clicked.connect(self.print_dataset)

        btndescribe = QPushButton('Describe Data', self)
        btndescribe.setToolTip('Press for Summarising the dataset')
        btndescribe.move(100, 170)
        btndescribe.clicked.connect(self.desc_dataset)

        btnplot = QPushButton('Plot info and Visualation', self)
        btnplot.setToolTip('Press for Visualising the Earthquake')
        btnplot.setGeometry(100, 170, 150, 40)
        btnplot.move(80, 270)
        btnplot.clicked.connect(self.plot_eq)

        btnbk = QPushButton('Back', self)
        btnbk.setToolTip('Press for going Back')
        btnbk.move(580, 420)
        btnbk.resize(100, 30)
        btnbk.clicked.connect(self.back)

        self.show()

    # @pyqtSlot()
    def print_dataset(self):
        self.textEdit.clear()
        print("Dataset samples after filtering")
        print(earth.head(20))

    def desc_dataset(self):
        self.textEdit.clear()
        print("The summarize of data of the Magnitudes")
        print(earth["Magnitude"].describe())

    def plot_eq(self):
        self.ui = plot()
        myStream = MyStream()
        myStream.message.connect(self.ui.on_myStream_message)

        sys.stdout = myStream
        self.close()

    def on_myStream_message(self, message):
        self.textEdit.moveCursor(QtGui.QTextCursor.End)
        self.textEdit.insertPlainText(message)

    def back(self):
        self.close()
        self.ui=Window()

class plot(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUi()

    def initUi(self):

        self.setGeometry(50, 50, 700, 500)
        self.setWindowTitle("ML Project")
        self.lable = QLabel(self)
        self.lable.setPixmap(QPixmap("Parallax-Background-Dark-08.jpg"))
        self.lable.resize(700, 480)
        self.statusBar().showMessage("Data Visualation and Prediction")
        self.statusBar().showMessage("Earthquake Plotting and Visualation")

        self.affected = earth[earth["Magnitude"] >= 4]
        self.longitudes = self.affected["Longitude"].tolist()
        self.latitudes = self.affected["Latitude"].tolist()

        self.lineedit = QLineEdit(self)
        self.label = QLabel("Enter the magnitude of earthquake to plot in map", self)
        self.label.move(100, 10)
        self.label.setStyleSheet('color: White')
        self.lineedit.move(100, 40)
        self.lineedit.resize(100, 20)
        # text=5

        btnsub = QPushButton('Submit', self)
        btnsub.setToolTip('Press button to Submit')
        btnsub.move(100, 65)
        btnsub.resize(70, 20)
        btnsub.clicked.connect(self.submit)

        btnmap = QPushButton('Map Plotting', self)
        btnmap.setToolTip('Press button for map plot')
        btnmap.move(100, 110)
        btnmap.clicked.connect(self.map_plot)

        btnbox = QPushButton('Box Plot', self)
        btnbox.setToolTip('Press button for box plot')
        btnbox.move(100, 170)
        btnbox.clicked.connect(self.box_plot)

        btnhist = QPushButton('Histogram plot', self)
        btnhist.setToolTip('Press button for Histogram plot')
        btnhist.move(100, 230)
        btnhist.clicked.connect(self.hist_plot)

        btnScatter = QPushButton('Scatterplot', self)
        btnScatter.setToolTip('Press button for Scatter plot')
        btnScatter.move(100, 290)
        btnScatter.clicked.connect(self.scatter_plot)

        btnmonth = QPushButton('Monthly Earthquakes', self)
        btnmonth.setToolTip('Press button for Monthly Occurance plot')
        btnmonth.move(90, 350)
        btnmonth.resize(120,30)
        btnmonth.clicked.connect(self.month_plot)

        btnyear = QPushButton('Yearly Earthquake', self)
        btnyear.setToolTip('Press button for Yearly Occurance plot')
        btnyear.move(90, 410)
        btnyear.resize(120, 30)
        btnyear.clicked.connect(self.year_plot)

        self.lable = QLabel("Output Here", self)

        self.lable.move(250, 50)
        self.lable.setStyleSheet('color: White')

        # self.table = QTableView()
        # self.table.move(100, 200)

        self.textEdit = QTextEdit(self)
        self.textEdit.move(250, 75)
        self.textEdit.resize(400, 300)
        self.textEdit.setStyleSheet('color: Grey')

        btnbk = QPushButton('Back', self)
        btnbk.setToolTip('Press for going Back')
        btnbk.move(580, 420)
        btnbk.resize(100, 30)
        #btnbk.icon(QIcon("large.png"))
        btnbk.clicked.connect(self.back)

        self.show()

    @pyqtSlot()
    def map_plot(self):
        # print(type(x),len(x),type(latitudes),len(latitudes))
        m = Basemap(projection='mill', llcrnrlat=-80, urcrnrlat=80, llcrnrlon=-180, urcrnrlon=180, lat_ts=20,
                    resolution='c')
        x, y = m(self.longitudes, self.latitudes)
        fig = plt.figure(figsize=(12, 10))
        plt.title("All affected area")

        m.plot(x, y, "o", markersize=3, color='blue')
        m.drawcoastlines()
        m.fillcontinents(color='coral', lake_color='aqua')
        m.drawmapboundary()
        m.drawcountries()
        plt.show()

    def box_plot(self):
        self.textEdit.clear()
        plt.boxplot(earth["Magnitude"])
        plt.title("Box Plot of the Earthquake")
        plt.ylabel("Earthquake Magnitudes")

        plt.show()
        print(earth["Magnitude"].describe())
        # plt.earth["Magnitude"].plot(kind="box").show()

    def hist_plot(self):
        self.textEdit.clear()
        (n, bins, patches) = plt.hist(earth["Magnitude"], range=(0, 10), bins=10)
        plt.title("Histogram plot by Occurance Magnitude")
        plt.xlabel("Earthquake Magnitudes")
        plt.ylabel("Number of Occurences")
        plt.title("Overview of earthquake magnitudes")
        plt.show()

        print("Magnitude" + "   " + "Number of Occurence")
        for i in range(5, len(n)):
            print(str(i) + "-" + str(i + 1) + "         " + str(n[i]))
            # earth["Magnitude"].hist().show()

    def month_plot(self):
        abc = self.affected
        self.textEdit.clear()
        # print(abc.head())
        abc["Month"] = abc["Date"].dt.month
        month_occurrence = abc.groupby("Month").groups
        # print(month_occurrence[1])
        # print(month_occurrence)
        # month = [i for i in range(1, 13)]
        month = abc["Month"].values
        month = list(set(month))
        #print(month)
        occurrence = []

        for i in range(len(month)):
            val = month_occurrence[month[i]]
            occurrence.append(len(val))

        # print(occurrence)
        # print(sum(occurrence))

        fig, ax = plt.subplots(figsize=(10, 8))
        bar_positions = np.arange(len(month)) + 0.5

        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        num_cols = month
        bar_heights = occurrence

        print("Month :" + "     " + "Occurrence")

        for k, v in month_occurrence.items():
            print(str(k) + "      " + str(len(v)))

        ax.bar(bar_positions, bar_heights)
        tick_positions = np.arange(1, 13)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(num_cols, rotation=90)
        plt.title("Frequency by Month")
        plt.xlabel("Months")
        plt.ylabel("Frequency")
        plt.show()

    def year_plot(self):
        abc = self.affected
        self.textEdit.clear()

        abc["Year"] = abc['Date'].dt.year
        # print(abc.head())
        year_occurrence = abc.groupby("Year").groups
        # print(year_occurrence[1])
        # year = [i for i in range(1965, 2017)]
        year = abc["Year"].values
        year = list(set(year))
        #print(year)
        # year = [i for i in range(1965, 2017)]
        occurrence = []
        #print(year)

        for i in range(len(year)):
            #print(i)
            val = year_occurrence[year[i]]
            occurrence.append(len(val))

        #print(year_occurrence)
        maximum = max(occurrence)
        minimum = min(occurrence)
        print("Maximum no in Year", maximum)
        print("Minimum no in year", minimum)

        print("Year :" + "     " + "Occurrence")

        for k, v in year_occurrence.items():
            print(str(k) + "      " + str(len(v)))
        fig = plt.figure(figsize=(10, 6))
        plt.plot(year, occurrence)
        plt.xticks(rotation=90)
        plt.xlabel("Year")
        plt.ylabel("Number of Occurrence")
        plt.title("Frequency of Earthquakes by Year")
        plt.xlim(1965, 2017)
        plt.show()

    def scatter_plot(self):
        scatter_matrix(self.affected)
        plt.show()

    def submit(self):

        text = self.lineedit.text()
        text = float(text)

        self.affected = earth[earth["Magnitude"] >= text]
        self.longitudes = self.affected["Longitude"].tolist()
        self.latitudes = self.affected["Latitude"].tolist()

    def on_myStream_message(self, message):
        self.textEdit.moveCursor(QtGui.QTextCursor.End)
        self.textEdit.insertPlainText(message)

    def back(self):
        self.close()
        self.ui = visualation()
        myStream = MyStream()
        myStream.message.connect(self.ui.on_myStream_message)

        sys.stdout = myStream



app = QApplication(sys.argv)
GUI = Window()


sys.exit(app.exec())
