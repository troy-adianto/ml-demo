# Import libraries
import joblib
import flask
from flask import Flask, jsonify, request
#Packages related to general operating system & warnings
import os
import warnings
warnings.filterwarnings('ignore')
#Packages related to data importing, manipulation, exploratory data #analysis, data understanding
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from termcolor import colored as cl # text customization
#Packages related to data visualizaiton
import seaborn as sns
import matplotlib.pyplot as plt
#Setting plot sizes and type of plot
plt.rc("font", size=14)
plt.rcParams['axes.grid'] = True
plt.figure(figsize=(6,3))
plt.gray()
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.preprocessing import  PolynomialFeatures, KBinsDiscretizer, FunctionTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer, OrdinalEncoder
import statsmodels.formula.api as smf
import statsmodels.tsa as tsa
from sklearn.linear_model import LogisticRegression, LinearRegression, ElasticNet, Lasso, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz
from sklearn.ensemble import BaggingClassifier, BaggingRegressor,RandomForestClassifier,RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.svm import LinearSVC, LinearSVR, SVC, SVR
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def train_model():
    # Import Dataset
    data=pd.read_csv("creditcard.csv")

    # Check transaction distribution
    Total_transactions = len(data)
    normal = len(data[data.Class == 0])
    fraudulent = len(data[data.Class == 1])
    fraud_percentage = round(fraudulent/normal*100, 2)
    print(cl('Total number of Trnsactions are {}'.format(Total_transactions), attrs = ['bold']))
    print(cl('Number of Normal Transactions are {}'.format(normal), attrs = ['bold']))
    print(cl('Number of fraudulent Transactions are {}'.format(fraudulent), attrs = ['bold']))
    print(cl('Percentage of fraud Transactions is {}'.format(fraud_percentage), attrs = ['bold']))

    # Check for null values
    data.info()

    # Scale Variable
    sc = StandardScaler()
    amount = data['Amount'].values
    data['Amount'] = sc.fit_transform(amount.reshape(-1, 1))

    # Drop Time variable
    data.drop(['Time'], axis=1, inplace=True)

    # Drop Duplicates
    data.drop_duplicates(inplace=True)

    # Train & Test Split
    X = data.drop('Class', axis = 1).values
    y = data['Class'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)

    # Model Building
    # XGBoost
    xgb = XGBClassifier(max_depth = 4)
    xgb.fit(X_train, y_train)
    xgb_yhat = xgb.predict(X_test)

    joblib.dump(xgb, 'xgboost-model.model')

    print('Accuracy score of the XGBoost model is {}'.format(accuracy_score(y_test, xgb_yhat)))
    print('F1 score of the XGBoost model is {}'.format(f1_score(y_test, xgb_yhat)))

    return accuracy_score(y_test, xgb_yhat)


app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_card_fraud():
    accuracy_score = train_model()

    return str(accuracy_score)

@app.route('/', methods=['GET'])
def index():
    return 'Machine Learning Inference'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
