# Simple Linear Regression

#Importing Libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler  #For feature scaling
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def main():
    # Laoding Dataset
    dataset = pd.read_csv('Datasets/Salary_Data.csv')
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 1].values

    #Splitting the dataset into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1/3,random_state=0)

    # # Feature Scaling
    # sc_X = StandardScaler()
    # X_train = sc_X.fit_transform(X_train)
    # X_test = sc_X.fit_transform(X_test)

    # sc_y = StandardScaler()
    # y_train = sc_y.fit_transform(y_train)
    # y_test = sc_y.fit_transform(y_test)

    #Fitting Simple Linear Regression Model to train the set
    regressor = LinearRegression()
    regressor.fit(X_train,y_train)

    # Predicting the Test set results
    y_pred = regressor.predict(X_test)

    # Visualising the Training set results
    plt.scatter(X_train, y_train, color='red')
    plt.plot(X_train, regressor.predict(X_train), color='blue')
    plt.title('Salary vs Experience (Training set)')
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary')
    plt.show()

    # Visualising the Test set results
    plt.scatter(X_test, y_test, color='red')
    plt.plot(X_train, regressor.predict(X_train), color='blue')
    plt.title('Salary vs Experience (Test set)')
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary')
    plt.show()

if __name__ == '__main__':
    main()