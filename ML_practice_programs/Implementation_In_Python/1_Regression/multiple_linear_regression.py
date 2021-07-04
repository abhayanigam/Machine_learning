# Multiple Linear Regression

# Importing Libraries.
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

def main():
    # Importing thr datasets.
    dataset = pd.read_csv("Datasets/50_Startups.csv")
    X = dataset.iloc[:, :1].values
    y = dataset.iloc[:, 4].values

    # Encoding categorial data
    labelencoder = LabelEncoder()
    X[: , 3] = labelencoder.fit_transform(X[ : , 3])
    onehotencoder = OneHotEncoder(categorical_features = [3]) 
    ct = ColumnTransformer([("State", OneHotEncoder(), [3])], remainder = 'passthrough')
    X = ct.fit_transform(X)

    # Avoiding the Dummy Variable Trap
    X = X[:,1:]

    # Splitting the dataset into the training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2 ,random_state=0)

    # Feature Scaling
    """
    sc_X = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)

    sc_Y = StandardScaler()
    y_train = sc.fit_transform(y_train)
    y_test = sc.fit_transform(y_test)
    """

    # Fitting multiple linear regression to the training set 
    regressor = LinearRegression()
    regressor.fit(X_train,y_train)

    # Predicting the test set results
    y_pred = regressor.predict(X_test)

    print(r2_score(y_test, y_pred))

if __name__ == "__main__":
    main()