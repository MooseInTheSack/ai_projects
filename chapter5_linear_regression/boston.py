import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def predictUsingModel(model, LSTAT, RM):
    print(model.predict([[LSTAT,RM]]))

def printActualVsPredicted(Y_test, price_pred):
    from sklearn.metrics import mean_squared_error

    mse = mean_squared_error(Y_test, price_pred)
    print(mse)

    plt.scatter(Y_test, price_pred)
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Actual prices vs Predicted prices")
    plt.show()

def trainModel(df):
    #Now train the model
    x = pd.DataFrame(np.c_[df['LSTAT'], df['RM']], columns = ['LSTAT', 'RM'])
    Y = df['MEDV']

    #split the dataset into 70 percent for training and 30 percent for testing
    from sklearn.model_selection import train_test_split
    x_train, x_test, Y_train, Y_test = train_test_split(x, Y, test_size=0.3, random_state=5)

    #print the training set number of rows and coumns for x and y
    #print(x_train.shape)
    #print(Y_train.shape)

    #print the testing set number of rows and coumns for x and y
    #print(x_test.shape)
    #print(Y_test.shape)

    #Now to train our model
    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    model.fit(x_train, Y_train)

    #use testing set to perform predictions
    price_pred = model.predict(x_test)

    #To learn how well our model performed, use R-squared method which tells you how close the test data fits the regression line.
    print('R-squared: %.4f' % model.score(x_test, Y_test))

    #printActualVsPredicted(Y_test, price_pred)

    modelIntercept = model.intercept_
    modelCoefficients = model.coef_
    print('Intercept: %s' % modelIntercept)
    print('Coefficients: %s' % modelCoefficients)

    #use model to predict price
    predictUsingModel(model, 30, 5)

    #Stopped at page 133

def printScatterPlots(df):
    #Plot a scatter plot showing relationship between LSTAT feature and MEDV label
    plt.scatter(df['LSTAT'], df['MEDV'], marker='o')
    plt.xlabel('LSTAT')
    plt.ylabel('MEDV')
    plt.show()

    #Plot a scatter plot showing relationship between RM feature and MEDV label
    plt.scatter(df['RM'], df['MEDV'], marker='o')
    plt.xlabel('RM')
    plt.ylabel('MEDV')
    plt.show()

def getTopFeatures(df, numFeatures):
    #Get the top numFeatures features with the highest correlation
    print(corr.abs().nlargest(numFeatures, 'MEDV').index)

    #print the top numFeatures correlation values
    print(corr.abs().nlargest(numFeatures, 'MEDV').values[:,13])

def checkForMissingValues(df):
    #Check to see if there are any missing values
    print(df.isnull().sum())

def printCorrelation(df):
    #Choose the features that directly influence the result (which is the house price)
    corr = df.corr()
    print(corr)

def main():
    print("Hello World!")
    
    from sklearn.datasets import load_boston
    dataset = load_boston()

    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)

    #Add the prices of the houses to the DataFrame, so add a new column and call it MEDV
    df['MEDV'] = dataset.target

    #getTopXFeatures
    #getNumFeatures(df, 3)

    #Now let's plot them on a 3D Chart...
    #from mpl_toolkits.mplot3d import Axes3D

    #fig = plt.figure(figsize=(18,15))
    #ax = fig.add_subplot(111, projection='3d')

    #ax.scatter(df['LSTAT'], df['RM'], df['MEDV'], c='b')

    #ax.set_xlabel("LSTAT")
    #ax.set_ylabel("RM")
    #ax.set_zlabel("MEDV")
    #plt.show()

    trainModel(df)

if __name__ == "__main__":
    main()
