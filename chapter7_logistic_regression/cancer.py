# Chapter 7 Excersises
#
# P / (1-P) <=== Probability of Success/ Probability of Failure
#
# When you apply natural log function to odds ^ you get the logit function.
# The logit function is the logarithm of the odds.
# The logit function transfers variables on (0,1) into a new variable (-infinity, +infinity)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def logit(x):
    return np.log(x/(1-x))

def showLogitFunctionGraph():
    x = np.arange(0.001, 0.999, 0.0001)
    y = [logit(n) for n in x]
    plt.plot(x,y)
    plt.xlabel("Probability")
    plt.ylabel("Logit - L")
    plt.show()

#
# The sigmoid function maps the real-number system to the probabilities
# This is done by flipping the axes of the logit function, so the sigmoid function is the inverse of the logit function
# Sigmoid function transforms values to a range from 0 to 1

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

def showSigmoidCurve():
    x = np.arange(-10, 10, 0.0001)
    y = [sigmoid(n) for n in x]
    plt.plot(x,y)
    plt.xlabel("Logit - L")
    plt.ylabel("Probability")
    plt.show()

#
# WISCONSIN CANCER CENTER DATA
#

def getConfusionMatrix(preds, test_labels):
    import pandas as pd
    #---generate table of predictions vs actual---
    print("---Confusion Matrix---")
    print(pd.crosstab(preds, test_labels))


def testModel(log_regress, test_set, test_labels):
    import pandas as pd

    #---get the predicted probablities and convert into a dataframe---
    preds_prob = pd.DataFrame(log_regress.predict_proba(X=test_set))

    #---assign column names to prediction---
    preds_prob.columns = ["Malignant", "Benign"]

    #---get the predicted class labels---
    preds = log_regress.predict(X=test_set)
    preds_class = pd.DataFrame(preds)
    preds_class.columns = ["Prediction"]

    #---actual diagnosis---
    original_result = pd.DataFrame(test_labels)
    original_result.columns = ["Original Result"]

    #---merge the three dataframes into one---
    result = pd.concat([preds_prob, preds_class, original_result], axis=1)
    print(result.head())

    getConfusionMatrix(preds, test_labels)    

def trainModelUsingAllFeatures():
    from sklearn.datasets import load_breast_cancer
    cancer = load_breast_cancer()        # Load dataset

    from sklearn.model_selection import train_test_split
    train_set, test_set, train_labels, test_labels = train_test_split(
                              cancer.data,               # features
                              cancer.target,             # labels
                              test_size = 0.25,          # split ratio
                              random_state = 1,          # set random seed
                              stratify = cancer.target)  # randomize based on labels
    
    from sklearn import linear_model
    x = train_set[:,0:30]         # mean radius
    y = train_labels              # 0: malignant, 1: benign
    log_regress = linear_model.LogisticRegression(max_iter=2000)
    log_regress.fit(X = x,
                    y = y)

    print(log_regress.intercept_)     #
    print(log_regress.coef_)          #

    testModel(log_regress, test_set, test_labels)

def makePrediction(log_regress):
    #This is fucked, I have no clue what the problem is...
    print(log_regress.predict_proba(20)) # [[0.93489354 0.06510646]]
    print(log_regress.predict(20)[0])    # 0

def findInterceptAndCoefficient(x, y):
    from sklearn import linear_model
    import numpy as np

    log_regress = linear_model.LogisticRegression()

    #---train the model---
    log_regress.fit(X = np.array(x).reshape(len(x),1),
                    y = y)

    #---print trained model intercept---
    print(log_regress.intercept_)     # [ 8.19393897]

    #---print trained model coefficients---
    print(log_regress.coef_)          # [[-0.54291739]]

    #makePrediction(log_regress)


def trainWithOneFeature():
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    from sklearn.datasets import load_breast_cancer

    cancer = load_breast_cancer()   # Load dataset
    x = cancer.data[:,0]            # mean radius
    y = cancer.target               # 0: malignant, 1: benign
    colors = {0:'red', 1:'blue'}    # 0: malignant, 1: benign

    plt.scatter(x,y,
                facecolors='none',
                edgecolors=pd.DataFrame(cancer.target)[0].apply(lambda x: colors[x]),
                cmap=colors)

    plt.xlabel("mean radius")
    plt.ylabel("Result")

    red   = mpatches.Patch(color='red',   label='malignant')
    blue  = mpatches.Patch(color='blue',  label='benign')

    plt.legend(handles=[red, blue], loc=1)
    #plt.show()
    findInterceptAndCoefficient(x, y)


#Plot Features in 3D
def plotFeatures3D():
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn.datasets import load_breast_cancer

    cancer = load_breast_cancer()

    #---copy from dataset into a 2-d array---
    X = []
    for target in range(2):
        X.append([[], [], []])
        for i in range(len(cancer.data)):    # target is 0,1
            if cancer.target[i] == target:
                X[target][0].append(cancer.data[i][0])
                X[target][1].append(cancer.data[i][1])
                X[target][2].append(cancer.data[i][2])

    colours = ("r", "b")   # r: malignant, b: benign
    fig = plt.figure(figsize=(18,15))
    ax = fig.add_subplot(111, projection='3d')
    for target in range(2):
        ax.scatter(X[target][0],
                X[target][1],
                X[target][2],
                c=colours[target])

    ax.set_xlabel("mean radius")
    ax.set_ylabel("mean texture")
    ax.set_zlabel("mean perimeter")
    plt.show()


#Plot Features in 2D
def plotFeatures2D():
    #%matplotlib inline
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_breast_cancer

    cancer = load_breast_cancer()

    #---copy from dataset into a 2-d list---
    X = []
    for target in range(2):
        X.append([[], []])
        for i in range(len(cancer.data)):              # target is 0 or 1
            if cancer.target[i] == target:
                X[target][0].append(cancer.data[i][0]) # first feature - mean radius
                X[target][1].append(cancer.data[i][1]) # second feature — mean texture

    colours = ("r", "b")   # r: malignant, b: benign
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    for target in range(2):
        ax.scatter(X[target][0],
                X[target][1],
                c=colours[target])

    ax.set_xlabel("mean radius")
    ax.set_ylabel("mean texture")
    plt.show()

#Wisconsin cancer center problem:
def getCancerData():
    from sklearn.datasets import load_breast_cancer
    cancer = load_breast_cancer()

trainModelUsingAllFeatures()