#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 22:35:35 2020

@author: tahir
"""
# ## CSC311 - Assignment 1
# By: Tahir Muhammad <br>
# Student Number: 1002537613

# import relevant libraries
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import sklearn.linear_model as lin
import bonnerlib3 as bl3d
import pickle as pickle
import time

from sklearn.neighbors import KNeighborsClassifier
from numpy.linalg import inv


# Set seet for data repretition 
rnd.seed(3)

############################################################
######################## QUESTION 1 ########################
############################################################

# QUESTION 1A
print("\n\nQuestion 1(a):")
print("--------------")
B = np.random.rand(4, 5)
print(B)


# QUESTION 1B
print("\n\nQuestion 1(b):")
print("--------------")
y = rnd.rand(4,1)
print(y)


# Question 1C
print("\n\nQuestion 1(c):")
print("--------------")
C = np.reshape(B,(2, 10))
print(C)


# Question 1D
print("\n\nQuestion 1(d):")
print("--------------")
D = B - y
print(D)


# Question 1E
print("\n\nQuestion 1(e):")
print("--------------")
z = np.reshape(y,(4))
print(z)


# Question 1F
print("\n\nQuestion 1(f):")
print("--------------")
B[:,3] = z
print(B)


# Question 1G
print("\n\nQuestion 1(g):")
print("--------------")
# Compute the addition and Assign it to the first column (column 0) of D.
D[:,0] = B[:,2] + z
print(D)


# Question 1H
print("\n\nQuestion 1(h):")
print("--------------")
print(B[:3])


# Question 1I
print("\n\nQuestion 1(i):")
print("--------------")
print( B[:, [1,3]] )


# Question 1J
print("\n\nQuestion 1(j):")
print("--------------")
print(np.log(B))


# Question 1K
print("\n\nQuestion 1(k):")
print("--------------")
print(np.sum(B))


# Question 1L
print("\n\nQuestion 1(l):")
print("--------------")
# print(B.shape)
print(B.max(axis=0))

# Question 1M
print("\n\nQuestion 1(m):")
print("--------------")
print(np.max(np.sum(B, axis=0)))

# Question 1N
print("\n\nQuestion 1(n):")
print("--------------")
print(np.matmul(B.transpose(),D))

# Question 1O
print("\n\nQuestion 1(o):")
print("--------------")
print(np.matmul(((np.matmul(y.transpose(),D))),((np.matmul(D.transpose(),y)))))


############################################################
######################## QUESTION 2 ########################
############################################################

#################
## Question 2A ##
#################
print("\n\nQuestion 2(a-b):")
print("--------------")
print("The above part(s) had no explicit output to printout.") 
def matrix_poly(A):
    """
    Compute the Polynomial A + A^2 + A^3, where A^2 = A*A
    and A^3 = A*A*A. 
    """
    # We want to Implement A+A*(A+A*A)
    # Get the Addition, A + A
    A_Plus_A = matrixAddition(A)
    # Get the parentheses (A+A*A)
    A_Plus_A_TimesA = MatrixMult(A, A_Plus_A)
    # Get the given polynomial, A+A*(A+A*A)
    polynomial = MatrixMult(A_Plus_A, A_Plus_A_TimesA)
    return polynomial
    

# Helper Function for 2A
def matrixAddition(A):
    """
    Given a Matrix A, add every element in the Matrix by itself.
    Return A+A.
    """ 
    # Quick check to see if A is a square matrix
    x = A.shape
    if x[0] != x[1]:
        print("Please input a square matrix")
        
    # Initialize an empty matrix for the result
    Result = np.zeros(A.shape)
    # For each row in the matrix
    for i in range(len(A)):
        # For each element inside of that row
        for j in range(len(A[0])):
            # Add that element to itself, and store the result
            # inside of the corresponding index of Result
            Result[i,j] = A[i,j] + A[i,j]

    return Result

def MatrixMult(A, B):
    """
    Given two matrices, preform Matrix multiplication on them.
    Return A*B
    """
    Ashape = A.shape
    Bshape = B.shape 
    if Ashape[1] != Bshape[0]:
        print("The Following Matrices cannot be multiplied")
    
    # Initialize array for result
    result = np.zeros((Ashape[0], Bshape[1]))

    # for each row in Matrix A
    for i in range(len(A)):
        newRow = []
        # for each element 
        for j in range(len(B[i])):
            newEntry = 0
            for k in range(len(A[i])):
                newEntry += A[i][k]*B[k][j]
            newRow.append(newEntry)
        # The len of the final matrix will be the same
        # Aslong as you can multiply the matrices. 
        result[i] = newRow
        
    return result

#################
## Question 2B ##
#################

def timing(N):
    """ 
    Return the Excecution Speed of the Given Polynomial
    Excecuted at different sizes. 
    """
    # Random NxN Matrix
    A = np.random.rand(N,N)
    
    # Measure Execution Time of Non-Vectorized Code (Used Loops)
    start_time = time.time()
    B1 = matrix_poly(A)
    time1 = time.time() - start_time
    print("    The Excecution time of the matrix_poly(A) is:", time1)
    
    # Measure the Excution time of Vectorized Code 
    start_time2 = time.time()
    B2 = np.matmul( (A+A), (np.matmul((A+A),A)))
    time2 = time.time() - start_time2
    print("    The Excecution time of the vectorized polynomial is:", time2)
    
    magnitude = np.sqrt(np.sum( (B1 - B2)**2 ))
    print("    The Magnitude of the above matrices is: ", magnitude)

#################
## Question 2C ##
#################

print("\n\nQuestion 2(c):")
print("--------------")
print("This is the Execution Time for N = 100: \n") 
timing(100)
print(" ")
print("This is the Execution Time for N = 300: \n")
timing(300)
print(" ")
print("This is the Execution Time for N = 1000: \n")
timing(1000)


############################################################
######################## QUESTION 3 ########################
############################################################

#################
## Question 3A ##
#################
print("\n\nQuestion 3(a-c):")
print("--------------")
print("The following part(s) had no specific output to print.")

def least_squares(x,t):
    """ 
    Take a vector x of input values, and vector t of target values
    Return the Optimal values of A and B, as (b,a). 
    """
    # Bais vector, nx1
    bais = np.ones(len(x))
    # Add the bais and x values into one matrix, of shape nx2
    X = np.column_stack((bais,x))
    # Compute the least squares formula (given from slides) 
    w = inv(X.T.dot(X)).dot(X.T).dot(t)
    return w

#################
## Question 3B ##
#################

def plot_data(x,t):
    """ 
    Plot the data and the line of best fit.
    """
    #  Getting the optimal B0 and B1
    w = least_squares(x,t)    
    # Find the minimum and maximum points for plotting
    x1 , x2 = np.min(x), np.max(x)
    x_values = [x1,x2]
    # find the corresponding y values
    y1, y2 = w[1]*x1+w[0], w[1]*x2 +w[0]
    y_values = [y1,y2]
    
    # Plot the points, title, axis, and line of best fit. 
    plt.figure()
    plt.title("Question 3(b):  the fitted line")
    plt.xlabel("x Values")
    plt.ylabel("Target Values")
    plt.scatter(x,t,color="blue")
    # Plot a line between the min and max predicted points. 
    plt.plot(x_values,y_values,color='red')
    
    return w

#################
## Question 3C ##
#################

def error(a,b,X,T):
    """ 
    Compute how well a line fits the data.
    Returns the Mean Squared Error of the line with the data.
    """
    #y_pred = a*x+b
    y_pred = np.dot(X, (np.append(b,a)) )
    mse = np.mean((T - y_pred)**2)
    return mse

#################
## Question 3D ##
#################
print("\n\nQuestion 3(d):")
print("--------------")

# Read in the Data file.
with open("dataA1Q3.pickle", "rb") as f:
    dataTrain,dataTest = pickle.load(f)
    
# Make the data a numpy array
trainMatrix = np.array(dataTrain)
testMatrix = np.array(dataTest)

# read in the train data into corresponding variables.
# x and t are both 30x1. 
x_train = np.array(trainMatrix[0]).transpose()
t_train = np.array(trainMatrix[1]).transpose()

# read in the train data into corresponding variables
x_test = np.array(testMatrix[0]).transpose()
t_test = np.array(testMatrix[1]).transpose()

# Call the polt data function to plot the data, and get optimal weights
beta_hat = plot_data(x_train,t_train)

# Print the values of a and b.
a = beta_hat[1]
b = beta_hat[0]
print("This is the value of a: ", a)
print("This is the value of b: ", b)

# Compute and print the training error
bais = np.ones(len(x_train))
X_matTrain = np.column_stack((bais,x_train))
mse_train = error(a,b,X_matTrain, t_train)
print("This is the training error: ", mse_train)

# Compute and print the testing error
bais = np.ones(len(x_test))
X_matTest = np.column_stack((bais,x_test))
mse_test = error(a,b,X_matTest, t_test)
print("This is the testing error: ", mse_test)

############################################################
######################## QUESTION 4 ########################
############################################################

#################
## Question 4A ##
#################
print("\n\nQuestion 4(a):")
print("--------------")

# Retrieve the training and test data
with open("dataA1Q4v2.pickle","rb") as f:
    Xtrain,Ttrain,Xtest,Ttest = pickle.load(f)
    
# Create a classification object, clf
clf = lin.LogisticRegression()
# Learn a logistic-regression classifier
clf.fit(Xtrain,Ttrain)
# weight vector 
w = clf.coef_[0]
# bais term
w0 = clf.intercept_[0]
    
print("The wieght vector for the logistic regression model is:\n ", w )
print( " ")
print("The bais term of the logistic regression model is: \n", w0)
    

#################
## Question 4B ##
#################
print("\n\nQuestion 4(b):")
print("--------------")

# Compute the test accuracy of your Logistic Regression Model
# Method (1): Using the Score method. 
accuracy1 = clf.score(Xtest,Ttest)

# Method (2): Using the formulas on the lecture slides.
# Compute your wieght matrix, and add the bais term. 
bais = np.ones(len(Xtest))  # nx1
X = np.column_stack((bais,Xtest))  # nx4
# Get the Optimal Wieghts, beta_hat
Beta_hat = np.append(w0,w)
# This is MLR model, Y = X*B
z = np.dot(X,Beta_hat)
# Pass it in through the sigmoid
yhat = 1 / (1 + np.exp(-z))
yhat = np.where(yhat > 0.5, 1, 0)
accuracy2 = np.mean(np.equal(yhat,Ttest))
difference = accuracy1 - accuracy2
print("Accuracy1 is:\n {}\nAccuracy2 is:\n {} \nThe difference is:\n {}".format(accuracy1, accuracy2, difference))
    

#################
## Question 4C ##
#################
# plt.figure() # Not needed here as we are using the profs imported functions. 
bl3d.plot_db(Xtrain,Ttrain,w,w0,30,5)
plt.suptitle("Question  4(c):  Training data and decision boundary", fontsize=12)
plt.show()

#################
## Question 4D ##
#################
bl3d.plot_db(Xtrain,Ttrain,w,w0,30,20)
plt.suptitle("Question  4(d):  Training data and decision boundary", fontsize=12)
plt.show()

print("\n\nQuestion 4(c-f):")
print("--------------")
print("The following part(s) had no specific output to print.")

############################################################
######################## QUESTION 5 ########################
############################################################

# Question 5
print("\n\nQuestion 5(a):")
print("--------------")
print("The following part did not require specific output.")


# Helper Functions for Q5 
# ----------------------- 

def sigmoid(z):
    """ 
    Return the sigmoid version of the given equation.
    """
    return 1 / ( 1+ np.exp(-z) )

def predict(X,w):
    """
    Given the Matrix X, and vector of weights w, 
    Return the prediction. 
    """
    # Linear Model 
    z = np.dot(X,w)
    # Hypothesis
    y = sigmoid(z)
    return y
    
def cross_entropy(y,t):
    """ 
    Return the Cross Entropy loss between predicted y and t
    """
    return -t*np.log(y) - (1-t)*np.log(1-y)

def predicted_labels(y):
    """
    Take in the predicted values of Yhat, and get labels for them.  
    """
    return np.where(y > 0.5, 1, 0)

def get_accuracy(y,t):
    """
    Given two nx1 vectors; predicted, and the True labels, 
    Return how accurate each prediction was. 
    """
    return np.mean(np.equal(y,t))


def gd_logreg(lrate):
    
    """
    Given the learning rate, lrate, preform gradient descent. 
    The following function should work for any dimensionality.
    Assume the data variables are global variables.
    Return the wieghts vector, and record 
    ----------
    lrate : TYPE: Int
        The Learning Rate Hypyer Parameter in Gradient Descent.

    Returns
    -------
    Wieght Vector, w. 
    """
    
    # Part A
    # Set seed for repetion 
    rnd.seed(3)
    
    # Add a column of 1s to the Xtrain and Ttrain for bais
    X_train = np.column_stack(( np.ones(len(Xtrain)) , Xtrain ))
    X_test = np.column_stack(( np.ones(len(Xtest)) , Xtest ))
    
    # Reshape Labels from (n,) to (n,1)
    T_train = Ttrain.reshape( (len(Ttrain), 1) )
    T_test = Ttest.reshape( (len(Ttest), 1) )
    
    # Initialize the weight vector with the bais term. 
    # Add 1 for the bais term.
    cols  = ( Xtrain.shape[1] ) + 1
    w = np.random.randn(cols,1) / 1000

    # Initialize some Arrays to record progress
    TrainLCE, TestLCE, TrainAccuracy, TestAccuracy = [], [], [], []
    
    difference = 1
    iterations = 0
    # Main Gradient Descent Loop
    while (difference >= 1e-10):
        # Get the prediction for training data
        y_train = predict(X_train, w)
     
        # Do the same for Test Data 
        y_test = predict(X_test,w)
    
        # backprop -- Compute the derivate
        dLdw = np.dot(X_train.T,(y_train-T_train))
        # Take the average cost derivative for each feature
        dLdw /= len(Xtrain)
        # Update the wieghts
        w = w - lrate*dLdw
        
        # Preform cross entropy loss
        test_cost = np.mean(cross_entropy(y_test ,T_test)) 
        train_cost = np.mean(cross_entropy(y_train,T_train))
    
        # Record test and training accuracies and cost for progress. 
        # find accuracys for the train and test datasets
        y_train, y_test = predicted_labels(y_train), predicted_labels(y_test)
        TrainAccuracy.append(get_accuracy(y_train,T_train))
        TestAccuracy.append( get_accuracy(y_test,T_test) )
        
        # Add costs to the lists
        TrainLCE.append(train_cost)
        TestLCE.append(test_cost)
        
        # Find the difference between two consecutive updates
        if len(TrainLCE) == 1:
            difference = TrainLCE[0]     #Get the first element
        else: 
            difference = TrainLCE[-2] - TrainLCE[-1] #Get the 2nd last - last
            
        # Increase iteration count 
        iterations+=1
        
    # PART (f)
    # Plot the Training & Test Loss.  
    plt.figure()
    # Plot the training cross entropies (blue)
    plt.plot(TrainLCE,color='blue')
    # Plot the testing cross entropies (red) 
    plt.plot(TestLCE,color='red')
    # Plot title and x and y axis labels 
    plt.suptitle("Question 5: Training and test loss v.s. iterations")
    plt.xlabel("Iteration number")
    plt.ylabel("Cross entropy")
    
    
    # PART (g) 
    # Plot the previous figure using a log scale for horizontal axis.
    plt.figure()
    # Plot the training cross entropies (blue)
    plt.semilogx(TrainLCE,color='blue')
    # Plot the testing cross entropies (red) 
    plt.semilogx(TestLCE,color='red')
    # Plot title and x and y axis labels 
    plt.title("Question 5: Training and test loss v.s. iterations (logscale)")
    plt.xlabel("Iteration number")
    plt.ylabel("Cross entropy")
    
    
    # PART (h)
    # Plot the list of Training and Testing Accuracies along a log scale
    plt.figure()
    # Plot the training accuracy (blue)
    plt.semilogx(TrainAccuracy,color='blue')
    # Plot the testing accuracy (red) 
    plt.semilogx(TestAccuracy,color='red')
    # Plot title and x and y axis labels 
    plt.title("Question 5: Training and test accuracy v.s. iterations (log scale)")
    plt.xlabel("Iteration number")
    plt.ylabel("Accuracy")
    
    
    # PART (i)
    # Plot the last 100 training cross entropies.  
    plt.figure()
    plt.plot(TrainLCE[-100:],color='blue')
    # Plot title and x and y axis labels 
    plt.title("Question 5: last 100 training cross entropies")
    plt.xlabel("Iteration number")
    plt.ylabel("Cross entropy")
    
    
    # PART (j)
    # Plot all but the first 50 test cross entropies
    plt.figure()
    plt.semilogx(TestLCE[51:],color='blue')
    # Plot title and x and y axis labels 
    plt.title("Question 5: test loss from iteration 50 on (log scale)")
    plt.xlabel("Iteration number")
    plt.ylabel("Cross entropy")
    
    return w, iterations, lrate
    
LogReg_GD = gd_logreg(1)
print(f"The weight vector is:\n {LogReg_GD[0]}")
print(f"The number of iterations are: {LogReg_GD[1]}")
print(f"The Learning Rate is: {LogReg_GD[2]}")
print(f"Q4 bais: {w0}\nQ4 weights: \n {w} ")

# PART (k)
# Extract the bais and weights.
w0 = LogReg_GD[0][0]
w = LogReg_GD[0][1:]
bl3d.plot_db(Xtrain,Ttrain,w,w0,30,5)
plt.suptitle("Question 5: Training data and decision boundary", fontsize=12)
plt.show() 


############################################################
######################## QUESTION 6 ########################
############################################################

# Question 6
print("\n\nQuestion 6(a-b):")
print("--------------")
print("The following part(s) did not require specific output.")

# Read in the data File
with open("mnistTVT.pickle","rb") as f:
    Xtrain,Ttrain,Xval,Tval,Xtest,Ttest = pickle.load(f)
 
    
#####################
##### PART (a)  #####
#####################

def get_digits(a,b,TrainData,TestData):
    """
    Given two numbers a,and b, along with 2 datasets -- Train & Test,
    Get a reduced dataset with just the specified digits. 
    Parameters
    ----------
    a : A digit from 1 to 10.
    b : A second digit from 1 to 10.
    TrainData: The Training Dataset 
    TestData: The Testing Dataset

    Returns
    -------
    Reduced Version of the Data (Features and Labels), which 
    contains only the digits specified. 
    """
    # Get the digit indices from there labels, then get the corresponding rows.
    SubsetFeatures = TrainData[(TestData == a ) | (TestData == b)] 
    # Extract the labels for the specified digits
    SubsetLabels =  TestData[(TestData == a) | (TestData == b)]
    return SubsetFeatures, SubsetLabels

# Get digits 5 and 6 from MNST dataset. 
TrainSubset, TrainLabels = get_digits(5,6,Xtrain,Ttrain)  # (9444, 784) & (9444,)
ValSubset, ValLabels = get_digits(5,6,Xval,Tval)          # (1895, 784) & (1895,) 
TestSubset, TestS_Labels = get_digits(5,6,Xtest,Ttest)      # (1850, 784) & (1850,)

# Smaller Version of Training Data for testing error. 
TrainError, TE_labels = TrainSubset[:2000], TrainLabels[:2000] # (2000, 784) & (2000,)

#####################
##### PART (b)  #####
#####################

# Doesnt work yet. 
def plot_digits(data, question):
    """
    Given a Dataset, extract and display the first 16 digits. 
    
    Parameters
    ----------
    data : A dataset of numbers in pixels, each row corresponds to 1 number.
    question: The corresponding question number to plot the data for
    
    Returns
    -------
    None.

    """
    img = 1
    plt.figure()
    for image in data:
        plt.subplot(4,4,img)
        plt.axis("off")
        plt.imshow(image.reshape(28,28),cmap="Greys", aspect="auto", interpolation="nearest")
        img +=1
    plt.suptitle("Question {}: 16 MNIST training images".format(question))
    return None 

# Plot Figures for Part 6(b).
data = TrainError[:16]
question = "6(b)"
plot_digits(data, question) 

#####################
##### PART (c)  #####
#####################

def KNN(k,TrainData, TrainLabels, TestData, TestLabels):
    """
    Given the Training and and Testing Datasets, return how accurate
    the Model was for that specific K.
    
    Parameters
    ----------
    TrainData : Training Data.
    TrainLabels : Training Labels.
    TestData : Data to evaluate our Model. This can be  Validation, Test, or Training Error Data.
    TestLabels : Labels to evaluate our Model. This can be  Validation, Test, or Training Error Data

    Returns
    -------
    The accuracy of the model with the given K, and k itself.

    """
    # Fit the model
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(TrainData,TrainLabels)
        
    # Compute Accuracy on Testing Dataset
    ModelScore = model.score(TestData,TestLabels)
    
    return ModelScore
      
def FindOptimal_K(TrainData, TrainLabels, ValData, ValLabel, TEdata, TElabels):
    """
    Given the Training and and Testing Datasets, find the optimal
    value of K, the HyperParameter for K-Nearest Neighbour Algorithm 
    by evaluating the model on Odd Values of 1-19 as K. 
    
    Parameters
    ----------
    TrainData : Training Data.
    TrainLabels : Training Labels.
    ValData : Validation Data
    ValLabels : Validation Labels
    TEdata: The smaller training dataset. (2000,784) 
    TElabels: The smaller training dataset labels (2000,)
    Returns
    -------
    The Best Value of K, and the Corresponding Validation & Test Error.

    """
    
    val_acclst = []   # The validation Accuracy
    TE_acclst = []    # The small version of the Training Dataset
    kvalue = []       # The values of k we are on
    for K in range(1,20,2):
        val_acclst.append( KNN(K,TrainData, TrainLabels, ValData, ValLabel) )
        TE_acclst.append( KNN(K,TrainData, TrainLabels, TEdata, TElabels) )
        kvalue.append(K)
        
    # Determine the Optimal Value of K from the Validation set
    HighestIndex = int(np.argmax(val_acclst))
    Best_k = kvalue[HighestIndex]
    # Get that highest value. 
    Highest_val_acc = np.max(val_acclst)
    
    return val_acclst, TE_acclst, Best_k, Highest_val_acc

def Plot_Figure(Question, ValAccuracy, TEaccuracy, a,b):
    """
    Given the question number, and two list of accuracies, plot them in a 
    single figure. 
    
    Parameters
    ----------
    a: One of the Digits in Binary Classification (5 or 4 in our case)
    b: The other digit in Binary Classification  (6 or 7 in our case)
    Question : The corresponding question number, and the part. Ex: 6(a) 
    ValAccuracy : The list of Validation Accuracies.
    TEaccuracy : The list of Training Error Accuracies .

    Returns
    -------
    None.
    """
    
    plt.figure()
    # Plot the training accuracies (blue)
    plt.plot(TEaccuracy,color='blue')
    # Plot the validation accuracies (red) 
    plt.plot(ValAccuracy,color='red')
    # Plot title and x and y axis labels 
    plt.title("Question {}: Training and Validation Accuracy for KNN, digits {} and {}".format(Question,a,b))
    plt.xlabel("Number of Neighbours, K")
    plt.ylabel("Accuracy")
    
    return None 


# Get the Optimal K and Validation Accuracy List, Training Accuracy List. 
VAL_accLst, TE_accLst, Best_K, ValAcc = FindOptimal_K(TrainSubset, TrainLabels, ValSubset, ValLabels, TrainError,TE_labels)

# Plot the Validation and Training Accuracy in 1 figure.
question = "6(c)"
Plot_Figure(question,VAL_accLst,TE_accLst, 5, 6)

# Use the Best Value of K on the Reduced Test Set. 
TestAccuracy = KNN(Best_K,TrainSubset,TrainLabels,TestSubset, TestS_Labels)

print("\nQuestion 6(c):")
print("--------------")
print("The Best Value of K is: {}".format(Best_K))
print("The Validation Accuracy for the Best K is: {}".format(ValAcc))
print("The Test Accuracy for the Best K is: {}".format(TestAccuracy))
    
#####################
##### PART (d)  #####
#####################

# Repeat Part A: 
# Get digits 4 and 7 from MNST dataset. 
TrainSubset2, TrainLabels2 = get_digits(4,7,Xtrain,Ttrain)  
ValSubset2, ValLabels2 = get_digits(4,7,Xval,Tval)          
TestSubset2, TestS_Labels2 = get_digits(4,7,Xtest,Ttest)      
# Smaller Version of Training Data for testing error. 
TrainError2, TE_labels2 = TrainSubset2[:2000], TrainLabels2[:2000] 

# Repeat Part B:
# Plot Figures for Part 6(d).
data = TrainError2[:16]
question = "6(d)"
plot_digits(data, question)


# Repeat Part C:
# Get the Optimal K and Validation Accuracy List, Training Accuracy List for digits 4 and 7 now. 
VAL_accLst, TE_accLst, Best_K, ValAcc = FindOptimal_K(TrainSubset2, TrainLabels2, ValSubset2, ValLabels2, TrainError2,TE_labels2)

# Plot the Validation and Training Accuracy in 1 figure for digits 4 and 7

question = "6(d)"
Plot_Figure(question,VAL_accLst,TE_accLst, 4, 7)

# Use the Best Value of K on the Reduced Test Set for Digits 4 and 7. 
TestAccuracy = KNN(Best_K,TrainSubset2,TrainLabels2,TestSubset2, TestS_Labels2)

print("\nQuestion 6(d):")
print("--------------")
print("The Best Value of K is: {}".format(Best_K))
print("The Validation Accuracy for the Best K is: {}".format(ValAcc))
print("The Test Accuracy for the Best K is: {}".format(TestAccuracy))


print("\n\nQuestion 6(e-g):")
print("--------------")
print("The following part(s) did not require specific output.")
