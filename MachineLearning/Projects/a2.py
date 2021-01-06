"""
Created on Tue Nov 10 20:25:12 2020

@author: Tahir Muhammad
@StudentID: 1002537613
"""

# import relevant libraries
import numpy as np
import pickle as pickle
import matplotlib.pyplot as plt
import bonnerlib2D as bl2d
import sklearn.linear_model as lin
import sklearn.neural_network as nn
import sklearn.utils

from scipy.stats import multivariate_normal
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

# Set the QT5 Backgroung
# import matplotlib
# matplotlib.use("Qt5Agg")


#----------------------------------------------------------------#
#----------------------------------------------------------------#
#                           QUESTION 1                           #
#----------------------------------------------------------------#
#----------------------------------------------------------------#


# Read in the Data file.  
with open("dataA2Q1.pickle", "rb") as f:
    dataTrain,dataTest = pickle.load(f) 
    
	
# Define the Helper functions

def featurize(data,K):
    """
    Use basis expansion to featurize your data.
    
    Parameters
    ----------
    data : Any time of data of (x,y) pairs.
    K : The hyperparameter K, encodes x onto 2K + 1 featured values.

    Returns
    -------
    z : The featurized matrix of the data. 
    """
    vec1 = np.arange(1,K+1,1) # Values 1, ... , K.
    vec2 = np.dot(data[:,None],vec1[None]) # Encode the value of x for each value 1, ... , K. 
    sinv = np.sin(vec2) # Apply the sin function to all those values
    cosv = np.cos(vec2) # Apply the cosine function to all those values
    vector = np.concatenate((sinv,cosv),axis=1) # Add them both horizontally
    z = np.column_stack((np.ones(len(data)),vector))  # Add column of 1s to every row (at the front) 

    # Return the featurized matrix z. 
    return z

def error(y, t):
    """
    Given the predicted values and the true labels, return the mse loss.

    Parameters
    ----------
    y : Predicted vector.
    t : Ground Truth labels.

    Returns
    -------
    MSE Loss
    """
    
    return np.mean((t - y)**2) 

def printOutput(K, trainError, testError, w):
    """
    Print out the results from fitplot.
    """
    print("The value of K is:{}".format(K))
    print(f"The training error is: {trainError}")
    print(f"The testing error is: {testError}")
    print(f"The weight vectoris:\n{w}")

    return None

# Main Function
def fit_plot(dataTrain,dataTest,K):
	"""
	Given the dataset and a value of K, featurize the dataset
	and plot the corresponding figures.
	
	Parameters
	----------
	dataTrain : The Training Data.
	dataTest : The Testing Data.
	K : The Hyperparameter K. It represents the lenght of the featurized vector.
	
	Returns
	-------
	w : The weight vector, of optimal weights from trianing.
	train_error : The Training error.
	test_error : The Testing error.
	
	"""
	
	# get the featurized matrix z 
	z = featurize(dataTrain[0],K) # This is (25,2k+1)
	
	# Preform the Least squares regression fit
	w = np.linalg.lstsq(z,dataTrain[1], rcond=None)[0] # (2k+1,)
	w = w[:, np.newaxis] # For matmult, shape (2k+1, 1) now
	
	# Compute the prediction
	y_pred = np.matmul(w.T ,z.T) 
	
	# Compute the Train error
	train_error = error(y_pred,dataTrain[1])
	
	# Compute the Test error
	test_z = featurize(dataTest[0],K) # This is (1000,2k+1)# Featurize the x values
	test_y = np.matmul(w.T, test_z.T) 
	test_error = error(test_y,dataTest[1])
	
	# Plot the figures
	data = dataTrain
	# Get X vector of 1000 values between min and max
	xList = np.linspace(np.max(data[0]) ,np.min(data[0]) ,1000)
	
	# Featurize X vector, and matrix multiply with the wieghts to get corresponding y
	zList = featurize(xList,K)
	yList = np.matmul(w.T, zList.T)  
	
	# plotting
	xList = xList[:, np.newaxis] #Need this to match (1000,1) for y. 
	
	# plt.figure()
	plt.ylim(ymax = np.max(data[1])+5, ymin= np.min(data[1]) -5)
	plt.scatter(data[0], data[1], c="b", s=20) # scatter plot 
	plt.plot(xList, yList.T, c = "r")
	plt.xlabel("x")
	plt.ylabel("y")
	# plt.show()
	
	# Return weights, Test and Train MSE.
	return w, train_error, test_error


#####################
##### PART (b)  #####
#####################

print("\n\nQuestion 1(b):")
print("--------------")
plt.figure()
w, train_err, test_err = fit_plot(dataTrain,dataTest,K=3)
plt.suptitle("Question 1(b): the fitted function (K=3).")
printOutput(3, train_err, test_err, w)
plt.show() 
plt.close()
print(" ")

#####################
##### PART (c)  #####
#####################

print("\n\nQuestion 1(c):")
print("--------------")
plt.figure() 
w, train_err, test_err = fit_plot(dataTrain,dataTest,K=9)
plt.suptitle("Question 1(c): the fitted function (K=9).")
printOutput(9, train_err, test_err, w)
plt.show()
plt.close()
print(" ")

#####################
##### PART (d)  #####
#####################

print("\n\nQuestion 1(d):")
print("--------------")
plt.figure() 
w, train_err, test_err = fit_plot(dataTrain,dataTest,K=12)
plt.suptitle("Question 1(d): the fitted function (K=12).")
printOutput(12, train_err, test_err, w)
plt.show()
plt.close() 
print(" ")

#####################
##### PART (e)  #####
#####################

Question1eFigure = plt.figure()
for i in range(1,13):
    Question1eFigure.add_subplot(4,3,i)
    fit_plot(dataTrain,dataTest,i)
    Question1eFigure.suptitle("Question 1(e): fitted functions for many values of K.")	
plt.show() 
plt.close() 
print(" ")

#####################
##### PART (f)  #####
#####################

print("\n\nQuestion 1(f):")
print("--------------")

# Get the data 
train_x = dataTrain[0]
train_y = dataTrain[1]

# split the data into 5 equal folds, where each row represents a fold. 
train_x = np.array( np.split(train_x,5) )  
train_y = np.array( np.split(train_y,5) ) # Corresponding labels of the fold. 

# Initialize some arrays for tracking 
avg_trianError, avg_valError, K_values = [], [], []

# For each value of K
for k in range(1,13):

    # Tracking the errors for each fold and the value of K.
    val_error, train_error, K_value = [], [], []

    # Validate on 1 fold, train on rest, for all 5 folds.
    for j in range(5):

        # Keep the j'th fold for validating
        valset = train_x[j]
        valLabels = train_y[j] # Get it's corresponding labels

        # Rest of the folds are for training 
        trainset = np.concatenate([train_x[:j],train_x[j+1:]], axis=0)  # Train on rest (besides Jth fold)
        trainLabels = np.concatenate([train_y[:j],train_y[j+1:]], axis=0) # Get the corresponding y values for that train set

        # Reshape them to be (n, 1) for featurization to work
        trainset = trainset.reshape((20,))
        trainLabels = trainLabels.reshape((20,))

        # Train the model on 4/5 folds, and compute prediction 
        z = featurize(trainset,k)
        w = np.linalg.lstsq(z,trainLabels, rcond=None)[0]
        w = w[:, np.newaxis]
        y = np.matmul(w.T, z.T)

        # Do the same for validation fold
        val_z = featurize(valset,k)
        val_y = np.matmul(w.T, val_z.T)

        # Compute and store the corresponding errors.
        train_error.append( error(y,trainLabels) )
        val_error.append( error(val_y,valLabels) )

    # Compute the average error for this value of K. 
    avg_trianError.append( np.mean(train_error) )
    avg_valError.append( np.mean(val_error) )
    K_values.append(k)
    
# Plot the figure
plt.figure()
plt.semilogy(K_values, avg_trianError, c="b")
plt.semilogy(K_values, avg_valError, c="r")
plt.xlabel("K")
plt.ylabel("Mean Error")
plt.title("Question 1(f): mean training and validation error.")
plt.show()
plt.close() 


# Choose the K with the smallest mean validation error.
# pnp.min(avg_valError)
# We can see that the smallest validation error is K = 2. 

# Repeat part B from 
plt.figure()
w, train_err, test_err = fit_plot(dataTrain,dataTest,K=2)
plt.suptitle("Question 1(f): the best fitted line (K=2).")
printOutput(2, train_err, test_err, w)
plt.show()
plt.close() 

#----------------------------------------------------------------#
#----------------------------------------------------------------#
#                           QUESTION 2                           #
#----------------------------------------------------------------#
#----------------------------------------------------------------#

#####################
##### PART (a)  #####
#####################

# Read in the data
with open("dataA2Q2.pickle", "rb") as file:
    dataTrain, dataTest = pickle.load(file)
Xtrain, Ttrain = dataTrain
Xtest, Ttest = dataTest

def plot_data(X,T):
	"""
	Return the plot with 0.1 margins, and colours with different classes.
	"""
	lowerMargin = np.min(X[:,0]) - 0.1 
	upperMargin = np.max(X[:,1]) + 0.1
	colors = np.array(["red", "blue", "green"])

	plt.figure()
	plt.scatter(X[:, 0], X[:, 1], s=2, c=colors[T]) # scatter plot
	plt.xlim(lowerMargin, upperMargin)
	plt.ylim(lowerMargin, upperMargin)
	return None

plt.figure()
plot_data(Xtrain,Ttrain)
plt.suptitle("Training data for Question 2")
plt.show()
plt.close() 

#####################
##### PART (b)  #####
#####################
print("\n\nQuestion 2(b):")
print("--------------")

# Method 1
# Create a classification object, clf
clf = lin.LogisticRegression(solver = "lbfgs", multi_class="multinomial")
clf.fit(Xtrain,Ttrain) # Learn a multi-nomial logistic-regression classifier
accuracy1 = clf.score(Xtest,Ttest) # Get accuracy

# Method (2): Using the method described on the assignment page
def accuracyLR(clf, X, T):
    """
     Compute the accuracy of the multinomial Logistic regression classifier
    """
    # Compute the vectorized multiple regression 
    z =  clf.intercept_ + np.matmul(X, clf.coef_.T)  # Compute the vectorized multiple regression 
    #  Normally we use softmax to transform it into probabilities for each class, which sum to 1. 
	# However, we don't actually need that for prediction itself.
    # Take the maximum value from each row
    labeled = np.argmax(z, axis = 1)
    # Count how many were labeled correctly, and divide by N. 
    return np.mean(labeled==T)

accuracy2 = accuracyLR(clf, Xtest, Ttest)
difference = accuracy2 - accuracy1
print("Accuracy1 is: ", accuracy1)
print("Accuracy2 is: ", accuracy2)
print("The difference is: ", difference)


# Plot the training data with the decision boundaries 
plt.figure()
plot_data(Xtrain,Ttrain)
bl2d.boundaries(clf)
plt.title("Question 2(b): decision boundaries for logistic regression")
plt.show()
plt.close() 


######################
##### PART 2(c)  #####
######################

print("\n\nQuestion 2(c):")
print("--------------")

# Gaussian Discriminant Analysis, Method 1 for calculating accuracy. 
clf = QuadraticDiscriminantAnalysis(store_covariance=True)
clf.fit(Xtrain,Ttrain)
accuracy1 = clf.score(Xtest,Ttest) 

# Method 2 
def accuracyQDA(clf,X,T):
	"""
	Compute and return the accuracy of Quadratic Discriminant Analysis classifier

	Parameters
	----------
	clf : QDA Classifier.
	X : Training Data.
	T : True Labels.

	Returns
	-------
	Accuracy of the classifier.

	"""
	# Get the corresponding attributes from the QDA classifier 
	mu, cov, priors = clf.means_,  clf.covariance_ , clf.priors_
	
	# Initialize an empty array to store the prosterior probabilities 
	Prosteriors = np.zeros(shape=(X.shape[0],len(mu)))  # This will be of shape (1800,3). 
	
	for i in range(len(mu)):
		pdf = multivariate_normal.pdf(X, mu[i], cov[i]) # Get the P(x|class)
		Prosteriors[:, i] =  pdf*priors[i] # Numerator of the bayes class. 
		
	# predict the function
	y = np.argmax(Prosteriors,axis=1)
		
	return np.mean(y==T)

accuracy2 = accuracyQDA(clf, Xtest, Ttest)
difference = accuracy2 - accuracy1
print("Accuracy1 is: ", accuracy1)
print("Accuracy2 is: ", accuracy2)
print("The difference is: ", difference)


# Plot the training data with the decision boundaries (QDA) 
plt.figure()
plot_data(Xtrain,Ttrain)
bl2d.boundaries(clf)
plt.title("Question 2(c): decision boundaries for quadratic discriminant analysis")
plt.show()
plt.close() 

######################
##### PART 2(d)  #####
######################

print("\n\nQuestion 2(d):")
print("--------------")


clf = GaussianNB()  # Gaussian Naive Bayes Classifier
clf.fit(Xtrain,Ttrain)  # Fit the data to the model
accuracy1 = clf.score(Xtest,Ttest) # Compute the score

def MultivariateNormalNB(clf, X, T):
	
	mu, simga = clf.theta_,  clf.sigma_
	return None 

# Method 2 
def accuracyNB(clf,X,T):
	"""
	Naive bayes classifier

	Parameters
	----------
	clf : The NB Classifier 
	X : The training data matrix
	T : The corresponding output 

	Returns
	-------
	Accuracy2.
	"""
	
	# Get mean and sigma 
	mu, sigma = clf.theta_,  clf.sigma_ 

	# Broadcast the input vector and mean into the correct shapes. 
	X = np.array(X)[:, np.newaxis, :]
	mu = np.array(mu)[np.newaxis, :, : ]

	# class k, feature xi has a Gaussian distribution with mu = 0, sigma=1
	p_x_givenK = ( np.exp( -((X-mu)**2)) ) / (2*(sigma**2))
	p_x_givenK = np.prod( (p_x_givenK/ (np.sqrt((2*np.pi*sigma))) ) , 2) 
	P_Class_givenX = clf.class_prior_ * p_x_givenK
	
	# Predict and get accuracy2
	y = np.argmax(P_xX, axis=1)
	accuracy2 = np.mean(y == T)
	return accuracy2 

accuracy2 = accuracyNB(clf, Xtest, Ttest)
difference = accuracy1 - accuracy2
print("Accuracy1 is: ", accuracy1)
print("Accuracy2 is: ", accuracy2)
print("The difference is: ", difference)

# Plot the training data with the decision boundaries for Naive Bayes Classifier
plt.figure() 
plot_data(Xtrain,Ttrain)
bl2d.boundaries(clf)
plt.title("Question 2(d): decision boundaries for Gaussian naive Bayes")
plt.show()
plt.close() 


#----------------------------------------------------------------#
#----------------------------------------------------------------#
#                           QUESTION 3                           #
#----------------------------------------------------------------#
#----------------------------------------------------------------#

######################
##### PART 3(a)  #####
######################

# Read in the data
with open("dataA2Q2.pickle", "rb") as file:
    dataTrain, dataTest = pickle.load(file)
Xtrain, Ttrain = dataTrain
Xtest, Ttest = dataTest


######################
##### PART 3(b)  #####
######################

print("\n\nQuestion 3(b):")
print("--------------")


# set seed for reproducability 
np.random.seed(0)  #set seed for reproducability 

def MLP_Classifier(Xtrain, Ttrain, Xtest, Ttest, hidden_layers, initial_lr, max_iters):
	"""
	Given training data and testing data along with a few other parameters of the MLP classifier,
	compute the MLP classification and return the accuracy of the model. 
	
	Parameters
	----------
	Xtrain : X values of the training set.
	Ttrain : T values of the training set.
	Xtest : X values of the testing set.
	Ttest : T values of the testing set.
	hidden_layers : A tuple of the number of layers.
	initial_lr : The initial learning rate value. 
	max_iterations : The training iteration values. 

	Returns
	-------
	accuracy : The score of how accurate was the MLP classifier. 

	"""
	clf = nn.MLPClassifier(activation="logistic",
						solver="sgd", 
						hidden_layer_sizes=(hidden_layers), 
						learning_rate_init=initial_lr,
						tol=10e-6,
						max_iter=max_iters)
	
	clf.fit(Xtrain,Ttrain)
	accuracy  = clf.score(Xtest,Ttest)
	print("The Test Accuracy is: ", accuracy)

    # Plot the training data with the decision boundaries
	plot_data(Xtrain,Ttrain)
	bl2d.boundaries(clf)
	
	return accuracy

plt.figure()
MLP1 = MLP_Classifier(Xtrain, Ttrain, Xtest, Ttest, hidden_layers=(1,), initial_lr=0.01, max_iters=1000)
plt.suptitle("Question 3(b): Neural Net with 1 hidden unit")
print("The Test Accuracy with 1 Hidden layer is: ", MLP1)
plt.show() 
plt.close() 

######################
##### PART 3(c)  #####
######################

# set seed for reproducability 
np.random.seed(0) 

print("\n\nQuestion 3(c):")
print("--------------")

plt.figure()
MLP2 = MLP_Classifier(Xtrain, Ttrain, Xtest, Ttest, hidden_layers=(2,), initial_lr=0.01, max_iters=1000)
plt.suptitle("Question 3(c): Neural Net with 2 hidden unit")
print("The Test Accuracy with 2 Hidden layer is: ", MLP2)
plt.show()
plt.close() 

######################
##### PART 3(d)  #####
######################
# set seed for reproducability 
np.random.seed(0)

print("\n\nQuestion 3(d):")
print("--------------")
plt.figure() 
MLP3 = MLP_Classifier(Xtrain, Ttrain, Xtest, Ttest, hidden_layers=(9,), initial_lr=0.01, max_iters=1000)
plt.suptitle("Question 3(d): Neural Net with 9 hidden unit")
print("The Test Accuracy with 9 Hidden layer is: ", MLP3)
plt.show()
plt.close() 
 

######################
##### PART 3(e)  #####
######################

print("\n\nQuestion 3(e):")
print("--------------")
# set seed for reproducability 
 # Not too sure why subplot wasnt working.  
np.random.seed(0)

"""
 # COMMENTING OUT THE CODE AS IT IS NOT IN 1 PLOT -- 
 # BUT THE PLOTS ARE CORRECT! Please Print and see.
 # Not too sure why subplot wasnt working.  
Question3eFigure = plt.figure()
for i in range(1,10):
	Question3eFigure.add_subplot(3,3,i)
	MLP_Classifier(Xtrain, Ttrain, Xtest, Ttest, hidden_layers=(7,), initial_lr=0.01, max_iters=2**(i+1))
	Question3eFigure.suptitle("Question 3(e): different number of epochs.")
plt.show()
plt.close()
print(" ")
 """

######################
##### PART 3(f)  #####
######################

# set seed for reproducability 
np.random.seed(0)

""" COMMENTING OUT THE CODE AS IT IS NOT IN 1 PLOT -- 
 BUT THE PLOTS ARE CORRECT! Please Print and see.
 # Not too sure why subplot wasnt working. 
 
Question3fFigure = plt.figure()
for i in range(1,10):
	np.random.seed(i)
	Question3fFigure.add_subplot(3,3,i)
	MLP_Classifier(Xtrain, Ttrain, Xtest, Ttest, hidden_layers=(5,), initial_lr=0.01, max_iters=1000)
Question3fFigure.suptitle("Question 3(f): different initial weights.")
plt.show()
plt.close()
print(" ")

 """

######################
##### PART 3(g)  #####
######################

print("\n\nQuestion 3(g):")
print("--------------")

# Helper Function
def sigmoid(z):
    """ 
    Return the sigmoid version of the given equation.
    """
    return 1 / ( 1 + np.exp(-z) )	

def accuracyNN(clf,X,T):
	"""
	Ccomputes and returns the accuracy of classifier clf on data X,T, where clf
	is a neural network with one hidden layer.

	Parameters
	----------
	clf : The MLP Classification object
	X : X values of the testing data.
	T : The corresponding true labels of the data.

	Returns
	-------
	The score of how accurate was the MLP Classifier with 1 Hidden layer.
	
	"""
	
	# Compute the forward propogation 
	z1 = np.dot(X,clf.coefs_[0]) + clf.intercepts_[0]   # weighted sum of the inputs
	h1 = sigmoid(z1)						            # First hidden layer
	z2 = np.dot(h1, clf.coefs_[1]) + clf.intercepts_[1] # weighted sum passed onto next layer 
	y = np.argmax(z2, axis=1)                           # output layer
	
	return np.mean(y==T)

# Script for calling accuracy1 and accuracy 2, for Question 3(g) 
# Set seed for reproducability
np.random.seed(0) 

clf = nn.MLPClassifier(activation="logistic",
					   solver="sgd", 
					   hidden_layer_sizes=((9,)), 
					   learning_rate_init=0.01,
					   tol=10e-6,
					   max_iter=1000)

clf.fit(Xtrain,Ttrain)
accuracy1 = clf.score(Xtest,Ttest)
accuracy2 = accuracyNN(clf,Xtest,Ttest)

# print statements 
difference = accuracy2 - accuracy1
print("Accuracy1 is: ", accuracy1)
print("Accuracy2 is: ", accuracy2)
print("The difference is: ", difference)


######################
##### PART 3(h)  #####
######################

print("\n\nQuestion 3(h):")
print("--------------")

def softmax(z):
    """
	The softmax activate funcion 
    Return the probability of each class
    """
    denominator = np.sum( np.exp(z), axis=1)
    return np.exp(z) / denominator.reshape(denominator.shape[0],1)

def ceNN(clf, X, T):
	"""
	Compute and return the cross entropy of the MLP classifier in two ways. 

	Parameters
	----------
	clf : The MLP classifier.
	X : The X testing points.
	T : The corresponding True labels.

	Returns
	-------
	CE_1 : Cross Entropy Loss computed from the sklearn build in methods
	CE_2 : Cross Entropy Loss computed from scratch. 
	"""
	# Method 1
	# Get the logarithm of the probabilities for each class. 
	logProbabilities = clf.predict_log_proba(X) 
	
	# Encode labels a one hot vector  	   # We use np.unique() to be able to do
	labels = np.eye(len(np.unique(T)))[T]  # this for any # of classes
	   
	# Method 2
	# Compute the forward propogation 
	z1 = np.dot(X,clf.coefs_[0]) + clf.intercepts_[0]   # weighted sum of the inputs
	h1 = sigmoid(z1)						            # First hidden layer
	z2 = np.dot(h1, clf.coefs_[1]) + clf.intercepts_[1] # weighted sum passed onto next layer 
	y = softmax(z2) 		                            # Use softmax to get the probability for each class. 
	
	# Compute the Cross entropy loss --  
	CE1 = np.sum( -labels * logProbabilities ) / len(labels)
	CE2 = np.sum( -labels * np.log(y) ) / len(labels)
	
	return CE1, CE2

# Set seed for reproducability
np.random.seed(0) 

clf = nn.MLPClassifier(activation="logistic",
					   solver="sgd", 
					   hidden_layer_sizes=((9,)), 
					   learning_rate_init=0.01,
					   tol=10e-6,
					   max_iter=1000)
   
# Define the classifier 
clf.fit(Xtrain,Ttrain)
CE1, CE2 = ceNN(clf, Xtest, Ttest)

difference = CE2 - CE1
print("The Cross Entropy Loss, CE1 is: ", CE1)
print("The Cross Entropy Loss, CE2 is: ", CE2)
print("The difference is: ", difference)

# Question 4 -- All non programming question, in pdf attached. 

#----------------------------------------------------------------#
#----------------------------------------------------------------#
#                           QUESTION 5                           #
#----------------------------------------------------------------#
#----------------------------------------------------------------#

######################
##### PART 5(a)  #####
######################

# Read in the data File
with open("mnistTVT.pickle","rb") as f:
    Xtrain,Ttrain,Xval,Tval,Xtest,Ttest = pickle.load(f)
	
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

# Get digits 5 and 6 from MNST dataset (subset of the data)
sub_Xtrain, sub_Ttrain = get_digits(5,6,Xtrain,Ttrain)  # (9444, 784) & (9444,)
sub_Xtest, sub_Ttest = get_digits(5,6,Xtest,Ttest)      # (1850, 784) & (1850,)

# Encode the values into 1s and 0s
sub_Ttrain = np.where(sub_Ttrain == 5,1,0)
sub_Ttest = np.where(sub_Ttest == 5, 1, 0)


######################
##### PART 5(b)  #####
######################

# Helper Functions
def sigmoid(z):
    """ 
    Return the sigmoid version of the given equation.
    """
    return 1 / ( 1 + np.exp(-z) )	

def cross_entropy(y,t):
    """ 
    Return the Cross Entropy loss between predicted y and t 
	(This is for a binary classification). 
    """
    return -t*np.log(y) - (1-t)*np.log(1-y)

def tanh(z):
	return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

def softmax(z):
    """
	The softmax activate funcion 
    Return the probability of each class
    """
    
    return np.exp(z) / np.sum(np.exp(z), axis=0)

def get_accuracy(clf,X,T):
	
	# Compute Accuracies 
	Accuracy1 = clf.score(X,T)
	
	# Method 2
	# Compute the forward propogation 
	z1 = np.matmul(X,clf.coefs_[0]) + clf.intercepts_[0]   # weighted sum of the inputs
	h1 = tanh(z1)						                # First hidden layer, tan activation
	z2 = np.matmul(h1, clf.coefs_[1]) + clf.intercepts_[1] # weighted sum passed onto next layer 
	h2 = tanh(z2) 										# Second hidden layer, tan activation 
	z3 = np.matmul(h2, clf.coefs_[2] + clf.intercepts_[2]) # output layer
	y = sigmoid(z3)                                     # Sigmoid for binary classification
	y = np.squeeze(np.where(y>0.5,1,0))                 # If value greater than 0.5, 1 else 0. 
	Accuracy2 = np.sum (np.equal(y,T)) / len(T) 
	
	return Accuracy1, Accuracy2  

def get_CrossEntropy(clf, X,T):
	# Method 1
	# Get the logarithm of the probabilities for each class. 
	logProbabilities = clf.predict_log_proba(X) 
	# Encode labels a one hot vector  	  # We use np.unique() to be able to do
	labels = np.eye(len(np.unique(T)))[T]  # this for any # of classes
	#labels = sigmoid(logProbabilities)
	CE1 = np.sum( -labels * logProbabilities ) / len(labels)
	
	# Method 2
	# Compute the forward propogation 
	z1 = np.matmul(X,clf.coefs_[0]) + clf.intercepts_[0]       # weighted sum of the inputs
	h1 = np.tanh(z1)						                # First hidden layer, tan activation
	z2 = np.matmul(h1, clf.coefs_[1]) + clf.intercepts_[1]     # weighted sum passed onto next layer 
	h2 = np.tanh(z2) 										# Second hidden layer, tan activation 
	z3 = np.matmul(h2, clf.coefs_[2] + clf.intercepts_[2])     # output layer
	# Pass it into sigmoid for binary classification
	y = np.squeeze(sigmoid(z3)) 
	# Compute the Cross entropy loss 
	CE2 = np.mean( (cross_entropy(y,T) ))
	
	return CE1, CE2


def evaluateNN(clf, X, T):
	
	# Get Accuracys from Method1 and Method2 	
	Accuracy1, Accuracy2 = get_accuracy(clf,X,T)
	
	# Get the Cross Entropy Loss from Method1, and Method2 
	CE1, CE2 = get_CrossEntropy(clf, X,T)
	
	return Accuracy1, Accuracy2, CE1, CE2

######################
##### PART 5(c)  #####
######################

# Set seed for reproducability
np.random.seed(0) 

# Define a MLP classifier object 
clf = nn.MLPClassifier(activation="tanh",
					   solver="sgd", 
					   hidden_layer_sizes=(100,100), 
					   learning_rate_init=0.01,
					   tol=10e-6,
					   batch_size=100,
					   max_iter=100)
					               

# Fit the data to the model
clf = clf.fit(sub_Xtrain, sub_Ttrain)
acc1, acc2, CE1, CE2 = evaluateNN(clf, sub_Xtest, sub_Ttest)

print("\n\nQuestion 5(c):")
print("--------------")
difference1 = acc1 - acc2
difference2 = CE2 - CE1
print("The accuracy from method 1 is: ", acc1)
print("The accuracy from method 2 is: ", acc2)
print("The difference is: ", difference1)

print("The Cross Entropy Loss, CE1 is: ", CE1)
print("The Cross Entropy Loss, CE2 is: ", CE2)
print("The difference is: ", difference2)


######################
##### PART 5(d)  #####
######################
	
# Record Accuracies and cross entropys
Acc_lst, CE_lst, batch_lst = [], [], []

# Model each Classifier with different batch size value
for i in range(0,14):
	
	# Set seed for reproducability
	np.random.seed(0)
	
	# Define the MLP classifier object
	clf = nn.MLPClassifier(activation="tanh",
					 solver="sgd",
					 hidden_layer_sizes=(100,100),
					 learning_rate_init=0.001,
					 tol = 10e-6,
					 batch_size=2**i,
					 max_iter=1)

	# Fit the data to the model
	clf = clf.fit(sub_Xtrain, sub_Ttrain)
	acc1, acc2, CE1, CE2 = evaluateNN(clf, sub_Xtest, sub_Ttest)
	
	# Store it for results 
	Acc_lst.append(acc2)
	CE_lst.append(CE2)
	batch_lst.append(2**i) 

# Plotting for 5(d)
plt.figure()
plt.semilogx(batch_lst, Acc_lst,color="blue")
# Plot the title and the x and y labels
plt.title("Question 5(d): Accuracy v.s batch size")
plt.xlabel("batch size")
plt.ylabel("Accuracy")

# Plotting for 5(d) -- CE vs batch size 
plt.figure(2)
plt.semilogx(batch_lst, CE_lst, color="blue")
# Plot the title and the x and y labels
plt.title("Question 5(d): Cross Entropy v.s batch size")
plt.xlabel("batch size")
plt.ylabel("cross entropy")

# part(e) is on the written part of the assignment. 

######################
##### PART 5(f)  #####
######################
print("\n\nQuestion 5(f):")
print("--------------")
print(" ")
#

# Get digits 5 and 6 from MNST dataset (subset of the data)
sub_Xtrain, sub_Ttrain = get_digits(5,6,Xtrain,Ttrain)  # (9444, 784) & (9444,)
sub_Xtest, sub_Ttest = get_digits(5,6,Xtest,Ttest)      # (1850, 784) & (1850,)

# Get the reduced dataset 
# Encode the values into 1s and 0s
X = sub_Xtrain 
T = np.where(sub_Ttrain == 5,1,0)
Test_X = sub_Xtest
Test_T = np.where(sub_Ttest == 5, 1, 0)

# helper functions 
def tanh(z):
	return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

def get_Acc(o,labels):
	""" 
	Get the accuracy of the prediction from our MLP classifier 
	"""
	o = np.squeeze(np.where(o>0.5,1,0))
	return np.sum (np.equal(o,labels)) / len(labels) 

def get_CE(o, T):
	""" 
	Get the Cross Entropy loss of our prediction from the MLP classifier. 
	"""
	o = np.squeeze(o) 
	# Compute the Cross entropy loss 
	cross_entropy = -T*np.log(o) - (1-T)*np.log(1-o)
	return np.mean( cross_entropy )

# Set seed
np.random.seed(0)

# initialize wieghts
W = np.random.normal(0, 1, (sub_Xtrain.shape[1], 100))  # this is of shaoe (input features, 100)
V = np.random.normal(0, 1, (100, 100))  # Hidden layer 1 of shape 100, 100 cz next layer also has 100 
U = np.random.normal(0, 1, (100, 1))    # Hidden layer 2 of shape 100, 1 as 1 output (sigmoid) 

# initialize bais terms same shape as the Wieght matrices they are being added with. 
w0 = np.zeros(100)
v0 = np.zeros(100)
u0 = np.zeros(1)

# Initialize the learning rate
learning_rate = 0.1

# Main Gradient Descent Loop
for iteration in range(0,11): 
	
	# Compute the Forward Pass for Training Data
	# Lets just call the ~ on top of variables t or tilda 
	x_t = np.matmul(X,W) + w0  	#(9334, 100) + bais 
	h = tanh(x_t)			    # First hidden layer, tan  # (9334,100)
	h_t = np.matmul(h,V) + v0 # weighted sum passed onto next layer  (9334,100)
	g = tanh(h_t) 				# Second hidden layer, tan activation 
	g_t = np.matmul(g,U) + u0 # Connecting layer between hidden layer2 and output layer
	o = sigmoid(g_t)           # Sigmoid results for binary classification
	
	# Compute the Forward Pass for TESTING ONLY
	x_t_test = np.matmul(Test_X,W) + w0  	#(9334, 100) + bais 
	h_test = tanh(x_t_test)			       # First hidden layer, tan  # (9334,100)
	h_t_test = np.matmul(h_test,V) + v0    # weighted sum passed onto next layer  (9334,100)
	g_test = tanh(h_t_test) 				# Second hidden layer, tan activation 
	g_t_test = np.matmul(g_test,U) + u0   # Connecting layer between hidden layer2 and output layer
	o_test = np.squeeze(sigmoid(g_t_test))            # Sigmoid results for binary classification
	
	
	print(f"Iteration:{ iteration }, Test accuracy: { get_Acc(o_test,Test_T) }")
	
	# Compute the Backward pass
	DC_DGtilda = o - T.reshape(len(T),1)
	DC_DU = np.matmul(g.T,DC_DGtilda)  
	DC_DG = np.matmul(DC_DGtilda,U.T)
	DC_DHtilda = (1 - g**2) * DC_DG
	DC_DV = np.matmul(h.T,DC_DHtilda) 
	DC_DH = np.matmul(DC_DHtilda, V.T)
	DC_DXtilda = (1-h**2)*DC_DH
	DC_DW = np.matmul(X.T, DC_DXtilda)
	preds = np.argmax(probas, axis=1)
	# Gradients with respect to the bais
	DC_du0 = DC_DGtilda.sum(axis=0)
	DC_dv0 = DC_DHtilda.sum(axis=0)
	DC_dw0 = DC_DXtilda.sum(axis=0) 
	

	# Preform weight updates, using Average Gradient
	W -= DC_DW * learning_rate / sub_Xtrain.shape[0]
	V -= DC_DV * learning_rate / sub_Xtrain.shape[0]
	U -= DC_DU * learning_rate / sub_Xtrain.shape[0]
	
	# Update the bais term 
	w0 -= DC_dw0 * learning_rate / sub_Xtrain.shape[0]
	v0 -= DC_dv0 * learning_rate / sub_Xtrain.shape[0]
	u0 -= DC_du0 * learning_rate / sub_Xtrain.shape[0] 
	
print(" ")
print(f"The final Test Accuracy is: { get_Acc(o_test,Test_T) }")	
print(f"The final Cross Entropy loss is: {get_CE(o_test,Test_T)}")


##############################################################################
########################### PART 5(g)  #######################################
##############################################################################

# Get digits 5 and 6 from MNST dataset (subset of the data)
sub_Xtrain, sub_Ttrain = get_digits(5,6,Xtrain,Ttrain)  # (9444, 784) & (9444,)
sub_Xtest, sub_Ttest = get_digits(5,6,Xtest,Ttest)      # (1850, 784) & (1850,)

# Get the reduced dataset 
# Encode the values into 1s and 0s
X = sub_Xtrain 
T = np.where(sub_Ttrain == 5,1,0)
Test_X = sub_Xtest
Test_T = np.where(sub_Ttest == 5, 1, 0)

def StochasticGD(X, T, Test_X, Test_T):
	# Set seed
	np.random.seed(0)
	
	# initialize wieghts
	W = np.random.normal(0, 1, (sub_Xtrain.shape[1], 100))  # this is of shaoe (input features, 100)
	V = np.random.normal(0, 1, (100, 100))  # Hidden layer 1 of shape 100, 100 cz next layer also has 100 
	U = np.random.normal(0, 1, (100, 1))    # Hidden layer 2 of shape 100, 1 as 1 output (sigmoid) 
	
	# initialize bais terms same shape as the Wieght matrices they are being added with. 
	w0 = np.zeros(100)
	v0 = np.zeros(100)
	u0 = np.zeros(1)
	
	# Initialize the learning rate
	learning_rate = 0.1
	
	# Main Gradient Descent Loop
	for epoch in range(0,11):
		
		# Initialize some arrays for tracking progress
		Test_Acc, CE_Acc = [], [] 
		
		# split the data into 10 equal folds, where each row represents a fold. 
		X = np.array( np.array_split(X,10) )  
		T = np.array( np.array_split(X,10) )  
		Test_X = np.array( np.array_split(Test_X,10) ) 
		Test_T = np.array( np.array_split(Test_T,10) ) 
		
		i = 0
		for i in range(10): 
			
			# Randomly shuffle the training data 
			X, T = sklearn.utils.shuffle(X,T)
			
			# Compute the Forward Pass
			# Lets just call the ~ on top of variables t or tilda 
			x_t = np.matmul(X,W) + w0  	#(9334, 100) + bais 
			h = tanh(x_t)			    # First hidden layer, tan  # (9334,100)
			h_t = np.matmul(h,V) + v0 # weighted sum passed onto next layer  (9334,100)
			g = tanh(h_t) 				# Second hidden layer, tan activation 
			g_t = np.matmul(g,U) + u0 # Connecting layer between hidden layer2 and output layer
			o = sigmoid(g_t)           # Sigmoid results for binary classification
			
			# Compute the Forward Pass for TESTING DATA ONLY
			x_t_test = np.matmul(Test_X,W) + w0  	#(9334, 100) + bais 
			h_test = tanh(x_t_test)			       # First hidden layer, tan  # (9334,100)
			h_t_test = np.matmul(h_test,V) + v0    # weighted sum passed onto next layer  (9334,100)
			g_test = tanh(h_t_test) 				# Second hidden layer, tan activation 
			g_t_test = np.matmul(g_test,U) + u0   # Connecting layer between hidden layer2 and output layer
			o_test = np.squeeze(sigmoid(g_t_test))            # Sigmoid results for binary classification
			
			# Compute the Backward pass
			DC_DGtilda = o - T.reshape(len(T),1)
			DC_DU = np.matmul(g.T,DC_DGtilda)  
			DC_DG = np.matmul(DC_DGtilda,U.T)
			DC_DHtilda = (1 - g**2) * DC_DG
			DC_DV = np.matmul(h.T,DC_DHtilda) 
			DC_DH = np.matmul(DC_DHtilda, V.T)
			DC_DXtilda = (1-h**2)*DC_DH
			DC_DW = np.matmul(X.T, DC_DXtilda)
			
			# Gradients with respect to the bais
			DC_du0 = DC_DGtilda.sum(axis=0)
			DC_dv0 = DC_DHtilda.sum(axis=0)
			DC_dw0 = DC_DXtilda.sum(axis=0) 
			
			
			# Preform weight updates, using Average Gradient
			W -= DC_DW * learning_rate / sub_Xtrain.shape[0]
			V -= DC_DV * learning_rate / sub_Xtrain.shape[0]
			U -= DC_DU * learning_rate / sub_Xtrain.shape[0]
			
			# Update the bais term 
			w0 -= DC_dw0 * learning_rate / sub_Xtrain.shape[0]
			v0 -= DC_dv0 * learning_rate / sub_Xtrain.shape[0]
			u0 -= DC_du0 * learning_rate / sub_Xtrain.shape[0] 
			
			# Update test and loss 
			Test_Acc.append( get_Acc(o_test,Test_T) )
			CE_Acc.append(get_CE(o_test,Test_T))
		
	return None
