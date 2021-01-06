"""
Created on Tue Nov 10 20:25:12 2020

@author: Tahir Muhammad
"""

# import relevant libraries
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.utils.testing import ignore_warnings
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.utils import resample
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from numpy import linalg as lin

#----------------------------------------------------------------#
#----------------------------------------------------------------#
#                           QUESTION 1                           #
#----------------------------------------------------------------#
#----------------------------------------------------------------#

# Read in the corresponding data file
with open("mnistTVT.pickle", "rb") as f:
	Xtrain, Ttrain, Xval, Tval, Xtest, Ttest = pickle.load(f)

Xtrain = Xtrain.astype(np.float64)
Xval = Xval.astype(np.float64)
Xtest = Xtest.astype(np.float64)
	
######################
##### PART 1(a)  #####
######################

def my_PCA(train_X,test_X, num_components):
	pca = PCA(n_components = num_components)
	pca.fit(train_X)
	reducedData = pca.transform(test_X)
	ProjectedData = pca.inverse_transform(reducedData)
	
	# Plotting -- Taken from A1 solutions, as I didn't want to mess 
	# up my plots this time :) 
	X = np.reshape(ProjectedData, [-1,28,28])
	plt.figure()

	for i in range(25):
		plt.subplot(5,5,i+1)
		plt.imshow(X[i], cmap="Greys")
		plt.axis("off")
		
	return None

# Call the function and plot it with 30 componenents
my_PCA(Xtrain,Xtest,30)
plt.suptitle("Question 1(a):  MNIST test data projected onto 30 dimensions.")


######################
##### PART 1(b)  #####
######################
my_PCA(Xtrain,Xtest,3)
plt.suptitle("Question 1(b):  MNIST test data projected onto 3 dimensions.")

######################
##### PART 1(c)  #####
######################
my_PCA(Xtrain,Xtest,300)
plt.suptitle("Question 1(c):  MNIST test data projected onto 300 dimensions.")

######################
##### PART 1(d)  #####
######################

def Cov(Xmat, mean):
	numrows = Xmat.shape[0]
	mean = np.mean(Xmat - mean, axis=0)
	numerator = np.transpose(Xmat - mean) @ (Xmat - mean )
	return (numerator / numrows)

def myPCA(X,K):

	# compute the mean which will represent the new origin 
	mu = np.mean(X, axis=0)
	X = X - mu
	
	# calcualte the covariance matrix
	cov_matrix = np.cov(X.T)

	# use eigh to get the eighn vector and values
	eigen_vals, eigen_vecs = lin.eigh(cov_matrix)

	# sort the eigen values and eigen vectors
	sorted_indices = np.argsort(eigen_vals)[::-1]
	eigen_vals, eigen_vecs = eigen_vals[sorted_indices] , eigen_vecs[:, sorted_indices]
	
	# Top K components 
	reduced_Evec = eigen_vecs[:,:K]
	
	# Lower dimensional representation 
	z = np.dot(X, reduced_Evec)
	# Project on the reduced dimensions
	proj = np.dot(z,reduced_Evec.T)
	# Add the mean back
	Xp = mu + proj  
	
	return Xp

######################
##### PART 1(f)  #####
######################

myXtrainP = myPCA(Xtrain,100)

def plot_PCA(X):
	X = np.reshape(X, [-1,28,28])
	plt.figure()
	
	for i in range(25):
		plt.subplot(5,5,i+1)
		plt.imshow(X[i], cmap="Greys")
		plt.axis("off")

	return None

plot_PCA(myXtrainP)
plt.suptitle("Question 1(f): MNISTdata projected onto 100 dimensions (mine)")

# With SKlearn
pca = PCA(n_components = 100, svd_solver="full" )
pca.fit(Xtrain)
reducedData = pca.transform(Xtrain)
XtrainP = pca.inverse_transform(reducedData)

# Plot the 25 digits made with sklearn
plot_PCA(XtrainP)
plt.suptitle("Question 1(f): MNISTdata projected onto 100 dimensions (sklearn)")

# Compute RMS
rms = np.sqrt( np.mean((XtrainP-myXtrainP)**2) )

print("\n\nQuestion 1(f):")
print("--------------")
print("The RMS is: ", rms)


#----------------------------------------------------------------#
#----------------------------------------------------------------#
#                           QUESTION 2                           #
#----------------------------------------------------------------#
#----------------------------------------------------------------#

# Read in the corresponding data file
with open("mnistTVT.pickle", "rb") as f:
	Xtrain, Ttrain, Xval, Tval, Xtest, Ttest = pickle.load(f)

Xtrain = Xtrain.astype(np.float64)
Xval = Xval.astype(np.float64)
Xtest = Xtest.astype(np.float64)
	
######################
##### PART 2(a)  #####
######################

# Extract the first 200 data MNST data points 
small_Xtrain = Xtrain[:200, :]
small_Ttrain = Ttrain[:200]

# Gaussian Discriminant Analysis, Method 1 for calculating accuracy. 
clf = QuadraticDiscriminantAnalysis(store_covariance=True)
ignore_warnings(clf.fit)(small_Xtrain, small_Ttrain)
train_acc = clf.score(small_Xtrain, small_Ttrain)
test_acc = clf.score(Xtest,Ttest) 

print("\n\nQuestion 2(a):")
print("--------------")
print("The training accuracy is: ", train_acc)
print("The testing accuracy is: ", test_acc)

######################
##### PART 2(b)  #####
######################

# Initialize arrays and counter for keeping track -- Think of Lambda as the regularization param
train_acc, val_acc, Lambda = [], [], []

# For each value of the lambda ... 
for i in range(0,21):
	# Gaussian Discriminant Analysis, Method 1 for calculating accuracy. 
	clf = QuadraticDiscriminantAnalysis(reg_param=2**-i)
	ignore_warnings(clf.fit)(small_Xtrain, small_Ttrain)
	Lambda.append(2**-i)
	train_acc.append( clf.score(small_Xtrain, small_Ttrain) )
	val_acc.append( clf.score(Xval,Tval) )

# Plotting the figure
plt.figure()
plt.semilogx(Lambda, train_acc, "b")
plt.semilogx(Lambda,val_acc, "r")
plt.xlabel("Regularization parameter")
plt.ylabel('Accuracy')
plt.suptitle('Question 2(b): Training and Validation Accuracy for Regularized QDA')

# Get the highest index val accuarcy
val_index = np.argmax(val_acc)

print("\n\nQuestion 2(b):")
print("--------------")
print("The Maximum Validation Accuracy is: ", val_acc[val_index])
print("The corresponding training Accuracy is: ", train_acc[val_index])
print("The corresponding value of reg_param, or Lambda is: ", Lambda[val_index])

######################
##### PART 2(c)  #####
######################

# Writing part
	

######################
##### PART 2(d)  #####
######################

def train2d(K,X,T):
	   
	# Reduce Dimensions with PCAPCA(n_components = num_components)
	pca = PCA(n_components = K, svd_solver="full" )
	pca.fit(X)
	reducedData = pca.transform(X)
	
	# Train the QDA classifier on the reduced dataset
	qda = QuadraticDiscriminantAnalysis()
	ignore_warnings(qda.fit)(reducedData,T)
	
	# Compute accuracy
	train_acc = qda.score(reducedData,T)
	
	return pca, qda, train_acc


def test2d(pca,qda,X,T):
	
	# Use PCA instance to reduce dataset
	reducedData = pca.transform(X)
	
	# Compute Accuracy from QDA
	accuracy = qda.score(reducedData,T)
	
	return accuracy


# Extract the first 200 data MNST data points 
small_Xtrain = Xtrain[:200, :]
small_Ttrain = Ttrain[:200]

# initialize arrays for storage
train_acc, val_acc, Kvalue = [], [], []

# Run a loop for 1-50 Values of k, and trian and test different components. 
for i in range(1,51):
	# Fit the classifiers on the 200 rows datasets
	PCA_, QDA_, Acc = train2d(i, small_Xtrain, small_Ttrain)
	train_acc.append(Acc)
	
	# Test these values 
	val_acc.append(test2d(PCA_, QDA_, Xval, Tval))
	Kvalue.append(i)
	
# Get the highest index val accuarcy
val_index = np.argmax(val_acc)

print("\n\nQuestion 2(d):")
print("--------------")
print("The Maximum Validation Accuracy is: ", val_acc[val_index])
print("The corresponding training Accuracy is: ", train_acc[val_index])
print("The corresponding value of K is: ", Kvalue[val_index])
	
# Plotting the required figure
plt.figure()
plt.plot(Kvalue, train_acc, "b")
plt.plot(Kvalue,val_acc, "r")
plt.xlabel("Reduced Dimension")
plt.ylabel('Accuracy')
plt.suptitle('Question 2(d): Training and Validation Accuracy for PCA + QDA')

######################
##### PART 2(e)  #####
######################

# Written Part

######################
##### PART 2(f)  #####
######################

def Optimization(X,T,Val_X,Val_T):
	
	# Initialize arrays for keeping track
	train_acc, val_acc, reg_paramater, Kvalue, accMaxK = [], [], [], [], []
	
	# We first want to get the reduced dataset
	for k in range(1,51):
		
		clf_pca = PCA(n_components = k, svd_solver="full" )
		clf_pca.fit(small_Xtrain)
		reducedData = clf_pca.transform(small_Xtrain)
		reducedValData = clf_pca.transform(Xval)
		
		
		# Store accuracy values for max K accuracy
		Acc_list = []
		
		# Then try different values of regularization on that dataset
		for i in range(0,21):
			# Define the QDA classifier
			clf_qda = QuadraticDiscriminantAnalysis(reg_param=2**-i)
			# Fit the reduced data from PCA
			ignore_warnings(clf_qda.fit)(reducedData,small_Ttrain)
			
			# Compute the training and validation accuracy
			train_accuracy = clf_qda.score(reducedData, small_Ttrain)
			val_acccuracy = clf_qda.score(reducedValData,Tval)
			
			# Append to corresponding lists
			reg_paramater.append(2**-i)
			train_acc.append(train_accuracy)
			val_acc.append(val_acccuracy)
			Acc_list.append(val_acccuracy)
			Kvalue.append(k)
		
		# Store the maximum K from the reg_params 
		accMaxK.append(max(Acc_list))
			
	# Get the highest index val accuarcy
	val_index = np.argmax(val_acc)
	accMax = val_acc[val_index]
	
	print("\n\nQuestion 2(f):")
	print("--------------")
	print("The Max Accuracy, accMax is: ", accMax)
	print("The corresponding training Accuracy is: ", train_acc[val_index])
	print("The corresponding value of the regularization parameter is: ", reg_paramater[val_index])
	print("The corresponding K value is: ", Kvalue[val_index])
	
	return accMaxK

MaximumKvalues = Optimization(small_Xtrain, small_Ttrain,  Xval, Tval)

# Plotting the required figure
yaxis = np.arange(1,51,1).tolist()
plt.figure()
plt.plot(yaxis, MaximumKvalues, "b")
plt.xlabel("Reduced Dimension")
plt.ylabel('Maximum Accuracy')
plt.suptitle('Question 2(f): Maximum validation accuracy for QDA')


#----------------------------------------------------------------#
#----------------------------------------------------------------#
#                           QUESTION 3                           #
#----------------------------------------------------------------#
#----------------------------------------------------------------#

# Read in the corresponding data file
with open("mnistTVT.pickle", "rb") as f:
	Xtrain, Ttrain, Xval, Tval, Xtest, Ttest = pickle.load(f)

Xtrain = Xtrain.astype(np.float64)
Xval = Xval.astype(np.float64)
Xtest = Xtest.astype(np.float64)

# Extract the first 200 data MNST data points 
small_Xtrain = Xtrain[:200, :]
small_Ttrain = Ttrain[:200]


######################
##### PART 3(a)  #####
######################

def myBootstrap(X,T):
	
	# Run loop until we get the required bootstrap sample
	unique = 0
	while(unique<3):
		# Get the bootstrap sample
		bootstrap_data, bootstrap_labels = resample(X, T, replace=True)
		
		# Test the booststrapped sample for unique values
		labels = len(np.unique(bootstrap_labels))
		if labels >= 4:
			unique = 3
		else:
			unique = 0
	
	return bootstrap_data,bootstrap_labels

######################
##### PART 3(b)  #####
######################

print("\n\nQuestion 3(b):")
print("--------------")

# Quadratic Discriminant Analysis -- Base Classifier 
clf = QuadraticDiscriminantAnalysis(reg_param=0.004)
ignore_warnings(clf.fit)(small_Xtrain, small_Ttrain)
val_acc = clf.score(Xval,Tval) 
print("The Validation Accuracy for the base classifier is: ", val_acc)


def Qda_Fit(X,T):
	""" A helper function to simplify the bootrapping and fitting of the QDA classifier"""
	
	# Get a bootstrap sample
	bootstrap_data, bootstrap_labels = myBootstrap(X, T)
	
	# Fit the QDA classifier to it
	qda_classifier = QuadraticDiscriminantAnalysis(reg_param=0.004)
	ignore_warnings(qda_classifier.fit)(bootstrap_data, bootstrap_labels)
	
	return qda_classifier


# Initialize a counter and an empty array to store probability matrices 
nrows = Xval.shape[0]
labels = len(np.unique(small_Ttrain))
Probablity_matrices = np.ones((nrows,labels))

for i in range(50):
	
	qda_classifier = Qda_Fit(small_Xtrain, small_Ttrain)
	# Predict on the validation dataset, and store the results
	Probablity_matrices += qda_classifier.predict_proba(Xval)
	
# Compute the average probabilities and then the number of correct predictions
avrg_probabilities = Probablity_matrices / len(Probablity_matrices)
predictions = np.argmax( avrg_probabilities, axis=1)
Accuracy = np.sum(predictions==Tval) / len(predictions)
print("The Validation Accuracy for the bagged classifier is: ", Accuracy)


######################
##### PART 3(c)  #####
######################

# Initialize arrays for keeping track 
val_acc, boostrapped_samples = [], []
nrows = Xval.shape[0]
labels = len(np.unique(small_Ttrain))
Probablity_matrices = np.ones((nrows,labels))

num_sample = 0
for i in range(500):
	
	# Get the bootstrap sample and fit to the QDA classifier
	qda_classifier = Qda_Fit(small_Xtrain, small_Ttrain)
	boostrapped_samples.append(i) #1 sample completed
	
	# Predict on the validation dataset, and store the results
	Probablity_matrices += qda_classifier.predict_proba(Xval)
	# Compute the average probabilities as sample size increases
	num_sample += 1
	avrg_prob = Probablity_matrices / num_sample
	predictions = np.argmax( avrg_prob, axis=1)
	val_acc.append( np.mean(predictions==Tval) )
				 
	
# Plotting the required figure
plt.figure()
plt.plot(boostrapped_samples, val_acc, "b")
plt.xlabel("Number of Booststrap samples")
plt.ylabel('Accuracy')
plt.suptitle('Question 3(c): Validation Accuracy')
	
# Plotting the required figure
plt.figure()
plt.semilogx(boostrapped_samples, val_acc, "b")
plt.xlabel("Number of Booststrap samples (log)")
plt.ylabel('Accuracy')
plt.suptitle('Question 3(c): Validation Accuracy (log scale)')


######################
##### PART 3(d)  #####
######################

def train3d(K,R,X,T):
	""" Use PCA to reduce the dimensionaility of the data, and then train a QDA classifier to it """
	
	# Reduce the dimensioonality of the data with PCA
	pca = PCA(n_components=K)
	pca.fit(X)
	reducedData = pca.transform(X)
	
	# Train the QDA classifier on the reduced dataset
	qda = QuadraticDiscriminantAnalysis(reg_param=R)
	ignore_warnings(qda.fit)(reducedData,T)
	
	return pca, qda

def proba3d(pca, qda, X):
	
	# first reduce the dimensionality
	reducedData = pca.transform(X)
	# Predict probabilities from qda classifier on the reduced data
	predict_probs = qda.predict_proba(reducedData)
	
	return predict_probs

######################
##### PART 3(e)  #####
######################

# Remember out global data variables are
# Small_Xtrain, small_Ttrain (the reduced datasets)

# Define the global variables again just incase
small_Xtrain = Xtrain[:200, :]
small_Ttrain = Ttrain[:200]

def myBag(K,R):
	
	# Train the PCA and QDA classifiers
	pca_, qda_ = train3d(K,R,small_Xtrain, small_Ttrain)
	# Use PCA instance to reduce the validation dataset
	reduced_ValData = pca_.transform(Xval)
	val_accuracy_base = qda_.score(reduced_ValData,Tval)
	
	
	# Initialize an empty array numpy array to hold probability matrices
	nrows = Xval.shape[0]
	labels = len(np.unique(small_Ttrain))
	Probablity_matrices = np.ones((nrows,labels))

	# Reapeat 200 times
	for i in range(200):
		
		# Generate a bootstrap sample
		bootstrap_data, bootstrap_labels = myBootstrap(small_Xtrain, small_Ttrain)
		
		# Use Train3D to create another classifier and fit it to the bootstrap sample
		pca2, qda2 = train3d(K, R, bootstrap_data, bootstrap_labels)
		
		# Use proba3d to compute a matrix of predicted probabil-ities from the MNIST validation data.
		Probablity_matrices += proba3d(pca2, qda2, Xval)
		
	# Compute the avrg probability and the 
	avrg_probabilities = Probablity_matrices / len(Probablity_matrices)
	predictions = np.argmax(avrg_probabilities, axis=1)
	val_accuracy_bagged =  np.mean(predictions == Tval) # Bagged accuracy of the validation data
	
	return val_accuracy_base, val_accuracy_bagged

######################
##### PART 3(f)  #####
######################

print("\n\nQuestion 3(f):")
print("--------------")

val_accuracy_base , val_accuracy_bagged = myBag(100, 0.01)

print("The Validation Accuracy for the base classifier is: ", val_accuracy_base)
print("The Validation Accuracy for the bagged classifier is: ", val_accuracy_bagged)


######################
##### PART 3(g)  #####
######################

# We want to write something which allows us to call my bag on uniformly distributed
# Values of K and R, on specific intervals.

# Get the values for K and R 
sample_Ks = (np.random.uniform(1,10,50)).astype(int)
sample_Rs = np.random.uniform(0.2,1,50)

# Initialize empty arrays to keep track
val_acc_base, val_acc_bagged = [], []

for i in range(50):
	base, bagged = myBag(sample_Ks[i], sample_Rs[i])
	val_acc_base.append(base)
	val_acc_bagged.append(bagged) 



# Plotting the required figure
x, y = [0,1] , [0,1]
plt.figure()
plt.scatter(val_acc_base, val_acc_bagged, c="blue")
plt.plot(x,y, c="red")
plt.xlabel("Base validation accuracy")
plt.ylabel("Bagged validation accuracy")
# Set the max x and y limits
plt.xlim(0,1)
plt.ylim(0,1)   
plt.suptitle('Question 3(g): Bagged v.s base validation accuracy')

######################
##### PART 3(h)  #####
######################

print("\n\nQuestion 3(h):")
print("--------------")

# Get the values for K and R  K - between 50, 20 and R is [0,0.05]
sample_Ks2 = np.random.randint(50,200,50)
sample_Rs2 = np.random.uniform(0, 0.05, 50)

# Initialize empty arrays to keep track
val_acc_base2, val_acc_bagged2 = [], []

for i in range(50):
	base2, bagged2 = myBag(sample_Ks2[i], sample_Rs2[i])
	val_acc_base2.append(base2)
	val_acc_bagged2.append(bagged2) 

# Plotting the required figure
plt.figure()
plt.scatter(val_acc_base2, val_acc_bagged2, c="blue")
plt.axhline(y=max(val_acc_bagged2),c="red")
plt.xlabel("Base validation accuracy")
plt.ylabel("Bagged validation accuracy")
# Set the max y lim
plt.ylim(0,1)   
plt.suptitle('Question 3(h): Bagged v.s. base validation accuracy')

# Print out the Maximum bagged validation accuracy
print("Max bagged validation accuracy for the bagged Classifier =", max(val_acc_bagged2))


#----------------------------------------------------------------#
#----------------------------------------------------------------#
#                           QUESTION 4                           #
#----------------------------------------------------------------#
#----------------------------------------------------------------#

# Read in the corresponding dataset
# Read in the data
with open("dataA2Q2.pickle", "rb") as file:
    dataTrain, dataTest = pickle.load(file)
Xtrain, Ttrain = dataTrain
Xtest, Ttest = dataTest

######################
##### PART 4(a)  #####
######################

def plot_clusters(X,R,Mu):
	
	# X us a Nx2 data matrix
	# R is a Nx3 matrix
	# Mu is a 3x2 matrix of cluster centers
	
	# Sum up the corresponding columns vertically and then sort R
	sum_of_R = np.sum(R, axis=0)
	sum_of_R_sorted = np.argsort(sum_of_R)
	R = R[:, sum_of_R_sorted] 

	# Get the center of each column
	col1_means, col2_means = Mu[:,0] , Mu[:,1]
	
	# Plot the required figure
	plt.figure()
	plt.scatter(X[:, 0], X[:, 1], color=R, s=5)
	plt.scatter(col1_means, col2_means, color="black")
	
	return None

######################
##### PART 4(b)  #####
######################

# Call the KMeans classifier with 3 clusters
clustering = KMeans(n_clusters=3)
# Fit the training data on the clusters
clustering.fit(Xtrain)

# Extract the mean from the classifier
mu = clustering.cluster_centers_ 

# get the R vector -- This is HARD Kmeans, so the corresponding cluster should be a 1 or 0
# We need to do 1 hot encoding which is basically R.
predictions = clustering.predict(Xtrain)
# We use len(np.unique) to make it work for any number of classes. 
Responsibilities = np.eye(len(np.unique(predictions)))[predictions]
   
# plotting the required figure
plot_clusters(Xtrain, Responsibilities, mu)
plt.suptitle("Question 4(b): K means")

# Compute the score on training and test data 
train_score = clustering.score(Xtrain)
test_score = clustering.score(Xtest)


# Printing scores
print("\n\nQuestion 4(b):")
print("--------------")
print("The score on training data for K means is: ", train_score)
print("The score on test data for K means is: ", test_score)


######################
##### PART 4(c)  #####
######################

# Call the GGM classifier
GMM = GaussianMixture(n_components=3, covariance_type='spherical')
# Fit the training data on the clusters
GMM.fit(Xtrain)

# Extract the mean from the classifier
mu = GMM.means_ 

# For Gaussian Mixture Models, the Predicted Probabilities is the Responsibilities function. 
predict_proba = GMM.predict_proba(Xtrain)
   
# plotting the required figure
plot_clusters(Xtrain, predict_proba, mu)
plt.suptitle("Question 4(c): Gaussian mixture model (Spherical)")

# Compute the score on training and test data 
train_score = GMM.score(Xtrain)
test_score = GMM.score(Xtest)

# Printing scores
print("\n\nQuestion 4(c):")
print("--------------")
print("The score on training data for GMM (spherical) is: ", train_score)
print("The score on test data for GMM (spherical) is: ", test_score)

######################
##### PART 4(d)  #####
######################

# Call the GGM classifier, and set covariance_type = Full
GMM2 = GaussianMixture(n_components=3, covariance_type='full')
# Fit the training data on the clusters
GMM2.fit(Xtrain)

# Extract the mean from the classifier
mu2 = GMM2.means_ 

# For Gaussian Mixture Models, the Predicted Probabilities is the Responsibilities function. 
predict_proba2 = GMM2.predict_proba(Xtrain)
   
# plotting the required figure
plot_clusters(Xtrain, predict_proba2, mu2)
plt.suptitle("Question 4(d): Gaussian mixture model (full)")

# Compute the score on training and test data 
train_score2 = GMM2.score(Xtrain)
test_score2 = GMM2.score(Xtest)

# Printing scores
print("\n\nQuestion 4(d):")
print("--------------")
print("The score on training data for GMM (full) is: ", train_score2)
print("The score on test data for GMM (full) is: ", test_score2)
print("Q4d-Q4c test scores = ", test_score2 - test_score )

	
######################
##### PART 4(e)  #####
######################

def myKmeans(X,K,I):
	# X us a Nx2 data matrix
	# R is a Nx3 matrix
	# Mu is a 3x2 matrix of cluster centers

	
	# Initialize array for keeping track, data rows and data columns 
	scores, num_rows, num_features = [] , X.shape[0] , X.shape[1]
	
	# Set K cluster meansm1,...,mKto random value
	# Random initialization of so initial clusters are uniformly distributed around the dataset 
	cluster_centers = np.random.rand(K,X.shape[1])
	
	# we loop till the number of iterations, I and search for K clusters in datamatrix X
	for i in range(I):
		
		# To be able to subtract the mean from each datapoint, we need to make it correct dimenstions first, of Nx3x2
		
		data = np.array(X)[:, np.newaxis, :]
		means = np.array(mu)[np.newaxis, :, : ]

		# Compute the distance and sum across the dimension
		# for every datapoint, now we have it's distance to each centroid
		distance = np.sum( (data - means)**2,axis=2) # This is 1800,3
		
		# Get the minimum distance for each datapoint 
		smallest_distance_index = np.argmin(distance, axis=1)
		
		# Compute the score -- Optimize the sum of the squared distances between all datapoints and it's closest center
		sum_distance = np.sum(distance[smallest_distance_index], axis=0)
		scores.append(sum_distance)
		
		# Convert the smallest distance to a one hot vector for the responsiblity vector
		# We use len(np.unique) to make it work for any number of classes. 
		Responsibilities = np.eye(len(np.unique(smallest_distance_index)))[smallest_distance_index]
		
		# Each centre is set to mean of data assigned to it (find m_k)
		cluster_center = np.sum(np.dot(Responsibilities.T,X)) / np.sum(Responsibilities) 
		
	return cluster_centers, Responsibilities, scores

mu, r, scores = myKmeans(Xtrain,3,100)
plot_clusters(Xtrain,r,mu)
plt.suptitle("Question 4(e): Data clustered by K means")

######################
##### PART 4(f)  #####
######################

def myGMM(X,K,I):
	
	# Initialize array for keeping track, data rows and data columns 
	scores, num_rows, num_features = [] , X.shape[0] , X.shape[1]
	
	# Initialize the means mu nd mixing coefficients pi
	data = np.array(X)[:, np.newaxis, :]
	means = np.array(mu)[np.newaxis, :, : ]
	
	# The pis need to sum up to one 
	pi = np.random.random(100)
	pi /= pi.sum()
	
	# Initialize array for keeping track, data rows and data columns 
	scores, num_rows, num_features = [] , X.shape[0] , X.shape[1]
	for i in range(I):
		
		# Compute R_k given the current parameters, mu, x and pi
		distance = np.sum( (data - means)**2,axis=2) # This is 1800,3
		numerator = pi * np.exp((-1/2)*distance)
		denom = np.sum(numerator, axis=1) # Sum over each cluster for each and every datapoint
		r_k = numerator/denom 
		
		# Compute the log likelihood using the multivariate normal distribution 
		multivariate_norm = numerator / (2*np.pi)**(X.shape[1]/2) 
		log_likelihood = np.mean(np.log(multivariate_norm)) 
		scores.append(log_likelihood)
		
		# Re-estimate the parameters given current responsibilities
		Mu_k = 1/len(X) * (np.sum(r_k*data))
		pi_k =  np.sum(r_k) / len(X)
		
		return Mu_k, pi_k
		
		
		
		
		
		

	
	



	
