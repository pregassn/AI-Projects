#!/usr/bin/python

#"""
#Author: Nirmal Pregassame
#file:
#"""
import math
import numpy as np


##############################
# PART 1 and 2:  kNN Algorithm
##############################

### Function ReadData to read training and test data from csv format
def ReadData(filename, typef):
    # Convert csv format into numpy array
    dataArray = np.genfromtxt(filename, delimiter=",")
    
    # According to the type (training or test), the outliner is removed
    if typef == "training":
        dataArray = np.delete(dataArray, (465), axis=0)
    
    # Split the array into data features array and result array
    dataInfo = np.array(dataArray[:,:-1], dtype=float)
    dataResult = np.array(dataArray[:,-1], dtype=float)
    return (dataInfo,dataResult)




### Function calculateDistances to calaculate (Euclidean) distance between query and training data and return the 3 nearest neighbors
def calculateDistances(data, query):
    # nb of neibhors to return
    k = 3
    
    calcDistance_tmp = data.copy()
    
    # Values range for ordinal, integer features
    BiRadMax = 5
    BiRadMin = 1
    AgeMax = np.max(data[:,1], axis=0)
    AgeMin = np.min(data[:,1], axis=0)
    
    # Normalized values of the 2 first column (ordinal and int features)
    normalized_val = np.array([BiRadMax-BiRadMin, AgeMax-AgeMin], dtype=float)
    calcDistance_tmp[:,0:2] = (data[:,0:2] - query[0:2])/normalized_val
    
    # Hamming distance value for the 3rd and 4th features 
    calcDistance_tmp[:,2:4] = (data[:,2:4] != query[2:4])
    
    # Normalized values of the last column (ordinal feature)
    calcDistance_tmp[:,4] = (data[:,4] - query[4]) / (np.max(data[:,4], axis=0) - np.min(data[:,4], axis=0))
    
    # Euclidean distance calculation
    calcDistance_tmp = np.sum(calcDistance_tmp**2, axis=1)
    euclDistance = np.sqrt(calcDistance_tmp)
    
    # Return Euclidean distance array and the 3 nearest neighbors indexes
    index_sort = np.argsort(euclDistance)[:k]
    return (euclDistance, index_sort)




### Function calculateDistanceskp to calaculate (Minkowski) distance between query and training data and return the k nearest neighbors for Part 2
def calculateDistanceskp(data, query, k, p):
    calcDistance_tmp = data.copy()
    
    # Values range for ordinal, integer features
    BiRadMax = 5
    BiRadMin = 1
    AgeMax = np.max(data[:,1], axis=0)
    AgeMin = 0
    
    # Normalized values of the 2 first column (ordinal and int features)
    normalized_val = np.array([BiRadMax-BiRadMin, AgeMax-AgeMin], dtype=float)
    calcDistance_tmp[:,0:2] = np.absolute(data[:,0:2] - query[0:2])/normalized_val
    
    # Hamming distance value for the 3rd and 4th features
    calcDistance_tmp[:,2:4] = (data[:,2:4] != query[2:4])
    
    # Normalized values of the last column (ordinal feature)
    calcDistance_tmp[:,4] = np.absolute(data[:,4] - query[4]) / (np.max(data[:,4], axis=0) - np.min(data[:,4], axis=0))
    
    # Minkowski distance calculation
    calcDistance_tmp = np.sum(calcDistance_tmp**p, axis=1)
    p = float(p)
    Distance = calcDistance_tmp**(1/p)
    
    # Return Minkowski distance array and the 3 nearest neighbors indexes
    index_sort = np.argsort(Distance)[:k]
    return (Distance, index_sort)




### Function queryResult to return test prediction value without distance weigthing
def queryResult(dataResult, index_sort):
    # Sum of the result values and then take the mean of it and return the nearest integer 
    dataResultSum = np.sum(dataResult[index_sort])
    qResult = round(dataResultSum/len(index_sort))
    return qResult



### Function queryResult to return test prediction value with distance weigthing
def queryDistWeightVoteResult(dataResult, index_sort, Distance):
    # Array of invert distance
    DistWeight = np.reciprocal(Distance[index_sort])
    
    # Bool array of the Training Result (True = Malignant)
    dataResultBool = np.array(dataResult[index_sort], dtype=bool)
    
    # Sum of invert distance of Malignant data
    dataResultMalignantSum = np.sum(DistWeight[dataResultBool])
    
    # Sum of invert distance of Benign data
    dataResultBeningnSum = np.sum(DistWeight[np.invert(dataResultBool)])
    
    # Voting system
    if dataResultMalignantSum >= dataResultBeningnSum:
        qResult = 1.0
    else:
        qResult = 0.0
    return qResult




### Function AccuracyResult to return the Accuracy of the test prediction result
def AccuracyResult(testDataResult, testDataPrediction):
	# Number of the correct prediction
	correct = 0
	for x in range(len(testDataResult)):
		if testDataResult[x] == testDataPrediction[x]:
			correct += 1
	
	# Return the accuracy rate
	return (correct/float(len(testDataResult))) * 100.0







################################################################################
#        TEST
################################################################################



(trainDataInfo, trainDataResult) = ReadData("./cancer2/trainingData2.csv", "training")
(testDataInfo, testDataResult) = ReadData("./cancer2/testData2.csv", "test")



############################################################
# PART 1: Result for a KNN of 3 and euclidean distance
############################################################

print("###########################################################")
print("# PART 1: Result with a KNN of 3 and euclidean distance   #")
print("###########################################################")
print("")

TestDataPredictionOriginal = []
for i in range(len(testDataInfo)):
    (euclDistance, index_sort) = calculateDistances(trainDataInfo, testDataInfo[i])
    prediction = queryResult(trainDataResult, index_sort)
    TestDataPredictionOriginal.append(prediction)

TestDataPredictionDistWeight = []
for i in range(len(testDataInfo)):
    (euclDistance, index_sort) = calculateDistances(trainDataInfo, testDataInfo[i])
    prediction = queryDistWeightVoteResult(trainDataResult, index_sort, euclDistance)
    TestDataPredictionDistWeight.append(prediction)

AcurracyOriginal = AccuracyResult(testDataResult, TestDataPredictionOriginal)
AcurracyDistWeight = AccuracyResult(testDataResult, TestDataPredictionDistWeight)
print("Acuracy of k=3, dist=2, non weighted dist:", AcurracyOriginal)
print("Acuracy of k=3, dist=2, weighted dist    :", AcurracyDistWeight)
print("")
print("")
print("")

############################################################
# PART 2: Investigating kNN variants and hyper-parameters
############################################################

print("###########################################################")
print("# PART 2: Investigating kNN variants and hyper-parameters #")
print("###########################################################")
print("")

for dist in (0.25, 0.5, 1, 2, 4, 8): 
    print("#################### Minkowski distance: ", dist)
    for k in (3, 4, 5, 6, 11, 12, 33, 34, 63, 64): 
	TestDataPredictionOriginal = []
	for i in range(len(testDataInfo)):
	    (Distance, index_sort) = calculateDistanceskp(trainDataInfo, testDataInfo[i], k, dist)
	    prediction = queryResult(trainDataResult, index_sort)
	    TestDataPredictionOriginal.append(prediction)
	AcurracyOriginal = AccuracyResult(testDataResult, TestDataPredictionOriginal)
        print("Acuracy of k=", k, "dist=", dist, "non weighted dist:", AcurracyOriginal)

	TestDataPredictionDistWeight = []
	for i in range(len(testDataInfo)):
	    (Distance, index_sort) = calculateDistanceskp(trainDataInfo, testDataInfo[i], k, dist)
	    prediction = queryDistWeightVoteResult(trainDataResult, index_sort, Distance)
	    TestDataPredictionDistWeight.append(prediction)
	AcurracyDistWeight = AccuracyResult(testDataResult, TestDataPredictionDistWeight)
        print("Acuracy of k=", k, "dist=", dist, "weighted dist    :", AcurracyDistWeight)
    
    print("")


