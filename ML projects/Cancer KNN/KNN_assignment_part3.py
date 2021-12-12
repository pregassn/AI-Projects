#!/usr/bin/python

#"""
#Author: Nirmal Pregassame
#file:
#"""
import math
import numpy as np

### Function ReadData to read training and test data from csv format
def ReadData(filename):
    # Convert csv format into numpy array
    dataArray = np.genfromtxt(filename, delimiter=",")
    
    # Split the array into data features array and result array
    dataInfo = np.array(dataArray[:,:-1], dtype=float)
    dataResult = np.array(dataArray[:,-1], dtype=float)
    return (dataInfo,dataResult)




### Function calculateDistances to calaculate (Euclidean) distance between query and training data and return the 3 nearest neighbors
def calculateDistanceskp(data, query, k, p):
    calcDistance_tmp = data.copy()
    
    #number of feature: number of column of data
    nbfeatures = calcDistance_tmp.shape[1]
    
    #Evaluation of min and Max for each feature and build an array of Max and an array of Min
    FeaturesMax = []
    FeaturesMin = []
    for f in range(nbfeatures):
        FeaturesMax.append(np.max(calcDistance_tmp[:,f], axis=0))
	FeaturesMin.append(np.min(calcDistance_tmp[:,f], axis=0))
    FeaturesMax = np.array(FeaturesMax, dtype=float)
    FeaturesMin = np.array(FeaturesMin, dtype=float)
    
    # Normalized values of all the features.
    normalized_val = FeaturesMax-FeaturesMin
    calcDistance_tmp = (np.absolute(data - query))/normalized_val
    
    # Minkowski distance calculation
    calcDistance_tmp = np.sum(calcDistance_tmp**p, axis=1)
    p = float(p)
    Distance = calcDistance_tmp**(1/p)
    
    # Return Minkowski distance array and the 3 nearest neighbors indexes
    index_sort = np.argsort(Distance)[:k]
    return (Distance, index_sort)




### Function queryDistWeightVoteResult to return test prediction value with distance weigthing : addition of the weigthing distance parameter n
def queryDistWeightVoteResult(dataResult, index_sort, Distance, n):
    # Array of invert distance weigthed by the parameter n
    WeightCoef = np.reciprocal(Distance[index_sort])**n
    
    # Sum of distance weigthed with the WeightCoef array
    DistWeight = np.sum(WeightCoef*dataResult[index_sort])
    
    # Return the predictive result of the regression test
    qResult = DistWeight/np.sum(WeightCoef)
    return qResult




### Function AccuracyResult to return the R Squared value of the regression test result
def AccuracyResult(testDataResult, testDataPrediction):
	# Sum of squared residuals
	SSR = np.sum((testDataResult-testDataPrediction)**2)
	
	# Construction of an array of mean value 
	meanVal = np.mean(testDataResult)
	meanValArray = np.full(len(testDataPrediction), meanVal)
	
	# Total Sum of Squares
	TSS = np.sum((meanValArray-testDataPrediction)**2)
	
	# Return the R squared value
	Rsquare = 1- (SSR/TSS)
	return Rsquare








################################################################################
#        TEST
################################################################################




(trainDataInfo, trainDataResult) = ReadData("./regressionData/trainingData.csv")
(testDataInfo, testDataResult) = ReadData("./regressionData/testData.csv")

print("###########################################################")
print("# PART 3: Devoloping kNN for regression problems          #")
print("###########################################################")
print("")

for dist in (1, 2, 4): 
    print("#################### Minkowski distance: ", dist)
    for n in (1, 2, 4, 8, 16): 
        print("### Parameter n: ", n)
	for k in (3, 5, 11, 21, 51, 75, 100, 200):
	    TestDataPredictionDistWeight = []
	    for i in range(len(testDataInfo)):
	        (Distance, index_sort) = calculateDistanceskp(trainDataInfo, testDataInfo[i], k, dist)
	        prediction = queryDistWeightVoteResult(trainDataResult, index_sort, Distance, n)
	        TestDataPredictionDistWeight.append(prediction)
	    AcurracyDistWeight = AccuracyResult(testDataResult, TestDataPredictionDistWeight)
            print("RSquare of k=", k, " Mdist=", dist, " weighted dist    :", AcurracyDistWeight)
    
        print("")

