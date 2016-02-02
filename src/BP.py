import numpy as np
from common  import *
import math


def BackPropagation(dataHistory, forecastedWeekDayTotalVisitNum, lastStartWeeks, lastEndWeeks, progFile):
    totalUsers = len(forecastedWeekDayTotalVisitNum)
    progFile.write("%s backPropagation(), start week %d,  end week %d, %d users \n" % (getCurrentTime(), lastStartWeeks, lastEndWeeks, totalUsers))

    forecastedVisitNum = {}

    trainingDays = (lastEndWeeks - lastStartWeeks) * 7

    runForUser = 0

    for userId in forecastedWeekDayTotalVisitNum:
        runForUser += 1
        progFile.write("%s calculated users: %d / %d , [%s], history data: \n%s\n" % (getCurrentTime(), runForUser, totalUsers, userId, convertMatToStr(dataHistory[userId])))

        visitNumTrainMat = np.mat(np.ones((3, trainingDays)))

        # 过去n周每天总的访问量
        visitNumTrainMat[2, 0:trainingDays] = dataHistory[userId][SITES][:, lastStartWeeks*7:lastEndWeeks*7]

        forecastedVisitNum[userId] = np.mat(np.zeros((SITES, 7)))

        for website in range(SITES):
            # 过去n周每天某个website的访问量， matrix 类型
            visitNumTrainMat[1, 0:trainingDays] = dataHistory[userId][website][:, lastStartWeeks*7:lastEndWeeks*7]

            if (visitNumTrainMat[1].sum() == 0):
                progFile.write("visit num of week %d - %d weeks of website [%d] is 0, skip!\n" % (lastStartWeeks, lastEndWeeks, website))                
                continue

            progFile.write("[%s] website %d training mat \n%s\n" % (userId, website, convertMatToStr(visitNumTrainMat)))

            weights = BPImpl(visitNumTrainMat.T, progFile)

            if (weights[1, 0] > 0.01):
                forecastedVisitNum[userId][website] = (forecastedWeekDayTotalVisitNum[userId] - weights[0, 0]) / weights[1, 0]
            else:
                forecastedVisitNum[userId][website] = forecastedWeekDayTotalVisitNum[userId] - weights[0, 0]

        roundBPMat(forecastedVisitNum[userId])

    return forecastedVisitNum

def roundBPMat(forecastedVisitNum):
    rows, cols = np.shape(forecastedVisitNum)
    for ri in range(rows):
        for ci in range(cols):
            if (forecastedVisitNum[ri, ci] < 0):
                forecastedVisitNum[ri, ci] = 0
            elif (forecastedVisitNum[ri, ci] > 0 and forecastedVisitNum[ri, ci] < 1):
                forecastedVisitNum[ri, ci] = 1

    return 0

# def sigmoid(x, deriv, progFile):
#     if(deriv==True):
#         return x.A*(1-x.A)

#     return 1/(1+np.exp(-x))

def sigmoid(x, deriv, progFile):
    y = 1/(1+np.exp(-x))
    if(deriv==True):
        return y.A * (1 - y.A)
    else:
        return y

def changePercent(deltaWeight, percent):
    m, n = np.shape(deltaWeight)
    for mi in range(m):
        for ni in range(n):
            if (deltaWeight[mi, ni] > percent):
                return True
    return False

def roundBPTrainingMat(visitNumTrainMat):
    m, n = np.shape(visitNumTrainMat)
    for mi in range(m):
        for ni in range(n):
            if (visitNumTrainMat[mi, ni] > 0):
                visitNumTrainMat[mi, ni] = 1

def BPImpl(visitNumTrainMat, progFile):
    roundBPTrainingMat(visitNumTrainMat)
    m, n = np.shape(visitNumTrainMat)

    xMat = visitNumTrainMat[:, 0:n-1]
    yMat = visitNumTrainMat[:, n-1:n]

    weights = np.mat(np.ones((n-1, 1)))

    progFile.write("xMat is %s" % convertMatToStr(xMat.T))
    progFile.write("yMat is %s" % convertMatToStr(yMat.T))

    convergent = 0

    for itor in range(1000):
        #progFile.write("weight is \n%s\n" % convertMatToStr(weights, "%.4f "))
        l1 = sigmoid(xMat * weights, False, progFile)
        l1_error = yMat - l1

        # sigmoid(l1,True) 表示 sigmoid 函数在 l1 处的斜率， 斜率越小，曲线越平缓，表示l1的值越大或越小
        # 则意味着网络越确定 l1 的分类，l1_error * sigmoid(l1,True) 就越小， 则 l1 上的权值需要调整的幅度就越小
        l1_delta = np.mat(l1_error.A * sigmoid(l1, True,  progFile))
        deltaWeight = xMat.T * l1_delta

        tmp = deltaWeight / weights

        if (not changePercent(np.abs(tmp), 0.01)):
            convergent += 1
            if (convergent > 10):
                break

        weights += deltaWeight

    tmp = np.mat(np.zeros((np.shape(xMat)[0], 2)))
    tmp[:, 0] = yMat
    tmp[:, 1] = xMat * weights
    progFile.write("%s after trainning,  weights %s\nyMat : xMat * weights\n%s\n" % (getCurrentTime(), convertMatToStr(weights), convertMatToStr(tmp)))

    return weights

'''
# input dataset
X = np.array([ [0,0,1], [0,1,1],[1,0,1],[1,1,1] ])

# output dataset
y = np.array([[0,0,1,1]]).T

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2*np.random.random((3,1)) - 1
#syn0 = np.ones((3,1))

for itor in range(1):
    # forward propagation
    l0 = X
    l1 = sigmoid(np.dot(l0,syn0))

    # how much did we miss?
    l1_error = y - l1

    l1_delta = l1_error * sigmoid(l1,True)

    # update weights
    syn0 += np.dot(l0.T,l1_delta)

    print ("Output After Training:")
    print (l1)

print("syn0 is ", syn0)
print("X * syn0 is ", np.dot(X, syn0))、
'''
