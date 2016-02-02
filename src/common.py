import time
import numpy as np
import math

ISOTIMEFORMAT="%Y-%m-%d %X"
WEEKS = 7
SITES = 10
totalUsers = 0
runMaxUser = np.inf

DRIVE_LETTER = "g"

def getCurrentTime():
    return time.strftime(ISOTIMEFORMAT, time.localtime())

def convertMatToStr(matrix, fmtStr="%4d"):
	rows, cols = np.shape(matrix)
	matrixToString = []
	for arow in range(rows):
		matrixToString.append("[")
		for acol in range(cols):
			# if acol < cols -1:
			#     matrixToString.append(fmtStr % matrix[arow, acol])
   #              matrixToString.append(",")
			# else:
				matrixToString.append(fmtStr % matrix[arow, acol])
		matrixToString.append("]\n")
	return "".join(matrixToString)


def loadData(fileName):
    dataHistory = {}
    dataFile = open(fileName)
    lineIdx = 0;

    for aline in dataFile.readlines():
        lineIdx += 1
        curLine = aline.split("\t")
        if len(curLine) == 0:
        	continue

        userId = curLine[0]

        if (not userId in dataHistory.keys()):
            dataHistory[userId] = ([[0 for column in range(WEEKS * 7)] for row in range(SITES + 1)])

        week = int(curLine[1][1])
        day =  (week - 1) * 7 + int(curLine[1][3]) - 1
        website = curLine[2]
        if (len(website) == 3):
            website = int(website[1:3])
        else:
            website = int(website[1:2])
        visitNum = int(curLine[3])

        dataHistory[userId][website-1][day] = visitNum

    dataFile.close()

    for userId in dataHistory:
        dataHistory[userId] = np.mat(dataHistory[userId]) # transform narray into matrix
        dataHistory[userId][SITES] = dataHistory[userId].sum(axis=0)

    global totalUsers

    totalUsers = len(dataHistory)
    
    return dataHistory



def calculatCosin(data1, data2):
    # print("calculatCosin data1 %s \n" % type(data1), data1)
    # print("calculatCosin data2 %s \n" % type(data2), data2)
    if (data1.sum() == 0 or data2.sum() == 0):
        return 0.0

    array1 = np.array(data1)
    array2 = np.array(data2)
    mat1 = np.mat(data1)
    mat2 = np.mat(data2)
    return (array1 * array2).sum()/ (math.sqrt(mat1 * mat1.T) * math.sqrt(mat2 * mat2.T))

def verification(dataHistory, forecastedVisitNum, verifyWeek, progFile):    
    verifyData = {};
    totalUsers = len(dataHistory)
    calculatedUser = len(forecastedVisitNum)

    logMsg = "verification(): %d are  going to verify, verifyWeek %d\n" % (calculatedUser, verifyWeek)
    if progFile is not None:
        progFile.write(logMsg)
    else:
        print(logMsg)

    realUserCount = 0 #历史数据中有访问量的用户数
    hitUserCount = 0 # 预测出来有访问量的用户数
    similarity = 0.0

    for userId in forecastedVisitNum:
        tmp = "1f+59QdDaN9Vl9p8Ne47CA=="
        visitNumHistory = dataHistory[userId]
        visitNumForecast = forecastedVisitNum[userId].flatten()

        visitNumHistoryVerify = visitNumHistory[:, verifyWeek * 7:(verifyWeek + 1) * 7] #截取用于验证的历史记录
        visitNumHistoryVerify = visitNumHistoryVerify[0:SITES, :] # 最后一行为总的访问量，去掉

        visitNumHistoryVerify = visitNumHistoryVerify.flatten()

        # 若该用户在 verifyWeek 这一周没有历史访问量则不算进 realUserCount
        if (visitNumHistoryVerify.sum() > 0):         
            realUserCount += 1

        # 若该用户的预测值都为0则不算进 hitUserCount
        if (visitNumForecast.sum() >0):
            hitUserCount += 1

        similarity += calculatCosin(visitNumHistoryVerify, visitNumForecast)

    if hitUserCount == 0:
        precision = 0
    else:
        precision = similarity / hitUserCount

    recall = hitUserCount / realUserCount

    if recall == 0 and precision == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    logMsg = "verification: totalUsers %d, realUserCount %d, hitUserCount %d, similarity %f,  precision %f, recall %f, f1 %f\n" % (totalUsers, realUserCount, hitUserCount, similarity, precision, recall, f1)

    if progFile is not None:
        progFile.write(logMsg)
    else:
        print(logMsg)

    return f1


def leastSquareImpl(trainingMat):
    n = np.shape(trainingMat)[1]
    xMat = trainingMat[:, 0:n-1]
    yMat = trainingMat[:, n-1:n]
    xTx = xMat.T  * xMat
    if (np.linalg.det(xTx) == 0.0):
        return None

    factorVec = xTx.I * xMat.T * yMat
    return factorVec


def roundVisitNum(forcastedVisitNumMat):
    forcastedVisitNumMat = np.round(forcastedVisitNumMat)
    rows, columns = np.shape(forcastedVisitNumMat)
    for row in range(rows):
        for col in range(columns):
            if (forcastedVisitNumMat[row, col] <= 0):
                forcastedVisitNumMat[row, col] = 0

    return forcastedVisitNumMat
    