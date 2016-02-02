import numpy as np
from common  import *
import math

# sliding average , last N weeks
def SlidingAvg(dataHistory, lastStartWeeks, lastEndWeeks, progFile):
    progFile.write("\n\n\n%s calWebSiteDayVisitNum_SlidingAvg() ... " % getCurrentTime())

    global totalUsers

    forecastedWeekDayTotalVisitNum = {} 
    userIdx = 1;
    for userId in dataHistory:
        forecastedWeekDayTotalVisitNum[userId] = np.zeros((SITES, 7))
        progFile.write("calculating [%s] %d/%d   \n" % (userId, userIdx, totalUsers))

        for website in range(SITES):
            # 过去n周每天某个website的访问量， matrix 类型
            #lastWeeksEachDayVisitNum = dataHistory[userId][website][:, (WEEKS-lastWeeks)*7 : WEEKS*7]
            lastWeeksEachDayVisitNum = dataHistory[userId][website][:, lastStartWeeks*7 : lastEndWeeks*7]

            if (lastWeeksEachDayVisitNum.sum() == 0):
                progFile.write("visit num of last %d weeks  of website [%d] is 0, skip!\n" % (lastWeeks, website))
                continue

            for week_day in range(7):
                lastWeeksTotalVisitNum = lastWeeksEachDayVisitNum[:, week_day:lastWeeks * 7].sum()
                lastWeeksTotalVisitNum += forecastedWeekDayTotalVisitNum[userId][0:week_day].sum()
                forecastedWeekDayTotalVisitNum[userId][website, week_day] = round(lastWeeksTotalVisitNum / (lastWeeks * 7))

        forecastedWeekDayTotalVisitNum[userId] = np.mat(forecastedWeekDayTotalVisitNum[userId])
        if (forecastedWeekDayTotalVisitNum[userId].sum() == 0):
            forecastedWeekDayTotalVisitNum.pop(userId)
            progFile.write("    all forecast data of %s is 0, skip!\n" % userId)

        if (userIdx > runMaxUser):
            break
        userIdx += 1

    progFile.write("\n%s leaving calWebSiteDayVisitNum_SlidingAvg() ... \n" % getCurrentTime())

    return forecastedWeekDayTotalVisitNum




# def leastSquare(trainingMat, userId, website):
#     n = np.shape(trainingMat)[1]
#     xMat = trainingMat[:, 0:n-1]
#     yMat = trainingMat[:, n-1:n]
#     xTx = xMat.T  * xMat
#     if (np.linalg.det(xTx) == 0.0):        
#         return None

#     factorVec = xTx.I * xMat.T * yMat
#     return factorVec



def roundFactorMat(factorMat):
    rows, columns = np.shape(factorMat)
    for row in range(rows):
        for col in range(columns):
            if (factorMat[row, col] >= -0.001 and factorMat[row, col] <= 0.001):
                factorMat[row, col] = 0
    return factorMat


def LeastSquare(dataHistory, forecastedWeekDayTotalVisitNum, progFile):
    progFile.write("%s calWebSiteDayVisitNum_LeastSquare()...\n" % getCurrentTime())

    global totalUsers

    forecastedVisitNum = {}

    trainingDays = (WEEKS) * 7
    # 1 - 6 周作为训练数据
    #训练数据的第一行全为 1, 第二行为某website的访问量, 作为自变量
    #训练数据的最后一行是总的访问量， 作为因变量
    visitNumTrainMat = np.mat(np.ones((3, trainingDays)))

    factorMat = []
    zeroVisitNum =  np.zeros((1,7))#[0 for x in range(7)]

    runForUser = 0

    #for userId in dataHistory:
    for userId in forecastedWeekDayTotalVisitNum:
        progFile.write("\n\n\n%sforecastVisitNum for user %s %d/%d \n" % (getCurrentTime(), userId, runForUser, totalUsers))

        visitNumTrainMat[2] = dataHistory[userId][SITES][:, 0:trainingDays]
     
        forcastedVisitNumMat = np.zeros((SITES,7))

        progFile.write("dataHistory[%s]: => \n%s\n" % (userId, convertMatToStr(dataHistory[userId])))

        for website in range(0, SITES):
            visitNumTrainMat[1] = dataHistory[userId][website][:, 0:trainingDays]

            # shape(factorVec) = n * 1
            factorVec = leastSquareImpl(visitNumTrainMat.T)
            if (factorVec is not None):
                factorMat.append(factorVec.T[0].A)
                progFile.write("website [%d] factorVec is \n%s\n" % (website, convertMatToStr(factorVec.T)))

                if (factorVec[1,0] >= 0.01):
                    forcastedVisitNum = (forecastedWeekDayTotalVisitNum[userId] - factorVec[0,0])/factorVec[1,0]
                else:
                    forcastedVisitNum = forecastedWeekDayTotalVisitNum[userId] - factorVec[0,0]

                forcastedVisitNumMat[website] = forcastedVisitNum

        forcastedVisitNumMat = roundVisitNum(np.mat(forcastedVisitNumMat))

        if ( forcastedVisitNumMat.sum() == 0):
            progFile.write("%s : all forecast data are 0, skip!" % userId)
            continue

        progFile.write("%s [%s]   forceted website visite number \n%s\n" % (getCurrentTime(), userId, convertMatToStr(forcastedVisitNumMat)))

        forecastedVisitNum[userId] = forcastedVisitNumMat

        if (runForUser >= runMaxUser):
            break

        progFile.write("%s  calWebSiteDayVisitNum_LeastSquare() %d/%d\n" % (getCurrentTime(), runForUser, totalUsers))

        runForUser += 1

    progFile.write("%s leaving calWebSiteDayVisitNum_LeastSquare()... \n" % getCurrentT0me() )
    return forecastedVisitNum

def lwlrImpl(trainingMat, trainingDays, k, forcastedVisitNumMat, website, progFile):
    progFile.write("%s lwlrImpl website %d\n" % (getCurrentTime(), website))

    n = np.shape(trainingMat)[1] # shape = n * 3
    xMat = trainingMat[:, 0:n-1]
    yMat = trainingMat[:, n-1:n]

    days = np.shape(xMat)[0]

    factorVectors = []

    for day1 in range(7):
        weightsMat = np.mat(np.eye(trainingDays))
        for day2 in range(trainingDays):
            diffMat = xMat[trainingDays + day1] - xMat[day1 + day2]
            weightsMat[day2, day2] = np.exp(diffMat * diffMat.T/(-2 * (k**2)))

        if weightsMat.sum() == 0.0:
            factorVectors.append([0, 0])
            progFile.write(" weightsMat.sum() %d is 0, continue" % website)
            continue

        progFile.write(" weightsMat.sum() %d is %d" % (website, weightsMat.sum()))

        xMatTraining = xMat[day1 : day1+trainingDays]
        xTx = roundFactorMat(xMatTraining.T * weightsMat * xMatTraining)
        #xTx = np.round(xMatTraining.T * weightsMat * xMatTraining)

        if (np.linalg.det(xTx) == 0.0):
            factorVectors.append([0, 0])
            continue

        factorVec = xTx.I * xMatTraining.T * weightsMat * yMat[day1 : day1 + trainingDays] # shape(ws) = 2*1

        if math.isnan(factorVec[0,0]):
            factorVec[0,0] = 0
        else:
            factorVec[0,0] = round(factorVec[0, 0], 3)

        if math.isnan(factorVec[1,0]):
            factorVec[1,0] = 0
        else:
            factorVec[1,0] = round(factorVec[1, 0], 3)

        if factorVec[1,0] > 0.01:
            forcastedVisitNumMat[website, day1] = (xMat[trainingDays + day1, 1] - factorVec[0]) / factorVec[1]
        else:
            forcastedVisitNumMat[website, day1] = xMat[trainingDays + day1, 1] - factorVec[0]

        factorVectors.append([factorVec[0,0], factorVec[1,0]])

    progFile.write("%s lwlrImpl factorVectors \n%s\n" % (getCurrentTime(), convertMatToStr(np.mat(factorVectors), "%.3f ")))
    return factorVectors

# Locally Weighted Linear Regression
def LWLR(dataHistory, forecastedWeekDayTotalVisitNum, lastStartWeeks, lastEndWeeks, progFile):
    totalUsers = len(forecastedWeekDayTotalVisitNum)
    progFile.write("%s LWLR(), start week %d,  end week %d, %d users \n" % (getCurrentTime(), lastStartWeeks, lastEndWeeks, totalUsers))

    forecastedVisitNum = {}

    trainingDays = (lastEndWeeks - lastStartWeeks) * 7
    # 1 - 6 周作为训练数据
    #训练数据的第一行全为 1, 第二行为website总的访问量, 作为自变量
    #第二行的最后7个值为预测出来的总的访问量
    #训练数据的最后一行是某website的访问量， 作为因变量，最后7个值为
    #预测出来的每个website的访问量
    visitNumTrainMat = np.mat(np.ones((3, trainingDays + 7)))

    factorMat = []
    zeroVisitNum =  np.zeros((1,7))#[0 for x in range(7)]

    runForUser = 0

    #for userId in dataHistory:
    for userId in forecastedWeekDayTotalVisitNum:

        runForUser += 1
        progFile.write("%s calculated users: %d / %d \n" % (getCurrentTime(), runForUser, totalUsers))
        
        if forecastedWeekDayTotalVisitNum[userId].sum() == 0:
            progFile.write("%s forecasted total visit num are 0, skip" % userId)            
            continue

        progFile.write("forecasted week day total visite num for [%s]\n %s \n" % (userId, convertMatToStr(forecastedWeekDayTotalVisitNum[userId])))
        
        #总的访问量的历史记录
        visitNumTrainMat[1, 0:trainingDays] = dataHistory[userId][SITES][:, lastStartWeeks*7:lastEndWeeks*7]
        #预测出来的总的访问量
        visitNumTrainMat[1, trainingDays:trainingDays + 7] = forecastedWeekDayTotalVisitNum[userId]

        forcastedVisitNumMat = np.zeros((SITES,7))

        progFile.write("dataHistory[%s]: %s\n" % (userId, convertMatToStr(dataHistory[userId])))

        for website in range(SITES):
            visitNumTrainMat[2, 0:trainingDays] = dataHistory[userId][website][:, 0:trainingDays]

            # shape(factorVec) = n * (2 * 1)
            k = 0.9
            factorVectors = lwlrImpl(visitNumTrainMat.T, trainingDays, k, forcastedVisitNumMat, website, progFile)
            days = len(factorVectors)

        forcastedVisitNumMat = roundVisitNum(np.mat(forcastedVisitNumMat))

        progFile.write("%s [%s]   forecasted website visite number \n%s\n" % (getCurrentTime(), userId, convertMatToStr(forcastedVisitNumMat)))

        if ( forcastedVisitNumMat.sum() == 0):
            progFile.write( "%s: all forecasted data are 0, skip!\n" % userId)
            continue

        forecastedVisitNum[userId] = forcastedVisitNumMat

        if (runForUser >= runMaxUser):
            break

        progFile.write("%s  LWLR() %d/%d \n" % (getCurrentTime(), runForUser, totalUsers))
        

    progFile.write("%s leaving LWLR()... \n" % getCurrentTime() )
    return forecastedVisitNum
