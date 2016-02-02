import numpy as np
from common  import *
import math

def errorVariance(lessEqual, larger):
    return np.var(lessEqual[1]) * np.shape(lessEqual)[1] + np.var(larger[1]) * np.shape(larger)[1]

def errorLS(lessEqualMat, largerMat, visitNumTrainMat, eachday, website, progFile):
    lessEqualFac = leastSquareImpl(lessEqualMat)
    largerFac = leastSquareImpl(largerMat)

    if (lessEqualFac is None or largerFac is None):
        progFile.write("   <= or > are singluar when split on day %d website %d.\n" % (eachday, website))
        factor = leastSquareImpl(visitNumTrainMat)
        if (factor is None):
            return None
        splitErr = visitNumTrainMat[:, 0:2] * factor - visitNumTrainMat[:, 2]
        splitErr = np.power(splitErr, 2).sum()
        return splitErr, factor, factor

    lessEqualErr =  lessEqualMat[:, 0:2] * lessEqualFac - lessEqualMat[:, 2]

    largerErr = largerMat[:, 0:2] * largerFac - largerMat[:, 2]

    splitErr = np.power(lessEqualErr, 2).sum() + np.power(largerErr, 2).sum()

    return splitErr, lessEqualFac, largerFac

def findBestSplitPointlitPoint(visitNumTrainMat, website, progFile):
    bestSplitPoint = 0
    bestLessEqualFac = 0
    bestLargerFac = 0
    bestVal = np.inf
    bestLessEqualMat = 0
    bestLargerMat = 0

    trainArr = visitNumTrainMat.A

    trainingDays = np.shape(visitNumTrainMat)[1]

    # 过去n周每天总的访问量, 在该集合上遍历，根据 visitNumTrainMat[2] 将 visitNumTrainMat[1] 分成
    # <= 和 > 两部分，取得两部分的总方差之和， 导致总方差之和最小的
    # visitNumTrainMat[2] 上的点作为切分点， 对应的visitNumTrainMat[1]上的点取1 或0，根据 <=, >两部分
    # 中那个多取哪个
    dayidx = 0
    for eachday in range(trainingDays):
        
        dayidx += 1

        testingPoint = visitNumTrainMat[2, eachday]
        lessEqual = visitNumTrainMat[:, np.nonzero(trainArr[2] <= testingPoint)]
        lessEqual = lessEqual[:, 0]

        larger = visitNumTrainMat[:, np.nonzero(trainArr[2] > testingPoint)]
        larger = larger[:, 0]

        #progFile.write("SP %d, <= \n%s\n, > \n%s\n" % (eachday, convertMatToStr(lessEqual), convertMatToStr(larger)))

        splitVal, lessEqualFac, largerFac = errorLS(lessEqual.T, larger.T, visitNumTrainMat.T, eachday, website, progFile)
        if (splitVal is None):
            continue

        if (splitVal < bestVal):
            bestVal = splitVal
            bestSplitPoint = eachday
            bestLessEqualFac = lessEqualFac
            bestLargerFac = largerFac
            bestLessEqualMat = lessEqual
            bestLargerMat = larger

    return bestSplitPoint, bestLessEqualFac, bestLargerFac, bestLessEqualMat, bestLargerMat

def calculationXWithFactor(Y, factor):
    if (factor[1, 0] > 0.01 and factor[1, 0] != 0):
        return (Y - factor[0, 0])/factor[1, 0]
    else:
        return Y - factor[0, 0]


def treeRegression(dataHistory, forecastedWeekDayTotalVisitNum, lastStartWeeks, lastEndWeeks, progFile):
    totalUsers = len(forecastedWeekDayTotalVisitNum)
    progFile.write("%s treeReg(), start week %d,  end week %d, %d users \n" % (getCurrentTime(), lastStartWeeks, lastEndWeeks, totalUsers))

    forecastedVisitNum = {}

    trainingDays = (lastEndWeeks - lastStartWeeks) * 7

    runForUser = 0

    for userId in forecastedWeekDayTotalVisitNum:
        runForUser += 1
        progFile.write("%s calculated users: %d / %d , [%s], history data: \n%s\n" % (getCurrentTime(), runForUser, totalUsers, userId, convertMatToStr(dataHistory[userId])))

        visitNumTrainMat = np.mat(np.ones((3, trainingDays)))

        # 过去n周每天总的访问量
        visitNumTrainMat[2, 0:trainingDays] = dataHistory[userId][SITES][:, lastStartWeeks*7:lastEndWeeks*7]

        forecastedVisitNum[userId] = []

        for website in range(SITES):
            # 过去n周每天某个website的访问量， matrix 类型
            visitNumTrainMat[1, 0:trainingDays] = dataHistory[userId][website][:, lastStartWeeks*7:lastEndWeeks*7]

            if (visitNumTrainMat[1].sum() == 0):
                progFile.write("visit num of week %d - %d weeks of website [%d] is 0, skip!\n" % (lastStartWeeks, lastEndWeeks, website))
                forecastedVisitNum[userId].append([0,0,0,0,0,0,0])
                continue

            progFile.write("[%s] website %d training mat \n%s\n" % (userId, website, convertMatToStr(visitNumTrainMat)))

            websiteVisitNum = []
            bestSplitPoint, bestLessEqualFac, bestLargerFac, bestLessEqualMat, bestLargerMat = findBestSplitPointlitPoint(visitNumTrainMat, website, progFile)
            progFile.write("[%s] website (%d) best split point %d (%d) \n " % (userId, website, bestSplitPoint, visitNumTrainMat[2, bestSplitPoint]))
            progFile.write("[%s] website (%d) best <= mat \n%s\n best <= fac\n%s\n" % (userId, website, convertMatToStr(bestLessEqualMat), convertMatToStr(bestLessEqualFac)))
            progFile.write("[%s] website (%d) best > mat \n%s \n best > fac \n%s" % (userId, website, convertMatToStr(bestLargerMat), convertMatToStr(bestLargerFac)))

            for weekDay in range(7):
                totalVisitNum = forecastedWeekDayTotalVisitNum[userId][0, weekDay]
                if (totalVisitNum <= visitNumTrainMat[2, bestSplitPoint]):
                    websiteVisitNum.append(calculationXWithFactor(totalVisitNum, bestLessEqualFac))
                else:
                    websiteVisitNum.append(calculationXWithFactor(totalVisitNum, bestLargerFac))

            forecastedVisitNum[userId].append(websiteVisitNum)

        forecastedVisitNum[userId] = roundVisitNum(np.mat(forecastedVisitNum[userId]))

        progFile.write("%s [%s] forecasted visit number: \n %s\n" % (getCurrentTime(), userId, convertMatToStr(forecastedVisitNum[userId])))

    return forecastedVisitNum
