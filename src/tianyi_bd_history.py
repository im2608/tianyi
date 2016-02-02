import numpy as np
from common  import *
import math
import sys
import calWebSiteDayVisitNum
import forecastTotalVisitNum
import treeRegression
import BP

def printDataHistory():
    for userId in dataHistory:
        progFile.write("%s => %s \n" % (userId, convertMatrixToString(np.mat(dataHistory[userId]))))


def outputForecastedVisitNum(forecastedVisitNum, fileIdx):

    outputFile = open("%s:/workspace/tianyi_bd_history/output/forecast.%s" % (DRIVE_LETTER, fileIdx), "w")
    outputUsers = 0

    for userId in forecastedVisitNum:
        if (forecastedVisitNum[userId].sum() == 0):
            #progFile.write("all forecast of [%s] is 0, skip!\n" % userId)
            continue

        outputStr = "%s\t" % userId
        websites, days = np.shape(forecastedVisitNum[userId])
        for site in range(websites):
            for week_day in range(days):
                outputStr += "%d," % forecastedVisitNum[userId][site, week_day]

        outputFile.write(outputStr[0:len(outputStr) - 1] + "\n")
        outputUsers += 1

    outputFile.close()
    progFile.write("outputForecastedVisitNum(): %d user have been output\n" % outputUsers)
    return 0

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

##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################


fileIdx = sys.argv[1].split("=")[1]

dataFile = "%s:/Workspace/tianyi_bd_history/splitedHistory/datafile.%s" % (DRIVE_LETTER, fileIdx)
progFile = open("%s:/Workspace/tianyi_bd_history/logs/progress.%s" % (DRIVE_LETTER, fileIdx), "w")

progFile.write("%s file %s is analysing..." % (getCurrentTime(), dataFile))

if (False):
    dataFile = "%s:/Workspace/tianyi_bd_history/part-r-00000" % DRIVE_LETTER

progFile.write("reading data file %s \n" % dataFile)
##########################################################################################
# 读取历史记录
##########################################################################################

dataHistory = loadData(dataFile)


##########################################################################################
# 计算每日总的访问量
##########################################################################################
lastStartWeeks = 0 #   从 lastStartWeeks 开始， 0 based
lastEndWeeks = 7 # 到 lastEndWeeks 结束， 所以 1 based
algoForTotalVisitNum = "SLIDE" # SLIDE = Sliding Average, SP = Simple Proportion
progFile.write("using algo for total visit num: %s \n" % algoForTotalVisitNum)

if (algoForTotalVisitNum == "SLIDE"): #滑动平均法
    forecastedWeekDayTotalVisitNum = forecastTotalVisitNum.SlidingAverage(dataHistory, lastStartWeeks, lastEndWeeks, progFile)
elif (algoForTotalVisitNum == "SP"): # 简单比例法
    forecastedWeekDayTotalVisitNum = forecastTotalVisitNum.SimpleProportion(dataHistory, lastWeeks)

##########################################################################################
# 1. 根据训练数据得到每日每个website访问量的factor
# 2. 根据每日总的访问量和factor来进行预测
##########################################################################################

algoForDayFactors = "BP" # LS = LeastSquare, LWLR = Locallt Weighted Linear Regression, TREEREG = tree regression, BP = back propagation
progFile.write("using algo for each day visit num: %s \n" % algoForDayFactors)

if (algoForDayFactors == "LS"): #最小二乘
    forecastedVisitNum = calWebSiteDayVisitNum.LeastSquare(dataHistory, forecastedWeekDayTotalVisitNum, progFile)
elif (algoForDayFactors == "LWLR"): #局部加权线性回归
    forecastedVisitNum = calWebSiteDayVisitNum.LWLR(dataHistory, forecastedWeekDayTotalVisitNum, lastStartWeeks, lastEndWeeks, progFile)
elif (algoForDayFactors == "SLIDE"): #滑动平均
    forecastedVisitNum = calWebSiteDayVisitNum.SlidingAvg(dataHistory, lastStartWeeks, lastEndWeeks, progFile)
elif (algoForDayFactors == "TREEREG"): #树回归
    forecastedVisitNum = treeRegression.treeRegression(dataHistory, forecastedWeekDayTotalVisitNum, lastStartWeeks, lastEndWeeks, progFile)
elif (algoForDayFactors == "BP"): #反向传播
    forecastedVisitNum = BP.BackPropagation(dataHistory, forecastedWeekDayTotalVisitNum, lastStartWeeks, lastEndWeeks, progFile)    

verifyWeek = 6 # 用这一周来验证， 0 based

f1 = verification(dataHistory, forecastedVisitNum, verifyWeek, progFile)

##########################################################################################
##########################################################################################

outputForecastedVisitNum(forecastedVisitNum, fileIdx)

progFile.write("%s analysing %s has done... Exiting..." % (getCurrentTime(), dataFile))
progFile.close()
