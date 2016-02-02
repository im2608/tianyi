import numpy as np
from common  import *
import math



# Simple Proportion , N weeks
def SimpleProportion(dataHistory, lastWeeks, progFile):

    progFile.write("%s forecastTotalVisitNum_SP() ... " % getCurrentTime())

    global totalUsers

    forecastedWeekDayTotalVisitNum = {} 
    userIdx = 1;
    for userId in dataHistory:
        # print("\n\n\n %s forecastTotalVisitNum_SP4(%d) for user %s \n" % ( getCurrentTime(), lastWeeks, userId),  dataHistory[userId][:, (WEEKS-lastWeeks)*7 : WEEKS*7])
        progFile.write("%s forecastTotalVisitNum_SP4(%s) %d/%d \n" % (getCurrentTime(), userId, userIdx, totalUsers))
        userIdx += 1

        forecastedWeekDayTotalVisitNum[userId] = [0 for day in range(7)]
        last4WeeksTotalVisitNum = dataHistory[userId][SITES][:, (WEEKS-lastWeeks)*7 : WEEKS*7][0].sum()

        if (last4WeeksTotalVisitNum == 0):
            continue

        avgWeekTotalVisitNum = last4WeeksTotalVisitNum / 4

        last4Weeks_eachTotalVisitNum = np.zeros((lastWeeks+1, 7))
        for lastWeekN in range(lastWeeks):
            last4Weeks_eachTotalVisitNum[lastWeekN] = dataHistory[userId][SITES][:, (WEEKS-lastWeekN-1)*7 : (WEEKS-lastWeekN)*7]
            last4Weeks_eachTotalVisitNum[lastWeeks] = last4Weeks_eachTotalVisitNum[lastWeeks] + last4Weeks_eachTotalVisitNum[lastWeekN]

        forecastedVisitNum = (last4Weeks_eachTotalVisitNum[lastWeeks] / last4WeeksTotalVisitNum) * avgWeekTotalVisitNum

        forecastedWeekDayTotalVisitNum[userId] = forecastedVisitNum
        # print("%s last %d weeks data : \n" % (userId, lastWeeks), last4Weeks_eachTotalVisitNum)
        
        if (userIdx > runMaxUser):
            break

    progFile.write("%s leaving forecastTotalVisitNum_SP() ... " % getCurrentTime())

    return forecastedWeekDayTotalVisitNum


# sliding average , last N weeks
def SlidingAverage(dataHistory, lastStartWeeks, lastEndWeeks, progFile):

    progFile.write("%s forecastTotalVisitNum_SlidingAvg() ... \n" % getCurrentTime())

    forecastedWeekDayTotalVisitNum = {} 
    userIdx = 1;
    noVisitNumUsers = 0
    for userId in dataHistory:
        forecastedWeekDayTotalVisitNum[userId] = np.zeros((1, 7))

        # 过去n周每天总的访问量， matrix 类型
        #lastWeeksEachdayTotalVisiNum = dataHistory[userId][SITES][:, (WEEKS-lastWeeks)*7 : WEEKS*7]
        lastWeeksEachdayTotalVisiNum = dataHistory[userId][SITES][:, lastStartWeeks*7 : lastEndWeeks*7]

        if (lastWeeksEachdayTotalVisiNum.sum() == 0):
            noVisitNumUsers += 1
            continue

        for week_day in range(7):
            lastWeeksTotalVisitNum = lastWeeksEachdayTotalVisiNum[:, week_day:lastEndWeeks * 7].sum()
            lastWeeksTotalVisitNum += forecastedWeekDayTotalVisitNum[userId][0:week_day].sum()
            forecastedWeekDayTotalVisitNum[userId][0][week_day] = round(lastWeeksTotalVisitNum / (lastEndWeeks * 7))

        forecastedWeekDayTotalVisitNum[userId] = np.mat(forecastedWeekDayTotalVisitNum[userId])
        if (userIdx > runMaxUser):
            break
        userIdx += 1

    progFile.write("%s leaving forecastTotalVisitNum_SlidingAvg() ... %d users forecasted no visit num\n" % (getCurrentTime(), noVisitNumUsers))

    return forecastedWeekDayTotalVisitNum
