import sys
import subprocess  
import os  
import time
from common import *

def splitHistoryData(fileName):
    print(" reading data file ", fileName)
    dataHistory = {}
    historyDataFile = open(fileName, "r")
    print("splited_files is  %d" %splited_files)

    splitedFileHandle = []
    for fileIdx in range(splited_files):
        splitedFileName = "%s:/Workspace/tianyi_bd_history/splitedHistory/datafile.%03d" % (DRIVE_LETTER, fileIdx)
        dataFile = open(splitedFileName, "w")
        splitedFileHandle.append(dataFile)
        print("%s created" % splitedFileName)

    lineIdx = 0

    for aline in historyDataFile.readlines():
        userId = aline.split("\t")[0]
        fileIdx = hash(userId) % splited_files
        splitedFileHandle[fileIdx].write(aline)
        lineIdx += 1

    print("history data file is read")

    for fileIdx in range(splited_files):
    	splitedFileHandle[fileIdx].close()

    historyDataFile.close()

    return 0

def runAnalysis():
    runningSubProcesses = {}
    for fileIdx in range(splited_files):
        cmdLine = "python tianyi_bd_history.py fileIdx=%03d" % (fileIdx + started_file)
        sub = subprocess.Popen(cmdLine, shell=True)
        runningSubProcesses[fileIdx] = sub
        print(cmdLine)

    while True:
        endedProcessIdx = waitSubprocesses(runningSubProcesses)
        if endedProcessIdx >=0 and endedProcessIdx < splited_files:            
            runningSubProcesses.pop(endedProcessIdx)

        if len(runningSubProcesses) == 0:
            break
    return 0

def waitSubprocesses(runningSubProcesses):
    for fileIdx in runningSubProcesses:
        sub = runningSubProcesses[fileIdx]
        ret = subprocess.Popen.poll(sub)
        if ret == 0:
            print("subprocess %d ended" % fileIdx)
            return fileIdx            
        elif ret is None:
            time.sleep(1) # running
        else:
            print("subporcess %d terminated" % fileIdx)
            runningSubProcesses.pop(fileIdx)
            return fileIdx
    return -1


def combineResultFiles():
    resultFile = open(finalResultFileName, "w")
    for fileIdx in range(splited_files):
        fileName = "%s:/workspace/tianyi_bd_history/output/forecast.%03d" % (DRIVE_LETTER, fileIdx + started_file)
        print("combineResultFiles() %s " % fileName)
        outputFile = open(fileName, "r")
        for aline in outputFile.readlines():
            resultFile.write(aline)
    return 0

def loadForecastedData():
    forecastedVisitNum = {}
    dataFile = open(finalResultFileName, "r")
    for aline in dataFile.readlines():
        userId, eachdayVisitNum = aline.split("\t")
        forecastedVisitNum[userId] = np.mat(np.zeros((SITES, WEEKS)))        
        eachdayVisitNum = eachdayVisitNum.split(",")
        for eachday in range(len(eachdayVisitNum)):
            forecastedVisitNum[userId][eachday/7, eachday%7] = int(eachdayVisitNum[eachday])

    return forecastedVisitNum

def finalVerification():
    forecastedVisitNum = loadForecastedData()
    historyData = {}

    for fileIdx in range(splited_files):        
        splitedFileName = "%s:/Workspace/tianyi_bd_history/splitedHistory/datafile.%03d" % (DRIVE_LETTER, fileIdx + started_file)
        print("finalVerification(): reading %s " % splitedFileName)
        historyData.update(loadData(splitedFileName))

    verification(historyData, forecastedVisitNum, 6, progFile=None)
    return 0



############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################

historyFileName = "%s:/Workspace/tianyi_bd_history/data/part-r-00000-new" % DRIVE_LETTER
finalResultFileName = "%s:/Workspace/tianyi_bd_history/output/forecast.dat" % (DRIVE_LETTER)
splited_files = 6
started_file = 4

if False:
    splitHistoryData(historyFileName)

if True:
    runAnalysis()
    combineResultFiles()

if False:
    finalVerification()