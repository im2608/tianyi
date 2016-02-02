letter = "G"
filename = "%s:/workspace/tianyi_bd_history/output/forecast.dat" % letter

dataFile = open(filename, "r")

for aline in dataFile.readlines():
	userId, visistNums = aline.split("\t")
	everyNum = visistNums.split(",")
	newline = []
	for number in everyNum:
		if (number != "0"):
			number = "1"
		newline.append(number)