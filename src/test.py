import numpy as np
from common import *

dataFile = open("G:/workspace/tianyi_bd_history/testdat.dat", "w")
a = np.mat(np.ones((3,5)))
a[1,3] = 99
astr = convertMatrixToString(a)
dataFile.write(astr)

dataFile.close()