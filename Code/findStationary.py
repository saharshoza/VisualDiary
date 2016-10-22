import numpy as np 
import sys
import matplotlib.pyplot as plt

threshold = 0.00005
stationaryLimit = []

def identify_stationary(newMat):
	stationaryFlag = 0
	for time in range(0,newMat.shape[0]-1):	
		print (newMat[time][1] - newMat[time+1][1])
		print (newMat[time][2] - newMat[time+1][2])
		print abs(threshold)
		if (abs(newMat[time][1] - newMat[time+1][1]) <= abs(threshold)) and (abs(newMat[time][2] - newMat[time+1][2]) <= abs(threshold)) and stationaryFlag == 0:
			stationaryLimit.append(time)
			stationaryFlag = 1
		if (abs(newMat[time][1] - newMat[time+1][1]) > abs(threshold)) and (abs(newMat[time][2] - newMat[time+1][2]) > abs(threshold)) and stationaryFlag == 1:
			stationaryLimit.append(time)
			stationaryFlag = 0

if __name__ == "__main__":
	
	#Get File
	gpsIn = sys.argv[1]
	gpsData = np.genfromtxt(gpsIn, delimiter=',')
	firstTimeStamp = int(gpsData[0][0])
	newMat = np.zeros((int(gpsData[gpsData.shape[0]-1][0]) - int(gpsData[0][0]),4))
	newMatIter = 0
	for i in range(0,gpsData.shape[0]-1):
		for newMatIter in range(int(gpsData[i][0]),int(gpsData[i+1][0])):
			newMat[newMatIter - firstTimeStamp] = gpsData[i]
			newMat[newMatIter - firstTimeStamp][0] = newMatIter
	np.savetxt('gps_change.csv',newMat,delimiter=',')
	#newMat[:,1] = (newMat[:,1] - np.mean(newMat[:,1]))/(newMat[newMat.shape[0]-1,1]-newMat[0,1])
	#newMat[:,2] = (newMat[:,2] - np.mean(newMat[:,2]))/(newMat[newMat.shape[0]-1,2]-newMat[0,2])
	newMat[:,1] = (newMat[:,1] - np.mean(newMat[:,1]))
	newMat[:,2] = (newMat[:,2] - np.mean(newMat[:,2]))
	identify_stationary(newMat)
	stationaryLimitArr = np.asarray(stationaryLimit)
	plt.plot((newMat[:,0]-int(firstTimeStamp)),newMat[:,1],label='Latitude')
	plt.plot((newMat[:,0]-int(firstTimeStamp)),newMat[:,2],label='Longitude')
	plt.legend()
	print stationaryLimit
	realStationary = []
	for i in range(0,len(stationaryLimit)-1,2):
		if(stationaryLimit[i+1] - stationaryLimit[i] > 100):
			realStationary.append(stationaryLimit[i])
			realStationary.append(stationaryLimit[i+1])
	for xc in realStationary:
		plt.axvline(x=xc)
	plt.xlabel('Time')
	plt.ylabel('GPS Data')
	plt.grid()
	plt.show()