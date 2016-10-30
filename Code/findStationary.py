import numpy as np 
import sys
import matplotlib.pyplot as plt
import gmplot
from geopy.geocoders import Nominatim

gpsValThreshold = 0.00005
stationaryTimeThreshold = 100
stationaryLimit = []
secondFilterFlag = 0
secondFilterList = []
stdThreshold = 0.00005
geolocator = Nominatim()

def identify_stationary(newMat):
	stationaryFlag = 0
	for time in range(0,newMat.shape[0]-1):	
		print (newMat[time][1] - newMat[time+1][1])
		print (newMat[time][2] - newMat[time+1][2])
		print abs(gpsValThreshold)
		if (abs(newMat[time][1] - newMat[time+1][1]) <= abs(gpsValThreshold)) and (abs(newMat[time][2] - newMat[time+1][2]) <= abs(gpsValThreshold)) and stationaryFlag == 0:
			stationaryLimit.append(time)
			stationaryFlag = 1
		if (abs(newMat[time][1] - newMat[time+1][1]) > abs(gpsValThreshold)) and (abs(newMat[time][2] - newMat[time+1][2]) > abs(gpsValThreshold)) and stationaryFlag == 1:
			stationaryLimit.append(time)
			stationaryFlag = 0

def generateAddressFromLocation(newMat, secondFilterList, latMean, longMean):
	for i in range(0,len(secondFilterList),2):
		timeStampMean = (secondFilterList[i]+secondFilterList[i+1])/2
		locationLat = newMat[timeStampMean][1] + latMean
		locationLong = newMat[timeStampMean][2] + longMean
		correspondingLocation = (locationLat,locationLong)
		stationaryLocation = geolocator.reverse(correspondingLocation)
		print "The user was stationary at " + stationaryLocation.address

def mapPlot(newMat,secondFilterList,latMean,longMean):
	latitudeList = []
	longitudeList = []
	gmap = gmplot.GoogleMapPlotter(30.28142551, -97.73600101, 16)
	for i in range(0,len(secondFilterList),2):
		time = secondFilterList[i] + ((secondFilterList[i+1] - secondFilterList[i])/2)
		latitudeList.append(newMat[time,1]+latMean)
		longitudeList.append(newMat[time,2]+longMean)
	print latitudeList
	print longitudeList
	gmap.heatmap(latitudeList,longitudeList)
	gmap.draw("heatMap.html")

if __name__ == "__main__":
	gpsIn = sys.argv[1]
	gpsData = np.genfromtxt(gpsIn, delimiter=',')
	firstTimeStamp = int(gpsData[0][0])
	newMat = np.zeros((int(gpsData[gpsData.shape[0]-1][0]) - int(gpsData[0][0]),4))
	newMatIter = 0
	for i in range(0,gpsData.shape[0]-1):
		for newMatIter in range(int(gpsData[i][0]),int(gpsData[i+1][0])):
			newMat[newMatIter - firstTimeStamp] = gpsData[i]
			newMat[newMatIter - firstTimeStamp][0] = newMatIter
	#np.savetxt('gps_change.csv',newMat,delimiter=',')
	latMean = np.mean(newMat[:,1])
	longMean = np.mean(newMat[:,2])
	newMat[:,1] = (newMat[:,1] - latMean)
	newMat[:,2] = (newMat[:,2] - longMean)
	identify_stationary(newMat)
	stationaryLimitArr = np.asarray(stationaryLimit)
	plt.plot((newMat[:,0]-int(firstTimeStamp)),newMat[:,1],label='Latitude')
	plt.plot((newMat[:,0]-int(firstTimeStamp)),newMat[:,2],label='Longitude')
	plt.legend()
	print stationaryLimit
	realStationary = []
	for i in range(0,len(stationaryLimit)-1,2):
		if(stationaryLimit[i+1] - stationaryLimit[i] > stationaryTimeThreshold):
			realStationary.append(stationaryLimit[i])
			realStationary.append(stationaryLimit[i+1])
	print realStationary
	for secondFilterIter in range(0,len(realStationary),2):
		print secondFilterIter
		frameAdaptive = 0
		for frameIter in range(realStationary[secondFilterIter], realStationary[secondFilterIter+1]):
			print frameIter
			if realStationary[secondFilterIter+1] - frameIter < stationaryTimeThreshold:
				if secondFilterFlag == 1:
					secondFilterList.append(frameIter+stationaryTimeThreshold)
					secondFilterFlag = 0
				break
			else:
				idx = frameIter - frameAdaptive
				print frameAdaptive
				print 'blah'
				latStd = np.std(newMat[ idx : frameIter+stationaryTimeThreshold,1])
				longStd = np.std(newMat[idx : frameIter+stationaryTimeThreshold,2])
				if latStd < stdThreshold and longStd < stdThreshold:
					if secondFilterFlag == 0:
						secondFilterList.append(frameIter)
						secondFilterFlag = 1
						frameAdaptive = 0
					else:
						frameAdaptive += 1
				else:
					if secondFilterFlag == 1:
						secondFilterList.append(frameIter+stationaryTimeThreshold)
						secondFilterFlag = 0
						frameAdaptive = 0
						break
	print secondFilterList
	print realStationary

	mapPlot(newMat,secondFilterList,latMean,longMean)
	generateAddressFromLocation(newMat,secondFilterList,latMean,longMean)

	for xc in realStationary:
		plt.axvline(x=xc,color='y')
	for yc in secondFilterList:
		plt.axvline(x=yc,color='r')
	plt.xlabel('Time')
	plt.ylabel('GPS Data')
	plt.grid()
	plt.show()