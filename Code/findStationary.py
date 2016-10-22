import numpy as np 
import sys
import matplotlib.pyplot as plt

if __name__ == "__main__":
	
	#Get File
	gpsIn = sys.argv[1]
	gpsData = np.genfromtxt(gpsIn, delimiter=',')
	#print gpsData[gpsData.shape[0]-1][0]
	firstTimeStamp = int(gpsData[0][0])
	newMat = np.zeros((int(gpsData[gpsData.shape[0]-1][0]) - int(gpsData[0][0]),4))
	newMatIter = 0
	for i in range(0,gpsData.shape[0]-1):
		for newMatIter in range(int(gpsData[i][0]),int(gpsData[i+1][0])):
			newMat[newMatIter - firstTimeStamp] = gpsData[i]
	np.savetxt('gps_change.csv',newMat,delimiter=',')
	plt.plot((newMat[:,0]-int(firstTimeStamp)),(newMat[:,1] - np.mean(newMat[:,1]))/np.std(newMat[:,1]))
	plt.plot((newMat[:,0]-int(firstTimeStamp)),(newMat[:,2] - np.mean(newMat[:,2]))/np.std(newMat[:,2]))
	plt.show()