import numpy as np 
import matplotlib.pyplot as plt
import gmplot

dataset = np.genfromtxt("rahul_gps.csv",delimiter=",")
latitudeList = []
longitudeList = []
gmap = gmplot.GoogleMapPlotter(30.28142551, -97.73600101, 16)

for rowIterator in dataset:
	latitudeList.append(rowIterator[1])
	longitudeList.append(rowIterator[2])
print latitudeList
print longitudeList
gmap.heatmap(latitudeList,longitudeList)
gmap.draw("myMap.html")