'''
Erik Skogfeldt	11/30/2018

Multiple Object Detection and Tracking from camera feed

'''
import numpy as np
import cv2 
import imutils
import datetime
import copy
import sys
import random

from munkres import Munkres,print_matrix
from kalmanFilter import kalman
from matrixClass import matrix

ID = 0
ageThreshold = 5

tracks = []

firstFrame = None
unmatchedDetections =[]
unmatchedTracks =[]
matches = []
solution = []

def getID():
	global ID 
	ID = ID + 1
	return ID
class Track:
	def __init__(self):
		self.x = 0
		self.y = 0
		self.sw = 0
		self.sy = 0 
		self.id = getID()
		self.age = 1
		self.visibleFrames = 1
		self.kalman = None

	def updateXY(self,x,y):
		self.x = x
		self.y = y
def detectObjects(firstFrame,gray):

	#background subtraction
	frameDelta = cv2.absdiff(firstFrame, gray)
	thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
 

	# dilate the thresholded image to fill in holes, then find contours
	# on thresholded image
	thresh = cv2.dilate(thresh, None, iterations=2)
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]

	detections = []
	count = 0
	# LOOP OVER THE CONTOURS AND ADD SIGNIFICANTLY LARGE COUNTOURS TO DETECTIONS
	for c in cnts:		
 
		# if the contour is too small, ignore it
		if cv2.contourArea(c) < 500:
			continue
		count += 1
		# compute the bounding box for the contour, draw it on the frame,
		# and update the text
		(x, y, w, h) = cv2.boundingRect(c)

		detections.append( [x ,y ,w,h] )
	cv2.imshow("threshold ", thresh)
	return detections

def detectionToTrackAssignment(unmatchedDetectionCost,unmatchedTrackCost,detections,tracks):

	rows = len(detections )
	cols = len(tracks)

	if (len(detections) == 0):
		return [[],tracks, [] ,[]]
 
 	#if no track exists, then all detections are unmatched detections
	if (len(tracks) == 0):
		return [detections,[], [] ,[]]

	#Create cost matrix
	cost = []
	for i in range(rows):
		row = []
		for j in range(cols):

			## non-padded cost matrix
			trackX = (tracks[j].kalman.predict())[2][0]
			trackY = (tracks[j].kalman.predict())[3][0]
			detectionX = detections[i][0]
			detectionY = detections[i][0]
			# Euclidean cost function
			row.append( ( detectionX - trackX)**2 + ( detectionY - trackY)**2  )
		for j in range(rows):
			row.append(unmatchedTrackCost)
		cost.append(row)
	for i in range(cols):
		row = []
		for j in range(cols):
			row.append( unmatchedDetectionCost )
		for j in range(rows):
			row.append(0)
		cost.append(row)
		
	m = Munkres()
	solution =  m.compute(cost)

	unmatchedTracks = []
	unmatchedDetections = []
	matches = []

	for x in range(len(solution)):
		if (solution[x][0] < rows and solution[x][1] < cols):
			matches.append(tracks[solution[x][1]] )
		elif (solution[x][0] < rows and solution[x][1] >= cols):
			unmatchedDetections.append(detections[solution[x][0]])
		elif (solution[x][0] >= rows and solution[x][1] < cols):
			unmatchedTracks.append(tracks[solution[x][1]])\

	return [unmatchedDetections,unmatchedTracks,matches,solution]

def predictNewLocationsOfTracks():
	global tracks
	for track in tracks:
		tmp = track.kalman.predict()
		track.x= int(tmp[2][0])
		track.y= int(tmp[3][0])

def updateAssignedTracks(solution):
	global tracks
	global detections

	rows = len(detections)
	cols = len(tracks)

	#update kalman filters with detected positions
	for x in range(len(solution)):
		if (solution[x][0] < rows and solution[x][1] < cols):

			k = tracks[solution[x][1]].kalman
			detectionX = detections[ solution[x][0]][0]
			detectionY = detections[ solution[x][0]][1]
			k.measure( [ [ detectionX], [detectionY ] ] )

			#update tracks position (from center )and size to match detection position and size
			tmp = tracks[solution[x][1]].kalman.predict()
			tracks[solution[x][1]].sx = detections[ solution[x][0]][2]
			tracks[solution[x][1]].sy = detections[ solution[x][0]][3]
			tracks[solution[x][1]].x = int(tmp[2][0]) +  (.5* tracks[solution[x][1]].sx)
			tracks[solution[x][1]].y = int(tmp[3][0]) +  (.5* tracks[solution[x][1]].sy)

			tracks[solution[x][1]].visibleFrames = tracks[solution[x][1]].visibleFrames +1

def updateUnassignedTracks(unassignedTracks):
	global tracks
	for i in range(len(unassignedTracks)):
		for j in range(len(tracks)):
			if (unassignedTracks[i].id == tracks[j].id):
				tracks[j].age = min(tracks[j].age + 1, 150)
				break
def deleteLostTracks():

	global tracks
	for t in tracks:
		if t.age >ageThreshold and t.visibleFrames/ float(t.age) < .6:
			tracks.remove(t)


def createNewTracks(unassignedDetections):
	global tracks
	for i in range(len(unassignedDetections)):
		delta = [[0,2],[1,3]]
		x = [[unassignedDetections[i][0] ],[unassignedDetections[i][1]],[unassignedDetections[i][0]],[unassignedDetections[i][1]]] # initial state (location x,y,x',y')

		P = [	[1000., 0., 0., 0.],
				[0., 1000., 0., 0.],
				[0., 0., 1000., 0.],
				[0., 0., 0., 1000.],] # initial uncertainty

		u = [	[0.], [0.],[0.],[0.]] # external motion

		F = [	[1., 0., 0., 0.],
				[0., 1., 0., 0.],
				[0., 0., 1., 0.],
				[0., 0., 0., 1.]] # next state function

		H = [	[0., 0., 1., 0.],
				[0., 0., 0., 1.]] # measurement function

		R = [	[10., 0.],
				[0., 10.]] # measurement uncertainty

		I = [	[1., 0., 0., 0.],
				[0., 1., 0., 0.],
				[0., 0., 1., 0.],
				[0., 0., 0., 1.]] # identity matrix
		t = Track()
		t.kalman = kalman(x,P,u,F,H,R,I,delta)
		t.x = unassignedDetections[i][0] #+ (.5 *unassignedDetections[i][2])
		t.y = unassignedDetections[i][1] #+ (.5 *unassignedDetections[i][3])
		t.sx = unassignedDetections[i][2]
		t.sy = unassignedDetections[i][3]

		tracks.append(t)


##MAIN LOOP
#######################################################################################################
cap = cv2.VideoCapture(1)

while(1):

	ret, frame = cap.read()
	frame = imutils.resize(frame, width=500)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21, 21), 0)
 
	# if the first frame is None, initialize the camera by taking consecutive camera reads
	if firstFrame is None:
		for i in range(25):
			ret, frame = cap.read();
		frame = imutils.resize(frame, width=500)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (21, 21), 0)
		firstFrame = gray
		continue

	detections =  detectObjects(firstFrame,gray)

	if (len(tracks) >= 1):
		predictNewLocationsOfTracks()

	unmatchedDetections,unmatchedTracks,matches,solution =  detectionToTrackAssignment(10**3,10**5,detections,tracks)

	if (len(matches) >= 1):
		updateAssignedTracks(solution)

	if (len(unmatchedTracks) >= 1):
		updateUnassignedTracks(unmatchedTracks)

	if (len(tracks) >= 1):
		deleteLostTracks()

	if ( len(unmatchedDetections) >= 1):
		createNewTracks(unmatchedDetections)
	#displayTrackingResults();

	for track in matches:
		cv2.rectangle(frame,(int(track.x-(.5*track.sx)),int(track.y-(.5*track.sy))),(int(track.x+ (.5*track.sx)),int(track.y+(.5*track.sy))),(255,0,0),1)
		cv2.circle(frame,(int(track.x),int(track.y) ),5,(255,0,0) )

	cv2.putText(frame,'|Matches| ={}, |Detections W/O Track|={}], |Tracks W/O Detection| ={}'.format(len(matches),len(unmatchedDetections),len(unmatchedTracks)),(10,350),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),2)
	cv2.imshow("Multiple object detection. ", frame)
	# LOOP ESCAPE AND CLEANUP
	###############################################################################################
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break
cap.release()
cv2.destroyAllWindows()