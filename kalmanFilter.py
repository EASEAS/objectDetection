import os
import time
import math
from matrixClass import matrix
##########################################################################



class kalman:
	def __init__(self,x,P,u,F,H,R,I,delta):

		# Model Matrices
		self.x = matrix(x) # initial state (location and velocity)
		self.P = matrix(P) # initial uncertainty
		self.u = matrix(u) # external motion
		#self.F = matrix(F) # next state function
		self.H = matrix(H) # measurement function
		self.R = matrix(R) # measurement uncertainty
		self.I = matrix(I) # identity matrix
		self.delta = 1

		for d in delta:
			F[d[0]][d[1]] = self.delta

		self.F = matrix(F)

		self.start = 	 time.time()
	#recieve a new measurement and return prediction for next measurement	
	def measure(self,m):
		#kalman measurement. 
		self.measurement = m
		self.end = time.time()
		self.delta = self.end - self.start
		##################################################################
		# Measurement
		# get measurement
		self.z = matrix(self.measurement)

		# get difference of measurement and actual value
		self.y = self.z - (self.H*self.x)
		


		# measurement with applied covariance + measurement noise
		self.S = (self.H * self.P * self.H.transpose()) + self.R

		# calculate gain
		self.k = self.P* self.H.transpose() * self.S.inverse()

    		#update measurement
		self.x = self.x +(self.k*self.y)
		#update covariance
		self.P = (self.I - (self.k*self.H))* self.P
		#restart delta timer 
		self.start = time.time()
		
		
		##################################################################
	def predict(self):
    		# prediction
		#calculate next state from current + external distrubances
		self.x = ( self.F * self.x ) + self.u
		#calculate new covariances
		self.P = self.F *self.P * self.F.transpose() 
			
		return self.x.value
	def getLastPrediction(self):
		return self.x.value