import numpy as np
import sys
import math

"""
/****************/
setWindow
/****************/
Type : function
Process : window procedure 
Input : Y, window
	Y -> <numpy array> dataset. shape is (n, )
	window -> <string> window function
		{hanning, rectangle}
Output : 
"""
def setWindow(Y, window):
	n = len(Y) #the number of data

	if window == "hanning":
		return Y*np.hanning(n)
	elif window == "rectangle":
		return Y
	else:
		print("error@octave.Fourier.setWindow")
		print("window "+str(window)+" hasn't been defined yet.")
		sys.exit()

"""
/****************/
checkDataSize
/****************/
Type : function
Process : check the number of data which is used for fourier transfer
Input : Y, overlap, framesize
	Y -> <numpy array> dataset
	overlap -> <float> overlap ratio (0 <= overlap < 1)
	framesize -> <int> frame size
Output : 
"""
def checkDataSize(Y, overlap, framesize):
	average = math.floor( (len(Y) - overlap*framesize)/((1 - overlap)*framesize) )
	return average, int(framesize + (average - 1)*(1-overlap)*framesize)

"""
/****************/
DFT1d
/****************/
Type : function
Process : discrete fourier transfer for 1d data
Input : Y, dx, window, overlap
	Y -> <numpy array> dataset
	dx -> <float> discrete step for example time step
	framesize -> <int> framesize. If None, all data are used.
	window -> <string> window function
	overlap -> <float> overlap ratio (0 <= overlap < 1). overlap*framesize should be int
Output : freq, Pf, theta
	freq -> <numpy array> frequency
	Pf -> <numpy array> Power
"""
def DFT1d(Y, dx, framesize = None, window = "hanning", overlap = 0):
	if framesize is None:
		framesize = len(Y)
	
	#####check overlap value
	if not(0<=overlap<1):
		print("error@octave.Fourier.DFT1d")
		print("overlap should be 0<=overlap<1")
		sys.exit()

	if not((overlap*framesize-int(overlap*framesize))==0):
		print("error@octave.Fourier.DFT1d")
		print("overlap*framesize should be int")
		sys.exit()
	
	freq = np.linspace(0., 1./dx, framesize)[:int(framesize/2)] #frequency step from zero to Nyquist frequency
	average, data_num = checkDataSize(Y, overlap, framesize)
	
	#####cut data
	Y = Y[:data_num]
	
	#####split data
	split_data = [np.expand_dims(
		Y[int(i*(1-overlap)*framesize) : int(i*(1-overlap)*framesize+framesize)], axis = 0) 
		for i in range(average)]
	split_data = np.concatenate(split_data, axis = 0) #shape is (average, framesize)

	#####window precedure
	window_data = [np.expand_dims(setWindow(sd, window), axis = 0) for sd in split_data]
	window_data = np.concatenate(window_data, axis = 0) #shape is (average, framesize)

	#####discrete fourier transfer
	Yf = [np.expand_dims(np.fft.fft(wd), axis = 0) for wd in window_data]

	Yf = np.concatenate(Yf, axis = 0) #shape is (average, framesize)
	Pf = np.mean(np.abs(Yf), axis = 0) #amplitude
	Pf = Pf[:int(len(Pf)/2)]

	#####normalized
	Pf /= (framesize/2.); Pf[0] *= 2.


	return freq, Pf