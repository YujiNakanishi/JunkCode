import numpy as np
import math

"""
/**************/
cvOctave
/**************/
Type : function
Process : convert Octave
Input : freq, dB, N, G
	freq -> <numpy array> Frequency. shape is (n, )
	dB -> <numpy array> dB. shape is (n, )
	N -> <int> Octave
	G -> <float> Octave ratio
Output : freq_octave, dB_octave
	freq_octave -> <numpy array> Frequency. shape is (n, )
	dB_octave -> <numpy array> dB. shape is (n, )
"""
def cvOctave(freq, dB, N, G = None):
	if G is None:
		G = 10**(0.3)

	#####calcurate frequency band
	if (N%2) == 0:
		min_band = math.floor(N*np.log(np.min(freq)/1000)/np.log(G) - 0.5)
		max_band = math.ceil(N*np.log(np.max(freq)/1000)/np.log(G) - 0.5)
		freq_octave = G**((2*np.arange(min_band, max_band+1)+1)/(2*N))*1000 #frequency for new octave band
	else:
		min_band = math.floor(N*np.log(np.min(freq)/1000)/np.log(G))
		max_band = math.ceil(N*np.log(np.max(freq)/1000)/np.log(G))
		freq_octave = G**(np.arange(min_band, max_band+1)/N)*1000 #frequency for new octave band

	freq_low = G**(-1/(2*N))*freq_octave #lower frequency
	freq_up = G**(1/(2*N))*freq_octave #upper frequency

	#####calculate new dB
	dB_octave = []

	for fl, fu in zip(freq_low, freq_up):
		mask = (freq>fl)*(freq<fu)

		if np.any(mask):
			dB_octave.append(10.*np.log10(np.sum(10.**(dB[mask]/10.))))
		else:
			dB_octave.append(0.0)

	return freq_octave, np.array(dB_octave)


"""
/**************/
Acalibration
/**************/
Type : function
Process : A calibration
Input : freq, dB
Output : dB_A
"""
def Acalibration(freq, dB):

	Ra = (12194.**2)*(freq**4) / ((freq**2)+(20.6**2)) / np.sqrt((freq**2+107.7**2)*(freq**2+737.9**2)) / ((freq**2)+12194.**2)+1e-20
	A = 20.*np.log10(Ra)+2.

	return dB + A

"""
/**************/
OA
/**************/
Type : function
Process : calc OA
Input : dB
Output : OA
"""
def OA(dB):
	return 10.*np.log10(np.sum(10.**(P/10.)))

"""
/**************/
OA
/**************/
Type : function
Process : calc dB
Input : pressure
Output : dB
"""
def dB(pressure):
	return 20*np.log10(pressure/2e-5)