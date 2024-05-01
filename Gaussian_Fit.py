import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import tstd
import math
import csv, sys

NUMBINS = 120
modals = ['gauss', 'twomodal', 'threemodal', 'fourmodal', 'fivemodal', 'sixmodal', 'sevenmodal', 'eightmodal', 'ninemodal', 'tenmodal']

#Takes in:
# 1) Input angles in .tsv format
# 2) Maximum number of Gaussians
# 3) Output text file to be written

# Read in data
with open(sys.argv[1]) as f:
	angle_df = pd.read_table(f, dtype=float)
	

# Definition of 1-10 peak gaussians
def gauss(x, mu, sigma, A):
    return abs(A*np.exp(-(x-mu)**2/2/sigma**2))

def twomodal(x, mu1, sigma1, A1, mu2, sigma2, A2):
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)

def threemodal(x, mu1, sigma1, A1, mu2, sigma2, A2, mu3, sigma3, A3):
	return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)+gauss(x,mu3,sigma3,A3)

def fourmodal(x, mu1, sigma1, A1, mu2, sigma2, A2, mu3, sigma3, A3, mu4, sigma4, A4):
	return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)+gauss(x,mu3,sigma3,A3)+gauss(x,mu4,sigma4,A4)

def fivemodal(x, mu1, sigma1, A1, mu2, sigma2, A2, mu3, sigma3, A3, mu4, sigma4, A4, mu5, sigma5, A5, ):
    return gauss(x, mu1, sigma1, A1)+gauss(x, mu2, sigma2, A2)+gauss(x, mu3, sigma3, A3)+gauss(x, mu4, sigma4, A4)+gauss(x, mu5, sigma5, A5)

def sixmodal(x, mu1, sigma1, A1, mu2, sigma2, A2, mu3, sigma3, A3, mu4, sigma4, A4, mu5, sigma5, A5, mu6, sigma6, A6, ):
    return gauss(x, mu1, sigma1, A1)+gauss(x, mu2, sigma2, A2)+gauss(x, mu3, sigma3, A3)+gauss(x, mu4, sigma4, A4)+gauss(x, mu5, sigma5, A5)+gauss(x, mu6, sigma6, A6)

def sevenmodal(x, mu1, sigma1, A1, mu2, sigma2, A2, mu3, sigma3, A3, mu4, sigma4, A4, mu5, sigma5, A5, mu6, sigma6, A6, mu7, sigma7, A7, ):
    return gauss(x, mu1, sigma1, A1)+gauss(x, mu2, sigma2, A2)+gauss(x, mu3, sigma3, A3)+gauss(x, mu4, sigma4, A4)+gauss(x, mu5, sigma5, A5)+gauss(x, mu6, sigma6, A6)+gauss(x, mu7, sigma7, A7)

def eightmodal(x, mu1, sigma1, A1, mu2, sigma2, A2, mu3, sigma3, A3, mu4, sigma4, A4, mu5, sigma5, A5, mu6, sigma6, A6, mu7, sigma7, A7, mu8, sigma8, A8, ):
    return gauss(x, mu1, sigma1, A1)+gauss(x, mu2, sigma2, A2)+gauss(x, mu3, sigma3, A3)+gauss(x, mu4, sigma4, A4)+gauss(x, mu5, sigma5, A5)+gauss(x, mu6, sigma6, A6)+gauss(x, mu7, sigma7, A7)+gauss(x, mu8, sigma8, A8)

def ninemodal(x, mu1, sigma1, A1, mu2, sigma2, A2, mu3, sigma3, A3, mu4, sigma4, A4, mu5, sigma5, A5, mu6, sigma6, A6, mu7, sigma7, A7, mu8, sigma8, A8, mu9, sigma9, A9, ):
    return gauss(x, mu1, sigma1, A1)+gauss(x, mu2, sigma2, A2)+gauss(x, mu3, sigma3, A3)+gauss(x, mu4, sigma4, A4)+gauss(x, mu5, sigma5, A5)+gauss(x, mu6, sigma6, A6)+gauss(x, mu7, sigma7, A7)+gauss(x, mu8, sigma8, A8)+gauss(x, mu9, sigma9, A9)

def tenmodal(x, mu1, sigma1, A1, mu2, sigma2, A2, mu3, sigma3, A3, mu4, sigma4, A4, mu5, sigma5, A5, mu6, sigma6, A6, mu7, sigma7, A7, mu8, sigma8, A8, mu9, sigma9, A9, mu10, sigma10, A10, ):
    return gauss(x, mu1, sigma1, A1)+gauss(x, mu2, sigma2, A2)+gauss(x, mu3, sigma3, A3)+gauss(x, mu4, sigma4, A4)+gauss(x, mu5, sigma5, A5)+gauss(x, mu6, sigma6, A6)+gauss(x, mu7, sigma7, A7)+gauss(x, mu8, sigma8, A8)+gauss(x, mu9, sigma9, A9)+gauss(x, mu10, sigma10, A10)

def generateExpected(reps, A, mu, sigma):
	expected = []
	for i in range(0, reps):
		expected.append(A)
		expected.append(mu)
		expected.append(sigma)
	return expected

outputDf = pd.DataFrame({"params":[0], "sigma":[0]})
figure, axis = plt.subplots(1, len(angle_df.columns), figsize=(len(angle_df.columns)*10,6))

for column in range(0,len(angle_df.columns)):

	data = angle_df.iloc[:,column].tolist()

	print("Generating Gaussians for %s dataset" % angle_df.columns.values[column])

	if angle_df.iloc[:,column].isnull().values.any():
		print("ERROR: NaN VALUES FOUND IN DATASET ")
		axis[column].set_title(angle_df.columns.values[column])
		pass
	else:
		# Generate histogram from data and add to plot
		try:
			bins = np.linspace(min(data), max(data), NUMBINS)
		except:
			print("Odd entries found in dataset %s" % angle_df.columns.values[column])
			pass
		y,x,_ = axis[column].hist(data, NUMBINS)
		lastSuccessful = 0
		bestResidual = 0
		lastResidual = 0
		residualSquares = 0

		''' expected = (x[np.where(y == max(y))[0][0]], tstd(x), max(y)) '''

		# For every n-modal function I have created here
		# Try to fit it to the histogram
		# If it doesn't find a fit, output that
		# Otherwise:
		#	 Create a curve fit and assign params and covariance
		#	 Update the last successful tally and last residual square values
		#	 Calculate new sume of residual squares
		#	 Check if that new one is better
		#	 Calculate a sigma value from the covariance
		#	 Print that this find was successful

		expA = max(y)
		expMu = x[np.where(y == max(y))[0][0]]
		expSig = tstd(x)

		for n in range(0, int(sys.argv[2])):
			expected = generateExpected(n+1, expA, expMu, expSig)
			try:
				curve_fit(locals()[modals[n]], bins, y)
			except:
				print("Could not find %s fit for histogram" % modals[n])
				pass
			else:
				params, cov = curve_fit(locals()[modals[n]], bins, y)
				lastSuccessful = n
				lastResidual = residualSquares
				residualSquares = np.sum(np.square(locals()[modals[n]](bins, *params) - y))
				if residualSquares < lastResidual:
					bestResidual = n
				sigma = np.sqrt(np.diag(cov))
				print("Found %s Gaussian fit with square residual sum of %s" % (modals[n], residualSquares))

		# Present curve with best residual

		paramsB, covB = curve_fit(locals()[modals[bestResidual]], bins, y)
		sigmaB = np.sqrt(np.diag(covB))
		print("Modal with best square residual sum was %s:" % modals[bestResidual])
		# print(pd.DataFrame(data={'params': paramsB, 'sigma': sigmaB}, index=locals()[modals[bestResidual]].__code__.co_varnames[1:]))
		outputDf = pd.concat([outputDf, pd.DataFrame(data={'params': [angle_df.columns.values[column]], 'sigma': bestResidual})])
		outputDf = pd.concat([outputDf, pd.DataFrame(data={'params': paramsB, 'sigma': sigmaB}, index=locals()[modals[bestResidual]].__code__.co_varnames[1:])])

		axis[column].plot(bins, locals()[modals[bestResidual]](bins, *paramsB), color='orange', lw=1, label=modals[bestResidual])
		for i in range(0, bestResidual+1):
			axis[column].plot(bins, gauss(bins, *paramsB[i*3:i*3+3]), color='orange', lw=1, ls='--', label=i) 

		axis[column].legend()
		axis[column].set_title(angle_df.columns.values[column])

figure.savefig("GaussianPlots.png")

file_out = open(sys.argv[3], 'w')
file_out.write(outputDf.to_string())
file_out.close()