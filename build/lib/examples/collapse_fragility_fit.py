import numpy as np 
from pyFragility.MLEClass import MaximumLikelihoodMethod



def collape_fragility_fit():
    hazardLevel = np.array([0.178, 0.274, 0.444, 0.56, 0.652, 0.79, 0.982, 1.246, 1.564, 2.014, 2.417, 3.021, 3.625, 4.028, 4.431, 5.035])
    numGM = 45 * np.ones(len(hazardLevel))
    returnPeriod = [15, 25, 50, 75, 100, 150, 250, 500, 1000, 2500, 2700, 3000, 3300, 3500, 3700, 4000]
    rate = [1/i for i in returnPeriod]
    numCount = np.array([0, 0, 0, 0, 0, 4, 13, 23, 38, 41, 44, 45, 45, 45, 45, 45])
    
    mle_fit = MaximumLikelihoodMethod(hazardLevel, numCount, numGM, rate)
    print (f'Theta is: {mle_fit.theta[0]}, Beta is {mle_fit.theta[1]}')


if __name__ == '__main__':
    collape_fragility_fit()