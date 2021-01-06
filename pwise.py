#!/usr/bin/python3
import unittest

import numpy as np
import scipy
from scipy.stats import entropy

def gauss(sample_phi,phi,sample_psi,psi,sigma):
    return np.exp((-1.0*((sample_phi-phi)**2 + (sample_psi-psi)**2))/(2.0*(sigma**2)))

def ParzenWindow(w, h, d=1):
    """""""""""""""""""""""
    Average of the gaussian window functions centered on each data point of w for marginal or joint density
    estimation at a particular point x
    Parameters
    ----------
        w = vector of distances from a point x to the other points
        h = window width
        d = length or the variable dimension (1 by default if unit variable for marginal density and 2 if bivariate
            variable for joint density.)
    Returns
    -------
        pw = Estimation of the parzen window function for the density estimation f(w)
    """""""""""""""""""""""
    if d>1:
        pw = np.sum(np.prod(GaussianWindow(list(w),h),1))
    else: pw = np.sum(GaussianWindow(w,h))#np.sum(np.exp(-w**2/(2*phi))/den)
    return pw

def GaussianWindow(w,h):
    """
    Gaussian kernel function with a variance of 2h^2
    :param w: vector of distances from a point x to the other points
    :param h: Window width
    :return: VAlue of the gaussian function
    """
    phi = 2*h**2
    den = (2*np.pi*phi)**(1/2)
    return den*np.exp(-np.power(w,2)/(2*phi))


def getPairWiseArray(dims):
    _unity = 1.0
    _diag = _unity

    data = np.zeros(dims, dtype=float)
    for i in range(min(dims)):
        data[i, i] = _diag

    return data


def setPairWiseArrayPair(data, row, col, p, q):
    """Set both P(i,j) and P(j,i) to p """
    assert (p >= 0.0 and p <= 1.0)
    data[row, col] = p
    data[col, row] = p if q is None else q


def getSelfSimilarity(data):
    rows, cols = data.shape
    ss = np.zeros((1,cols), dtype=float)
    count_entropy = np.log2(rows)
    # stats.entropy will normalize if row does not add up to 1
    # we normalize ourseleves
    for row in range(rows):
        pk = data[row,:]
        pk = 1.0 * pk / np.sum(pk, axis=0, keepdims=True)
        ss[0,row] = scipy.stats.entropy(pk, None, base=2) / count_entropy
    return ss



def getPairWiseArrayStats(data):
    """Return row means """

    stats = {}
    stats['means'] = np.mean(data, axis=0)
    stats['ranks'] = np.argsort(stats['means'])
    return stats




class TestMethods(unittest.TestCase):

    def test_array_create(self):
        dim = 3
        foo = getPairWiseArray((dim,dim))
        for row in range(dim):
            for col in range(dim):
                if row == col:
                    self.assertEqual(foo[row,col], 1.0)
                    continue
                self.assertEqual(foo[row,col], 0.0)

    def test_array_roll(self):
        dim = 3
        foo = getPairWiseArray((dim, dim))
        ## fill up with random
        for row in range(dim):
            for col in range(dim):
                if row == col:
                    continue
                setPairWiseArrayPair(foo, row, col, np.random.random_sample())
        # make a copy
        fcopy = np.copy(foo)
        # fill in for the new entry in the first tow
        for col in range(dim):
            if col == 0: continue
            setPairWiseArrayPair(fcopy, 0, col, np.random.random_sample())
        # roll by the number of unchanged that is dim - 1
        np.roll(fcopy, ((dim - 1) * dim, 0))

        # compare
        cmp = foo == fcopy
        for row in range(dim):
            for col in range(dim):
                if row == col or (row > 0 and col > 0):
                    self.assertEqual(cmp[row, col], True)
                    continue
                self.assertEqual(cmp[row, col], False)


    def test_scipy_entropy(self):

        data = [0.05, 0.5, 0.9, 0.3, 0.2]
        sum = np.sum(data)
        normed = np.divide(data,sum)
        print((sum, normed))
        entropy = 0
        for ee in normed:
            vv = ee * np.log2(ee) * -1.0
            entropy += vv

        entropy = entropy
        sci_entropy =  scipy.stats.entropy(normed, None, base=2)
        self.assertAlmostEqual(entropy,sci_entropy)

    def test_get_ss(self):
        data = [[1. ,        0.99822814, 0.99088965],
        [0.99822814 ,1. ,        0.99486904],
        [0.99088965 ,0.99486904 ,1.]]
        ss = getSelfSimilarity(np.asarray(data))
        print(('ss',ss))
        self.assertAlmostEqual(ss[0,0], 0.99999286, 5)
        self.assertAlmostEqual(ss[0,1], 0.99999793, 5)
        self.assertAlmostEqual(ss[0,2], 0.99999361, 5)



if __name__ == '__main__':
    unittest.main()
    
