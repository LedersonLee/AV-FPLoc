"""Regression for actual boolean localization outcomes in ray diagnostics."""
import unittest
import numpy as np
from analyze_rays import correlation

class CorrelationTests(unittest.TestCase):
 def test_boolean_outcomes(self):
  self.assertAlmostEqual(correlation([1,2,3,4],[False,False,True,True]),2/np.sqrt(5))
 def test_constant_boolean(self):
  self.assertIsNone(correlation([1,2,3],[True,True,True]))
 def test_nan_filter(self):
  self.assertAlmostEqual(correlation([1,2,3,float('nan')],[False,True,True,False]),np.sqrt(3)/2)
if __name__=='__main__':unittest.main()
