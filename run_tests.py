
import unittest 

from lr.tests import test_da
from lr.tests import test_lr
from lr.tests import test_diagnose

loader = unittest.TestLoader()
suite = unittest.TestSuite()

suite.addTest(loader.loadTestsFromModule(test_da))
suite.addTest(loader.loadTestsFromModule(test_lr))
suite.addTest(loader.loadTestsFromModule(test_diagnose))

runner = unittest.TextTestRunner(verbosity=2)
result = runner.run(suite)