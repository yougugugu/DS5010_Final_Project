import unittest
from ..lr import lm
from ..diagnose import diag

x = [['a', 'b', 'c', 'd'], ['1', '2', '3', '4'], ['2', '3', '4', '5'], ['5', '3', '2', '2'], ['2', '4', '1', '2'], ['2', '3', '4', '1'], ['1', '2', '3', '4'], ['4', '5', '2', '1']]

model = lm(x, ['a', 'b', 'c'], ['d'])

diagInstance = diag(model)

class TestDiagnose(unittest.TestCase):

	def test_residuals(self):
		self.assertEqual(diagInstance.residuals().round(6).tolist(), [0.054983, 1.95189, -0.027491, -0.082474, -2.04811, 0.054983, 0.09622])

	def test_leverage(self):
		self.assertEqual(diagInstance.leverage().round(6).tolist(), [0.390034, 0.415808, 0.972509, 0.752577, 0.415808, 0.390034, 0.66323])

	def test_stdResiduals(self):
		self.assertEqual(diagInstance.std_residuals().round(6).tolist(), [0.043037, 1.561162, -0.101361, -0.101361, -1.638121, 0.043037, 0.101361])

	def test_cooksDistances(self):
		self.assertEqual(diagInstance.cooks_distance().round(6).tolist(), [0.000296, 0.433683, 0.090861, 0.007812, 0.477495, 0.000296, 0.005058])