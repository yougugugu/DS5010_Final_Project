import unittest
from ..lr import lm

x = [['a', 'b', 'c', 'd'], ['1', '2', '3', '4'], ['2', '3', '4', '5'], ['5', '3', '2', '2'], ['2', '4', '1', '2'], ['2', '3', '4', '1'], ['1', '2', '3', '4'], ['4', '5', '2', '1']]

model1 = lm(x[:5], [0,1,2], [3])
model2 = lm(x, ['a', 'b', 'c'], ['d'])
model3 = lm(x, [0, 1, 2], [3], False)

class TestLr(unittest.TestCase):

	def test_NumOfData(self):
		self.assertEqual(model1.n, 4)
		self.assertEqual(model2.n, 7)
		self.assertEqual(model3.n, 7)

	def test_NumOfCoeff(self):
		self.assertEqual(model1.p, 4)
		self.assertEqual(model2.p, 4)
		self.assertEqual(model3.p, 4)

	def test_coeff(self):
		self.assertEqual(model1.get_coeff().tolist(), [3.250000, -0.494872, 0.174963, 1.383208])
		self.assertEqual(model2.get_coeff().tolist(), [2.714286, -0.426022, -0.751270, 0.097504])
		self.assertEqual(model3.get_coeff().tolist(), [5.369416, -0.281787, -0.702749, 0.087629])

	def test_intercept(self):
		self.assertEqual(model1.get_intercept(), 3.250000)
		self.assertEqual(model2.get_intercept(), 2.714286)
		self.assertEqual(model3.get_intercept(), 5.369416)

	def test_fittedValue(self):
		self.assertEqual(model1.get_fitted().tolist(), [4.0, 5.0, 2.0, 2.0])
		self.assertEqual(model2.get_fitted().tolist(), [3.945017, 3.04811, 2.027491, 2.082474, 3.04811, 3.945017, 0.90378])
		self.assertEqual(model3.get_fitted().tolist(), [3.945017, 3.04811, 2.027491, 2.082474, 3.04811, 3.945017, 0.90378])

	def test_SSE(self):
		self.assertEqual(round(model1.SSE, 6), 0.000000)
		self.assertEqual(round(model2.SSE, 6), 8.027491)
		self.assertEqual(round(model3.SSE, 6), 8.027491)

	def test_SSR(self):
		self.assertEqual(round(model1.SSR, 6), 6.750000)
		self.assertEqual(round(model2.SSR, 6), 7.401080)
		self.assertEqual(round(model3.SSR, 6), 7.401080)

	def test_MSE(self):
		self.assertEqual(model1.MSE, "inf")
		self.assertEqual(round(model2.MSE, 6), 2.675830)
		self.assertEqual(round(model3.MSE, 6), 2.675830)

	def test_MSR(self):
		self.assertEqual(round(model1.MSR, 6), 2.250000)
		self.assertEqual(round(model2.MSR, 6), 2.467027)
		self.assertEqual(round(model3.MSR, 6), 2.467027)

	def test_Rsquared(self):
		self.assertEqual(round(model1.Rsquare, 6), 1.0)
		self.assertEqual(round(model2.Rsquare, 6), 0.479700)
		self.assertEqual(round(model3.Rsquare, 6), 0.479700)

	def test_ajustedRSquared(self):
		self.assertEqual(model1.Rsquare_a, "inf")
		self.assertEqual(round(model2.Rsquare_a, 6), -0.040601)
		self.assertEqual(round(model3.Rsquare_a, 6), -0.040601)

	def test_FScore(self):
		self.assertEqual(model1.F, "inf")
		self.assertEqual(round(model2.F, 6), 0.921967)
		self.assertEqual(round(model3.F, 6), 0.921967)