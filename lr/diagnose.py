from .lr import lm
import numpy as np
import matplotlib.pyplot as plt

class diag:
	"""
	Diagnostic tools for linear regression including residuals, leverage, cook's distance and plots analysis. 
	"""

	def __init__(self, model):
		"""
		Initailize diagnostics tool for linear regression.
		"""
		if isinstance(model, lm):
			self.model = model
		else:
			raise TypeError("model type must be lm")


	def residuals(self):
		"""
		Compute Residuals for lm model.

		Param:
			None.

		Return
			Matrix of residuals.
		"""
		return (self.model.y - self.model.fitted).transpose()[0]

	def leverage(self):
		"""
		Compute leverage for lm model.

		Param:
			None.

		Return
			Matrix of leverage.
		"""
		x_trans = self.model.x.transpose()
		hmatrix = np.matmul(np.matmul(self.model.x, np.linalg.inv(np.matmul(x_trans, self.model.x))), x_trans)
		return np.diagonal(hmatrix)

	def std_residuals(self):
		"""
		Compute standardized Residuals for lm model.

		Param:
			None.

		Return
			Matrix of standardized residuals.
		"""
		return self.residuals() / ((self.model.MSE ** (1/2)) * (1 - self.leverage()) ** (1/2))
 
	def cooks_distance(self):
		"""
		Compute cook's distance for lm model.

		Param:
			None.

		Return
			Matrix of cook's distance.
		"""
		return ((self.std_residuals() ** 2) * (self.leverage() / (1 - self.leverage()))) / self.model.p
	
	def plot_fitVsRes(self):
		"""
		Plot fitted values versus residuals.

		Param:
			None.

		Return
			None.
		"""
		plt.plot(self.model.get_fitted(), self.residuals(), "o")
		plt.axhline(y = 0, ls = ":")
		plt.xlabel("Fitted values")
		plt.ylabel("Residuals")
		plt.title("Residuals vs Fitted")
		plt.show()

	def plot_fitVsStaRes(self):
		"""
		Plot sqrt of standardized residuals versus residuals

		Param:
			None.

		Return
			None.
		"""
		plt.plot(self.model.get_fitted(), (abs(self.std_residuals()) ** (1/2)).round(6), "o")
		plt.xlabel("Fitted values")
		plt.ylabel("Sqrt of standardized residuals")
		plt.title("Scale-Location")
		plt.show()

	def plot_cooksDis(self):
		"""
		Plot Cook's distance of each data

		Param:
			None.

		Return
			None.
		"""
		plt.stem(np.array([i+1 for i in range(len(self.model.y))]), self.cooks_distance(), "o")
		plt.xlabel("Obs. number")
		plt.ylabel("Cook's distance")
		plt.title("Cook's distance")
		plt.show()

	def plot_LeverVsStdRes(self):
		"""
		Plot leverage versus sqrt of standardized residuals

		Param:
			None.

		Return
			None.
		"""
		plt.plot(self.leverage(), self.std_residuals(), "o")
		plt.axhline(y = 0, ls = ":")
		plt.xlabel("Leverage")
		plt.ylabel("Standardized residuals")
		plt.title("Residuals vs Leverage")
		plt.show()

	def plot_cooksVsLever(self):
		"""
		Plot Cook's distance versus leverage

		Param:
			None.

		Return
			None.
		"""
		plt.plot(self.cooks_distance(), self.leverage(), "o")
		plt.axhline(y = 0, ls = ":")
		plt.xlabel("Leverage")
		plt.ylabel("Cook's distance")
		plt.title("Cook's dist vs Leverage")
		plt.show()

	def plot(self, type = 1):
		"""
		Compute Residuals for lm model.

		Param:
			param type: types of plots.

		Return
			None.
		"""
		if type == 1:
			self.plot_fitVsRes()

		elif type == 2:
			self.plot_fitVsStaRes()

		elif type == 3:
			self.plot_cooksDis()

		elif type == 4:
			self.plot_LeverVsStdRes()

		elif type == 5:
			self.plot_cooksVsLever()
			
		else:
			print("Type must be in 1 to 5.")