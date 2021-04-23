import numpy as np
import copy as copy
from . import da
from scipy import stats

class lm:
	"""
	Multiple linear regression based on ordinary least square. Allow to predict for new dataset.
	"""
	def __init__(self, data, explanatory, response, standardize = True):
		"""
		Inilize lm model with given data, explanatory value column, response value cloumn and standardization. 
		"""
		self.data = data
		if len(response) == 1:
			if all(isinstance(xi, int) for xi in explanatory):
				self.explan = da.select_byindex(self.data, explanatory)
				self.resp = da.to_float(da.select_byindex(self.data, response))
			elif all(isinstance(xi, str) for xi in explanatory):
				self.explan = da.select_byname(self.data, explanatory)
				self.resp = da.to_float(da.select_byname(self.data, response))
			else:
				raise TypeError("slice type can only be int or str")
		else:
			raise IndexError("can only have one response value")
		
		self.mean = da.get_mean(self.explan)
		self.std = da.get_std(self.explan)
		self.stand = standardize
		
		if isinstance(self.stand, bool):
			if self.stand == True:
				self.explan = [self.explan[0]] + ((da.to_matrix(self.explan) - self.mean)/self.std).round(6).tolist()
			else:
				self.explan = self.explan
		else:
			raise TypeError("standardize type must be bool")
		
		self.explan_1 = da.add_ahead(self.explan, 1)
		self.explan_1[0][0] = "intercept"
		self.x = da.to_matrix(self.explan_1)
		self.y = da.to_matrix(self.resp)
		self.n = len(self.x)
		self.coeff = self.compute_coefficient()
		self.p = len(self.coeff)
		self.intercept = self.coeff[0][0]
		self.fitted = self.compute_fitted()
		self.SSE = self.compute_SSE()
		self.SST = self.compute_SST()
		self.SSR = self.SST - self.SSE

		if self.n != self.p:		
			self.MSE = self.SSE / (self.n - self.p)
		else:
			self.MSE = "inf"

		self.MSR = self.SSR / (self.p - 1)

		if self.SST > 0:
			self.Rsquare = self.SSR / self.SST
		else:
			self.Rsquare = "inf"

		if self.SST > 0 and self.MSE != "inf" and self.n > 1:
			self.Rsquare_a = 1 - self.MSE / (self.SST / (self.n - 1))
		else:
			self.Rsquare_a = "inf"

		if self.MSE != "inf" and self.MSE != 0:
			self.F = self.MSR / self.MSE
		else:
			self.F = "inf"

		self.predicted = None
		
	def compute_coefficient(self):
		"""
		Compute coefficients of model.

		Param:
			None.

		Returns:
			A matrix of coefficient
		"""
		x_trans = self.x.transpose()
		coeff = np.matmul(np.matmul(np.linalg.inv(np.matmul(x_trans, self.x)), x_trans), self.y)
		return coeff.round(8)

	def compute_SSE(self):
		"""
		Compute sum square of errors of model.

		Param:
			None.

		Returns:
			Sum square of error.
		"""
		residual = self.y - self.fitted
		return np.matmul(residual.transpose(), residual)[0][0]
	
	def compute_SST(self):
		"""
		Compute sum square of total of model.

		Param:
			None.

		Returns:
			Sum square of total.
		"""
		y_mean = self.y.mean(axis = 0)
		total = self.y - y_mean
		return np.matmul(total.transpose(), total)[0][0]

	def compute_fitted(self):
		"""
		Compute fitted value.

		Param:
			None.

		Returns:
			Matrix of fitted values.
		"""
		return np.matmul(self.x, self.coeff)

	def get_coeff(self):
		"""
		Get coefficient with 6 decimal palces.

		Param:
			None.

		Returns:
			Matrix of ceofficient with 6 decimal palces.
		"""
		return self.coeff.round(6).transpose()[0]

	def get_intercept(self):
		"""
		Get intercep with 6 decimal palces.

		Param:
			None.

		Returns:
			Intercept 6 decimal palces.
		"""
		return self.intercept.round(6)

	def get_fitted(self):
		"""
		Get fitted values with 6 decimal palces.

		Param:
			None.

		Returns:
			Matrix of fitted values with 6 decimal palces.
		"""
		return self.fitted.round(6).transpose()[0]

	def show_coeff(self):
		"""
		Dispaly coefficents with coefficient names.

		Param:
			None.

		Returns:
			None.
		"""
		da.show([self.explan_1[0]] + [self.get_coeff().tolist()])

	def show_fitted(self):
		"""
		Dispaly fitted value with response value names.

		Param:
			None.

		Returns:
			None.

		"""
		da.show([self.resp[0]] + [self.get_fitted().tolist()])
	
	def summary(self):
		"""
		Dispaly summary information of coefficents, Std. Error, t-test, F-test, etc.

		Param:
			None.

		Returns:
			None.
		"""
		if self.MSE != "inf" and self.SST != 0:
			V = np.diagonal(np.linalg.inv(np.matmul(self.x.transpose(), self.x)))
			sbj = (V * self.MSE) ** (1/2)
			t = self.coeff.transpose()[0] / sbj
			pvalue = stats.t.sf(abs(t), self.n - self.p) * 2
			coeff_sum = np.concatenate((self.explan_1[0], self.get_coeff(), sbj.round(6), t.round(6), pvalue.round(6))).reshape(5, self.p).transpose().tolist()
		elif self.SST == 0:
			coeff_sum = np.concatenate((self.explan_1[0], self.get_coeff(), [0] * self.p, ["inf"] * self.p, ["inf"] * self.p)).reshape(5, self.p).transpose().tolist()
		else:
			coeff_sum = np.concatenate((self.explan_1[0], self.get_coeff(), ["inf"] * self.p, ["inf"] * self.p, ["inf"] * self.p)).reshape(5, self.p).transpose().tolist()

		da.show([["coeff name", "Estimated coeff", "Std. Error", "t-value", "P-value"]] + coeff_sum)
		print()
		print("Residual standard error:   {}   on   {}   degrees of freedom".format(round(self.MSE ** (1/2), 6) if self.MSE != "inf" else "inf", (self.n - self.p)))
		print("Multiple R-squared:   {},      Adjusted R-squared:   {}".format(round(self.Rsquare, 6) if self.Rsquare != "inf" else "inf", round(self.Rsquare_a, 6) if self.Rsquare_a != "inf" else "inf"))
		print("F-statistic:   {}   on   {}   and    {}    degrees of freedom,      p-value:    {}".format(round(self.F, 6) if self.F != "inf" else "inf", (self.p - 1), (self.n - self.p), round(stats.f.sf(self.F, (self.p - 1), (self.n - self.p)), 6) if self.F != "inf" else "inf"))

	def predict(self, data):
		"""
		Preicting new data with fitted model.

		Param:
			param data: 2-dimensional list with same of training data.

		Returns:
			Predicted values.
		"""
		if data[0] == self.explan[0]:
			if self.stand == True:
				x = ((da.to_matrix(data) - self.mean)/self.std)
				newx = np.array(da.add_ahead(x.tolist(), 1))
				self.predicted = np.matmul(newx, self.coeff)
			else:
				self.predicted = np.matmul(da.to_matrix(da.add_ahead(data, 1)), self.coeff)
		else:
			raise ValueError("columns of input must match fiited model")
		return self.predicted.tolist()

	def show_predicted(self):
		"""
		Dispaly predicted values.

		Param:
			None.

		Returns:
			None.
		"""
		if self.predicted is not None:
			da.show([self.resp[0]] + self.predicted.round(6).transpose().tolist())
		else:
			print("No data have been predicted yet")