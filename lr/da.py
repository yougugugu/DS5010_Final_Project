import numpy as np
import random as random
import copy as copy

def read_data(path, na = True):
	"""
	Read file from system loaction.
	
	Param:
		param path: A string containing file path.
		param na: True of False to determine whether or not to remove NULL values from file (default to True).
	
	Returns:
		2-Dimensional list.
	"""
	data = []
	if isinstance(path, str):
		with open(path, "r") as dataset:
			input_data = [line.strip().split(",") for line in dataset]
			if na == True:
				for i in range(len(input_data)):
					if "" not in input_data[i]:
						data.append(input_data[i])
			else:
				data = input_data
				print("choose not to remove null values may cause further problems")
	else:
		raise TypeError("path type must be str")
	return data

def show(data):
	"""
	Display dataset.

	Param:
		param data: A dataset of 2-Dimension.

	Returns:
		None.
	"""
	irow = []
	for i in range(len(data)):
		irow.append("".join(["{:<20}".format(xi) for xi in data[i]]))
	print("\n".join(irow))

def select_byindex(data, indexlist):
	"""
	Slice Dataset by column index.

	Param:
		param data: A 2-Dimensional list.
		param indexlist: A list of column index.

	Returns:
		2-dimensional list after slicing.
	"""
	index = []
	col_len = len(data[0])
	if isinstance(indexlist, list):
		for ind in indexlist:
			if isinstance(ind, int) and ind < col_len and ind >= -col_len:
				index.append(ind)
			elif not isinstance(ind, int):
				raise TypeError("index type must be int")
			else:
				raise IndexError("index out of range")
	else:
		raise TypeError("indexlist type must be list")
	data_select = [[] for _ in range(len(data))]
	for i in range(len(data)):
		for j in index:
			data_select[i].append(data[i][j])
	return data_select

def select_byname(data, namelist):
	"""
	Slice Dataset by column name.

	Param:
		param data: A 2-Dimensional list.
		param indexlist: A list of column name.

	Returns:
		2-dimensional list after slicing.
	"""
	index = []
	colname = data[0]
	if isinstance(namelist, list):
		for name in namelist:
			if isinstance(name, str) and name in colname:
				for i in range(len(colname)):
					if name == colname[i]:
						index.append(i)
			elif not isinstance(name, str):
				raise TypeError("name type must be str")
			else:
				raise LookupError("column is not found in dataset")
	else:
		raise TypeError("namelist type must be list")
	return select_byindex(data, index)

def partition(data, percentage, seed):
	"""
	Partition dataset into training and validation sets.

	Param:
		param data: A 2-Dimensional list.
		param percentage: Percentage of training set.
		param seed: Seed to control random.

	Returns:
		List of training and validation sets.
	"""
	train = []
	valid = []
	if percentage <= 1 and percentage >= 0:
		random.seed(seed)
		newdata = data[1:]
		random.shuffle(newdata)
		n = int(len(newdata) * percentage)
		train = [data[0]] + newdata[:n]
		valid = [data[0]] + newdata[n:]
	else:
		raise Error("percentage must between 0 and 1")
	return [train, valid]

def to_float(data):
	"""
	Transform string to float except for first row.

	Param:
		param data: A 2-Dimensional list with elements can be transformed to float.

	Returns:
		2-dimensional list of float elements except for first row.
	"""
	newdata = [[] for _ in range(len(data) - 1)]
	for i in range(1, len(data)):
		newdata[i - 1] = list(map(float, data[i]))
	return [data[0]] + newdata

def to_matrix(data):
	"""
	Transform 2-Dimensional list into matrix except for first row. 

	Param:
		param data: A 2-Dimensional list only containing float elements. 

	Returns:
		2-dimensional matrix of float elements.
	"""
	number = to_float(data)[1:]
	return np.array(number, dtype = "float64")

def get_mean(data):
	"""
	Compute mean of each column.

	Param:
		param data: A 2-Dimensional list only containing float elements.

	Returns:
		A 1-Dimensional matrix of mean of each column.
	"""
	return np.mean(to_matrix(data), axis = 0)

def get_std(data):
	"""
	Compute standar deviation of each column.

	Param:
		param data: A 2-Dimensional list only containing float elements.

	Returns:
		A 1-Dimensional matrix of standard deviation of each column.
	"""
	return np.std(to_matrix(data), axis = 0, ddof = 1) 

def add_ahead(data, value):
	"""
	Add value at the beginning of list.

	Param:
		param data: A 2-Dimensional list.
		param value: Value needed to be added.

	Returns:
		A 2-Dimensional list after adding the value.
	"""
	newdata = copy.deepcopy(data)
	for i in range(len(data)):
		newdata[i][:0] = [value]
	return newdata