import unittest
from .. import da

x = [['a', 'b', 'c', 'd'], ['1', '2', '3', '4'], ['2', '3', '4', '5'], ['5', '3', '2', '2'], ['2', '4', '1', '2']]
select_x = [['b', 'c'], ['2', '3'], ['3', '4'], ['3', '2'], ['4', '1']]
float_x = [['a', 'b', 'c', 'd'], [1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0], [5.0, 3.0, 2.0, 2.0], [2.0, 4.0, 1.0, 2.0]]
one_x = [['abc', 'a', 'b', 'c', 'd'], ['abc', '1', '2', '3', '4'], ['abc', '2', '3', '4', '5'], ['abc', '5', '3', '2', '2'], ['abc', '2', '4', '1', '2']]

class TestDa(unittest.TestCase):

	def test_readData(self):
		readData = da.read_data("test_data.txt")
		self.assertEqual(readData, x)

	def test_selectIndex(self):
		indexData = da.select_byindex(x, [1, 2])
		self.assertEqual(indexData, select_x)

	def test_selectName(self):
		nameData = da.select_byname(x, ['b', 'c'])
		self.assertEqual(nameData, select_x)

	def test_partition(self):
		test_train, test_valid = da.partition(x, 0.5, 111)
		self.assertEqual(len(test_train[1:]), 2)
		self.assertEqual(len(test_valid[1:]), 2)

	def test_toFloat(self):
		floatData = da.to_float(x)
		self.assertEqual(floatData, float_x)

	def test_addAhead(self):
		addData = da.add_ahead(x, "abc")
		self.assertEqual(addData, one_x)