import correldata
import numpy

def test_basic():
	cdata = correldata.CorrelData(
		Name = ['foo', 'bar', 'baz'],
		S = [1., 2., 3.],
		X = [1., 2., 3.],
		SE_X = [0.1, 0.1, 0.1],
		Y = [0.1, 0.1, 0.1],
		covar_Y = numpy.array([[1.0, 0.1, 0.1], [0.1, 1.0, 0.1], [0.1, 0.1, 1.0]]),
		Z = [1., 2., 3.],
		SE_Z = [0.1, 0.1, 0.1],
		correl_Z = [[1.0, 0.1, 0.1], [0.1, 1.0, 0.1], [0.1, 0.1, 1.0]],
	)
	print(cdata)
	print(cdata.str())

	names = cdata['Name']
	S = cdata['S']*1.1
	X = cdata['X']*2.0
	Y = cdata['Y']*3.0
	Z = cdata['Z']*4.0

	cdata = correldata.CorrelData(Name = names, S = S, X = X, Y = Y, Z = Z)
	print(cdata.str())

if __name__ == '__main__':
	test_basic()
