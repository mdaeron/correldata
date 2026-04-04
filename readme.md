# correldata

[![PyPI Version](https://img.shields.io/pypi/v/correldata.svg)](https://pypi.python.org/pypi/correldata)

Dataframe-like tables of data with correlated uncertainties

These data are stored in a dictionary, whose values are numpy arrays
with elements which may be strings, floats, or floats with associated uncertainties
as defined in the [uncertainties](https://pypi.org/project/uncertainties) library.

When reading data from a csv file, column names are interpreted in the following way:

Column names are interpreted in the following way:
* In most cases, each columns is converted to a dict value, with the corresponding
dict key being the column's label.
* Columns whose label starts with `SE` are interpreted as specifying the standard
error for the latest preceding data column.
* Columns whose label starts with `correl` are interpreted as specifying the
correlation matrix for the latest preceding data column. In that case, column labels
are ignored for the rest of the columns belonging to this matrix.
* Columns whose label starts with `covar` are interpreted as specifying the
covariance matrix for the latest preceding data column. In that case, column labels
are ignored for the rest of the columns belonging to this matrix.
* `SE`, `correl`, and `covar` may be specified for any arbitrary variable other than
the latest preceding data column, by adding an underscore followed by the variable's
label (ex: `SE_foo`, `correl_bar`, `covar_baz`).
* `correl`, and `covar` may also be specified for any pair of variable, by adding an
underscore followed by the two variable labels, joined by a second underscore
(ex: `correl_foo_bar`, `covar_X_Y`). The elements of the first and second variables
correspond, respectively, to the lines and columns of this matrix.

## Example

```py
import correldata

data  = '''
Sample, Tacid,  D47,   SE,         correl,,,  D48, covar,,,          correl_D47_D48
   FOO,   90., .245, .005,      1, 0.5, 0.5, .145,  4e-4, 1e-4, 1e-4, 0.5,   0,   0
   BAR,   90., .246, .005,    0.5,   1, 0.5, .146,  1e-4, 4e-4, 1e-4,   0, 0.5,   0
   BAZ,   90., .247, .005,    0.5, 0.5,   1, .147,  1e-4, 1e-4, 4e-4,   0,   0, 0.5
'''[1:-1]

cdata = correldata.read_str(data)
print(cdata)
```
yields:
```
{
  'Sample': array(['FOO', 'BAR', 'BAZ'], dtype='<U3'),
  'Tacid':  array([90., 90., 90.]),
  'D47':    uarray([0.245+/-0.005, 0.246+/-0.005, 0.247+/-0.005], dtype=object),
  'D48':    uarray([0.145+/-0.02, 0.146+/-0.02, 0.147+/-0.02], dtype=object)
}
```
and 
```py
print(cdata.str(correl_format = '.2f'))
```
yields:
``` 
Sample, Tacid,   D47, SE_D47, correl_D47,     ,     ,   D48, SE_D48, correl_D48,     ,     , correl_D47_D48,      ,      
   FOO,    90, 0.245,  0.005,       1.00, 0.50, 0.50, 0.145,   0.02,       1.00, 0.25, 0.25,           0.50, -0.00, -0.00
   BAR,    90, 0.246,  0.005,       0.50, 1.00, 0.50, 0.146,   0.02,       0.25, 1.00, 0.25,           0.00,  0.50, -0.00
   BAZ,    90, 0.247,  0.005,       0.50, 0.50, 1.00, 0.147,   0.02,       0.25, 0.25, 1.00,          -0.00,  0.00,  0.50
```

## Documentation / API

[https://mdaeron.github.io/correldata](https://mdaeron.github.io/correldata)