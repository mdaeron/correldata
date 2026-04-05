# correldata

[![PyPI Version](https://img.shields.io/pypi/v/correldata.svg)](https://pypi.python.org/pypi/correldata)
[![docs](https://shields.io/badge/docs-correldata-success?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxOTIiIGhlaWdodD0iMTkyIj48cGF0aCBkPSJNNjcuODA3IDIzLjI0NGMtMTguMDMyIDUuNTIzLTI0LjI3MyA5LjQxMS0zMy4xOTMgMTYuMTY5UzE5LjQwNyA1MS4xMjUgOS4zNCA2OS43NkMzLjI3IDgwLjk5Ni44MjcgOTkuMjcxIDEgMTExLjY3OWMuMTE0IDguMTY5IDIuMTg4IDEzLjU1OSAyLjE4OCAxMy41NTlTNDIuMzI2IDI0Ny4yNyAyMC40MzUgMjc4LjEwNWMtMTAuNDg4IDE0Ljc3MyA4NC40MjIgMjcuMDA3IDgwLjA0NyAxMC40NS0xMC4zOTctMzkuMzQ4LTI3LjU5NS05Ny40MDktMzEuOTA4LTExMy42NjctMy4wNDctMTEuNDg0IDguMTYtMjEuOTk1IDE2LjI5NS0yNy4zNDMgMy4zODMtMi4yMjQgOC4wNTEtMy43ODYgOC4wNTEtMy43ODZsNTEuOTI0LTE1LjU5NXMyMy44MDktOC41NzkgMzQuMDYyLTI2LjU2NyAxMi42ODktMzcuNTg0IDEyLjY4OS0zNy41ODQuNDk1LTE2LjU4OC01LjkzOS0yMy4xNC0yLjczMi01Ljg2Ni0xOC42NDYtMTIuNDk4LTQ5LjEyMi0xMC4wNTMtNDkuMTIyLTEwLjA1My0zMi4wNDktLjYwMi01MC4wODEgNC45MjF6IiBmaWxsPSIjMjBhZDZjIi8+PGcgdHJhbnNmb3JtPSJyb3RhdGUoMzQzLjk5MykiPjxlbGxpcHNlIGN4PSI1Mi4yNzYiIGN5PSI5NC4wODkiIGZpbGw9IiNmZmYiIHJ4PSIyNy44NTIiIHJ5PSIyNy44MjMiLz48ZWxsaXBzZSBjeD0iNTIuNTM4IiBjeT0iOTQuNjg5IiByeD0iMTguNjg4IiByeT0iMTguMzMxIiBmaWxsPSIjMTA1YTQ4Ii8+PGVsbGlwc2UgY3g9IjYzLjMzMyIgY3k9Ijg2LjMiIGZpbGw9IiNmZmYiIHJ4PSI3LjU5NiIgcnk9IjcuNTg4Ii8+PC9nPjxnIGZpbGw9IiMxMDVhNDgiPjxlbGxpcHNlIGN4PSI5NC4xNTQiIGN5PSIxMzUuNzM0IiByeD0iNC41MDUiIHJ5PSI1LjI1NyIgdHJhbnNmb3JtPSJyb3RhdGUoMzI1LjAxNSkiLz48ZWxsaXBzZSBjeD0iMTY3Ljg2IiBjeT0iNTYuOTUxIiByeD0iNC40ODQiIHJ5PSI1LjIzMSIgdHJhbnNmb3JtPSJyb3RhdGUoMzU4Ljc3OSkiLz48L2c+PHBhdGggZD0iTTE4OC45NjIgNzcuNDU4bC0zNC4yNDggMTAuMTcxYy02Ljg3OCAxLjk3My05Ljk3NCAzLjM4OC0xNC44ODggMy42NnMtMTAuODY4LTEuODY5LTEwLjg2OC0xLjg2OWwtMi4zNDIgOC42MzNjMi4xOTcuOTQ0IDkuOTY5IDIuMzkgMTQuNzU2IDIuNTkyczE1LjU4Mi0yLjQ2IDE1LjU4Mi0yLjQ2bDI3LjQ3MS03LjUzNGMxLjk1MS00LjcwOSAzLjQzMi05LjIzOCA0LjUzNy0xMy4xOTN6IiBmaWxsPSIjZTc4MzYxIi8+PC9zdmc+)](https://mdaeron.github.io/correldata)

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

## Documentation

[https://mdaeron.github.io/correldata](https://mdaeron.github.io/correldata)
