import pdoc
import pathlib

libname = next(pathlib.Path(".").glob("*/__init__.py")).parent.stem

pdoc.render.configure(search = False)

with open('../docs/index.html', 'w') as fid:
	fid.write(pdoc.pdoc(libname))
