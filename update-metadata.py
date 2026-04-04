import tomllib
import pathlib
import datetime

srcpath = next(pathlib.Path("src").glob("*/__init__.py")).parent

with open('pyproject.toml', 'rb') as fid:
	toml = tomllib.load(fid)

with open(srcpath / '_metadata.py', 'w') as fid:

	version = toml['project']['version']
	fid.write(f'__version__     = "{version}"\n')

	description = toml['project']['description']
	fid.write(f'__description__ = "{description}"\n')

	author = toml['project']['authors'][0]['name']
	fid.write(f'__author__      = "{author}"\n')

	contact = toml['project']['authors'][0]['email']
	fid.write(f'__contact__     = "{contact}"\n')

	license = toml['project']['license']['text']
	fid.write(f'__license__     = "{license}"\n')

	fid.write(f'__copyright__   = "{author}"\n')

	today = datetime.date.today().isoformat()
	fid.write(f'__date__        = "{today}"\n')

with open(srcpath / '_metadata.py', 'r') as fid:
	print(fid.read())
