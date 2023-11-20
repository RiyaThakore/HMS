## Referred from https://github.com/RichieHakim/ROICaT/blob/main/setup.py.

## setup.py file for my_pipeline
from pathlib import Path

from distutils.core import setup
import copy

dir_parent = Path(__file__).parent

def read_requirements():
    with open(str(dir_parent / "requirements.txt"), "r") as req:
        content = req.read()  ## read the file
        requirements = content.split("\n") ## make a list of requirements split by (\n) which is the new line character

    ## Filter out any empty strings from the list
    requirements = [req for req in requirements if req]
    ## Filter out any lines starting with #
    requirements = [req for req in requirements if not req.startswith("#")]
    ## Remove any commas, quotation marks, and spaces from each requirement
    requirements = [req.replace(",", "").replace("\"", "").replace("\'", "").strip() for req in requirements]

    return requirements

deps_all = read_requirements()

## Dependencies: latest versions of requirements
### remove everything starting and after the first =,>,<,! sign
deps_names = [req.split('=')[0].split('>')[0].split('<')[0].split('!')[0] for req in deps_all]
deps_all_latest = copy.deepcopy(deps_names)

## Get README.md
with open(str(dir_parent / "README.md"), "r") as f:
    readme = f.read()

## Get version number
with open(str(dir_parent / "my_pipeline" / "__init__.py"), "r") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().replace("\"", "").replace("\'", "")
            break

setup(
    name='my_pipeline',
    version=version,
    author='Riya Thakore',
    description='',
    long_description=readme,
    long_description_content_type="text/markdown",
    url='',

    packages=[
        'my_pipeline',
    ],

    install_requires=[],
    extras_require={
        'all': deps_all,
        'all_latest': deps_all_latest,
    },
)
