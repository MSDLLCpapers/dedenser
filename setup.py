import os
import logging
import sys
import setuptools
from setuptools import setup , find_packages

long_description = 'Dedenser A Python tool for creating and downsampling chemical pointclouds.\n Software developed by MSD, https://www.msd.com.'



def package_to_path(package):
    """
    Convert a package (as found by setuptools.find_packages)
    e.g. "foo.bar" to usable path
    e.g. "foo/bar"
    No idea if this works on windows
    """
    return package.replace('.', '/')

def find_subdirectories(package):
    """
    Get the subdirectories within a package
    This will include resources (non-submodules) and submodules
    """
    try:
        subdirectories = next(os.walk(package_to_path(package)))[1]
    except StopIteration:
        subdirectories = []
    return subdirectories

def subdir_findall(dir, subdir):
    """
    Find all files in a subdirectory and return paths relative to dir
    This is similar to (and uses) setuptools.findall
    However, the paths returned are in the form needed for package_data
    """
    strip_n = len(dir.split('/'))
    path = '/'.join((dir, subdir))
    return ['/'.join(s.split('/')[strip_n:]) for s in setuptools.findall(path)]

def find_package_data(packages):
    """
    For a list of packages, find the package_data
    This function scans the subdirectories of a package and considers all
    non-submodule subdirectories as resources, including them in
    the package_data
    Returns a dictionary suitable for setup(package_data=<result>)
    """
    skip_tests = True
    package_data = {}
    for package in packages:
        package_data[package] = []
        for subdir in find_subdirectories(package):
            if '.'.join((package, subdir)) in packages:  # skip submodules
                logging.debug("skipping submodule %s/%s" % (package, subdir))
                continue
            if skip_tests and (subdir == 'tests'):  # skip tests
                logging.debug("skipping tests %s/%s" % (package, subdir))
                continue
            package_data[package] += subdir_findall(package_to_path(package), subdir)
    return package_data

# ----------- Override defaults here ----------------

packages = None
package_name = None
package_data = None

if packages is None:
    packages = setuptools.find_packages()

if len(packages) == 0:
    raise Exception("No valid packages found")

if package_name is None:
    package_name = packages[0]

if package_data is None:
    package_data = find_package_data(packages)


install_requires=['numpy>=1.24.4','pandas','openpyxl', 'mordred>=1.2.0', 'rdkit', 'scikit-learn>=1.3.0', 'alphashape==1.3.1', 'scipy>=1.10.1',
					'point-cloud-utils==0.30.4', 'umap-learn>=0.5.5', 'matplotlib',  'future']

setup(
	name = 'dedenser',
	description = 'An application for downsampling chemical point clouds.',
    long_description = long_description,
	version = '0.11b',
    url='https://github.com/MSDLLCpapers/dedenser',
	packages = find_packages(),
    install_requires = install_requires,
    author = 'Armen G. Beck',
    author_email = 'armen.beck@merck.com',
    python_requires='>=3.8.2', 
	test_suite="tests", # where to find tests
	entry_points = {
		'console_scripts': [
			'dedenser = dedenser.__main__:main' # got to module convert.__main__ and run the method called main
			]
		},
	classifiers=[
		"Development Status :: 4 - Beta",
		"Environment :: Console",
		"Intended Audience :: Science/Research",
		"License :: OSI Approved :: BSD License",
		"Operating System :: Microsoft :: Windows",
		"Operating System :: Unix",
		"Programming Language :: Python :: 3",
		"Topic :: Scientific/Engineering :: Chemistry"
    	]
	)
