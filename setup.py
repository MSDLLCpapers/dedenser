import setuptools
from setuptools import setup , find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires=['numpy','pandas','openpyxl', 'mordred', 'rdkit', 'scikit-learn', 'alphashape',
					'point-cloud-utils', 'umap-learn', 'matplotlib',  'future']

setup(
	name = 'Dedenser',
	description = 'An application for downsampling chemical point clouds.',
    long_description = long_description,
	version = '0.10',
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
		"Development Status :: Beta",
		"Environment :: Console",
		"Intended Audience :: Chemists",
		"Intended Audience :: Science/Research",
		"License :: OSI Approved :: BSD License",
		"Operating System :: Microsoft :: Windows",
		"Operating System :: Unix",
		"Programming Language :: Python :: 3",
		"Topic :: Scientific/Engineering",
		"Topic :: Cheminformatics",
    	]
	)
