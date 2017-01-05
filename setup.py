from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy

extensions = [Extension("*", ["src/*.pyx"])]

setup(
	cmdclass = {'build_ext':build_ext},
	ext_modules = cythonize(extensions),
	include_dirs =[numpy.get_include()]
	)
 
 