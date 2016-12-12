from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy

extensions = [Extension("convolve1", ["convolve1.pyx"], include_dirs = [numpy.get_include()], extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp'])]

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules=cythonize(extensions)
)
