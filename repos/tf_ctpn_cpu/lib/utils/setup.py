from Cython.Build import cythonize
import numpy as np
from distutils.core import setup


setup(ext_modules=cythonize(["bbox.pyx","cython_nms.pyx"],include_path=[np.get_include()]
    ))

