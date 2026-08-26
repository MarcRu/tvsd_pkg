
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [ \
        Extension("proj_div_0", ["proj_div_0.pyx"]), \
        Extension(name="proj_div_0_low_cap_base", 
                  sources=["proj_div_0_low_cap_base.pyx"],
                  include_dirs=[np.get_include()],
                  extra_compile_args=["-O3"],  # Optimization, can include other flags like -g for debugging), \
                )\
        ]


setup(
    name="example",
    ext_modules=cythonize(extensions),
)
