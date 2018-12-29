
# python3 setup.py build_ext --inplace

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

name = "export"
sources = ["export.pyx", ]
libraries = []
include_dirs = ["blender/source/blender/makesdna",
                "blender/source/blender/blenlib", ]
extra_compile_args = []
extra_link_args = []
extensions = [Extension(name=name,
                        sources=sources,
                        language="c++",
                        include_dirs=include_dirs,
                        libraries=libraries,
                        extra_compile_args=extra_compile_args,
                        extra_link_args=extra_link_args, ), ]

setup(name=name, ext_modules=cythonize(extensions), )
