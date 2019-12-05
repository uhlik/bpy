
# python3 setup.py build_ext --inplace

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

# import Cython.Compiler.Options
# Cython.Compiler.Options.annotate = True

name = "export_obj"
include_dirs = ["blender/source/blender/makesdna",
                "blender/source/blender/blenlib", ]
extensions = [Extension(name=name,
                        sources=["export_obj.pyx", ],
                        language="c++",
                        include_dirs=include_dirs,
                        libraries=[],
                        extra_compile_args=[],
                        extra_link_args=[], ), ]

setup(name=name, ext_modules=cythonize(extensions), )
