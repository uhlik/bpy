
# python3 setup.py build_ext --inplace

# /Applications/Blender/blender-2.83.4.app/Contents/Resources/2.83/python/bin/pip3 install cython
# https://blender.stackexchange.com/a/107381
# /Applications/Blender/blender-2.83.4.app/Contents/Resources/2.83/python/bin/python3.7m setup.py build_ext --inplace


'''
cd /Applications/Blender/blender-2.83.6.app/Contents/Resources/2.83/python/bin/
./python3.7m -m ensurepip
./python3.7m -m pip install -U pip
./pip3 install cython

cd ~/Library/Application Support/Blender/2.83/scripts/addons/io_mesh_fast_obj
git clone git://git.blender.org/blender.git
cd blender
git checkout v2.83.6

/Applications/Blender/blender-2.83.6.app/Contents/Resources/2.83/python/bin/python3.7m setup.py build_ext --inplace
'''

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

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
