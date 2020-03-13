# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software Foundation,
#  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# ##### END GPL LICENSE BLOCK #####

# part of "Point Cloud Visualizer" blender addon
# author: Jakub Uhlik
# (c) 2019, 2020 Jakub Uhlik

bl_info = {"name": "PCV Instance Visualizer",
           "description": "",
           "author": "Jakub Uhlik",
           "version": (0, 0, 6),
           "blender": (2, 80, 0),
           "location": "View3D > Sidebar > PCVIV",
           "warning": "",
           "wiki_url": "https://github.com/uhlik/bpy",
           "tracker_url": "https://github.com/uhlik/bpy/issues",
           "category": "3D View", }

if('bpy' in locals()):
    import importlib
    importlib.reload(debug)
    importlib.reload(mechanist)
    importlib.reload(overseer)
    importlib.reload(props)
    importlib.reload(ops)
    importlib.reload(ui)
else:
    from . import debug
    from . import mechanist
    from . import overseer
    from . import props
    from . import ops
    from . import ui

import bpy

classes = props.classes + ops.classes + ui.classes
if(debug.debug_mode()):
    classes += props.classes_debug + ops.classes_debug + ui.classes_debug


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    mechanist.PCVIVMechanist.deinit()
    
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
