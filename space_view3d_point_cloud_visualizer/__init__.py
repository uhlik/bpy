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

bl_info = {"name": "Point Cloud Visualizer",
           "description": "Display, edit, filter, render, convert, generate and export colored point cloud PLY files.",
           "author": "Jakub Uhlik",
           "version": (0, 9, 30),
           "blender": (2, 81, 0),
           "location": "View3D > Sidebar > Point Cloud Visualizer",
           "warning": "",
           "wiki_url": "https://github.com/uhlik/bpy",
           "tracker_url": "https://github.com/uhlik/bpy/issues",
           "category": "3D View", }

if('bpy' in locals()):
    import importlib
    importlib.reload(debug)
    importlib.reload(prefs)
    importlib.reload(io_ply)
    importlib.reload(machine)
    importlib.reload(props)
    importlib.reload(ui)
    importlib.reload(ops)
    importlib.reload(filters)
    importlib.reload(edit)
    importlib.reload(render)
    importlib.reload(convert)
    importlib.reload(generate)
else:
    from . import debug
    from . import prefs
    from . import io_ply
    from . import machine
    from . import props
    from . import ui
    from . import ops
    from . import filters
    from . import edit
    from . import render
    from . import convert
    from . import generate

import bpy

classes = props.classes + prefs.classes + ui.classes + ops.classes + convert.classes + render.classes + edit.classes + generate.classes + filters.classes
if(debug.debug_mode()):
    classes += generate.classes_dev + ui.classes_dev
    classes += ops.classes_debug + ui.classes_debug


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    
    ui.update_panel_bl_category(None, bpy.context)


def unregister():
    machine.PCVSequence.deinit()
    machine.PCVManager.deinit()
    
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


# NOTE $ pycodestyle --ignore=W293,E501,E741,E402 --exclude='io_mesh_fast_obj/blender' .

if __name__ == "__main__":
    """
    > Well, that doesn't explain... why you've come all the way out here, all the way out here to hell.
    > I, uh, have a job out in the town of Machine.
    > Machine? That's the end of the line.
    Jim Jarmusch, Dead Man (1995)
    """
    register()
