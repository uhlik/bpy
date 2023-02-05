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

# Copyright (c) 2023 Jakub Uhlik

bl_info = {"name": "Color Management Presets",
           "description": "Presets support for Render > Color Management panel",
           "author": "Jakub Uhlik",
           "version": (0, 0, 1),
           "blender": (2, 80, 0),
           "location": "Properties > Render > Color Management",
           "warning": "",
           "wiki_url": "",
           "tracker_url": "",
           "category": "System", }


import os
import textwrap

import bpy
from bpy.types import Menu, Operator
from bpy.props import StringProperty
from bl_operators.presets import AddPresetBase


class CMP_MT_presets(Menu):
    """Color Management Presets List"""
    bl_label = "Color Management Presets"
    preset_subdir = "color_management"
    preset_operator = "script.execute_preset"
    draw = Menu.draw_preset


class CMP_OT_add(AddPresetBase, Operator):
    """Add / Remove Preset"""
    bl_idname = 'color_management_presets.add'
    bl_label = 'Add / Remove Preset'
    preset_menu = 'CMP_MT_presets'
    preset_subdir = 'color_management'
    preset_defines = ['s = bpy.context.scene', ]
    preset_values = ['s.display_settings.display_device',
                     's.view_settings.view_transform',
                     's.view_settings.exposure',
                     's.view_settings.gamma',
                     's.view_settings.look',
                     's.view_settings.use_curve_mapping',
                     's.sequencer_colorspace_settings.name', ]


def CMP_UI_draw(self, context, ):
    l = self.layout
    r = l.row(align=True)
    r.menu('CMP_MT_presets', text=bpy.types.CMP_MT_presets.bl_label, )
    r.operator('color_management_presets.add', text="", icon='ADD', )
    r.operator('color_management_presets.add', text="", icon='REMOVE', ).remove_active = True
    l.separator()


def default_presets():
    d = {'filmic_high_contrast': ('sRGB', 'Filmic', 0.0, 1.0, 'Filmic - High Contrast', False, 'sRGB', ),
         'filmic_medium_high_contrast': ('sRGB', 'Filmic', 0.0, 1.0, 'Filmic - Medium High Contrast', False, 'sRGB', ),
         'standard': ('sRGB', 'Standard', 0.0, 1.0, 'None', False, 'sRGB', ), }
    return d


def setup():
    subdir = bpy.types.CMP_MT_presets.preset_subdir
    path = os.path.join(bpy.utils.user_resource('SCRIPTS'), 'presets', subdir, )
    preset_paths = bpy.utils.preset_paths(subdir)
    if(path not in preset_paths):
        if(not os.path.exists(path)):
            os.makedirs(path)
    
    preset_paths = bpy.utils.preset_paths(subdir)
    found = []
    for p in preset_paths:
        files = [f for f in os.listdir(p) if os.path.isfile(os.path.join(p, f))]
        for f in files:
            if(f.endswith(".py")):
                found.append(os.path.splitext(f)[0])
    
    if(len(found) == 0):
        d = default_presets()
        t = textwrap.dedent('''\
            import bpy
            s = bpy.context.scene
            
            s.display_settings.display_device = '{}'
            s.view_settings.view_transform = '{}'
            s.view_settings.exposure = {}
            s.view_settings.gamma = {}
            s.view_settings.look = '{}'
            s.view_settings.use_curve_mapping = {}
            s.sequencer_colorspace_settings.name = '{}'\
        ''')
        for n, p in d.items():
            s = t.format(*p)
            with open(os.path.join(path, "{}.py".format(n)), mode='w', encoding='utf-8') as f:
                f.write(s)


classes = (
    CMP_MT_presets,
    CMP_OT_add,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.RENDER_PT_color_management.prepend(CMP_UI_draw)
    setup()


def unregister():
    bpy.types.RENDER_PT_color_management.remove(CMP_UI_draw)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
