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

import bpy
from bpy.types import AddonPreferences
from bpy.props import BoolProperty, StringProperty, FloatVectorProperty, EnumProperty

from . import ui


class PCV_preferences(AddonPreferences):
    # bl_idname = __name__
    bl_idname = __package__
    
    default_vertex_color: FloatVectorProperty(name="Default", default=(0.65, 0.65, 0.65, ), min=0, max=1, subtype='COLOR', size=3, description="Default color to be used upon loading PLY to cache when vertex colors are missing", )
    normal_color: FloatVectorProperty(name="Normal", default=((35 / 255) ** 2.2, (97 / 255) ** 2.2, (221 / 255) ** 2.2, ), min=0, max=1, subtype='COLOR', size=3, description="Display color for vertex normals lines", )
    selection_color: FloatVectorProperty(name="Selection", description="Display color for selection", default=(1.0, 0.0, 0.0, 0.5), min=0, max=1, subtype='COLOR', size=4, )
    convert_16bit_colors: BoolProperty(name="Convert 16bit Colors", description="Convert 16bit colors to 8bit, applied when Red channel has 'uint16' dtype", default=True, )
    gamma_correct_16bit_colors: BoolProperty(name="Gamma Correct 16bit Colors", description="When 16bit colors are encountered apply gamma as 'c ** (1 / 2.2)'", default=False, )
    shuffle_points: BoolProperty(name="Shuffle Points", description="Shuffle points upon loading, display percentage is more useable if points are shuffled", default=True, )
    category: EnumProperty(name="Tab Name", items=[('POINT_CLOUD_VISUALIZER', "Point Cloud Visualizer", ""),
                                                   ('PCV', "PCV", ""), ], default='POINT_CLOUD_VISUALIZER', description="To have PCV in its own separate tab, choose one", update=ui.update_panel_bl_category, )
    category_custom: BoolProperty(name="Custom Tab Name", default=False, description="Check if you want to have PCV in custom named tab or in existing tab", update=ui.update_panel_bl_category, )
    category_custom_name: StringProperty(name="Name", default="View", description="Custom PCV tab name, if you choose one from already existing tabs it will append to that tab", update=ui.update_panel_bl_category, )
    
    def draw(self, context):
        l = self.layout
        r = l.row()
        r.prop(self, "default_vertex_color")
        r.prop(self, "normal_color")
        r.prop(self, "selection_color")
        r = l.row()
        r.prop(self, "shuffle_points")
        r.prop(self, "convert_16bit_colors")
        c = r.column()
        c.prop(self, "gamma_correct_16bit_colors")
        if(not self.convert_16bit_colors):
            c.active = False
        
        f = 0.5
        r = l.row()
        s = r.split(factor=f)
        c = s.column()
        c.prop(self, "category")
        if(self.category_custom):
            c.enabled = False
        s = s.split(factor=1.0)
        r = s.row()
        r.prop(self, "category_custom")
        c = r.column()
        c.prop(self, "category_custom_name")
        if(not self.category_custom):
            c.enabled = False
    
    # @classmethod
    # def prefs(cls, context=None, ):
    #     if(context is None):
    #         context = bpy.context
    #     return context.preferences.addons[__name__].preferences


classes = (PCV_preferences, )
