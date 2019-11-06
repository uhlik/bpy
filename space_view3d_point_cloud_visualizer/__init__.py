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

bl_info = {"name": "Point Cloud Visualizer",
           "description": "Display, edit, filter, render, convert, generate and export colored point cloud PLY files.",
           "author": "Jakub Uhlik",
           "version": (0, 9, 30),
           "blender": (2, 80, 0),
           "location": "View3D > Sidebar > Point Cloud Visualizer",
           "warning": "",
           "wiki_url": "https://github.com/uhlik/bpy",
           "tracker_url": "https://github.com/uhlik/bpy/issues",
           "category": "3D View", }

if('bpy' in locals()):
    import importlib
    importlib.reload(debug)
    importlib.reload(io_ply)
    importlib.reload(machine)
    importlib.reload(props)
    importlib.reload(ui)
    importlib.reload(ops)
    importlib.reload(ops_filter)
    importlib.reload(convert)
    importlib.reload(sample)
    importlib.reload(instavis)
else:
    from . import debug
    from . import io_ply
    from . import machine
    from . import props
    from . import ui
    from . import ops
    from . import ops_filter
    from . import convert
    from . import sample
    from . import instavis


import bpy
from bpy.props import PointerProperty, BoolProperty, StringProperty, FloatProperty, IntProperty, FloatVectorProperty, EnumProperty, CollectionProperty
from bpy.types import AddonPreferences


# NOTE $ pycodestyle --ignore=W293,E501,E741,E402 --exclude='io_mesh_fast_obj/blender' .

# TODO: split operators to logic / algorithm, or by type display, filter, etc..
# TODO: instavis cleanup


class PCV_preferences(AddonPreferences):
    bl_idname = __name__
    
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


classes = (
    instavis.PCVIV2_properties,
    instavis.PCVIV2_generator_properties,
    instavis.PCVIV2_UL_materials,
    
    props.PCV_properties,
    
    PCV_preferences,
    
    ui.PCV_PT_panel,
    ui.PCV_PT_clip,
    ui.PCV_PT_edit,
    ui.PCV_PT_filter,
    ui.PCV_PT_filter_simplify,
    ui.PCV_PT_filter_project,
    ui.PCV_PT_filter_boolean,
    ui.PCV_PT_filter_remove_color,
    ui.PCV_PT_filter_merge,
    ui.PCV_PT_filter_join,
    ui.PCV_PT_filter_color_adjustment,
    ui.PCV_PT_render,
    ui.PCV_PT_convert,
    ui.PCV_PT_generate,
    ui.PCV_PT_export,
    ui.PCV_PT_sequence,
    
    ops.PCV_OT_load,
    ops.PCV_OT_draw,
    ops.PCV_OT_erase,
    ops.PCV_OT_render,
    ops.PCV_OT_render_animation,
    ops.PCV_OT_convert,
    ops.PCV_OT_reload,
    ops.PCV_OT_export,
    ops.PCV_OT_edit_start,
    ops.PCV_OT_edit_update,
    ops.PCV_OT_edit_end,
    ops.PCV_OT_edit_cancel,
    ops.PCV_OT_sequence_preload,
    ops.PCV_OT_sequence_clear,
    ops.PCV_OT_generate_point_cloud,
    ops.PCV_OT_reset_runtime,
    
    ops_filter.PCV_OT_filter_simplify,
    ops_filter.PCV_OT_filter_remove_color,
    ops_filter.PCV_OT_filter_remove_color_delete_selected,
    ops_filter.PCV_OT_filter_remove_color_deselect,
    ops_filter.PCV_OT_filter_project,
    ops_filter.PCV_OT_filter_merge,
    ops_filter.PCV_OT_filter_boolean_intersect,
    ops_filter.PCV_OT_filter_boolean_exclude,
    ops_filter.PCV_OT_color_adjustment_shader_reset,
    ops_filter.PCV_OT_color_adjustment_shader_apply,
    ops_filter.PCV_OT_filter_join,
    
    ui.PCV_PT_development,
    ops.PCV_OT_generate_volume_point_cloud,
    
    instavis.PCVIV2_PT_panel,
    instavis.PCVIV2_PT_generator,
    instavis.PCVIV2_PT_display,
    instavis.PCVIV2_PT_debug,
    instavis.PCVIV2_OT_init,
    instavis.PCVIV2_OT_deinit,
    instavis.PCVIV2_OT_reset,
    instavis.PCVIV2_OT_reset_all,
    instavis.PCVIV2_OT_update,
    instavis.PCVIV2_OT_update_all,
    
    instavis.PCVIV2_OT_dev_transform_normals,
    ops.PCV_OT_clip_planes_from_bbox,
    ops.PCV_OT_clip_planes_reset,
    ops.PCV_OT_clip_planes_from_camera_view,
    
    ui.PCV_PT_debug,
    
    ops.PCV_OT_init,
    ops.PCV_OT_deinit,
    ops.PCV_OT_gc,
    ops.PCV_OT_seq_init,
    ops.PCV_OT_seq_deinit,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    
    ui.update_panel_bl_category(None, bpy.context)


def unregister():
    machine.PCVSequence.deinit()
    machine.PCVManager.deinit()
    # instavis.PCVIV2Manager.deinit()
    
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    """
    > Well, that doesn't explain... why you've come all the way out here, all the way out here to hell.
    > I, uh, have a job out in the town of Machine.
    > Machine? That's the end of the line.
    Jim Jarmusch, Dead Man (1995)
    """
    register()
