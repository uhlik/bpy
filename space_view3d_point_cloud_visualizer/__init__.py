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
# (c) 2019 Jakub Uhlik

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
    importlib.reload(instavis)
    importlib.reload(instavis3)
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
    from . import instavis
    from . import instavis3


import bpy


# NOTE $ pycodestyle --ignore=W293,E501,E741,E402 --exclude='io_mesh_fast_obj/blender' .


classes = (
    # instavis.PCVIV2_properties, instavis.PCVIV2_generator_properties, instavis.PCVIV2_UL_materials,
    
    props.PCV_properties,
    prefs.PCV_preferences,
    
    ui.PCV_PT_panel, ui.PCV_PT_clip, ui.PCV_PT_edit, ui.PCV_PT_filter, ui.PCV_PT_filter_simplify, ui.PCV_PT_filter_project, ui.PCV_PT_filter_boolean,
    ui.PCV_PT_filter_remove_color, ui.PCV_PT_filter_merge, ui.PCV_PT_filter_join, ui.PCV_PT_filter_color_adjustment, ui.PCV_PT_render, ui.PCV_PT_convert,
    ui.PCV_PT_generate, ui.PCV_PT_export, ui.PCV_PT_sequence,
    
    ops.PCV_OT_load, ops.PCV_OT_draw, ops.PCV_OT_erase,
    ops.PCV_OT_reload, ops.PCV_OT_export,
    ops.PCV_OT_sequence_preload, ops.PCV_OT_sequence_clear,
    ops.PCV_OT_reset_runtime,
    ops.PCV_OT_clip_planes_from_bbox, ops.PCV_OT_clip_planes_reset, ops.PCV_OT_clip_planes_from_camera_view,
    
    convert.PCV_OT_convert,
    render.PCV_OT_render, render.PCV_OT_render_animation,
    edit.PCV_OT_edit_start, edit.PCV_OT_edit_update, edit.PCV_OT_edit_end, edit.PCV_OT_edit_cancel,
    generate.PCV_OT_generate_point_cloud,
    
    filters.PCV_OT_filter_simplify, filters.PCV_OT_filter_remove_color, filters.PCV_OT_filter_remove_color_delete_selected,
    filters.PCV_OT_filter_remove_color_deselect, filters.PCV_OT_filter_project, filters.PCV_OT_filter_merge,
    filters.PCV_OT_filter_boolean_intersect, filters.PCV_OT_filter_boolean_exclude, filters.PCV_OT_color_adjustment_shader_reset,
    filters.PCV_OT_color_adjustment_shader_apply, filters.PCV_OT_filter_join,
    
    # instavis.PCVIV2_PT_panel, instavis.PCVIV2_PT_generator, instavis.PCVIV2_PT_display, instavis.PCVIV2_PT_debug, instavis.PCVIV2_OT_init,
    # instavis.PCVIV2_OT_deinit, instavis.PCVIV2_OT_reset, instavis.PCVIV2_OT_reset_all, instavis.PCVIV2_OT_update, instavis.PCVIV2_OT_update_all,
    
    ui.PCV_PT_development,
    generate.PCV_OT_generate_volume_point_cloud,
    # instavis.PCVIV2_OT_dev_transform_normals,
    
    ui.PCV_PT_debug,
    ops.PCV_OT_init, ops.PCV_OT_deinit, ops.PCV_OT_gc, ops.PCV_OT_seq_init, ops.PCV_OT_seq_deinit,
    
    # instavis3 props
    instavis3.PCVIV3_psys_properties, instavis3.PCVIV3_object_properties, instavis3.PCVIV3_material_properties,
    # instavis3 ops
    instavis3.PCVIV3_OT_init, instavis3.PCVIV3_OT_deinit,
    instavis3.PCVIV3_OT_register,
    # instavis3.PCVIV3_OT_update,
    # instavis3 ui
    instavis3.PCVIV3_PT_panel,
    
    # instavis3 tests
    instavis3.PCVIV3_OT_test_generator_speed, instavis3.PCVIV3_OT_test_generator_profile, instavis3.PCVIV3_OT_test_generator_draw,
    instavis3.PCVIV3_PT_tests,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    
    ui.update_panel_bl_category(None, bpy.context)


def unregister():
    machine.PCVSequence.deinit()
    machine.PCVManager.deinit()
    # instavis.PCVIV2Manager.deinit()
    instavis3.PCVIV3Manager.deinit()
    
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
