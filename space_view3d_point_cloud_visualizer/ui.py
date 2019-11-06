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

import os
import numpy as np

import bpy
from bpy.types import Panel


from .debug import log, debug_mode
from .machine import PCVManager, PCVSequence, preferences
from . import ops
from . import ops_filter
from . import instavis


def update_panel_bl_category(self, context, ):
    _main_panel = PCV_PT_panel
    # NOTE: maybe generate those from 'classes' tuple, or, just don't forget to append new panel also here..
    _sub_panels = (
        PCV_PT_clip,
        PCV_PT_edit, PCV_PT_filter, PCV_PT_filter_simplify, PCV_PT_filter_project, PCV_PT_filter_boolean, PCV_PT_filter_remove_color,
        PCV_PT_filter_merge, PCV_PT_filter_join, PCV_PT_filter_color_adjustment, PCV_PT_render, PCV_PT_convert, PCV_PT_generate, PCV_PT_export, PCV_PT_sequence,
        PCV_PT_development,
        
        instavis.PCVIV2_PT_panel, instavis.PCVIV2_PT_generator, instavis.PCVIV2_PT_display, instavis.PCVIV2_PT_debug,
        
        PCV_PT_debug,
    )
    try:
        p = _main_panel
        bpy.utils.unregister_class(p)
        for sp in _sub_panels:
            bpy.utils.unregister_class(sp)
        prefs = preferences()
        c = prefs.category_custom
        n = ''
        if(c):
            n = prefs.category_custom_name
        else:
            v = prefs.category
            ei = prefs.bl_rna.properties['category'].enum_items
            for e in ei:
                if(e.identifier == v):
                    n = e.name
        if(n == ''):
            raise Exception('Name is empty string')
        p.bl_category = n
        bpy.utils.register_class(p)
        for sp in _sub_panels:
            bpy.utils.register_class(sp)
    except Exception as e:
        log('PCV: setting tab name failed ({})'.format(str(e)))


class PCV_PT_panel(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    # bl_category = "View"
    bl_category = "Point Cloud Visualizer"
    bl_label = "Point Cloud Visualizer"
    # bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        # o = context.active_object
        # if(o):
        #     return True
        # return False
        return True
    
    def draw(self, context):
        o = context.active_object
        if(not o):
            self.layout.label(text='Select an object..', icon='ERROR', )
            return
        
        pcv = context.object.point_cloud_visualizer
        l = self.layout
        sub = l.column()
        
        if(pcv.instance_visualizer_active_hidden_value):
            r = sub.row()
            r.alert = True
            r.prop(pcv, 'instance_visualizer_active', toggle=True, icon='ERROR', )
            sub.separator()
        
        # edit mode, main pcv object panel
        if(pcv.edit_initialized):
            sub.label(text='PCV Edit in progress..', icon='ERROR', )
            sub.separator()
            sub.operator('point_cloud_visualizer.edit_cancel')
            return
        
        # edit mode, helper object panel
        if(pcv.edit_is_edit_mesh):
            sub.label(text='PCV Edit helper mesh', icon='INFO', )
            sub.separator()
            c = sub.column()
            c.label(text='• Transform, delete and duplicate vertices.')
            c.label(text='• Update button will refresh point cloud.')
            c.label(text='• End button will refresh point cloud and delete helper mesh.')
            c.label(text='• All other functions are disabled until finished.')
            c.scale_y = 0.66
            
            sub.separator()
            
            sub.prop(pcv, 'edit_overlay_alpha')
            sub.prop(pcv, 'edit_overlay_size')
            
            sub.separator()
            
            r = sub.row(align=True)
            r.operator('point_cloud_visualizer.edit_update')
            r.operator('point_cloud_visualizer.edit_end')
            
            if(context.mode != 'EDIT_MESH'):
                sub.label(text="Must be in Edit Mode", icon='ERROR', )
            
            sub.enabled = ops.PCV_OT_edit_update.poll(context)
            
            return
        
        # ----------->>> file selector
        def prop_name(cls, prop, colon=False, ):
            for p in cls.bl_rna.properties:
                if(p.identifier == prop):
                    if(colon):
                        return "{}:".format(p.name)
                    return p.name
            return ''
        
        # f = 0.275
        f = 0.33
        
        r = sub.row(align=True, )
        s = r.split(factor=f)
        s.label(text=prop_name(pcv, 'filepath', True, ))
        s = s.split(factor=1.0)
        r = s.row(align=True, )
        c = r.column(align=True)
        c.prop(pcv, 'filepath', text='', )
        c.enabled = False
        r.operator('point_cloud_visualizer.load_ply_to_cache', icon='FILEBROWSER', text='', )
        
        r.operator('point_cloud_visualizer.reload', icon='FILE_REFRESH', text='', )
        
        # <<<----------- file selector
        
        # ----------->>> info block
        def human_readable_number(num, suffix='', ):
            # https://stackoverflow.com/questions/1094841/reusable-library-to-get-human-readable-version-of-file-size
            f = 1000.0
            for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z', ]:
                if(abs(num) < f):
                    return "{:3.1f}{}{}".format(num, unit, suffix)
                num /= f
            return "{:.1f}{}{}".format(num, 'Y', suffix)
        
        l0c0 = "Selected: "
        l0c1 = "{}".format("n/a")
        l1c0 = "Displayed: "
        # l1c1 = "{} of {}".format("0.0", "n/a")
        l1c1 = "{}".format("n/a")
        
        if(pcv.filepath != ""):
            _, t = os.path.split(pcv.filepath)
            l0c1 = "{}".format(t)
            if(pcv.uuid in PCVManager.cache):
                l0c0 = "Loaded: "
                l0c1 = "{}".format(t)
                cache = PCVManager.cache[pcv.uuid]
                
                n = human_readable_number(cache['display_length'])
                # don't use it when less or equal to 999
                if(cache['display_length'] < 1000):
                    n = str(cache['display_length'])
                
                if(not cache['draw']):
                    # n = "0.0"
                    n = "0"
                nn = human_readable_number(cache['stats'])
                if(nn.endswith('.0')):
                    nn = nn[:-2]
                l1c1 = "{} of {}".format(n, nn)
        
        f = 0.33
        c = sub.column()
        c.scale_y = 0.66
        r = c.row()
        s = r.split(factor=f)
        s.label(text=l0c0)
        s = s.split(factor=1.0)
        s.label(text=l0c1)
        r = c.row()
        s = r.split(factor=f)
        s.label(text=l1c0)
        s = s.split(factor=1.0)
        s.label(text=l1c1)
        
        sub.separator()
        # <<<----------- info block
        
        e = not (pcv.filepath == "")
        r = sub.row(align=True)
        r.operator('point_cloud_visualizer.draw')
        r.operator('point_cloud_visualizer.erase')
        r.scale_y = 1.5
        r.enabled = e
        r = sub.row()
        r.prop(pcv, 'display_percent')
        r.enabled = e
        r = sub.row()
        r.prop(pcv, 'point_size')
        r.enabled = e
        
        r = sub.row()
        r.prop(pcv, 'global_alpha')
        r.enabled = e
        
        r = sub.row(align=True)
        r.prop(pcv, 'vertex_normals', toggle=True, icon_only=True, icon='SNAP_NORMAL', )
        r.prop(pcv, 'vertex_normals_size')
        r.enabled = e
        if(not pcv.has_normals):
            r.enabled = False
        
        pcv = context.object.point_cloud_visualizer
        ok = False
        zero_length = False
        for k, v in PCVManager.cache.items():
            if(v['uuid'] == pcv.uuid):
                if(v['ready']):
                    if(v['draw']):
                        ok = True
                    if(len(v['vertices']) == 0):
                        zero_length = True
        
        if(ok):
            if(not pcv.has_normals):
                if(not zero_length):
                    sub.label(text="Missing vertex normals.", icon='ERROR', )
        
        c = sub.column()
        r = c.row(align=True)
        r.prop(pcv, 'illumination', toggle=True, )
        r.prop(pcv, 'illumination_edit', toggle=True, icon_only=True, icon='TOOL_SETTINGS', )
        # r.prop(pcv, 'illumination_edit', toggle=True, icon_only=True, icon='SETTINGS', )
        if(ok):
            if(not pcv.has_normals):
                # c.label(text="Missing vertex normals.", icon='ERROR', )
                c.enabled = False
        else:
            c.enabled = False
        if(pcv.illumination_edit):
            cc = c.column()
            cc.prop(pcv, 'light_direction', text="", )
            ccc = cc.column(align=True)
            ccc.prop(pcv, 'light_intensity')
            ccc.prop(pcv, 'shadow_intensity')
            if(not pcv.has_normals):
                cc.enabled = e
            
            sub.separator()
        
        # # other shaders
        # e = ok
        # c = sub.column()
        # r = c.row(align=True)
        # r.prop(pcv, 'dev_depth_enabled', toggle=True, )
        # r.prop(pcv, 'dev_depth_edit', toggle=True, icon_only=True, icon='TOOL_SETTINGS', )
        # if(pcv.dev_depth_edit):
        #     cc = c.column(align=True)
        #     cc.prop(pcv, 'dev_depth_brightness')
        #     cc.prop(pcv, 'dev_depth_contrast')
        #     c.prop(pcv, 'dev_depth_false_colors')
        #     r = c.row(align=True)
        #     r.prop(pcv, 'dev_depth_color_a', text="", )
        #     r.prop(pcv, 'dev_depth_color_b', text="", )
        #     r.enabled = pcv.dev_depth_false_colors
        #
        #     sub.separator()
        # c.enabled = e
        #
        # c = sub.column()
        # c.prop(pcv, 'dev_normal_colors_enabled', toggle=True, )
        # c.enabled = e
        #
        # c = sub.column()
        # c.prop(pcv, 'dev_position_colors_enabled', toggle=True, )
        # c.enabled = e
        
        # other shaders
        c = sub.column()
        c.enabled = ok
        r = c.row(align=True)
        r.prop(pcv, 'dev_depth_enabled', toggle=True, )
        # r.prop(pcv, 'dev_normal_colors_enabled', toggle=True, )
        cc = r.column(align=True)
        cc.prop(pcv, 'dev_normal_colors_enabled', toggle=True, )
        if(ok):
            if(not pcv.has_normals):
                cc.enabled = False
        r.prop(pcv, 'dev_position_colors_enabled', toggle=True, )
        
        # r = c.row(align=True)
        # r.prop(pcv, 'debug_shader', expand=True, )
        
        if(pcv.dev_depth_enabled):
            cc = c.column(align=True)
            cc.prop(pcv, 'dev_depth_brightness')
            cc.prop(pcv, 'dev_depth_contrast')
            c.prop(pcv, 'dev_depth_false_colors')
            r = c.row(align=True)
            r.prop(pcv, 'dev_depth_color_a', text="", )
            r.prop(pcv, 'dev_depth_color_b', text="", )
            r.enabled = pcv.dev_depth_false_colors
            # sub.separator()
        if(pcv.dev_normal_colors_enabled):
            pass
        if(pcv.dev_position_colors_enabled):
            pass
        
        # r = c.row(align=True)
        # r.prop(pcv, 'dev_bbox_enabled', toggle=True, icon='SHADING_BBOX', text="", icon_only=True, )
        # if(pcv.dev_bbox_enabled):
        #     r.prop(pcv, 'dev_bbox_color', text="", )
        #     r = c.row(align=True)
        #     r.prop(pcv, 'dev_bbox_size')
        #     r.prop(pcv, 'dev_bbox_alpha')
        
        # c.prop(pcv, 'dev_bbox_enabled', toggle=True, )
        # if(pcv.dev_bbox_enabled):
        #     r = c.row()
        #     r.prop(pcv, 'dev_bbox_color', text="", )
        #     c.prop(pcv, 'dev_bbox_size')
        #     c.prop(pcv, 'dev_bbox_alpha')


class PCV_PT_render(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "View"
    bl_label = "Render"
    bl_parent_id = "PCV_PT_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        o = context.active_object
        if(o is None):
            return False
        
        if(o):
            pcv = o.point_cloud_visualizer
            if(pcv.edit_is_edit_mesh):
                return False
            if(pcv.edit_initialized):
                return False
        return True
    
    def draw(self, context):
        pcv = context.object.point_cloud_visualizer
        l = self.layout
        sub = l.column()
        
        c = sub.column()
        c.prop(pcv, 'render_display_percent')
        c.prop(pcv, 'render_point_size')
        c.prop(pcv, 'render_supersampling')
        r = c.row()
        r.prop(pcv, 'render_smoothstep')
        ok = False
        # if(not pcv.illumination and not pcv.override_default_shader):
        if(not pcv.override_default_shader):
            ok = True
        r.enabled = ok
        
        c = sub.column()
        
        f = 0.33
        r = sub.row(align=True, )
        s = r.split(factor=f)
        s.label(text='Output:')
        s = s.split(factor=1.0)
        r = s.row(align=True, )
        c = r.column(align=True)
        c.prop(pcv, 'render_path', text='', )
        
        r = sub.row(align=True)
        c0 = r.column(align=True)
        c0.prop(pcv, 'render_resolution_linked', toggle=True, text='', icon='LINKED' if pcv.render_resolution_linked else 'UNLINKED', icon_only=True, )
        c0.prop(pcv, 'render_resolution_linked', toggle=True, text='', icon='LINKED' if pcv.render_resolution_linked else 'UNLINKED', icon_only=True, )
        c0.prop(pcv, 'render_resolution_linked', toggle=True, text='', icon='LINKED' if pcv.render_resolution_linked else 'UNLINKED', icon_only=True, )
        c1 = r.column(align=True)
        if(pcv.render_resolution_linked):
            render = context.scene.render
            c1.prop(render, 'resolution_x')
            c1.prop(render, 'resolution_y')
            c1.prop(render, 'resolution_percentage')
            c1.active = False
        else:
            c1.prop(pcv, 'render_resolution_x')
            c1.prop(pcv, 'render_resolution_y')
            c1.prop(pcv, 'render_resolution_percentage')
        
        r = sub.row(align=True)
        r.operator('point_cloud_visualizer.render')
        r.operator('point_cloud_visualizer.render_animation')
        
        sub.enabled = ops.PCV_OT_render.poll(context)


class PCV_PT_convert(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "View"
    bl_label = "Convert"
    bl_parent_id = "PCV_PT_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        o = context.active_object
        if(o is None):
            return False
        
        if(o):
            pcv = o.point_cloud_visualizer
            if(pcv.edit_is_edit_mesh):
                return False
            if(pcv.edit_initialized):
                return False
        return True
    
    def draw(self, context):
        pcv = context.object.point_cloud_visualizer
        l = self.layout
        sub = l.column()
        c = sub.column()
        c.prop(pcv, 'mesh_type')
        
        f = 0.245
        r = c.row(align=True)
        s = r.split(factor=f, align=True, )
        s.prop(pcv, 'mesh_all', toggle=True, )
        s = s.split(factor=1.0, align=True, )
        s.prop(pcv, 'mesh_percentage')
        if(pcv.mesh_all):
            s.enabled = False
        
        cc = c.column()
        cc.prop(pcv, 'mesh_size')
        
        if(pcv.mesh_type in ('INSTANCER', 'PARTICLES', )):
            cc.prop(pcv, 'mesh_base_sphere_subdivisions')
        
        cc_n = cc.row()
        cc_n.prop(pcv, 'mesh_normal_align')
        if(not pcv.has_normals):
            cc_n.enabled = False
        
        cc_c = cc.row()
        cc_c.prop(pcv, 'mesh_vcols')
        if(not pcv.has_vcols):
            cc_c.enabled = False
        
        if(pcv.mesh_type == 'VERTEX'):
            cc.enabled = False
        
        # c.operator('point_cloud_visualizer.convert')
        
        r = c.row(align=True)
        r.operator('point_cloud_visualizer.convert')
        r.prop(pcv, 'mesh_use_instancer2', toggle=True, text='', icon='AUTO', )
        
        c.enabled = ops.PCV_OT_convert.poll(context)


class PCV_PT_filter(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "View"
    bl_label = "Filter"
    bl_parent_id = "PCV_PT_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        o = context.active_object
        if(o is None):
            return False
        
        if(o):
            pcv = o.point_cloud_visualizer
            if(pcv.edit_is_edit_mesh):
                return False
            if(pcv.edit_initialized):
                return False
        return True
    
    def draw(self, context):
        pcv = context.object.point_cloud_visualizer
        l = self.layout
        c = l.column()


class PCV_PT_filter_simplify(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "View"
    bl_label = "Simplify"
    bl_parent_id = "PCV_PT_filter"
    bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        o = context.active_object
        if(o is None):
            return False
        
        if(o):
            pcv = o.point_cloud_visualizer
            if(pcv.edit_is_edit_mesh):
                return False
            if(pcv.edit_initialized):
                return False
        return True
    
    def draw(self, context):
        pcv = context.object.point_cloud_visualizer
        l = self.layout
        c = l.column()
        
        a = c.column(align=True)
        a.prop(pcv, 'filter_simplify_num_samples')
        a.prop(pcv, 'filter_simplify_num_candidates')
        
        c.operator('point_cloud_visualizer.filter_simplify')
        
        c.enabled = ops_filter.PCV_OT_filter_simplify.poll(context)


class PCV_PT_filter_project(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "View"
    bl_label = "Project"
    bl_parent_id = "PCV_PT_filter"
    bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        o = context.active_object
        if(o is None):
            return False
        
        if(o):
            pcv = o.point_cloud_visualizer
            if(pcv.edit_is_edit_mesh):
                return False
            if(pcv.edit_initialized):
                return False
        return True
    
    def draw(self, context):
        pcv = context.object.point_cloud_visualizer
        l = self.layout
        c = l.column()
        c.prop(pcv, 'filter_project_object')
        
        a = c.column(align=True)
        a.prop(pcv, 'filter_project_search_distance')
        r = a.row(align=True)
        r.prop(pcv, 'filter_project_negative', toggle=True, )
        r.prop(pcv, 'filter_project_positive', toggle=True, )
        
        c.prop(pcv, 'filter_project_discard')
        
        cc = c.column(align=True)
        f = 0.5
        r = cc.row(align=True, )
        s = r.split(factor=f, align=True, )
        s.prop(pcv, 'filter_project_colorize', toggle=True, )
        s = s.split(factor=1.0, align=True, )
        r = s.row(align=True, )
        ccc = r.column(align=True)
        ccc.prop(pcv, 'filter_project_colorize_from', text="", )
        ccc.enabled = pcv.filter_project_colorize
        
        c.prop(pcv, 'filter_project_shift')
        c.operator('point_cloud_visualizer.filter_project')
        
        # conditions are the same, also `filter_project_object` has to be set
        c.enabled = ops_filter.PCV_OT_filter_simplify.poll(context)
        
        if(pcv.filepath != '' and pcv.uuid != ''):
            if(not pcv.has_normals):
                c.label(text="Missing vertex normals.", icon='ERROR', )
                c.enabled = False


class PCV_PT_filter_remove_color(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "View"
    bl_label = "Remove Color"
    bl_parent_id = "PCV_PT_filter"
    bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        o = context.active_object
        if(o is None):
            return False
        
        if(o):
            pcv = o.point_cloud_visualizer
            if(pcv.edit_is_edit_mesh):
                return False
            if(pcv.edit_initialized):
                return False
        return True
    
    def draw(self, context):
        pcv = context.object.point_cloud_visualizer
        l = self.layout
        c = l.column()
        r = c.row()
        r.prop(pcv, 'filter_remove_color', text='', )
        
        a = c.column(align=True)
        r = a.row(align=True)
        r.prop(pcv, 'filter_remove_color_delta_hue_use', text='', toggle=True, icon_only=True, icon='CHECKBOX_HLT' if pcv.filter_remove_color_delta_hue_use else 'CHECKBOX_DEHLT', )
        cc = r.column(align=True)
        cc.prop(pcv, 'filter_remove_color_delta_hue')
        cc.active = pcv.filter_remove_color_delta_hue_use
        
        r = a.row(align=True)
        r.prop(pcv, 'filter_remove_color_delta_saturation_use', text='', toggle=True, icon_only=True, icon='CHECKBOX_HLT' if pcv.filter_remove_color_delta_saturation_use else 'CHECKBOX_DEHLT', )
        cc = r.column(align=True)
        cc.prop(pcv, 'filter_remove_color_delta_saturation')
        cc.active = pcv.filter_remove_color_delta_saturation_use
        
        r = a.row(align=True)
        r.prop(pcv, 'filter_remove_color_delta_value_use', text='', toggle=True, icon_only=True, icon='CHECKBOX_HLT' if pcv.filter_remove_color_delta_value_use else 'CHECKBOX_DEHLT', )
        cc = r.column(align=True)
        cc.prop(pcv, 'filter_remove_color_delta_value')
        cc.active = pcv.filter_remove_color_delta_value_use
        
        cc = c.column(align=True)
        r = cc.row(align=True)
        r.operator('point_cloud_visualizer.filter_remove_color')
        r.operator('point_cloud_visualizer.filter_remove_color_deselect', text="", icon='X', )
        cc.operator('point_cloud_visualizer.filter_remove_color_delete_selected')
        
        c.enabled = ops_filter.PCV_OT_filter_remove_color.poll(context)


class PCV_PT_filter_merge(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "View"
    bl_label = "Merge"
    bl_parent_id = "PCV_PT_filter"
    bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        o = context.active_object
        if(o is None):
            return False
        
        if(o):
            pcv = o.point_cloud_visualizer
            if(pcv.edit_is_edit_mesh):
                return False
            if(pcv.edit_initialized):
                return False
        return True
    
    def draw(self, context):
        pcv = context.object.point_cloud_visualizer
        l = self.layout
        c = l.column()
        
        c.operator('point_cloud_visualizer.filter_merge')
        
        c.enabled = ops_filter.PCV_OT_filter_merge.poll(context)


class PCV_PT_filter_join(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "View"
    bl_label = "Join"
    bl_parent_id = "PCV_PT_filter"
    bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        o = context.active_object
        if(o is None):
            return False
        
        if(o):
            pcv = o.point_cloud_visualizer
            if(pcv.edit_is_edit_mesh):
                return False
            if(pcv.edit_initialized):
                return False
        return True
    
    def draw(self, context):
        pcv = context.object.point_cloud_visualizer
        l = self.layout
        c = l.column()
        
        c.prop(pcv, 'filter_join_object')
        c.operator('point_cloud_visualizer.filter_join')
        
        ok = False
        for k, v in PCVManager.cache.items():
            if(v['uuid'] == pcv.uuid):
                if(v['ready']):
                    if(v['draw']):
                        ok = True
                break
        c.enabled = ok


class PCV_PT_filter_boolean(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "View"
    bl_label = "Boolean"
    bl_parent_id = "PCV_PT_filter"
    bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        o = context.active_object
        if(o is None):
            return False
        
        if(o):
            pcv = o.point_cloud_visualizer
            if(pcv.edit_is_edit_mesh):
                return False
            if(pcv.edit_initialized):
                return False
        return True
    
    def draw(self, context):
        pcv = context.object.point_cloud_visualizer
        l = self.layout
        c = l.column()
        
        c.prop(pcv, 'filter_boolean_object')
        c.operator('point_cloud_visualizer.filter_boolean_intersect')
        c.operator('point_cloud_visualizer.filter_boolean_exclude')
        
        c.enabled = ops_filter.PCV_OT_filter_merge.poll(context)


class PCV_PT_filter_color_adjustment(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "View"
    bl_label = "Color Adjustment"
    bl_parent_id = "PCV_PT_filter"
    bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        o = context.active_object
        if(o is None):
            return False
        
        if(o):
            pcv = o.point_cloud_visualizer
            if(pcv.edit_is_edit_mesh):
                return False
            if(pcv.edit_initialized):
                return False
        return True
    
    def draw(self, context):
        pcv = context.object.point_cloud_visualizer
        l = self.layout
        c = l.column()
        
        c.prop(pcv, 'color_adjustment_shader_enabled')
        cc = c.column(align=True)
        cc.prop(pcv, 'color_adjustment_shader_exposure')
        cc.prop(pcv, 'color_adjustment_shader_gamma')
        cc.prop(pcv, 'color_adjustment_shader_brightness')
        cc.prop(pcv, 'color_adjustment_shader_contrast')
        cc.prop(pcv, 'color_adjustment_shader_hue')
        cc.prop(pcv, 'color_adjustment_shader_saturation')
        cc.prop(pcv, 'color_adjustment_shader_value')
        cc.prop(pcv, 'color_adjustment_shader_invert')
        r = cc.row(align=True)
        r.operator('point_cloud_visualizer.color_adjustment_shader_reset')
        r.operator('point_cloud_visualizer.color_adjustment_shader_apply')
        cc.enabled = pcv.color_adjustment_shader_enabled
        
        c.enabled = ops_filter.PCV_OT_filter_merge.poll(context)


class PCV_PT_clip(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "View"
    bl_label = "Clip"
    bl_parent_id = "PCV_PT_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        o = context.active_object
        if(o is None):
            return False
        
        if(o):
            pcv = o.point_cloud_visualizer
            if(pcv.edit_is_edit_mesh):
                return False
            if(pcv.edit_initialized):
                return False
        return True
    
    def draw(self, context):
        pcv = context.object.point_cloud_visualizer
        l = self.layout
        c = l.column()
        c.prop(pcv, 'clip_shader_enabled', toggle=True, text='Enable Clipping Planes Shader', )
        
        a = l.column()
        c = a.column(align=True)
        r = c.row(align=True)
        r.prop(pcv, 'clip_plane0_enabled', text='', toggle=True, icon_only=True, icon='HIDE_OFF' if pcv.clip_plane0_enabled else 'HIDE_ON', )
        r.prop(pcv, 'clip_plane0', )
        r = c.row(align=True)
        r.prop(pcv, 'clip_plane1_enabled', text='', toggle=True, icon_only=True, icon='HIDE_OFF' if pcv.clip_plane1_enabled else 'HIDE_ON', )
        r.prop(pcv, 'clip_plane1', )
        r = c.row(align=True)
        r.prop(pcv, 'clip_plane2_enabled', text='', toggle=True, icon_only=True, icon='HIDE_OFF' if pcv.clip_plane2_enabled else 'HIDE_ON', )
        r.prop(pcv, 'clip_plane2', )
        r = c.row(align=True)
        r.prop(pcv, 'clip_plane3_enabled', text='', toggle=True, icon_only=True, icon='HIDE_OFF' if pcv.clip_plane3_enabled else 'HIDE_ON', )
        r.prop(pcv, 'clip_plane3', )
        r = c.row(align=True)
        r.prop(pcv, 'clip_plane4_enabled', text='', toggle=True, icon_only=True, icon='HIDE_OFF' if pcv.clip_plane4_enabled else 'HIDE_ON', )
        r.prop(pcv, 'clip_plane4', )
        r = c.row(align=True)
        r.prop(pcv, 'clip_plane5_enabled', text='', toggle=True, icon_only=True, icon='HIDE_OFF' if pcv.clip_plane5_enabled else 'HIDE_ON', )
        r.prop(pcv, 'clip_plane5', )
        
        c = a.column(align=True)
        c.prop(pcv, 'clip_planes_from_bbox_object')
        r = c.row(align=True)
        r.operator('point_cloud_visualizer.clip_planes_from_bbox')
        r.operator('point_cloud_visualizer.clip_planes_reset', text='', icon='X', )
        
        a.enabled = pcv.clip_shader_enabled
        
        pcv = context.object.point_cloud_visualizer
        ok = False
        for k, v in PCVManager.cache.items():
            if(v['uuid'] == pcv.uuid):
                if(v['ready']):
                    if(v['draw']):
                        ok = True
        l.enabled = ok


class PCV_PT_edit(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "View"
    bl_label = "Edit"
    bl_parent_id = "PCV_PT_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        o = context.active_object
        if(o is None):
            return False
        
        if(o):
            pcv = o.point_cloud_visualizer
            if(pcv.edit_is_edit_mesh):
                return False
            if(pcv.edit_initialized):
                return False
        return True
    
    def draw(self, context):
        pcv = context.object.point_cloud_visualizer
        l = self.layout
        c = l.column()
        c.operator('point_cloud_visualizer.edit_start', text='Enable Edit Mode', )


class PCV_PT_export(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "View"
    bl_label = "Export"
    bl_parent_id = "PCV_PT_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        o = context.active_object
        if(o is None):
            return False
        
        if(o):
            pcv = o.point_cloud_visualizer
            if(pcv.edit_is_edit_mesh):
                return False
            if(pcv.edit_initialized):
                return False
        return True
    
    def draw(self, context):
        pcv = context.object.point_cloud_visualizer
        l = self.layout
        c = l.column()
        c.prop(pcv, 'export_use_viewport')
        cc = c.column()
        cc.prop(pcv, 'export_visible_only')
        if(not pcv.export_use_viewport):
            cc.enabled = False
        c.prop(pcv, 'export_apply_transformation')
        c.prop(pcv, 'export_convert_axes')
        c.operator('point_cloud_visualizer.export')
        
        c.enabled = ops.PCV_OT_export.poll(context)


class PCV_PT_sequence(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "View"
    bl_label = "Sequence"
    bl_parent_id = "PCV_PT_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        o = context.active_object
        if(o is None):
            return False
        
        if(o):
            pcv = o.point_cloud_visualizer
            if(pcv.edit_is_edit_mesh):
                return False
            if(pcv.edit_initialized):
                return False
        return True
    
    # def draw_header(self, context):
    #     pcv = context.object.point_cloud_visualizer
    #     l = self.layout
    #     l.label(text='', icon='EXPERIMENTAL', )
    
    def draw(self, context):
        pcv = context.object.point_cloud_visualizer
        l = self.layout
        # c = l.column()
        # c.prop(pcv, 'sequence_enabled')
        # c.enabled = ops.PCV_OT_sequence_preload.poll(context)
        
        c = l.column()
        
        # c.label(text='Experimental', icon='ERROR', )
        
        c.operator('point_cloud_visualizer.sequence_preload')
        if(pcv.uuid in PCVSequence.cache.keys()):
            c.label(text="Loaded {} item(s)".format(len(PCVSequence.cache[pcv.uuid]['data'])))
            # c.enabled = pcv.sequence_enabled
        else:
            c.label(text="Loaded {} item(s)".format(0))
            c.enabled = ops.PCV_OT_sequence_preload.poll(context)
        # c.enabled = pcv.sequence_enabled
        
        # c = l.column()
        # c.prop(pcv, 'sequence_frame_duration')
        # c.prop(pcv, 'sequence_frame_start')
        # c.prop(pcv, 'sequence_frame_offset')
        c.prop(pcv, 'sequence_use_cyclic')
        # c.enabled = False
        # if(pcv.sequence_enabled):
        #     c.enabled = True
        # c.enabled = (ops.PCV_OT_sequence_preload.poll(context) and pcv.sequence_enabled)
        # c.enabled = ops.PCV_OT_sequence_preload.poll(context)
        # c.enabled = pcv.sequence_enabled
        c.operator('point_cloud_visualizer.sequence_clear')
        
        l.enabled = not pcv.runtime


class PCV_PT_generate(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "View"
    bl_label = "Generate"
    bl_parent_id = "PCV_PT_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        o = context.active_object
        if(o is None):
            return False
        
        if(o):
            pcv = o.point_cloud_visualizer
            if(pcv.edit_is_edit_mesh):
                return False
            if(pcv.edit_initialized):
                return False
        return True
    
    def draw(self, context):
        pcv = context.object.point_cloud_visualizer
        l = self.layout
        c = l.column()
        
        def prop_name(cls, prop, colon=False, ):
            for p in cls.bl_rna.properties:
                if(p.identifier == prop):
                    if(colon):
                        return "{}:".format(p.name)
                    return p.name
            return ''
        
        def third_label_two_thirds_prop(cls, prop, uil, ):
            f = 0.33
            r = uil.row()
            s = r.split(factor=f)
            s.label(text=prop_name(cls, prop, True, ))
            s = s.split(factor=1.0)
            r = s.row()
            r.prop(cls, prop, text='', )
        
        third_label_two_thirds_prop(pcv, 'generate_source', c, )
        
        if(pcv.generate_source == 'PARTICLES'):
            third_label_two_thirds_prop(pcv, 'generate_source_psys', c, )
        
        if(pcv.generate_source in ('SURFACE', )):
            third_label_two_thirds_prop(pcv, 'generate_algorithm', c, )
        
        if(pcv.generate_source in ('SURFACE', )):
            if(pcv.generate_algorithm in ('WEIGHTED_RANDOM_IN_TRIANGLE', )):
                c.prop(pcv, 'generate_number_of_points')
                c.prop(pcv, 'generate_seed')
                c.prop(pcv, 'generate_exact_number_of_points')
            if(pcv.generate_algorithm in ('POISSON_DISK_SAMPLING', )):
                c.prop(pcv, 'generate_minimal_distance')
                c.prop(pcv, 'generate_sampling_exponent')
                # c.prop(pcv, 'generate_seed')
        
        third_label_two_thirds_prop(pcv, 'generate_colors', c, )
        if(pcv.generate_colors == 'CONSTANT'):
            r = c.row()
            third_label_two_thirds_prop(pcv, 'generate_constant_color', c, )
        
        c.operator('point_cloud_visualizer.generate_from_mesh')
        c.operator('point_cloud_visualizer.reset_runtime', text="Remove Generated", )
        
        c.enabled = ops.PCV_OT_generate_point_cloud.poll(context)


class PCV_PT_development(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "View"
    bl_label = "Development"
    bl_parent_id = "PCV_PT_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        if(not debug_mode()):
            return False
        
        o = context.active_object
        if(o is None):
            return False
        
        if(o):
            pcv = o.point_cloud_visualizer
            if(pcv.edit_is_edit_mesh):
                return False
            if(pcv.edit_initialized):
                return False
        return True
    
    def draw_header(self, context):
        pcv = context.object.point_cloud_visualizer
        l = self.layout
        # l.label(text='', icon='SETTINGS', )
        l.label(text='', icon='EXPERIMENTAL', )
    
    def draw(self, context):
        pcv = context.object.point_cloud_visualizer
        l = self.layout
        sub = l.column()
        
        # sub.label(text="Color Adjustment Shader:")
        # c = sub.column(align=True)
        # c.prop(pcv, 'color_adjustment_shader_enabled')
        # cc = c.column(align=True)
        # cc.prop(pcv, 'color_adjustment_shader_exposure')
        # cc.prop(pcv, 'color_adjustment_shader_gamma')
        # cc.prop(pcv, 'color_adjustment_shader_brightness')
        # cc.prop(pcv, 'color_adjustment_shader_contrast')
        # cc.prop(pcv, 'color_adjustment_shader_hue')
        # cc.prop(pcv, 'color_adjustment_shader_saturation')
        # cc.prop(pcv, 'color_adjustment_shader_value')
        # cc.prop(pcv, 'color_adjustment_shader_invert')
        # r = cc.row(align=True)
        # r.operator('point_cloud_visualizer.color_adjustment_shader_reset')
        # r.operator('point_cloud_visualizer.color_adjustment_shader_apply')
        # cc.enabled = pcv.color_adjustment_shader_enabled
        # sub.separator()
        
        sub.label(text="Shaders:")
        e = False
        for k, v in PCVManager.cache.items():
            if(v['uuid'] == pcv.uuid):
                if(v['ready']):
                    if(v['draw']):
                        e = True
        
        c = sub.column()
        
        cc = c.column(align=True)
        cc.prop(pcv, 'dev_selection_shader_display', toggle=True, )
        if(pcv.dev_selection_shader_display):
            r = cc.row(align=True)
            r.prop(pcv, 'dev_selection_shader_color', text="", )
        
        c.prop(pcv, 'dev_minimal_shader_enabled', toggle=True, text="Minimal Shader", )
        
        c.prop(pcv, 'dev_minimal_shader_variable_size_enabled', toggle=True, text="Minimal Shader With Variable Size", )
        
        cc = c.column(align=True)
        cc.prop(pcv, 'dev_minimal_shader_variable_size_and_depth_enabled', toggle=True, text="Minimal Shader With Variable Size And Depth", )
        if(pcv.dev_minimal_shader_variable_size_and_depth_enabled):
            cc.prop(pcv, 'dev_minimal_shader_variable_size_and_depth_brightness')
            cc.prop(pcv, 'dev_minimal_shader_variable_size_and_depth_contrast')
            cc.prop(pcv, 'dev_minimal_shader_variable_size_and_depth_blend')
        
        cc = c.column(align=True)
        cc.prop(pcv, 'dev_billboard_point_cloud_enabled', toggle=True, text='BIllboard Shader', )
        if(pcv.dev_billboard_point_cloud_enabled):
            cc.prop(pcv, 'dev_billboard_point_cloud_size')
        
        cc = c.column(align=True)
        cc.prop(pcv, 'dev_rich_billboard_point_cloud_enabled', toggle=True, text='Rich BIllboard Shader', )
        if(pcv.dev_rich_billboard_point_cloud_enabled):
            cc.prop(pcv, 'dev_rich_billboard_point_cloud_size')
            cc.prop(pcv, 'dev_rich_billboard_depth_brightness')
            cc.prop(pcv, 'dev_rich_billboard_depth_contrast')
            cc.prop(pcv, 'dev_rich_billboard_depth_blend')
        
        cc = c.column(align=True)
        cc.prop(pcv, 'dev_phong_shader_enabled', toggle=True, text='Phong Shader', )
        if(pcv.dev_phong_shader_enabled):
            cc.prop(pcv, 'dev_phong_shader_ambient_strength')
            cc.prop(pcv, 'dev_phong_shader_specular_strength')
            cc.prop(pcv, 'dev_phong_shader_specular_exponent')
        
        cc = c.column(align=True)
        cc.prop(pcv, 'clip_shader_enabled', toggle=True, text='Clip', )
        if(pcv.clip_shader_enabled):
            r = cc.row(align=True)
            r.prop(pcv, 'clip_plane0_enabled', text='', toggle=True, icon_only=True, icon='HIDE_OFF' if pcv.clip_plane0_enabled else 'HIDE_ON', )
            r.prop(pcv, 'clip_plane0', )
            r = cc.row(align=True)
            r.prop(pcv, 'clip_plane1_enabled', text='', toggle=True, icon_only=True, icon='HIDE_OFF' if pcv.clip_plane1_enabled else 'HIDE_ON', )
            r.prop(pcv, 'clip_plane1', )
            r = cc.row(align=True)
            r.prop(pcv, 'clip_plane2_enabled', text='', toggle=True, icon_only=True, icon='HIDE_OFF' if pcv.clip_plane2_enabled else 'HIDE_ON', )
            r.prop(pcv, 'clip_plane2', )
            r = cc.row(align=True)
            r.prop(pcv, 'clip_plane3_enabled', text='', toggle=True, icon_only=True, icon='HIDE_OFF' if pcv.clip_plane3_enabled else 'HIDE_ON', )
            r.prop(pcv, 'clip_plane3', )
            r = cc.row(align=True)
            r.prop(pcv, 'clip_plane4_enabled', text='', toggle=True, icon_only=True, icon='HIDE_OFF' if pcv.clip_plane4_enabled else 'HIDE_ON', )
            r.prop(pcv, 'clip_plane4', )
            r = cc.row(align=True)
            r.prop(pcv, 'clip_plane5_enabled', text='', toggle=True, icon_only=True, icon='HIDE_OFF' if pcv.clip_plane5_enabled else 'HIDE_ON', )
            r.prop(pcv, 'clip_plane5', )
            cc.prop(pcv, 'clip_planes_from_bbox_object')
            cc.operator('point_cloud_visualizer.clip_planes_from_bbox')
        
        cc = c.column(align=True)
        cc.prop(pcv, 'billboard_phong_enabled', toggle=True, text='Billboard Phong', )
        if(pcv.billboard_phong_enabled):
            cc.prop(pcv, 'billboard_phong_circles', toggle=True, )
            cc.prop(pcv, 'billboard_phong_size')
            cc.prop(pcv, 'billboard_phong_ambient_strength')
            cc.prop(pcv, 'billboard_phong_specular_strength')
            cc.prop(pcv, 'billboard_phong_specular_exponent')
        
        cc = c.column(align=True)
        cc.prop(pcv, 'skip_point_shader_enabled', toggle=True, text='Skip Point Shader', )
        if(pcv.skip_point_shader_enabled):
            cc.prop(pcv, 'skip_point_percentage')
        
        cc = c.column(align=True)
        cc.prop(pcv, 'dev_bbox_enabled', toggle=True, text='Bounding Box', )
        if(pcv.dev_bbox_enabled):
            cc.prop(pcv, 'dev_bbox_color')
            cc.prop(pcv, 'dev_bbox_size')
            cc.prop(pcv, 'dev_bbox_alpha')
        
        sub.separator()
        
        sub.label(text="Generate Volume:")
        c = sub.column(align=True)
        c.prop(pcv, 'generate_number_of_points')
        c.prop(pcv, 'generate_seed')
        c.operator('point_cloud_visualizer.generate_volume_from_mesh')
        
        sub.separator()
        
        """
        c.separator()
        
        c.label(text="new ui for shaders")
        c.separator()
        
        r = c.row(align=True)
        s = r.split(factor=0.25, align=True, )
        s.label(text='Shader:')
        s = s.split(factor=0.75, align=True, )
        r = s.row(align=True)
        r.prop(pcv, 'shader', text='', )
        s = s.split(factor=0.25, align=True, )
        
        cc = s.column(align=True)
        cc.prop(pcv, 'shader_illumination', text='', icon='LIGHT', toggle=True, icon_only=True, )
        if(pcv.shader not in ('DEFAULT', 'DEPTH', )):
            cc.enabled = False

        cc = s.column(align=True)
        cc.prop(pcv, 'shader_options_show', text='', icon='TOOL_SETTINGS', toggle=True, icon_only=True, )
        if(pcv.shader not in ('DEPTH', )):
            cc.enabled = False

        cc = s.column(align=True)
        cc.prop(pcv, 'shader_normal_lines', text='', icon='SNAP_NORMAL', toggle=True, icon_only=True, )
        
        c.separator()
        c.separator()
        
        r = c.row(align=True)
        r.prop(pcv, 'shader', expand=True, )
        
        cc = r.column(align=True)
        cc.prop(pcv, 'shader_illumination', text='', icon='LIGHT', toggle=True, icon_only=True, )
        if(pcv.shader not in ('DEFAULT', 'DEPTH', )):
            cc.enabled = False
        
        cc = r.column(align=True)
        cc.prop(pcv, 'shader_options_show', text='', icon='TOOL_SETTINGS', toggle=True, icon_only=True, )
        if(pcv.shader not in ('DEPTH', )):
            cc.enabled = False
        
        cc = r.column(align=True)
        cc.prop(pcv, 'shader_normal_lines', text='', icon='SNAP_NORMAL', toggle=True, icon_only=True, )
        
        if(pcv.shader_illumination):
            if(pcv.shader in ('DEFAULT', 'DEPTH', )):
                c.label(text='shader illumination options..')
        
        if(pcv.shader_options_show):
            if(pcv.shader in ('DEPTH', )):
                c.label(text='shader options..')
        
        if(pcv.shader_normal_lines):
            c.label(text='shader normal lines options..')
        
        c.separator()
        """
        
        sub.label(text="Numpy Vertices And Normals Transform")
        c = sub.column()
        c.prop(pcv, 'dev_transform_normals_target_object')
        c.operator('point_cloud_visualizer.pcviv_dev_transform_normals')
        
        sub.label(text="Clip To Active Camera Cone")
        c = sub.column()
        c.operator('point_cloud_visualizer.clip_planes_from_camera_view')


class PCV_PT_debug(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "View"
    bl_label = "Debug"
    bl_parent_id = "PCV_PT_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        o = context.active_object
        if(o is None):
            return False
        
        if(o):
            pcv = o.point_cloud_visualizer
            if(debug_mode()):
                return True
        return False
    
    def draw_header(self, context):
        pcv = context.object.point_cloud_visualizer
        l = self.layout
        l.label(text='', icon='SETTINGS', )
    
    def draw(self, context):
        pcv = context.object.point_cloud_visualizer
        l = self.layout
        sub = l.column()
        sub.operator('script.reload')
        sub.separator()
        
        b = sub.box()
        r = b.row()
        r.prop(pcv, 'debug_panel_show_properties', icon='TRIA_DOWN' if pcv.debug_panel_show_properties else 'TRIA_RIGHT', icon_only=True, emboss=False, )
        r.label(text="properties")
        if(pcv.debug_panel_show_properties):
            c = b.column()
            for k, p in pcv.bl_rna.properties.items():
                v = 'n/a'
                if(p.type == 'POINTER'):
                    v = 'POINTER'
                else:
                    v = p.default
                    if(k in pcv.keys()):
                        v = pcv[k]
                if(p.type == 'BOOLEAN'):
                    v = bool(v)
                c.label(text="{}: {}".format(k, v))
            c.scale_y = 0.5
        
        b = sub.box()
        r = b.row()
        r.prop(pcv, 'debug_panel_show_manager', icon='TRIA_DOWN' if pcv.debug_panel_show_manager else 'TRIA_RIGHT', icon_only=True, emboss=False, )
        r.label(text="manager")
        if(pcv.debug_panel_show_manager):
            c = b.column(align=True)
            rr = c.row(align=True)
            rr.operator('point_cloud_visualizer.init')
            rr.operator('point_cloud_visualizer.deinit')
            rr.operator('point_cloud_visualizer.gc')
            bb = b.box()
            c = bb.column()
            c.label(text="cache: {} item(s)".format(len(PCVManager.cache.items())))
            c.label(text="handle: {}".format(PCVManager.handle))
            c.label(text="initialized: {}".format(PCVManager.initialized))
            c.scale_y = 0.5
            
            if(len(PCVManager.cache)):
                b.label(text="cache details:")
                for k, v in PCVManager.cache.items():
                    bb = b.box()
                    r = bb.row()
                    r.prop(pcv, 'debug_panel_show_cache_items', icon='TRIA_DOWN' if pcv.debug_panel_show_cache_items else 'TRIA_RIGHT', icon_only=True, emboss=False, )
                    r.label(text=k)
                    if(pcv.debug_panel_show_cache_items):
                        c = bb.column()
                        c.scale_y = 0.5
                        for ki, vi in sorted(v.items()):
                            if(type(vi) == np.ndarray):
                                c.label(text="{}: numpy.ndarray ({} items)".format(ki, len(vi)))
                            elif(type(vi) == dict):
                                c.label(text="{}: dict ({} items)".format(ki, len(vi.keys())))
                                t = '    '
                                for dk, dv in vi.items():
                                    c.label(text="{}{}: {}".format(t, dk, dv))
                            else:
                                c.label(text="{}: {}".format(ki, vi))
        
        b = sub.box()
        r = b.row()
        r.prop(pcv, 'debug_panel_show_sequence', icon='TRIA_DOWN' if pcv.debug_panel_show_sequence else 'TRIA_RIGHT', icon_only=True, emboss=False, )
        r.label(text="sequence")
        if(pcv.debug_panel_show_sequence):
            c = b.column(align=True)
            rr = c.row(align=True)
            rr.operator('point_cloud_visualizer.seq_init')
            rr.operator('point_cloud_visualizer.seq_deinit')
            bb = b.box()
            c = bb.column()
            c.label(text="cache: {} item(s)".format(len(PCVSequence.cache.items())))
            c.label(text="initialized: {}".format(PCVSequence.initialized))
            c.scale_y = 0.5
            
            if(len(PCVSequence.cache)):
                b.label(text="cache details:")
                for k, v in PCVSequence.cache.items():
                    bb = b.box()
                    c = bb.column()
                    c.scale_y = 0.5
                    c.label(text="{}: {}".format('uuid', v['uuid']))
                    c.label(text="{}: {}".format('pcv', v['pcv']))
                    c.label(text="{}: {}".format('data', '{} item(s)'.format(len(v['data']))))
