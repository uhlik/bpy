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

bl_info = {"name": "Carbon Tools",
           "description": "Ever-evolving set of small tools, workflows and shortcuts focused mainly on processing photogrammetry scans.",
           "author": "Jakub Uhlik",
           "version": (0, 2, 3),
           "blender": (2, 80, 0),
           "location": "3D Viewport > Sidebar > Carbon Tools",
           "warning": "",
           "wiki_url": "",
           "tracker_url": "",
           "category": "Object", }


import os
import uuid
import sys
import math

import bpy
import bmesh
from bpy.types import Operator, Panel, AddonPreferences
from mathutils import Vector, Matrix
from bpy.props import FloatProperty, IntProperty, BoolProperty, StringProperty, EnumProperty, PointerProperty
from bpy.types import PropertyGroup
import numpy as np


DEBUG = False


def log(msg, indent=0, ):
    m = "{0}> {1}".format("    " * indent, msg)
    if(DEBUG):
        print(m)


class ObjDiff():
    _objects = None
    
    @classmethod
    def diff(cls):
        obs = [o for o in bpy.data.objects]
        if(cls._objects is None):
            cls._objects = obs
            return None, None, None
        a = list(set(obs) - set(cls._objects))
        d = list(set(cls._objects) - set(obs))
        u = list((set(cls._objects) - set(a)) - set(d))
        cls._objects = obs
        return a, d, u


class Utils():
    @classmethod
    def activate_object(cls, o):
        bpy.ops.object.select_all(action='DESELECT')
        context = bpy.context
        view_layer = context.view_layer
        o.select_set(True)
        view_layer.objects.active = o
    
    @classmethod
    def get_spaceview3d(cls):
        for a in bpy.context.screen.areas:
            if(a.type == "VIEW_3D"):
                return a.spaces[0]
        return None
    
    @classmethod
    def normalize(cls, v, vmin, vmax):
        return (v - vmin) / (vmax - vmin)
    
    @classmethod
    def interpolate(cls, nv, vmin, vmax):
        return vmin + (vmax - vmin) * nv
    
    @classmethod
    def map(cls, v, min1, max1, min2, max2):
        return cls.remap(v, min1, max1, min2, max2)
    
    @classmethod
    def remap(cls, v, min1, max1, min2, max2):
        v = cls.clamp(v, min1, max1)
        r = cls.interpolate(cls.normalize(v, min1, max1), min2, max2)
        r = cls.clamp(r, min2, max2)
        return r
    
    @classmethod
    def clamp(cls, v, vmin, vmax):
        if(vmax <= vmin):
            raise ValueError("Maximum value is smaller than or equal to minimum.")
        if(v <= vmin):
            return vmin
        if(v >= vmax):
            return vmax
        return v
    
    @classmethod
    def maprange(cls, v, ar, br, ):
        (a1, a2), (b1, b2) = ar, br
        return b1 + ((v - a1) * (b2 - b1) / (a2 - a1))
    
    @classmethod
    def distance_vectors(cls, a, b, ):
        return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2) ** 0.5
    
    @classmethod
    def distance_tuples(cls, a, b, ):
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5
    
    @classmethod
    def distance(cls, a, b, ):
        if(type(a) is Vector and type(b) is Vector):
            return cls.distance_vectors(a, b)
        return cls.distance_tuples(a, b)
    
    @classmethod
    def real_length_to_relative(cls, matrix, length):
        l, r, s = matrix.decompose()
        ms = Matrix.Scale(s.x, 4)
        l = Vector((length, 0, 0))
        v = ms.inverted() @ l
        return v.x
    
    @classmethod
    def shift_vert_along_normal(cls, tco, tno, v):
        co = Vector(tco)
        no = Vector(tno)
        return co + (no.normalized() * v)


class Progress():
    def __init__(self, total, indent=0, prefix="> ", ):
        self.current = 0
        self.percent = -1
        self.last = -1
        self.total = total
        self.prefix = prefix
        self.indent = indent
        self.t = "    "
        self.r = "\r"
        self.n = "\n"
    
    def step(self, numdone=1):
        if(not DEBUG):
            return
        
        self.current += numdone
        self.percent = int(self.current / (self.total / 100))
        if(self.percent > self.last):
            sys.stdout.write(self.r)
            sys.stdout.write("{0}{1}{2}%".format(self.t * self.indent, self.prefix, self.percent))
            self.last = self.percent
        if(self.percent >= 100 or self.total == self.current):
            sys.stdout.write(self.r)
            sys.stdout.write("{0}{1}{2}%{3}".format(self.t * self.indent, self.prefix, 100, self.n))


class CARBON_OT_quick_mesh_dyntopo_cleanup(Operator):
    bl_idname = "carbon_tools.quick_mesh_dyntopo_cleanup_setup"
    bl_label = "Quick Mesh Dyntopo Cleanup Setup"
    bl_description = "From Object mode go to Sculpt mode, enable Dyntopo and set zero strength to default brush. This is for mesh topology cleanup, not sculpting."
    bl_options = {'REGISTER', 'UNDO', }
    
    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return (ob and ob.type == 'MESH' and context.mode == 'OBJECT')
    
    def execute(self, context):
        ctp = context.scene.carbon_tools
        bpy.ops.object.select_all(action='DESELECT')
        bpy.context.object.show_wire = True
        bpy.ops.sculpt.sculptmode_toggle()
        bpy.data.brushes["SculptDraw"].strength = 0
        if(not context.sculpt_object.use_dynamic_topology_sculpting):
            bpy.ops.sculpt.dynamic_topology_toggle()
        # bpy.context.scene.tool_settings.sculpt.detail_refine_method = 'COLLAPSE'
        bpy.context.scene.tool_settings.sculpt.detail_refine_method = ctp.dyntopo_method
        bpy.context.scene.tool_settings.sculpt.detail_type_method = 'CONSTANT'
        # bpy.context.scene.tool_settings.sculpt.constant_detail_resolution = 500
        bpy.context.scene.tool_settings.sculpt.constant_detail_resolution = ctp.dyntopo_resolution
        bpy.context.scene.tool_settings.sculpt.use_symmetry_x = False
        return {'FINISHED'}


class CARBON_OT_extract_mesh_part(Operator):
    bl_idname = "carbon_tools.extract_mesh_part"
    bl_label = "Extract Mesh Part aka Extract Subtool"
    bl_description = "Extract selected part of mesh (in Edit mode) and put it in separate child object for faster mesh editing."
    bl_options = {'REGISTER', 'UNDO', }
    
    @classmethod
    def poll(cls, context):
        o = context.active_object
        # return (o and o.type == 'MESH' and context.mode == 'EDIT_MESH')
        return (o and o.type == 'MESH')
    
    def extract(self, context, ):
        o = context.active_object
        
        sm = context.tool_settings.mesh_select_mode[:]
        context.tool_settings.mesh_select_mode = (False, False, True)
        
        ObjDiff.diff()
        bpy.ops.mesh.separate(type='SELECTED')
        bpy.ops.object.mode_set(mode='OBJECT')
        a, _, _ = ObjDiff.diff()
        e = a[0]
        Utils.activate_object(e)
        n = o.name
        e.name = "{}-{}-extracted".format(n, uuid.uuid1())
        e.parent = o
        e.matrix_world = o.matrix_world.copy()
        
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='DESELECT')
        context.tool_settings.mesh_select_mode = (True, False, False)
        
        ctp = context.scene.carbon_tools
        if(ctp.extract_protect):
            bpy.ops.mesh.select_non_manifold()
            bpy.ops.mesh.hide(unselected=False)
        
        context.tool_settings.mesh_select_mode = sm
        
        return True
    
    def execute(self, context):
        r = self.extract(context)
        if(r is False):
            return {'CANCELLED'}
        return {'FINISHED'}


class CARBON_OT_insert_mesh_part(Operator):
    bl_idname = "carbon_tools.insert_mesh_part"
    bl_label = "Insert Mesh Part aka Insert Subtool"
    bl_description = "Insert separated mesh part back to original mesh and merge."
    bl_options = {'REGISTER', 'UNDO', }
    
    @classmethod
    def poll(cls, context):
        o = context.active_object
        # return (o and o.type == 'MESH' and context.mode == 'EDIT_MESH')
        return (o and o.type == 'MESH' and context.mode == 'OBJECT')
    
    def insert(self, context):
        o = context.active_object
        p = o.parent
        if(p is None):
            self.report({'ERROR'}, "Not a subtool.")
            return False
        
        # bpy.ops.object.mode_set(mode='OBJECT')
        if(context.mode != 'EDIT'):
            Utils.activate_object(o)
            bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.reveal()
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.object.mode_set(mode='OBJECT')
        Utils.activate_object(p)
        o.select_set(True)
        bpy.ops.object.join()
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='DESELECT')
        context.tool_settings.mesh_select_mode = (True, False, False)
        bpy.ops.mesh.select_non_manifold()
        # t = real_length_to_relative(p.matrix_world, 0.00001)
        bpy.ops.mesh.remove_doubles(threshold=0.0)
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.object.mode_set(mode='OBJECT')
        
        return True
    
    def execute(self, context):
        r = self.insert(context)
        if(r is False):
            return {'CANCELLED'}
        return {'FINISHED'}


class CARBON_OT_extract_non_manifold_elements(Operator):
    bl_idname = "carbon_tools.extract_non_manifold_elements"
    bl_label = "Extract Non-Manifold"
    bl_description = "Extract non-manifold elements with part of mesh around them as subtool"
    bl_options = {'REGISTER', 'UNDO', }
    
    @classmethod
    def poll(cls, context):
        o = context.active_object
        return (o and o.type == 'MESH' and context.mode == 'EDIT_MESH')
    
    def execute(self, context):
        o = context.active_object
        sm = context.tool_settings.mesh_select_mode[:]
        context.tool_settings.mesh_select_mode = (True, False, False)
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.mesh.select_non_manifold()
        bpy.ops.object.mode_set(mode='OBJECT')
        ok = sum([v.select for v in o.data.vertices])
        bpy.ops.object.mode_set(mode='EDIT')
        if(not ok):
            self.report({'ERROR'}, "Nothing non-manifold found.")
            return {'CANCELLED'}
        
        for i in range(10):
            bpy.ops.mesh.select_more()
        
        ct = context.scene.carbon_tools
        ctep = ct.extract_protect
        ct.extract_protect = False
        bpy.ops.carbon_tools.extract_mesh_part()
        ct.extract_protect = ctep
        context.tool_settings.mesh_select_mode = sm
        return {'FINISHED'}


class CARBON_OT_copy_original_matrix(Operator):
    bl_idname = "carbon_tools.copy_original_matrix"
    bl_label = "Copy Original Matrix from Selected to Active"
    bl_description = "Select matrix source object first and then target object. Matrix will be copied while keeping visual transformation intact."
    bl_options = {'REGISTER', 'UNDO', }
    
    @classmethod
    def poll(cls, context):
        o = context.active_object
        return (o and o.type == 'MESH' and len(context.selected_objects) == 2)
    
    def execute(self, context):
        o = context.active_object
        s = [ob for ob in context.selected_objects if ob != o][0]
        
        m = s.matrix_world.copy()
        mi = m.inverted()
        
        me = o.data
        me.transform(o.matrix_world)
        o.matrix_world = Matrix()
        me.transform(mi)
        o.matrix_world = m
        
        me.update()
        
        return {'FINISHED'}


class CARBON_OT_copy_transformation(Operator):
    bl_idname = "carbon_tools.copy_transformation"
    bl_label = "Copy Transformation from Selected to Active"
    bl_description = "Copy transformation from selected to active. Useful for setting correct scale and orientation after initial import from PhotoScan."
    bl_options = {'REGISTER', 'UNDO', }
    
    @classmethod
    def poll(cls, context):
        o = context.active_object
        return (o and o.type == 'MESH' and len(context.selected_objects) == 2)
    
    def execute(self, context):
        o = context.active_object
        m = o.matrix_world.copy()
        s = [a for a in context.selected_objects if a != o][0]
        s.matrix_world = m
        return {'FINISHED'}


class CARBON_OT_export_obj_to_zbrush(Operator):
    bl_idname = "carbon_tools.export_obj_to_zbrush"
    bl_label = "Export OBJ to ZBrush"
    bl_description = "Export active mesh to ZBrush."
    bl_options = {'REGISTER', 'UNDO', }
    
    @classmethod
    def poll(cls, context):
        ob = context.active_object
        d = bpy.data
        # return (ob and ob.type == 'MESH' and context.mode == 'OBJECT' and d.is_saved and not d.is_dirty)
        return (ob and ob.type == 'MESH' and context.mode == 'OBJECT' and d.is_saved)
    
    def execute(self, context):
        b = bpy.data.filepath
        h, t = os.path.split(b)
        n, e = os.path.splitext(t)
        p = os.path.join(h, '{}.obj'.format(n))
        d = {'filepath': p,
             'apply_modifiers': False,
             'apply_transformation': True,
             'convert_axes': True,
             'triangulate': False,
             'use_normals': True,
             'use_uv': True,
             'use_vcols': False,
             'global_scale': 1.0,
             'precision': 6, }
        
        try:
            bpy.ops.export_mesh.fast_obj(**d)
        except AttributeError:
            self.report({'ERROR'}, "'Fast Wavefront^2' addon is not installed and activated")
            return {'CANCELLED'}
        
        return {'FINISHED'}


class CARBON_OT_import_obj_from_zbrush(Operator):
    bl_idname = "carbon_tools.import_obj_from_zbrush"
    bl_label = "Import OBJ from ZBrush"
    bl_description = "Import OBJ from ZBrush."
    bl_options = {'REGISTER', 'UNDO', }
    
    @classmethod
    def poll(cls, context):
        return (context.mode == 'OBJECT')
    
    def execute(self, context):
        d = {'convert_axes': True,
             'with_uv': True,
             'with_vertex_colors_mrgb': False,
             'with_vertex_colors_extended': False,
             'vcols_ext_with_gamma': False,
             'use_m_as_vertex_group': False,
             'with_polygroups': False,
             'global_scale': 1.0,
             'apply_conversion': True, }
        
        try:
            bpy.ops.import_mesh.fast_obj('INVOKE_DEFAULT', **d)
        except AttributeError:
            self.report({'ERROR'}, "'Fast Wavefront^2' addon is not installed and activated")
            return {'CANCELLED'}
        
        return {'FINISHED'}


class CARBON_OT_export_obj_to_photoscan(Operator):
    bl_idname = "carbon_tools.export_obj_to_photoscan"
    bl_label = "Export OBJ to PhotoScan"
    bl_description = "Export active mesh to PhotoScan."
    bl_options = {'REGISTER', 'UNDO', }
    
    @classmethod
    def poll(cls, context):
        ob = context.active_object
        d = bpy.data
        # return (ob and ob.type == 'MESH' and context.mode == 'OBJECT' and d.is_saved and not d.is_dirty)
        return (ob and ob.type == 'MESH' and context.mode == 'OBJECT' and d.is_saved)
    
    def execute(self, context):
        b = bpy.data.filepath
        h, t = os.path.split(b)
        n, e = os.path.splitext(t)
        p = os.path.join(h, '{}.obj'.format(n))
        d = {'filepath': p,
             'apply_modifiers': False,
             'apply_transformation': False,
             'convert_axes': False,
             'triangulate': False,
             'use_normals': True,
             'use_uv': True,
             'use_vcols': False,
             'global_scale': 1.0,
             'precision': 6, }
        
        try:
            bpy.ops.export_mesh.fast_obj(**d)
        except AttributeError:
            self.report({'ERROR'}, "'Fast Wavefront^2' addon is not installed and activated")
            return {'CANCELLED'}
        
        return {'FINISHED'}


class CARBON_OT_quick_texture_painting_setup(Operator):
    bl_idname = "carbon_tools.quick_texture_painting_setup"
    bl_label = "Quick Texture Painting Setup"
    bl_description = "Setup Texture painting, basically just set settings for External editing of viewport snapshots."
    bl_options = {'REGISTER', 'UNDO', }
    
    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return (ob and ob.type == 'MESH' and context.mode == 'OBJECT')
    
    def execute(self, context):
        bpy.ops.paint.texture_paint_toggle()
        ctp = context.scene.carbon_tools
        bpy.context.object.show_wire = False
        r = int(ctp.tp_ext_resolution)
        bpy.context.scene.tool_settings.image_paint.screen_grab_size[0] = r
        bpy.context.scene.tool_settings.image_paint.screen_grab_size[1] = r
        bpy.context.scene.tool_settings.image_paint.seam_bleed = 8
        bpy.data.brushes["TexDraw"].strength = 0
        
        v = Utils.get_spaceview3d()
        if(v is not None):
            v.shading.color_type = 'TEXTURE'
            v.shading.light = 'FLAT'
            v.overlay.show_overlays = False
        
        return {'FINISHED'}


class CARBON_OT_save_all_images(Operator):
    bl_idname = "carbon_tools.save_all_images"
    bl_label = "Save All Images"
    bl_description = "Shortcut to Texture Painting > Slots > Save All Images."
    bl_options = {'REGISTER', }
    
    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return (ob and ob.type == 'MESH' and context.mode == 'PAINT_TEXTURE')
    
    def execute(self, context):
        bpy.ops.image.save_all_modified()
        return {'FINISHED'}


class CARBON_OT_end_current_procedure(Operator):
    bl_idname = "carbon_tools.end_current_procedure"
    bl_label = "End Current Procedure"
    bl_description = "Well, that doesn't explain... why you've come all the way out here, all the way out here to hell. I, uh, have a job out in the town of Machine. Machine? That's the end of the line."
    bl_options = {'REGISTER', 'UNDO', }
    
    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return (ob and ob.type == 'MESH' and context.mode != 'OBJECT')
    
    def execute(self, context):
        ctp = context.scene.carbon_tools
        bpy.ops.object.mode_set(mode='OBJECT')
        if(not ctp.end_keep_wire):
            bpy.context.object.show_wire = False
        Utils.activate_object(context.active_object)
        
        v = Utils.get_spaceview3d()
        if(v is not None):
            v.shading.color_type = 'MATERIAL'
            v.shading.light = 'STUDIO'
            v.overlay.show_overlays = True
        
        return {'FINISHED'}


class CARBON_OT_select_seams(Operator):
    bl_idname = "carbon_tools.select_seams"
    bl_label = "Select Seams"
    bl_description = "Set seam edges selected"
    bl_options = {'REGISTER', 'UNDO', }
    
    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return (ob and ob.type == 'MESH' and (context.mode == 'EDIT_MESH' or context.mode == 'OBJECT'))
    
    def execute(self, context):
        if(context.mode == 'OBJECT'):
            bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='DESELECT')
        context.tool_settings.mesh_select_mode = (False, True, False)
        bpy.ops.object.mode_set(mode='OBJECT')
        o = context.active_object
        m = o.data
        for e in m.edges:
            if(e.use_seam):
                e.select = True
        bpy.ops.object.mode_set(mode='EDIT')
        return {'FINISHED'}


class CARBON_OT_mark_seams_from_uv_islands(Operator):
    bl_idname = "carbon_tools.mark_seams_from_uv_islands"
    bl_label = "Mark Seams From UV Islands"
    bl_description = "Mark Seams From UV Islands"
    bl_options = {'REGISTER', 'UNDO', }
    
    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return (ob and ob.type == 'MESH' and (context.mode == 'EDIT_MESH' or context.mode == 'OBJECT') and len(ob.data.uv_layers) > 0)
    
    def execute(self, context):
        e = True
        if(context.mode != 'EDIT_MESH'):
            e = False
            bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.uv.seams_from_islands()
        if(not e):
            bpy.ops.object.mode_set(mode='OBJECT')
        return {'FINISHED'}


class CARBON_OT_export_uv_layout(Operator):
    bl_idname = "carbon_tools.export_uv_layout"
    bl_label = "Export UV Layout"
    bl_description = "Export UV layout and png with resolution preset and fill opacity 1.0"
    bl_options = {'REGISTER', 'UNDO', }
    
    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return (ob and ob.type == 'MESH' and context.mode == 'OBJECT' and len(ob.data.uv_layers) > 0)
    
    def execute(self, context):
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        # original_type = context.area.type
        # context.area.type = "IMAGE_EDITOR"
        b = bpy.data.filepath
        h, t = os.path.split(b)
        n, e = os.path.splitext(t)
        p = os.path.join(h, '{}.png'.format(n))
        if(os.path.exists(p)):
            p = os.path.join(h, '{}-{}.png'.format(n, uuid.uuid1()))
        ctp = context.scene.carbon_tools
        r = int(ctp.export_uv_layout_resolution)
        # bpy.ops.uv.export_layout(filepath=p, check_existing=False, export_all=False, modified=False, mode='PNG', size=(r, r), opacity=1.0, tessellated=False, )
        bpy.ops.uv.export_layout(filepath=p, export_all=False, modified=False, mode='PNG', size=(r, r), opacity=1.0, )
        # context.area.type = original_type
        bpy.ops.object.mode_set(mode='OBJECT')
        return {'FINISHED'}


class CARBON_OT_toggle_unselected_wireframe(Operator):
    bl_idname = "carbon_tools.toggle_unselected_wireframe"
    bl_label = "Toggle Unselected Wireframe"
    bl_description = "Set object to draw wire, all edges and unselect object to be better visible in viewport"
    bl_options = {'REGISTER', 'UNDO', }
    
    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return (ob and ob.type == 'MESH' and context.mode == 'OBJECT')
    
    def execute(self, context):
        ob = context.active_object
        if(ob.show_wire):
            ob.show_wire = False
            ob.show_all_edges = False
            # ob.select = True
            ob.select_set(True)
        else:
            ob.show_wire = True
            ob.show_all_edges = True
            # ob.select = False
            ob.select_set(False)
        
        return {'FINISHED'}


class CARBON_OT_visualize_uv_seams_as_wireframe_mesh(Operator):
    bl_idname = "carbon_tools.visualize_uv_seams_as_wireframe_mesh"
    bl_label = "Visualize UV Seams As Wireframe Mesh"
    bl_description = "Copy seam edges to a new mesh object"
    bl_options = {'REGISTER', 'UNDO', }
    
    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return (ob and ob.type == 'MESH' and context.mode == 'OBJECT')
    
    def execute(self, context):
        ObjDiff.diff()
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.carbon_tools.select_seams()
        bpy.ops.mesh.duplicate(mode=1)
        ob = context.active_object
        bm = bmesh.from_edit_mesh(ob.data)
        bm.verts.ensure_lookup_table()
        ok = False
        for v in bm.verts:
            if(v.select):
                ok = True
                break
        if(not ok):
            self.report({'ERROR'}, "Mesh has no seams to be selected.")
            return {'CANCELLED'}
        bpy.ops.mesh.separate(type='SELECTED')
        a, _, _ = ObjDiff.diff()
        bpy.ops.object.mode_set(mode='OBJECT')
        o = a[0]
        Utils.activate_object(o)
        bpy.ops.carbon_tools.toggle_unselected_wireframe()
        return {'FINISHED'}


class CARBON_OT_select_non_manifold_extra(Operator):
    bl_idname = "carbon_tools.select_non_manifold_extra"
    bl_label = "Select Non-Manifold"
    bl_description = "Select non-manifold and optionally center camera view on selected"
    bl_options = {'REGISTER', 'UNDO', }
    
    @classmethod
    def poll(cls, context):
        o = context.active_object
        sm = tuple([i for i in context.tool_settings.mesh_select_mode]) != (False, False, True)
        return (o and o.type == 'MESH' and context.mode == 'EDIT_MESH' and sm)
    
    def execute(self, context):
        bpy.ops.mesh.select_non_manifold()
        ctp = context.scene.carbon_tools
        if(ctp.select_non_manifold_extra_auto_view):
            bpy.ops.view3d.view_selected(use_all_regions=False)
        return {'FINISHED'}


class CARBON_OT_texture_to_vertex_colors(Operator):
    bl_idname = "carbon_tools.texture_to_vertex_colors"
    bl_label = "UVTex > VCols"
    bl_description = "Copy image colors from active image texture node in active material using active UV layout to new vertex colors"
    bl_options = {'REGISTER', 'UNDO', }
    
    @classmethod
    def poll(cls, context):
        # o = context.active_object
        # o_ok = (o is not None and o.type == 'MESH' and context.mode == 'OBJECT')
        # m = o.active_material
        # m_ok = (m is not None)
        # n_ok = False
        # i_ok = False
        # u_ok = (o.data.uv_layers.active is not None)
        # if(m_ok):
        #     n = m.node_tree.nodes.active
        #     n_ok = (n.type == 'TEX_IMAGE')
        #     if(n_ok):
        #         if(n.image is not None):
        #             i = n.image
        #             i_ok = os.path.exists(i.filepath_from_user())
        # return (o_ok and m_ok and n_ok and i_ok and u_ok)
        
        if(context.mode != 'OBJECT'):
            return False
        o = context.active_object
        if(not o):
            return False
        if(o.type != 'MESH'):
            return False
        m = o.active_material
        if(not m):
            return False
        d = o.data
        if(not d.uv_layers.active):
            return False
        n = m.node_tree.nodes.active
        if(not n):
            return False
        if(n.type != 'TEX_IMAGE'):
            return False
        if(not n.image):
            return False
        i = n.image
        if(not os.path.exists(i.filepath_from_user())):
            return False
        return True
    
    def execute(self, context):
        log(self.bl_idname, 0)
        
        def remap(v, min1, max1, min2, max2):
            def clamp(v, vmin, vmax):
                if(vmax <= vmin):
                    raise ValueError("Maximum value is smaller than or equal to minimum.")
                if(v <= vmin):
                    return vmin
                if(v >= vmax):
                    return vmax
                return v
            
            def normalize(v, vmin, vmax):
                return (v - vmin) / (vmax - vmin)
            
            def interpolate(nv, vmin, vmax):
                return vmin + (vmax - vmin) * nv
            
            v = clamp(v, min1, max1)
            r = interpolate(normalize(v, min1, max1), min2, max2)
            r = clamp(r, min2, max2)
            return r
        
        o = context.active_object
        m = o.active_material
        n = m.node_tree.nodes.active
        im = n.image
        im.update()
        w, h = im.size
        
        a = np.asarray(im.pixels)
        a = a.reshape((h, w, 4))
        
        o.data.vertex_colors.new(name=im.name)
        bm = bmesh.new()
        bm.from_mesh(o.data)
        col_layer = bm.loops.layers.color.active
        uvl = bm.loops.layers.uv.active
        fs = bm.faces
        uv_layers = o.data.uv_layers
        
        prgs = Progress(len(fs), 1)
        for f in fs:
            prgs.step()
            ls = f.loops
            for l in ls:
                x, y = l[uvl].uv.to_tuple()
                xx = int(round(remap(x, 0.0, 1.0, 0, w - 1)))
                yy = int(round(remap(y, 0.0, 1.0, 0, h - 1)))
                l[col_layer] = a[yy][xx]
        
        bm.to_mesh(o.data)
        bm.free()
        log("done.", 1)
        
        return {'FINISHED'}


class CARBON_OT_vertex_group_to_vertex_colors(Operator):
    bl_idname = "carbon_tools.vertex_group_to_vertex_colors"
    bl_label = "Group > VCols"
    bl_description = "Active vertex group to new vertex colors, vertex weight to rgba(weight, weight, weight, 1.0)"
    bl_options = {'REGISTER', 'UNDO', }
    
    @classmethod
    def poll(cls, context):
        if(context.mode != 'OBJECT'):
            return False
        o = context.active_object
        if(not o):
            return False
        if(o.type != 'MESH'):
            return False
        if(not o.vertex_groups.active):
            return False
        return True
    
    def execute(self, context):
        log(self.bl_idname, 0)
        o = context.active_object
        me = o.data
        vg = o.vertex_groups.active
        
        # add zero to every vertex to have them all in group
        indexes = [i for i in range(len(me.vertices))]
        vg.add(indexes, 0.0, 'ADD')
        
        vc = me.vertex_colors.new(name=vg.name)
        me.vertex_colors.active = vc
        vcd = vc.data
        vcols = [None] * len(me.vertices)
        
        log("precalculating colors:", 1)
        prgs = Progress(len(me.vertices), 2)
        for v in me.vertices:
            prgs.step()
            i = v.index
            w = vg.weight(i)
            vcols[i] = (w, w, w, 1.0)
        
        log("assigning colors:", 1)
        prgs = Progress(len(me.loops), 2)
        for l in me.loops:
            prgs.step()
            li = l.index
            vi = l.vertex_index
            vcd[li].color = vcols[vi]
        
        log("done.", 1)
        
        return {'FINISHED'}


class CARBON_OT_vertex_colors_to_vertex_group(Operator):
    bl_idname = "carbon_tools.vertex_colors_to_vertex_group"
    bl_label = "VCols > Group"
    bl_description = "Active vertex colors to new vertex group, vertex weight by color perceived luminance"
    bl_options = {'REGISTER', 'UNDO', }
    
    @classmethod
    def poll(cls, context):
        if(context.mode != 'OBJECT'):
            return False
        o = context.active_object
        if(not o):
            return False
        if(o.type != 'MESH'):
            return False
        if(not o.data.vertex_colors.active):
            return False
        return True
    
    def execute(self, context):
        log(self.bl_idname, 0)
        o = context.active_object
        me = o.data
        vc = me.vertex_colors.active
        vs = me.vertices
        ls = me.loops
        
        vg = o.vertex_groups.new(name=vc.name)
        
        bm = bmesh.new()
        bm.from_mesh(me)
        
        lums = [None] * len(vs)
        cl = bm.loops.layers.color.active
        
        bm.verts.ensure_lookup_table()
        log("precalculating weights:", 1)
        prgs = Progress(len(bm.verts), 2)
        for i, v in enumerate(bm.verts):
            prgs.step()
            vi = v.index
            lums[vi] = []
            ll = v.link_loops
            for l in ll:
                r, g, b, _ = l[cl][:]
                # http://stackoverflow.com/questions/596216/formula-to-determine-brightness-of-rgb-color
                lum = math.sqrt(0.299 * r ** 2 + 0.587 * g ** 2 + 0.114 * b ** 2)
                lums[vi].append(lum)
        
        ws = [None] * len(vs)
        for i, a in enumerate(lums):
            ws[i] = sum(a) / float(len(a))
        
        log("assigning weights:", 1)
        prgs = Progress(len(vs), 2)
        for i, v in enumerate(vs):
            prgs.step()
            w = ws[i]
            # if l is None, means that vertex is not connected to any face > has no loops > has no color/weight
            if(l is not None):
                vg.add([i], w, 'REPLACE')
        
        bm.free()
        log("done.", 1)
        
        return {'FINISHED'}


class CARBON_OT_two_meshes_difference_to_vertex_group(Operator):
    bl_idname = "carbon_tools.two_meshes_difference_to_vertex_group"
    bl_label = "Difference > Group"
    bl_description = "Calculate difference between two selected meshes and write as vertex group to active mesh. Selected is consedered to be original, active to be modified. Objects should have the same transformation."
    bl_options = {'REGISTER', 'UNDO', }
    
    @classmethod
    def poll(cls, context):
        if(context.mode != 'OBJECT'):
            return False
        o = context.active_object
        if(not o):
            return False
        if(o.type != 'MESH'):
            return False
        obs = context.selected_objects
        if(len(obs) != 2):
            return False
        for o in obs:
            if(o.type != 'MESH'):
                return False
        return True
    
    def execute(self, context):
        log(self.bl_idname, 0)
        
        ctp = context.scene.carbon_tools
        
        modified = context.active_object
        original = [o for o in context.selected_objects if o is not modified][0]
        
        identical = True
        for i, r in enumerate(original.matrix_world):
            for j, c in enumerate(r):
                if(c != modified.matrix_world[i][j]):
                    identical = False
                    break
        
        if(not identical):
            log("Original object matrix and modified object matrix are not equal.", 1, )
            log("Will copy matrix from modified to original..", 1, )
            mos = repr(original.matrix_world).replace('        ', '').replace('\n', '')
            mms = repr(modified.matrix_world).replace('        ', '').replace('\n', '')
            log(mos, 1)
            log(mms, 1)
            original.matrix_world = modified.matrix_world
        
        m = modified.matrix_world
        max_search_distance = Utils.real_length_to_relative(m, ctp.diff_search_distance)
        log("search_distance: {0}, max_search_distance: {1}".format(ctp.diff_search_distance, max_search_distance), 1)
        
        log("calculating differences:", 1)
        mod_vs = [v.co for v in modified.data.vertices]
        mod_ns = [v.normal for v in modified.data.vertices]
        
        r = []
        prgr = Progress(len(mod_vs), 2)
        for i, v in enumerate(mod_vs):
            prgr.step()
            
            n = mod_ns[i]
            e0 = Utils.shift_vert_along_normal(v, n, -max_search_distance)
            e1 = Utils.shift_vert_along_normal(v, n, max_search_distance)
            pr = None
            nr = None
            
            if(ctp.diff_positive):
                _, loc, nor, ind = original.ray_cast(v, e1)
                if(ind != -1):
                    pr = (loc, nor, ind)
                    prd = Utils.distance_tuples(pr[0], v)
            
            if(ctp.diff_negative):
                _, loc, nor, ind = original.ray_cast(v, e0)
                if(ind != -1):
                    nr = (loc, nor, ind)
                    nrd = Utils.distance_tuples(nr[0], v)
            
            if(pr is not None and nr is not None):
                # hit in both directions
                # take closer point
                if(prd > nrd):
                    r.append((v, n, nr[0], nr[1], nr[2], nrd))
                else:
                    r.append((v, n, pr[0], pr[1], pr[2], -prd))
            elif(pr is not None):
                # hit positive
                r.append((v, n, pr[0], pr[1], pr[2], -prd))
            elif(nr is not None):
                # hit negative
                r.append((v, n, nr[0], nr[1], nr[2], nrd))
            else:
                # hit nothing..
                r.append((v, n, None, None, None, 0.0))
        
        log("adding vertex group..", 1)
        
        vg = modified.vertex_groups.new(name="Difference")
        indexes = [i for i in range(len(mod_vs))]
        modified.vertex_groups.active.add(indexes, 0.5, 'REPLACE')
        vg_index = modified.vertex_groups.active_index
        weights = [i[5] for i in r]
        
        arr = np.array(weights)
        mean = np.mean(arr)
        stdev = np.std(arr)
        log("mean: {0}, stdev: {1}".format(mean, stdev), 1)
        
        if(mean == 0.0 or stdev == 0.0):
            log("meshes looks identical..", 1)
            log("done.", 1)
            return {'FINISHED'}
        
        wmin = mean - (ctp.diff_sigma * stdev)
        wmax = mean + (ctp.diff_sigma * stdev)
        log("wmin: {0}, wmax: {1}".format(wmin, wmax), 1)
        
        log("calculating weights:", 1)
        prgr = Progress(len(mod_vs), 2)
        for i, v in enumerate(modified.data.vertices):
            w = weights[i]
            v.groups[vg_index].weight = Utils.map(w, wmin, wmax, 0.0, 1.0)
            prgr.step()
        
        log("done.", 1)
        
        return {'FINISHED'}


class CARBON_OT_calc_active_uv_coverage(Operator):
    bl_idname = "carbon_tools.calc_active_uv_coverage"
    bl_label = "UV Coverage"
    bl_description = ""
    bl_options = {'REGISTER', 'UNDO', }
    
    @classmethod
    def poll(cls, context):
        if(context.mode != 'OBJECT'):
            return False
        o = context.active_object
        if(not o):
            return False
        if(o.type != 'MESH'):
            return False
        if(not o.data.uv_layers.active):
            return False
        return True
    
    def execute(self, context):
        log(self.bl_idname, 0)
        
        o = context.active_object
        mesh = o.data
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bmesh.ops.triangulate(bm, faces=bm.faces)
        bm.faces.ensure_lookup_table()
        uvl = bm.loops.layers.uv.active
        total_area = 0.0
        for f in bm.faces:
            ls = f.loops
            uvv = []
            for l in ls:
                uvv.append(l[uvl].uv)
            a, b, c = uvv
            tri_area = ((a.x * (b.y - c.y)) + (b.x * (c.y - a.y)) + (c.x * (a.y - b.y))) / 2
            total_area += tri_area
        bm.free()
        
        self.report({'INFO'}, "Area used by UV: {:0.2f}%".format(total_area * 100))
        
        return {'FINISHED'}


class CARBON_properties(PropertyGroup):
    # utils_expanded: BoolProperty(default=False)
    # conversions_expanded: BoolProperty(default=False)
    
    extract_protect: BoolProperty(name="Protect Mesh Borders", default=False, description="Hide mesh borders to protect them from modification.", )
    dyntopo_resolution: IntProperty(name="Resolution", default=500, min=0, max=1000, description="Dyntopo constant resolution, this is just initial value. Change in Dyntopo settings afterwards.", )
    dyntopo_method: EnumProperty(name="Method", items=[('SUBDIVIDE', 'Subdivide', ''), ('COLLAPSE', 'Collapse', ''), ('SUBDIVIDE_COLLAPSE', 'Subdivide Collapse', ''), ], default='COLLAPSE', description="Dyntopo method, this is just initial value. Change in Dyntopo settings afterwards.", )
    tp_ext_resolution: EnumProperty(name="Resolution", items=[('512', '512', ''), ('1024', '1024', ''), ('2048', '2048', ''), ('4096', '4096', ''), ('8192', '8192', ''), ], default='2048', description="Image resolution presets", )
    end_keep_wire: BoolProperty(name="Keep Wireframe Display", default=False, description="Keep all wire display when finished with current procedure", )
    select_non_manifold_extra_auto_view: BoolProperty(name="Select Non-Manifold Extra Auto View", default=False, description="Center camera view on selected non-manifold elements", )
    export_uv_layout_resolution: EnumProperty(name="Resolution", items=[('1024', '1024', ''), ('2048', '2048', ''), ('4096', '4096', ''), ('8192', '8192', ''), ('16384', '16384', ''), ], default='8192', description="Image resolution presets", )
    
    diff_search_distance: FloatProperty(name="Distance", description="How far to look around for second object along vertex normal", default=10.0, precision=3, )
    diff_sigma: FloatProperty(name="Sigma", description="", default=1.0, precision=3, )
    diff_positive: BoolProperty(name="Positive", default=True, description="", )
    diff_negative: BoolProperty(name="Negative", default=True, description="", )
    
    version: StringProperty(name="Version", description="", default='.'.join([str(i) for i in bl_info['version']]), )
    
    @classmethod
    def register(cls):
        bpy.types.Scene.carbon_tools = PointerProperty(type=cls)
    
    @classmethod
    def unregister(cls):
        del bpy.types.Scene.carbon_tools


class CARBON_PT_carbon_tools(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'View'
    bl_label = 'Carbon Tools'
    bl_category = 'Carbon Tools'
    # bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        ctp = context.scene.carbon_tools
        l = self.layout
        
        c = l.column(align=True)
        c.label(text="Subtools")
        cc = c.column(align=True)
        r = cc.row(align=True)
        # r.scale_y = 1.5
        r.operator('carbon_tools.extract_mesh_part', text="Extract", )
        r.prop(ctp, 'extract_protect', toggle=True, text='', icon='LOCKED' if ctp.extract_protect else 'UNLOCKED', icon_only=True, )
        r = cc.row(align=True)
        # r.scale_y = 1.5
        r.operator('carbon_tools.insert_mesh_part', text="Insert", )
        if(context.mode in ('SCULPT', 'PAINT_TEXTURE', )):
            cc.active = False
        
        l.operator('carbon_tools.extract_non_manifold_elements', text="Extract Non-Manifold", )
        
        c = l.column(align=True)
        c.label(text="Dyntopo Setup")
        cc = c.column(align=True)
        r = cc.row(align=True)
        r.prop(ctp, 'dyntopo_resolution')
        r.prop(ctp, 'dyntopo_method', text='', )
        cc.operator('carbon_tools.quick_mesh_dyntopo_cleanup_setup', text="Dyntopo Setup", )
        if(context.mode != 'OBJECT'):
            cc.active = False
        
        ob = context.active_object
        toolsettings = context.tool_settings
        
        sculpt = toolsettings.sculpt
        c = l.column(align=True)
        c.label(text="Dyntopo Live Settings")
        cc = c.column(align=True)
        r = cc.row(align=True)
        r.prop(sculpt, 'constant_detail_resolution', )
        r.operator("sculpt.sample_detail_size", text="", icon='EYEDROPPER')
        cc.prop(sculpt, 'detail_refine_method', text="", )
        cc.prop(sculpt, 'detail_type_method', text="", )
        cc.enabled = False
        if(ob and ob.type == 'MESH' and context.mode == 'SCULPT'):
            if(context.sculpt_object.use_dynamic_topology_sculpting):
                cc.enabled = True
        
        c = l.column(align=True)
        c.label(text="Texture Paint Setup")
        cc = c.column(align=True)
        r = cc.row(align=True)
        r.operator('carbon_tools.quick_texture_painting_setup', text="TP Setup")
        r.prop(ctp, 'tp_ext_resolution', text="", )
        # if(context.mode != 'OBJECT'):
        #     cc.active = False
        if(ob and ob.type == 'MESH' and context.mode == 'OBJECT' and len(ob.data.uv_layers) > 0):
            cc.active = True
        else:
            cc.active = False
        
        ipaint = toolsettings.image_paint
        c = l.column(align=True)
        c.label(text="External TP Live Commands")
        cc = c.column(align=True)
        r = cc.row(align=True)
        # r.scale_y = 1.5
        r.operator("image.project_edit", text="Quick Edit")
        r.operator("image.project_apply", text="Apply")
        cc.operator('carbon_tools.save_all_images', text="Save All Images")
        cc.enabled = False
        if(ob and ob.type == 'MESH' and context.mode == 'PAINT_TEXTURE'):
            cc.enabled = True
        
        c = l.column(align=True)
        c.label(text="IO")
        c.operator('carbon_tools.import_obj_from_zbrush', text="Import from ZBrush")
        c.operator('carbon_tools.export_obj_to_zbrush', text="Export to ZBrush")
        
        c = l.column(align=True)
        c.operator('carbon_tools.copy_transformation', text="Transformation: Selected > Active")
        c.operator('carbon_tools.copy_original_matrix', text="Matrix: Selected > Active")
        c.operator('carbon_tools.export_obj_to_photoscan', text="Export to PhotoScan")
        
        l.separator()
        c = l.column(align=True)
        r = c.row(align=True)
        r.operator('carbon_tools.end_current_procedure', text="End", )
        r.prop(ctp, 'end_keep_wire', toggle=True, text='', icon='SHADING_WIRE', icon_only=True, )
        
        # l.separator()
        # b = l.box()
        # r = b.row()
        # r.prop(ctp, 'utils_expanded', icon='TRIA_DOWN' if ctp.utils_expanded else 'TRIA_RIGHT', icon_only=True, emboss=False, )
        # r.label(text="Utilities")
        # if(ctp.utils_expanded):
        #     c = b.column(align=True)
        #     c.operator('carbon_tools.calc_active_uv_coverage')
        #     c.operator('carbon_tools.mark_seams_from_uv_islands', text="Seams From Islands", )
        #     c.operator('carbon_tools.select_seams', text="Select Seams", )
        #     c.operator('carbon_tools.visualize_uv_seams_as_wireframe_mesh', text="Seams > Wireframe", )
        #
        #     r = c.row(align=True)
        #     r.operator('carbon_tools.export_uv_layout', text="Export UV Layout", )
        #     r.prop(ctp, 'export_uv_layout_resolution', text="", )
        #     if(ob and ob.type == 'MESH' and context.mode == 'OBJECT' and len(ob.data.uv_layers) > 0):
        #         r.active = True
        #     else:
        #         r.active = False
        #
        #     c.operator('carbon_tools.toggle_unselected_wireframe', text="Wireframe", )
        #
        #     r = c.row(align=True)
        #     r.operator('carbon_tools.select_non_manifold_extra')
        #     r.prop(ctp, 'select_non_manifold_extra_auto_view', toggle=True, text='', icon='HIDE_OFF' if ctp.select_non_manifold_extra_auto_view else 'HIDE_ON', icon_only=True, )
        
        # b = l.box()
        # r = b.row()
        # r.prop(ctp, 'conversions_expanded', icon='TRIA_DOWN' if ctp.conversions_expanded else 'TRIA_RIGHT', icon_only=True, emboss=False, )
        # r.label(text="Conversions")
        # if(ctp.conversions_expanded):
        #     c = b.column(align=True)
        #     c.label(text="Conversions")
        #     c.operator('carbon_tools.texture_to_vertex_colors')
        #     c.operator('carbon_tools.vertex_group_to_vertex_colors')
        #     c.operator('carbon_tools.vertex_colors_to_vertex_group')
        #
        #     c = b.column(align=True)
        #     c.prop(ctp, 'diff_search_distance')
        #     c.prop(ctp, 'diff_sigma')
        #     r = c.row(align=True)
        #     r.prop(ctp, 'diff_positive', toggle=True, )
        #     r.prop(ctp, 'diff_negative', toggle=True, )
        #     c.operator('carbon_tools.two_meshes_difference_to_vertex_group')
        #     c.enabled = CARBON_OT_two_meshes_difference_to_vertex_group.poll(context)
        
        r = l.row()
        r.label(text='Carbon Tools {}'.format(ctp.version))
        r.enabled = False


class CARBON_PT_carbon_tools_utils(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Carbon Tools'
    bl_label = "Utils"
    bl_parent_id = "CARBON_PT_carbon_tools"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        ctp = context.scene.carbon_tools
        ob = context.active_object
        
        l = self.layout
        sub = l.column()
        
        c = sub.column(align=True)
        c.operator('carbon_tools.calc_active_uv_coverage')
        c.operator('carbon_tools.mark_seams_from_uv_islands', text="Seams From Islands", )
        c.operator('carbon_tools.select_seams', text="Select Seams", )
        c.operator('carbon_tools.visualize_uv_seams_as_wireframe_mesh', text="Seams > Wireframe", )
        
        r = c.row(align=True)
        r.operator('carbon_tools.export_uv_layout', text="Export UV Layout", )
        r.prop(ctp, 'export_uv_layout_resolution', text="", )
        if(ob and ob.type == 'MESH' and context.mode == 'OBJECT' and len(ob.data.uv_layers) > 0):
            r.active = True
        else:
            r.active = False
        
        c.operator('carbon_tools.toggle_unselected_wireframe', text="Wireframe", )
        
        r = c.row(align=True)
        r.operator('carbon_tools.select_non_manifold_extra')
        r.prop(ctp, 'select_non_manifold_extra_auto_view', toggle=True, text='', icon='HIDE_OFF' if ctp.select_non_manifold_extra_auto_view else 'HIDE_ON', icon_only=True, )


class CARBON_PT_carbon_tools_conversions(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Carbon Tools'
    bl_label = "Conversions"
    bl_parent_id = "CARBON_PT_carbon_tools"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        ctp = context.scene.carbon_tools
        ob = context.active_object
        
        l = self.layout
        sub = l.column()
        
        c = sub.column(align=True)
        c.label(text="Conversions")
        c.operator('carbon_tools.texture_to_vertex_colors')
        c.operator('carbon_tools.vertex_group_to_vertex_colors')
        c.operator('carbon_tools.vertex_colors_to_vertex_group')
        
        c = sub.column(align=True)
        c.prop(ctp, 'diff_search_distance')
        c.prop(ctp, 'diff_sigma')
        r = c.row(align=True)
        r.prop(ctp, 'diff_positive', toggle=True, )
        r.prop(ctp, 'diff_negative', toggle=True, )
        c.operator('carbon_tools.two_meshes_difference_to_vertex_group')
        c.enabled = CARBON_OT_two_meshes_difference_to_vertex_group.poll(context)


def update_panel_bl_category(self, context, ):
    _main_panel = CARBON_PT_carbon_tools
    _sub_panels = (CARBON_PT_carbon_tools_utils, CARBON_PT_carbon_tools_conversions, )
    bpy.utils.unregister_class(_main_panel)
    for sp in _sub_panels:
        bpy.utils.unregister_class(sp)
    prefs = bpy.context.preferences.addons[__name__].preferences
    _main_panel.bl_category = prefs.tab_name
    bpy.utils.register_class(_main_panel)
    for sp in _sub_panels:
        bpy.utils.register_class(sp)


class CARBON_preferences(AddonPreferences):
    bl_idname = __name__
    tab_name: StringProperty(name="Tab Name", default="Carbon Tools", description="Custom tab name", update=update_panel_bl_category, )
    
    def draw(self, context):
        r = self.layout.row()
        r.prop(self, "tab_name")


classes = (
    CARBON_OT_quick_mesh_dyntopo_cleanup,
    CARBON_OT_extract_mesh_part,
    CARBON_OT_insert_mesh_part,
    CARBON_OT_extract_non_manifold_elements,
    CARBON_OT_copy_original_matrix,
    CARBON_OT_copy_transformation,
    CARBON_OT_export_obj_to_zbrush,
    CARBON_OT_import_obj_from_zbrush,
    CARBON_OT_export_obj_to_photoscan,
    CARBON_OT_quick_texture_painting_setup,
    CARBON_OT_save_all_images,
    CARBON_OT_end_current_procedure,
    CARBON_OT_select_seams,
    CARBON_OT_mark_seams_from_uv_islands,
    CARBON_OT_export_uv_layout,
    CARBON_OT_toggle_unselected_wireframe,
    CARBON_OT_visualize_uv_seams_as_wireframe_mesh,
    CARBON_OT_select_non_manifold_extra,
    CARBON_OT_texture_to_vertex_colors,
    CARBON_OT_vertex_group_to_vertex_colors,
    CARBON_OT_vertex_colors_to_vertex_group,
    CARBON_OT_two_meshes_difference_to_vertex_group,
    CARBON_OT_calc_active_uv_coverage,
    CARBON_properties,
    CARBON_preferences,
    CARBON_PT_carbon_tools,
    CARBON_PT_carbon_tools_utils,
    CARBON_PT_carbon_tools_conversions,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    
    update_panel_bl_category(None, bpy.context)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
