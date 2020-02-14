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

import numpy as np

import bpy
import bmesh
from bpy.types import Operator

from .debug import log, debug_mode
from .machine import PCVManager


class PCV_OT_edit_start(Operator):
    bl_idname = "point_cloud_visualizer.edit_start"
    bl_label = "Start"
    bl_description = "Start edit mode, create helper object and switch to it"
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        
        pcv = context.object.point_cloud_visualizer
        ok = False
        for k, v in PCVManager.cache.items():
            if(v['uuid'] == pcv.uuid):
                if(v['ready']):
                    if(v['draw']):
                        ok = True
        return ok
    
    def execute(self, context):
        pcv = context.object.point_cloud_visualizer
        c = PCVManager.cache[pcv.uuid]
        vs = c['vertices']
        ns = c['normals']
        cs = c['colors']
        
        # ensure object mode
        bpy.ops.object.mode_set(mode='OBJECT')
        
        # prepare mesh
        bm = bmesh.new()
        for v in vs:
            bm.verts.new(v)
        bm.verts.ensure_lookup_table()
        l = bm.verts.layers.int.new('pcv_indexes')
        for i in range(len(vs)):
            bm.verts[i][l] = i
        # add mesh to scene, activate
        nm = 'pcv_edit_mesh_{}'.format(pcv.uuid)
        me = bpy.data.meshes.new(nm)
        bm.to_mesh(me)
        bm.free()
        o = bpy.data.objects.new(nm, me)
        view_layer = context.view_layer
        collection = view_layer.active_layer_collection.collection
        collection.objects.link(o)
        p = context.object
        o.parent = p
        o.matrix_world = p.matrix_world.copy()
        bpy.ops.object.select_all(action='DESELECT')
        o.select_set(True)
        view_layer.objects.active = o
        # and set edit mode
        bpy.ops.object.mode_set(mode='EDIT')
        # set vertex select mode..
        bpy.ops.mesh.select_mode(use_extend=False, use_expand=False, type='VERT', )
        
        o.point_cloud_visualizer.edit_is_edit_uuid = pcv.uuid
        o.point_cloud_visualizer.edit_is_edit_mesh = True
        pcv.edit_initialized = True
        pcv.edit_pre_edit_alpha = pcv.global_alpha
        pcv.global_alpha = pcv.edit_overlay_alpha
        pcv.edit_pre_edit_display = pcv.display_percent
        pcv.display_percent = 100.0
        pcv.edit_pre_edit_size = pcv.point_size
        pcv.point_size = pcv.edit_overlay_size
        
        return {'FINISHED'}


class PCV_OT_edit_update(Operator):
    bl_idname = "point_cloud_visualizer.edit_update"
    bl_label = "Update"
    bl_description = "Update displayed cloud from edited mesh"
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        
        pcv = context.object.point_cloud_visualizer
        ok = False
        if(pcv.edit_is_edit_mesh):
            for k, v in PCVManager.cache.items():
                if(v['uuid'] == pcv.edit_is_edit_uuid):
                    if(v['ready']):
                        if(v['draw']):
                            if(context.mode == 'EDIT_MESH'):
                                ok = True
        return ok
    
    def execute(self, context):
        # get current data
        uuid = context.object.point_cloud_visualizer.edit_is_edit_uuid
        c = PCVManager.cache[uuid]
        vs = c['vertices']
        ns = c['normals']
        cs = c['colors']
        # extract edited data
        o = context.object
        bm = bmesh.from_edit_mesh(o.data)
        bm.verts.ensure_lookup_table()
        l = bm.verts.layers.int['pcv_indexes']
        edit_vs = []
        edit_indexes = []
        for v in bm.verts:
            edit_vs.append(v.co.to_tuple())
            edit_indexes.append(v[l])
        # combine
        u_vs = []
        u_ns = []
        u_cs = []
        for i, indx in enumerate(edit_indexes):
            u_vs.append(edit_vs[i])
            u_ns.append(ns[indx])
            u_cs.append(cs[indx])
        # display
        vs = np.array(u_vs, dtype=np.float32, )
        ns = np.array(u_ns, dtype=np.float32, )
        cs = np.array(u_cs, dtype=np.float32, )
        PCVManager.update(uuid, vs, ns, cs, )
        # update indexes
        bm.verts.ensure_lookup_table()
        l = bm.verts.layers.int['pcv_indexes']
        for i in range(len(vs)):
            bm.verts[i][l] = i
        
        return {'FINISHED'}


class PCV_OT_edit_end(Operator):
    bl_idname = "point_cloud_visualizer.edit_end"
    bl_label = "End"
    bl_description = "Update displayed cloud from edited mesh, stop edit mode and remove helper object"
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        
        pcv = context.object.point_cloud_visualizer
        ok = False
        if(pcv.edit_is_edit_mesh):
            for k, v in PCVManager.cache.items():
                if(v['uuid'] == pcv.edit_is_edit_uuid):
                    if(v['ready']):
                        if(v['draw']):
                            if(context.mode == 'EDIT_MESH'):
                                ok = True
        return ok
    
    def execute(self, context):
        # update
        bpy.ops.point_cloud_visualizer.edit_update()
        
        # cleanup
        bpy.ops.object.mode_set(mode='EDIT')
        o = context.object
        p = o.parent
        me = o.data
        view_layer = context.view_layer
        collection = view_layer.active_layer_collection.collection
        collection.objects.unlink(o)
        bpy.data.objects.remove(o)
        bpy.data.meshes.remove(me)
        # go back
        bpy.ops.object.select_all(action='DESELECT')
        p.select_set(True)
        view_layer.objects.active = p
        
        p.point_cloud_visualizer.edit_initialized = False
        p.point_cloud_visualizer.global_alpha = p.point_cloud_visualizer.edit_pre_edit_alpha
        p.point_cloud_visualizer.display_percent = p.point_cloud_visualizer.edit_pre_edit_display
        p.point_cloud_visualizer.point_size = p.point_cloud_visualizer.edit_pre_edit_size
        
        return {'FINISHED'}


class PCV_OT_edit_cancel(Operator):
    bl_idname = "point_cloud_visualizer.edit_cancel"
    bl_label = "Cancel"
    bl_description = "Stop edit mode, try to remove helper object and reload original point cloud"
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        
        pcv = context.object.point_cloud_visualizer
        if(pcv.edit_initialized):
            return True
        return False
    
    def execute(self, context):
        pcv = context.object.point_cloud_visualizer
        po = context.object
        nm = 'pcv_edit_mesh_{}'.format(pcv.uuid)
        view_layer = context.view_layer
        collection = view_layer.active_layer_collection.collection
        for o in po.children:
            if(o.name == nm):
                me = o.data
                collection.objects.unlink(o)
                bpy.data.objects.remove(o)
                bpy.data.meshes.remove(me)
                break
        
        pcv.edit_initialized = False
        pcv.global_alpha = pcv.edit_pre_edit_alpha
        pcv.edit_pre_edit_alpha = 0.5
        pcv.display_percent = pcv.edit_pre_edit_display
        pcv.edit_pre_edit_display = 100.0
        pcv.point_size = pcv.edit_pre_edit_size
        pcv.edit_pre_edit_size = 3
        
        # also beware, this changes uuid
        bpy.ops.point_cloud_visualizer.reload()
        
        return {'FINISHED'}
