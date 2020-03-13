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
from bpy.types import Operator

from .debug import debug_mode, log
from .mechanist import PCVIVMechanist


class PCVIV_OT_init(Operator):
    bl_idname = "point_cloud_visualizer.pcviv_init"
    bl_label = "Initialize"
    bl_description = "Initialize Instance Visualizer"
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        if(PCVIVMechanist.initialized):
            return False
        return True
    
    def execute(self, context):
        PCVIVMechanist.init()
        return {'FINISHED'}


class PCVIV_OT_deinit(Operator):
    bl_idname = "point_cloud_visualizer.pcviv_deinit"
    bl_label = "Deinitialize"
    bl_description = "Deinitialize Instance Visualizer"
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        if(not PCVIVMechanist.initialized):
            return False
        return True
    
    def execute(self, context):
        PCVIVMechanist.deinit()
        return {'FINISHED'}


class PCVIV_OT_force_update(Operator):
    bl_idname = "point_cloud_visualizer.pcviv_force_update"
    bl_label = "Force Update All"
    bl_description = "Force update all particle systems drawing"
    
    @classmethod
    def poll(cls, context):
        if(not PCVIVMechanist.initialized):
            return False
        return True
    
    def execute(self, context):
        PCVIVMechanist.force_update(with_caches=True, )
        return {'FINISHED'}


class PCVIV_OT_sync_instance_settings(Operator):
    bl_idname = "point_cloud_visualizer.pcviv_sync_instance_settings"
    bl_label = "Synchronize Settings To All"
    bl_description = "Synchronize point cloud generation settings to all in current collection"
    
    @classmethod
    def poll(cls, context):
        ok = False
        if(context.object is not None):
            o = context.object
            if(o.particle_systems.active is not None):
                pset = o.particle_systems.active.settings
                if(pset.render_type == 'COLLECTION'):
                    if(pset.instance_collection is not None):
                        ok = True
        return ok
    
    def execute(self, context):
        col = context.object.particle_systems.active.settings.instance_collection
        ao = col.objects[col.pcv_instavis.active_index]
        aoset = ao.pcv_instavis
        changed = []
        for o in col.objects:
            if(o is ao):
                continue
            ps = o.pcv_instavis
            ps.source = aoset.source
            ps.max_points = aoset.max_points
            ps.color_source = aoset.color_source
            ps.color_constant = aoset.color_constant
            ps.use_face_area = aoset.use_face_area
            ps.use_material_factors = aoset.use_material_factors
            ps.point_size = aoset.point_size
            ps.point_size_f = aoset.point_size_f
            changed.append(o)
        
        for o in changed:
            PCVIVMechanist.invalidate_object_cache(o.name)
        
        return {'FINISHED'}


class PCVIV_OT_sync_psys_settings(Operator):
    bl_idname = "point_cloud_visualizer.pcviv_sync_psys_settings"
    bl_label = "Synchronize Settings To All"
    bl_description = "Synchronize visualization settings to all particle systems on current target"
    
    @classmethod
    def poll(cls, context):
        ok = False
        if(context.object is not None):
            o = context.object
            if(o.particle_systems.active is not None):
                if(len(o.particle_systems) > 1):
                    ok = True        
        return ok
    
    def execute(self, context):
        o = context.object
        apsys = o.particle_systems.active
        apset = apsys.settings
        apcviv = apset.pcv_instavis
        update = False
        for psys in o.particle_systems:
            if(psys == apsys):
                continue
            pset = psys.settings
            pcviv = pset.pcv_instavis
            pcviv.point_scale = apcviv.point_scale
            pcviv.draw = apcviv.draw
            pcviv.display = apcviv.display
            pcviv.use_origins_only = apcviv.use_origins_only
            update = True
        
        if(update):
            PCVIVMechanist.force_update(with_caches=False, )
        
        return {'FINISHED'}


class PCVIV_OT_invalidate_caches(Operator):
    bl_idname = "point_cloud_visualizer.pcviv_invalidate_caches"
    bl_label = "Invalidate All Caches"
    bl_description = "Force refresh of all point caches"
    
    @classmethod
    def poll(cls, context):
        return True
    
    def execute(self, context):
        PCVIVMechanist.cache = {}
        PCVIVMechanist.update()
        return {'FINISHED'}


class PCVIV_OT_reset_viewport_draw(Operator):
    bl_idname = "point_cloud_visualizer.pcviv_reset_viewport_draw"
    bl_label = "Reset Viewport Draw Settings"
    bl_description = "Reset all viewport draw settings for all objects and particle systems in scene to defaults, in case something is meesed up after deinitialize"
    
    @classmethod
    def poll(cls, context):
        return True
    
    def execute(self, context):
        for o in context.scene.objects:
            o.display_type = 'TEXTURED'
            if(len(o.particle_systems)):
                for p in o.particle_systems:
                    p.settings.display_method = 'RENDER'
        
        return {'FINISHED'}


classes = ()
classes_debug = (PCVIV_OT_init, PCVIV_OT_deinit, PCVIV_OT_force_update, PCVIV_OT_sync_instance_settings, PCVIV_OT_reset_viewport_draw, PCVIV_OT_invalidate_caches, PCVIV_OT_sync_psys_settings, )
