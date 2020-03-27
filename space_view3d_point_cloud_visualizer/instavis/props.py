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
from bpy.props import PointerProperty, BoolProperty, StringProperty, FloatProperty, IntProperty, FloatVectorProperty, EnumProperty, CollectionProperty
from bpy.types import PropertyGroup

from .debug import debug_mode, log
from .mechanist import PCVIVMechanist


class PCVIV_preferences(PropertyGroup):
    
    def _switch_shader(self, context, ):
        PCVIVMechanist.cache = {}
        PCVIVMechanist.update()
    
    # shader quality, switch between basic pixel based and rich shaded geometry based, can be changed on the fly
    quality: EnumProperty(name="Quality", items=[('BASIC', "Basic", "Basic pixel point based shader with flat colors", ),
                                                 ('RICH', "Rich", "Rich billboard shader with phong shading", ),
                                                 ], default='RICH', description="Global quality settings for all", update=_switch_shader, )
    
    # exit display settings is used for file save and when instavis is deinitialized, just to prevent viewport slowdown
    use_exit_display: BoolProperty(name="Exit Display Setting Enabled", default=True, description="Switch display method/type of particles and instances objects when visualization is exited. When disabled, default values are used.", )
    exit_object_display_type: EnumProperty(name="Instanced Objects", items=[('BOUNDS', "Bounds", "", ), ('TEXTURED', "Textured", "", ), ], default='TEXTURED', description="To what set instance base objects Display Type when point cloud mode is exited", )
    exit_psys_display_method: EnumProperty(name="Particle Systems", items=[('NONE', "None", "", ), ('RENDER', "Render", "", ), ], default='RENDER', description="To what set particles system Display Method when point cloud mode is exited", )
    
    switch_origins_only: BoolProperty(name="Switch To Origins Only", default=True, description="Switch display to Origins Only for high instance counts", )
    switch_origins_only_threshold: IntProperty(name="Threshold", default=10000, min=1, max=2 ** 31 - 1, description="Switch display to Origins Only when instance count exceeds this value", )
    
    def _switch_update_method(self, context, ):
        if(PCVIVMechanist.initialized):
            PCVIVMechanist.deinit()
            PCVIVMechanist.init()
    
    update_method: EnumProperty(name="Update Method", items=[('MSGBUS', "MSGBUS", "Update using 'msgbus.subscribe_rna'", ),
                                                             ('DEPSGRAPH', "DEPSGRAPH", "Update using 'app.handlers.depsgraph_update_pre/post'", ),
                                                             ], default='MSGBUS', description="Switch update method of point cloud instance visualization", update=_switch_update_method, )
    
    @classmethod
    def register(cls):
        bpy.types.Scene.pcv_instavis = PointerProperty(type=cls)
    
    @classmethod
    def unregister(cls):
        del bpy.types.Scene.pcv_instavis


class PCVIV_psys_properties(PropertyGroup):
    # global point scale for all points, handy when points get too small to be visible, but you still want to keep different sizes per object
    point_scale: FloatProperty(name="Point Scale", default=1.0, min=0.001, max=10.0, description="Adjust point size of all points", precision=6, )
    # drawing on/off, monitor icon on particles system works too and is more general, so this is just some future special case if needed..
    draw: BoolProperty(name="Draw", default=True, description="Draw point cloud to viewport", )
    display: FloatProperty(name="Display", default=100.0, min=0.0, max=100.0, precision=0, subtype='PERCENTAGE', description="Adjust percentage of displayed instances", )
    
    use_origins_only: BoolProperty(name="Draw Origins Only", default=False, description="Draw only instance origins in a single draw pass", )
    origins_point_size: IntProperty(name="Size (Basic Shader)", default=6, min=1, max=10, subtype='PIXEL', description="Point size", )
    origins_point_size_f: FloatProperty(name="Size (Rich Shader)", default=0.05, min=0.001, max=1.0, description="Point size", precision=6, )
    
    def _use_update(self, context, ):
        if(not self.use):
            prefs = context.scene.pcv_instavis
            if(not prefs.use_exit_display):
                return
            pset = self.id_data
            pset.display_method = prefs.exit_psys_display_method
            if(pset.render_type == 'COLLECTION'):
                col = pset.instance_collection
                if(col is not None):
                    for co in col.objects:
                        co.display_type = prefs.exit_object_display_type
            elif(pset.render_type == 'OBJECT'):
                co = pset.instance_object
                if(co is not None):
                    co.display_type = prefs.exit_object_display_type
    
    # use: BoolProperty(default=False, options={'HIDDEN', }, update=_use_update, )
    use: BoolProperty(name="Use", default=False, description="Enable/disable instance visualization", update=_use_update, )
    
    @classmethod
    def register(cls):
        bpy.types.ParticleSettings.pcv_instavis = PointerProperty(type=cls)
    
    @classmethod
    def unregister(cls):
        del bpy.types.ParticleSettings.pcv_instavis


class PCVIV_object_properties(PropertyGroup):
    def _invalidate_object_cache(self, context, ):
        PCVIVMechanist.invalidate_object_cache(self.id_data.name)
    
    source: EnumProperty(name="Source", items=[('POLYGONS', "Polygons", "Mesh Polygons (constant or material viewport display color)"),
                                               ('VERTICES', "Vertices", "Mesh Vertices (constant color only)"),
                                               ], default='POLYGONS', description="Point cloud generation source", update=_invalidate_object_cache, )
    # max_points: IntProperty(name="Max. Points", default=100, min=1, max=10000, description="Maximum number of points per instance", update=_invalidate_object_cache, )
    max_points: IntProperty(name="Max. Points", default=500, min=1, max=10000, description="Maximum number of points per instance", update=_invalidate_object_cache, )
    color_source: EnumProperty(name="Color Source", items=[('CONSTANT', "Constant Color", "Use constant color value"),
                                                           ('OBJECT_VIEWPORT_DISPLAY_COLOR', "Object Viewport Display Color", "Use object viewport display color property"),
                                                           ('MATERIAL_VIEWPORT_DISPLAY_COLOR', "Material Viewport Display Color", "Use material viewport display color property"),
                                                           ], default='MATERIAL_VIEWPORT_DISPLAY_COLOR', description="Color source for generated point cloud", update=_invalidate_object_cache, )
    color_constant: FloatVectorProperty(name="Color", description="Constant color", default=(0.7, 0.7, 0.7, ), min=0, max=1, subtype='COLOR', size=3, update=_invalidate_object_cache, )
    
    use_face_area: BoolProperty(name="Use Face Area", default=True, description="Use mesh face area as probability factor during point cloud generation", update=_invalidate_object_cache, )
    use_material_factors: BoolProperty(name="Use Material Factors", default=False, description="Use material probability factor during point cloud generation", update=_invalidate_object_cache, )
    
    # point_size is for basic shader, point_size_f if for rich shader
    # point_size: IntProperty(name="Size (Basic Shader)", default=6, min=1, max=10, subtype='PIXEL', description="Point size", )
    point_size: IntProperty(name="Size (Basic Shader)", default=3, min=1, max=10, subtype='PIXEL', description="Point size", )
    # point_size_f: FloatProperty(name="Size (Rich Shader)", default=0.02, min=0.001, max=1.0, description="Point size", precision=6, )
    point_size_f: FloatProperty(name="Size (Rich Shader)", default=0.01, min=0.001, max=1.0, description="Point size", precision=6, )
    
    # def _target_update(self, context, ):
    #     if(not self.target):
    #         # if target is set to False, swap display to exit types
    #         prefs = context.scene.pcv_instavis
    #         if(not prefs.use_exit_display):
    #             return
    #         o = self.id_data
    #         ls = [ps.settings for ps in o.particle_systems]
    #         for pset in ls:
    #             pset.display_method = prefs.exit_psys_display_method
    #             if(pset.render_type == 'COLLECTION'):
    #                 col = pset.instance_collection
    #                 if(col is not None):
    #                     for co in col.objects:
    #                         co.display_type = prefs.exit_object_display_type
    #             elif(pset.render_type == 'OBJECT'):
    #                 co = pset.instance_object
    #                 if(co is not None):
    #                     co.display_type = prefs.exit_object_display_type
    #
    # target: BoolProperty(default=False, options={'HIDDEN', }, update=_target_update, )
    
    @classmethod
    def register(cls):
        bpy.types.Object.pcv_instavis = PointerProperty(type=cls)
    
    @classmethod
    def unregister(cls):
        del bpy.types.Object.pcv_instavis


class PCVIV_collection_properties(PropertyGroup):
    active_index: IntProperty(name="Index", default=0, description="", options={'HIDDEN', }, )
    
    @classmethod
    def register(cls):
        bpy.types.Collection.pcv_instavis = PointerProperty(type=cls)
    
    @classmethod
    def unregister(cls):
        del bpy.types.Collection.pcv_instavis


class PCVIV_material_properties(PropertyGroup):
    
    def _invalidate_object_cache(self, context, ):
        m = self.id_data
        for o in bpy.data.objects:
            for s in o.material_slots:
                if(s.material == m):
                    if(o.pcv_instavis.use_material_factors):
                        PCVIVMechanist.invalidate_object_cache(o.name)
    
    # this serves as material weight value for polygon point generator, higher value means that it is more likely for polygon to be used as point source
    factor: FloatProperty(name="Factor", default=0.5, min=0.001, max=1.0, precision=3, subtype='FACTOR', description="Probability factor of choosing polygon with this material", update=_invalidate_object_cache, )
    
    @classmethod
    def register(cls):
        bpy.types.Material.pcv_instavis = PointerProperty(type=cls)
    
    @classmethod
    def unregister(cls):
        del bpy.types.Material.pcv_instavis


classes = (PCVIV_preferences, PCVIV_psys_properties, PCVIV_object_properties, PCVIV_material_properties, PCVIV_collection_properties, )
classes_debug = ()
