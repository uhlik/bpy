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

import uuid
import time
import datetime
import random
import numpy as np

import bpy
import bmesh
from bpy.props import PointerProperty, BoolProperty, StringProperty, FloatProperty, IntProperty, FloatVectorProperty, EnumProperty, CollectionProperty
from bpy.types import PropertyGroup, Panel, Operator, UIList
from mathutils import Matrix, Vector, Quaternion, Color
from mathutils.geometry import barycentric_transform
from mathutils.interpolate import poly_3d_calc
from bpy.app.handlers import persistent
import bgl
from gpu.types import GPUShader, GPUBatch, GPUVertBuf, GPUVertFormat
from gpu_extras.batch import batch_for_shader

from .debug import log, debug_mode
from .machine import load_shader_code


class PCVIV3Config():
    # shuffling points in mesh sampler is very slow, PCV display percentage won't work as expected if points are not shuffled, but for instavis is not that important
    sampler_shuffle = False
    # default mesh sampler point color
    sampler_constant_color = (1.0, 0.0, 1.0, )
    # used when mesh data is not available, like when face sampler is used, but mesh has no faces, or vertex sampler with empty mesh
    sampler_error_color = (1.0, 0.0, 1.0, )


class PCVIV3FacesSampler():
    def __init__(self, target, count=-1, seed=0, colorize=None, constant_color=None, use_face_area=None, use_material_factors=None, ):
        if(colorize is None):
            colorize = 'CONSTANT'
        if(constant_color is None):
            constant_color = PCVIV3Config.sampler_constant_color
        if(use_material_factors is None and colorize == 'VIEWPORT_DISPLAY_COLOR'):
            use_material_factors = False
        if(use_material_factors):
            if(colorize == 'CONSTANT'):
                use_material_factors = False
        
        me = target.data
        
        if(len(me.polygons) == 0):
            # no polygons to generate from, use origin
            self.vs = np.array(((0.0, 0.0, 0.0, ), ), dtype=np.float32, )
            self.ns = np.array(((0.0, 0.0, 1.0, ), ), dtype=np.float32, )
            self.cs = np.array((PCVIV3Config.sampler_error_color, ), dtype=np.float32, )
            return
        if(colorize == 'VIEWPORT_DISPLAY_COLOR'):
            if(len(target.data.materials) == 0):
                # no materials, set to constant
                colorize = 'CONSTANT'
                constant_color = PCVIV3Config.sampler_error_color
                use_material_factors = False
            materials = target.data.materials
            if(None in materials[:]):
                # if there is empty slot, abort it and set to constant
                # TODO: make some workaround empty slots, this would require check for polygons with that empty material assigned and replacing that with constant color
                colorize = 'CONSTANT'
                constant_color = PCVIV3Config.sampler_error_color
                use_material_factors = False
        
        l = len(me.polygons)
        if(count == -1):
            count = l
        if(count > l):
            count = l
        
        np.random.seed(seed=seed, )
        
        centers = np.zeros((l * 3), dtype=np.float32, )
        me.polygons.foreach_get('center', centers, )
        centers.shape = (l, 3)
        
        normals = np.zeros((l * 3), dtype=np.float32, )
        me.polygons.foreach_get('normal', normals, )
        normals.shape = (l, 3)
        
        # TODO: following block can be skipped in some cases, like when requested point count is greater or equal to polygon count
        choices = np.indices((l, ), dtype=np.int, )
        choices.shape = (l, )
        weights = np.zeros(l, dtype=np.float32, )
        me.polygons.foreach_get('area', weights, )
        # make it all sum to 1.0
        weights *= 1.0 / np.sum(weights)
        indices = np.random.choice(choices, size=count, replace=False, p=weights, )
        
        if(colorize == 'VIEWPORT_DISPLAY_COLOR'):
            material_indices = np.zeros(l, dtype=np.int, )
            me.polygons.foreach_get('material_index', material_indices, )
            material_colors = np.zeros((len(materials), 3), dtype=np.float32, )
            material_factors = np.zeros((len(materials)), dtype=np.float32, )
            for i, m in enumerate(materials):
                mc = m.diffuse_color[:3]
                material_colors[i][0] = mc[0] ** (1 / 2.2)
                material_colors[i][1] = mc[1] ** (1 / 2.2)
                material_colors[i][2] = mc[2] ** (1 / 2.2)
                material_factors[i] = m.pcv_instance_visualizer3.factor
        
        if(use_material_factors):
            material_weights = np.take(material_factors, material_indices, axis=0, )
            material_weights *= 1.0 / np.sum(material_weights)
            if(use_face_area):
                weights = (weights + material_weights) / 2.0
            else:
                weights = material_weights
            indices = np.random.choice(choices, size=count, replace=False, p=weights, )
        
        li = len(indices)
        if(colorize == 'CONSTANT'):
            colors = np.column_stack((np.full(l, constant_color[0] ** (1 / 2.2), dtype=np.float32, ),
                                      np.full(l, constant_color[1] ** (1 / 2.2), dtype=np.float32, ),
                                      np.full(l, constant_color[2] ** (1 / 2.2), dtype=np.float32, ), ))
        elif(colorize == 'VIEWPORT_DISPLAY_COLOR'):
            colors = np.zeros((li, 3), dtype=np.float32, )
            colors = np.take(material_colors, material_indices, axis=0, )
        
        if(l == count):
            vs = centers
            ns = normals
            cs = colors
        else:
            vs = np.take(centers, indices, axis=0, )
            ns = np.take(normals, indices, axis=0, )
            cs = np.take(colors, indices, axis=0, )
        
        if(PCVIV3Config.sampler_shuffle):
            a = np.concatenate((vs, ns, cs), axis=1, )
            np.random.shuffle(a)
            vs = a[:, :3]
            ns = a[:, 3:6]
            cs = a[:, 6:]
        
        self.vs = vs
        self.ns = ns
        self.cs = cs


class PCVIV3VertsSampler():
    def __init__(self, target, count=-1, seed=0, constant_color=None, ):
        # NOTE: material display color is not useable here, material is assigned to polygons. checking each vertex for its polygon (in fact there can be many of them) will be cpu intensive, so leave vertex sampler for case when user wants fast results or need to see vertices because of low poly geometry..
        if(constant_color is None):
            constant_color = PCVIV3Config.sampler_constant_color
        
        me = target.data
        
        if(len(me.vertices) == 0):
            # no vertices to generate from, use origin
            self.vs = np.array(((0.0, 0.0, 0.0, ), ), dtype=np.float32, )
            self.ns = np.array(((0.0, 0.0, 1.0, ), ), dtype=np.float32, )
            self.cs = np.array((PCVIV3Config.sampler_error_color, ), dtype=np.float32, )
            return
        
        l = len(me.vertices)
        if(count == -1 or count > l):
            count = l
        
        locations = np.zeros((l * 3), dtype=np.float32, )
        me.vertices.foreach_get('co', locations, )
        locations.shape = (l, 3)
        
        normals = np.zeros((l * 3), dtype=np.float32, )
        me.vertices.foreach_get('normal', normals, )
        normals.shape = (l, 3)
        
        gc = [c ** (1 / 2.2) for c in constant_color]
        colors = np.column_stack((np.full(l, gc[0], dtype=np.float32, ),
                                  np.full(l, gc[1], dtype=np.float32, ),
                                  np.full(l, gc[2], dtype=np.float32, ), ))
        
        if(l == count):
            vs = locations
            ns = normals
            cs = colors
        else:
            # randomize points only when needed, when not using all vertices
            choices = np.indices((l, ), dtype=np.int, )
            choices.shape = (l, )
            np.random.seed(seed=seed, )
            indices = np.random.choice(choices, size=count, replace=False, )
            
            vs = np.take(locations, indices, axis=0, )
            ns = np.take(normals, indices, axis=0, )
            cs = np.take(colors, indices, axis=0, )
        
        if(PCVIV3Config.sampler_shuffle):
            a = np.concatenate((vs, ns, cs), axis=1, )
            np.random.shuffle(a)
            vs = a[:, :3]
            ns = a[:, 3:6]
            cs = a[:, 6:]
        
        self.vs = vs
        self.ns = ns
        self.cs = cs


class PCVIV3Manager():
    initialized = False
    
    registry = {}
    cache = {}
    flag = False
    buffer = []
    handle = None
    stats = 0
    
    cache_auto_update = True
    
    @classmethod
    def init(cls):
        if(cls.initialized):
            return
        log("init", prefix='>>>', )
        
        bpy.app.handlers.depsgraph_update_post.append(cls.depsgraph_update_post)
        
        bpy.app.handlers.load_pre.append(watcher)
        cls.initialized = True
        
        cls.handle = bpy.types.SpaceView3D.draw_handler_add(cls.draw, (), 'WINDOW', 'POST_VIEW')
        cls._redraw_view_3d()
        
        scene = bpy.context.scene
        depsgraph = bpy.context.evaluated_depsgraph_get()
        cls.depsgraph_update_post(scene, depsgraph, )
    
    @classmethod
    def deinit(cls):
        if(not cls.initialized):
            return
        log("deinit", prefix='>>>', )
        
        bpy.app.handlers.depsgraph_update_post.remove(cls.depsgraph_update_post)
        
        bpy.app.handlers.load_pre.remove(watcher)
        cls.initialized = False
        
        bpy.types.SpaceView3D.draw_handler_remove(cls.handle, 'WINDOW')
        cls._redraw_view_3d()
    
    @classmethod
    def register(cls, o, psys, ):
        pset = psys.settings
        pcviv = pset.pcv_instance_visualizer3
        if(pcviv.uuid == ''):
            pcviv.uuid = str(uuid.uuid1())
        else:
            raise Exception('{} . register() uuid exists, add some logic to handle that..'.format(cls.__class__.__name__))
        cls.registry[pcviv.uuid] = {'object': o, 'psys': psys, }
    
    @classmethod
    def depsgraph_update_post(cls, scene, depsgraph, ):
        # log("update!", prefix='>>>', )
        cls.update(depsgraph)
    
    @classmethod
    def update(cls, depsgraph, ):
        _t = time.time()
        
        # # NOTE: artificial updates (i need at least one when starting) are not detected
        # a = []
        # all_types = ['ACTION', 'ARMATURE', 'BRUSH', 'CAMERA', 'CACHEFILE', 'CURVE', 'FONT', 'GREASEPENCIL', 'COLLECTION', 'IMAGE', 'KEY', 'LIGHT', 'LIBRARY', 'LINESTYLE', 'LATTICE', 'MASK', 'MATERIAL', 'META', 'MESH', 'MOVIECLIP', 'NODETREE', 'OBJECT', 'PAINTCURVE', 'PALETTE', 'PARTICLE', 'LIGHT_PROBE', 'SCENE', 'SOUND', 'SPEAKER', 'TEXT', 'TEXTURE', 'WINDOWMANAGER', 'WORLD', 'WORKSPACE']
        # for t in all_types:
        #     a.append((t, depsgraph.id_type_updated(t), ))
        # print(a)
        
        # # NOTE: with this i could filter out updates i don't need, but lets keep it simple for now, react on all updates
        # hit = False
        # types = ['CAMERA', 'CURVE', 'COLLECTION', 'IMAGE', 'LIBRARY', 'MATERIAL', 'MESH', 'OBJECT', 'PARTICLE', 'SCENE', 'TEXTURE', ]
        # for t in types:
        #     hit = depsgraph.id_type_updated(t)
        #     if(hit):
        #         break
        # if(not hit):
        #     return
        
        if(cls.flag):
            return
        cls.flag = True
        
        registered = tuple([v['psys'] for k, v in cls.registry.items()])
        
        def pre():
            for o in bpy.data.objects:
                if(len(o.particle_systems) > 0):
                    for psys in o.particle_systems:
                        # skip unregistered systems
                        if(psys not in registered):
                            continue
                        
                        pset = psys.settings
                        if(pset.render_type == 'COLLECTION'):
                            col = pset.instance_collection
                            for co in col.objects:
                                co.display_type = 'BOUNDS'
                        elif(pset.render_type == 'OBJECT'):
                            co = pset.instance_object
                            co.display_type = 'BOUNDS'
                        pset.display_method = 'RENDER'
        
        def post():
            for o in bpy.data.objects:
                if(len(o.particle_systems) > 0):
                    for psys in o.particle_systems:
                        # skip unregistered systems
                        if(psys not in registered):
                            continue
                        
                        pset = psys.settings
                        pset.display_method = 'NONE'
                        if(pset.render_type == 'COLLECTION'):
                            col = pset.instance_collection
                            for co in col.objects:
                                co.display_type = 'TEXTURED'
                        elif(pset.render_type == 'OBJECT'):
                            co = pset.instance_object
                            co.display_type = 'TEXTURED'
        
        pre()
        
        depsgraph.update()
        
        buffer = [None] * len(depsgraph.object_instances)
        c = 0
        registered_uuids = tuple(cls.registry.keys())
        cls.stats = 0
        
        for instance in depsgraph.object_instances:
            if(instance.is_instance):
                ipsys = instance.particle_system
                ipset = ipsys.settings
                ipcviv = ipset.pcv_instance_visualizer3
                iuuid = ipcviv.uuid
                if(iuuid in registered_uuids):
                    m = instance.matrix_world.copy()
                    base = instance.object
                    instance_options = base.pcv_instance_visualizer3
                    
                    if(base.name not in cls.cache.keys()):
                        if(instance_options.source == 'VERTICES'):
                            sampler = PCVIV3VertsSampler(base,
                                                         count=instance_options.max_points,
                                                         seed=0,
                                                         constant_color=instance_options.color_constant, )
                        else:
                            sampler = PCVIV3FacesSampler(base,
                                                         count=instance_options.max_points,
                                                         seed=0,
                                                         colorize=instance_options.color_source,
                                                         constant_color=instance_options.color_constant,
                                                         use_face_area=instance_options.use_face_area,
                                                         use_material_factors=instance_options.use_material_factors, )
                        
                        vs, ns, cs = (sampler.vs, sampler.ns, sampler.cs, )
                        
                        if(ipcviv.quality == 'BASIC'):
                            # TODO: caching results of load_shader_code to memory, skip disk i/o
                            shader_data_vert, shader_data_frag, shader_data_geom = load_shader_code('instavis3_basic')
                            shader = GPUShader(shader_data_vert, shader_data_frag, )
                            batch = batch_for_shader(shader, 'POINTS', {"position": vs, "color": cs, }, )
                        else:
                            # TODO: caching results of load_shader_code to memory, skip disk i/o
                            shader_data_vert, shader_data_frag, shader_data_geom = load_shader_code('instavis3_rich')
                            shader = GPUShader(shader_data_vert, shader_data_frag, geocode=shader_data_geom, )
                            batch = batch_for_shader(shader, 'POINTS', {"position": vs, "normal": ns, "color": cs, }, )
                        
                        cls.cache[base.name] = (vs, ns, cs, shader, batch, )
                    else:
                        vs, ns, cs, shader, batch = cls.cache[base.name]
                    
                    if(ipcviv.quality == 'BASIC'):
                        draw_size = instance_options.point_size
                        draw_quality = 0
                    else:
                        draw_size = instance_options.point_size_f
                        draw_quality = 1
                    
                    buffer[c] = (draw_quality, shader, batch, m, draw_size, )
                    c += 1
                    cls.stats += len(vs)
        
        buffer = list(filter(None, buffer))
        cls.buffer = buffer
        
        post()
        cls.flag = False
        
        _d = datetime.timedelta(seconds=time.time() - _t)
        log("update: {}".format(_d), prefix='>>>', )
    
    @classmethod
    def draw(cls):
        # _t = time.time()
        
        bgl.glEnable(bgl.GL_PROGRAM_POINT_SIZE)
        bgl.glEnable(bgl.GL_DEPTH_TEST)
        # bgl.glEnable(bgl.GL_BLEND)
        
        # bgl.glEnable(bgl.GL_CULL_FACE)
        # # bgl.glCullFace(bgl.GL_BACK)
        
        buffer = cls.buffer
        
        for quality, shader, batch, matrix, size in buffer:
            shader.bind()
            
            if(quality == 0):
                
                perspective_matrix = bpy.context.region_data.perspective_matrix
                
                shader.uniform_float("perspective_matrix", perspective_matrix)
                shader.uniform_float("object_matrix", matrix)
                shader.uniform_float("size", size)
                shader.uniform_float("alpha", 1.0)
            else:
                
                view_matrix = bpy.context.region_data.view_matrix
                window_matrix = bpy.context.region_data.window_matrix
                light = view_matrix.copy().inverted()
                
                shader.uniform_float("model_matrix", matrix)
                shader.uniform_float("view_matrix", view_matrix)
                shader.uniform_float("window_matrix", window_matrix)
                shader.uniform_float("size", size)
                shader.uniform_float("alpha", 1.0)
                shader.uniform_float("light_position", light.translation)
                shader.uniform_float("light_color", (0.8, 0.8, 0.8, ))
                shader.uniform_float("view_position", light.translation)
                shader.uniform_float("ambient_strength", 0.5)
                shader.uniform_float("specular_strength", 0.5)
                shader.uniform_float("specular_exponent", 8.0)
            
            batch.draw(shader)
        
        bgl.glDisable(bgl.GL_PROGRAM_POINT_SIZE)
        bgl.glDisable(bgl.GL_DEPTH_TEST)
        # bgl.glDisable(bgl.GL_BLEND)
        
        # bgl.glDisable(bgl.GL_CULL_FACE)
        
        # _d = datetime.timedelta(seconds=time.time() - _t)
        # log("draw: {}".format(_d), prefix='>>>', )
    
    @classmethod
    def invalidate_cache(cls, name, ):
        if(not cls.cache_auto_update):
            return False
        
        if(name in PCVIV3Manager.cache.keys()):
            del PCVIV3Manager.cache[name]
            return True
        else:
            return False
    
    @classmethod
    def _redraw_view_3d(cls):
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if(area.type == 'VIEW_3D'):
                    area.tag_redraw()


@persistent
def watcher(scene):
    PCVIV3Manager.deinit()


class PCVIV3_psys_properties(PropertyGroup):
    # this is going to be assigned during runtime by manager if it detects new psys creation on depsgraph update
    uuid: StringProperty(default="", options={'HIDDEN', }, )
    
    def _draw_update(self, context, ):
        # PCVIV2Manager.draw_update(context.object, self.uuid, self.draw, )
        pass
    
    # draw: BoolProperty(name="Draw", default=True, description="Draw particle instances as point cloud", update=_draw_update, )
    # # this should be just safe limit, somewhere in advanced settings
    # max_points: IntProperty(name="Max. Points", default=1000000, min=1, max=10000000, description="Maximum number of points per particle system", )
    
    def _switch_shader(self, context, ):
        pass
    
    quality: EnumProperty(name="Quality", items=[('BASIC', "Basic", "", ),
                                                 ('RICH', "Rich", "", ),
                                                 ], default='RICH', description="", update=_switch_shader, )
    
    @classmethod
    def register(cls):
        bpy.types.ParticleSettings.pcv_instance_visualizer3 = PointerProperty(type=cls)
    
    @classmethod
    def unregister(cls):
        del bpy.types.ParticleSettings.pcv_instance_visualizer3


class PCVIV3_object_properties(PropertyGroup):
    
    def _invalidate_cache(self, context, ):
        PCVIV3Manager.invalidate_cache(context.object.name)
    
    source: EnumProperty(name="Source", items=[('POLYGONS', "Polygons", "Mesh Polygons (constant or material viewport display color)"),
                                               ('VERTICES', "Vertices", "Mesh Vertices (constant color only)"),
                                               ], default='POLYGONS', description="Point cloud generation source", update=_invalidate_cache, )
    max_points: IntProperty(name="Max. Points", default=100, min=1, max=10000, description="Maximum number of points per instance", update=_invalidate_cache, )
    color_source: EnumProperty(name="Color Source", items=[('CONSTANT', "Constant Color", "Use constant color value"),
                                                           ('VIEWPORT_DISPLAY_COLOR', "Material Viewport Display Color", "Use material viewport display color property"),
                                                           ], default='VIEWPORT_DISPLAY_COLOR', description="Color source for generated point cloud", update=_invalidate_cache, )
    color_constant: FloatVectorProperty(name="Color", description="Constant color", default=(0.7, 0.7, 0.7, ), min=0, max=1, subtype='COLOR', size=3, update=_invalidate_cache, )
    
    # def _method_update(self, context, ):
    #     if(not self.use_face_area and not self.use_material_factors):
    #         self.use_face_area = True
    #
    #     # simulated _invalidate_cache()
    #     o = context.object
    #     if(o.name in PCVIV3Manager.cache.keys()):
    #         del PCVIV3Manager.cache[o.name]
    
    # use_face_area: BoolProperty(name="Use Face Area", default=True, description="Use mesh face area as probability factor during point cloud generation", update=_method_update, )
    # use_material_factors: BoolProperty(name="Use Material Factors", default=False, description="Use material probability factor during point cloud generation", update=_method_update, )
    use_face_area: BoolProperty(name="Use Face Area", default=True, description="Use mesh face area as probability factor during point cloud generation", update=_invalidate_cache, )
    use_material_factors: BoolProperty(name="Use Material Factors", default=False, description="Use material probability factor during point cloud generation", update=_invalidate_cache, )
    
    # # NOTE: maybe don't use pixel points, they are faster, that's for sure, but in this case, billboard points give some depth sense..
    point_size: IntProperty(name="Size", default=6, min=1, max=10, subtype='PIXEL', description="Point size", )
    point_size_f: FloatProperty(name="Size", default=0.02, min=0.0, max=1.0, description="Point size", precision=6, )
    
    @classmethod
    def register(cls):
        bpy.types.Object.pcv_instance_visualizer3 = PointerProperty(type=cls)
    
    @classmethod
    def unregister(cls):
        del bpy.types.Object.pcv_instance_visualizer3


class PCVIV3_material_properties(PropertyGroup):
    
    def _invalidate_cache(self, context, ):
        PCVIV3Manager.invalidate_cache(context.object.name)
    
    # this serves as material weight value for polygon point generator, higher value means that it is more likely for polygon to be used as point source
    factor: FloatProperty(name="Factor", default=0.5, min=0.001, max=1.0, precision=3, subtype='FACTOR', description="Probability factor of choosing polygon with this material", update=_invalidate_cache, )
    
    @classmethod
    def register(cls):
        bpy.types.Material.pcv_instance_visualizer3 = PointerProperty(type=cls)
    
    @classmethod
    def unregister(cls):
        del bpy.types.Material.pcv_instance_visualizer3


class PCVIV3_OT_init(Operator):
    bl_idname = "point_cloud_visualizer.pcviv3_init"
    bl_label = "Initialize"
    bl_description = "Initialize Instance Visualizer"
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        return True
    
    def execute(self, context):
        PCVIV3Manager.init()
        return {'FINISHED'}


class PCVIV3_OT_deinit(Operator):
    bl_idname = "point_cloud_visualizer.pcviv3_deinit"
    bl_label = "Deinitialize"
    bl_description = "Deinitialize Instance Visualizer"
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        return True
    
    def execute(self, context):
        PCVIV3Manager.deinit()
        return {'FINISHED'}


class PCVIV3_OT_register(Operator):
    bl_idname = "point_cloud_visualizer.pcviv3_register"
    bl_label = "Register"
    bl_description = "Register particle system"
    
    @classmethod
    def poll(cls, context):
        ok = False
        if(context.object is not None):
            o = context.object
            if(o.particle_systems.active is not None):
                ok = True
        return ok
    
    def execute(self, context):
        PCVIV3Manager.register(context.object, context.object.particle_systems.active)
        return {'FINISHED'}


class PCVIV3_PT_panel(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "View"
    bl_label = "PCVIV3"
    bl_parent_id = "PCV_PT_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        o = context.active_object
        if(o is None):
            return False
        return True
    
    def draw(self, context):
        l = self.layout
        c = l.column()
        
        c.operator('point_cloud_visualizer.pcviv3_register')
        r = c.row(align=True)
        r.operator('point_cloud_visualizer.pcviv3_init')
        r.operator('point_cloud_visualizer.pcviv3_deinit')
        
        b = c.box()
        b.scale_y = 0.5
        b.label(text='registry:')
        for k, v in PCVIV3Manager.registry.items():
            b.label(text='{}:{}'.format(k, v['psys'].name))
        b = c.box()
        b.scale_y = 0.5
        b.label(text='cache:')
        for k, v in PCVIV3Manager.cache.items():
            b.label(text='{}'.format(k))
        
        def human_readable_number(num, suffix='', ):
            f = 1000.0
            for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z', ]:
                if(abs(num) < f):
                    return "{:3.1f}{}{}".format(num, unit, suffix)
                num /= f
            return "{:.1f}{}{}".format(num, 'Y', suffix)
        
        b = c.box()
        b.scale_y = 0.5
        b.label(text='stats:')
        b.label(text='points: {}'.format(human_readable_number(PCVIV3Manager.stats)))


class PCVIV3_PT_generator(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "View"
    bl_label = "PCVIV3 generator options"
    bl_parent_id = "PCV_PT_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        o = context.active_object
        if(o is None):
            return False
        return True
    
    def draw(self, context):
        pcviv = context.object.pcv_instance_visualizer3
        l = self.layout
        c = l.column()
        c.prop(pcviv, 'max_points')
        c.prop(pcviv, 'point_size')
        c.prop(pcviv, 'point_size_f')
        c.prop(pcviv, 'source')
        if(pcviv.source == 'VERTICES'):
            r = c.row()
            r.prop(pcviv, 'color_constant')
        else:
            c.prop(pcviv, 'color_source')
            if(pcviv.color_source == 'CONSTANT'):
                r = c.row()
                r.prop(pcviv, 'color_constant')
            else:
                c.prop(pcviv, 'use_face_area')
                c.prop(pcviv, 'use_material_factors')
        
        if(pcviv.use_material_factors):
            b = c.box()
            cc = b.column(align=True)
            for slot in context.object.material_slots:
                if(slot.material is not None):
                    cc.prop(slot.material.pcv_instance_visualizer3, 'factor', text=slot.material.name)


# ---------------------------------------------------------------------------- and now for something completely different. it's.. tests! who would have thought?


class PCVIV3_OT_test_generator_speed(Operator):
    bl_idname = "point_cloud_visualizer.test_generator_speed"
    bl_label = "test_generator_speed"
    
    def execute(self, context):
        log(self.bl_idname)
        
        o = context.object
        n = 100
        
        def run_test_polygons(d):
            _t = time.time()
            for i in range(n):
                s = PCVIV3FacesSampler(**d)
            _d = datetime.timedelta(seconds=time.time() - _t) / n
            return _d
        
        def run_test_vertices(d):
            _t = time.time()
            for i in range(n):
                s = PCVIV3VertsSampler(**d)
            _d = datetime.timedelta(seconds=time.time() - _t) / n
            return _d
        
        w = 170
        
        log('{}'.format('-' * (w + 4)), 1)
        log('polygons: ', 1)
        
        d = {'target': o, 'count': -1, 'seed': 0, 'colorize': None, 'constant_color': None, 'use_face_area': None, 'use_material_factors': None, }
        
        counts = [-1, 1000, 100, ]
        
        def run_variable_counts_with_arguments(d):
            for c in counts:
                d['count'] = c
                r = run_test_polygons(d)
                h = 'count: {}, colorize: {}, constant_color: {}, use_face_area: {}, use_material_factors: {}'.format(d['count'], d['colorize'], d['constant_color'], d['use_face_area'], d['use_material_factors'])
                log('{:>{w}}'.format('{} >> {} >> {:.3f} ms'.format(h, r, r.microseconds / 1000), w=w, ), 2)
            log('', 2)
        
        run_variable_counts_with_arguments(d)
        d['colorize'] = 'VIEWPORT_DISPLAY_COLOR'
        run_variable_counts_with_arguments(d)
        d['use_face_area'] = True
        run_variable_counts_with_arguments(d)
        d['use_face_area'] = False
        d['use_material_factors'] = True
        run_variable_counts_with_arguments(d)
        d['use_face_area'] = True
        d['use_material_factors'] = True
        run_variable_counts_with_arguments(d)
        
        d['constant_color'] = (0.1, 0.2, 0.3, )
        run_variable_counts_with_arguments(d)
        d['colorize'] = 'VIEWPORT_DISPLAY_COLOR'
        run_variable_counts_with_arguments(d)
        d['use_face_area'] = True
        run_variable_counts_with_arguments(d)
        d['use_face_area'] = False
        d['use_material_factors'] = True
        run_variable_counts_with_arguments(d)
        d['use_face_area'] = True
        d['use_material_factors'] = True
        run_variable_counts_with_arguments(d)
        
        log('{}'.format('-' * (w + 4)), 1)
        log('vertices: ', 1)
        log('{}'.format('-' * (w + 4)), 1)
        
        d = {'target': o, 'count': -1, 'seed': 0, 'constant_color': None, }
        
        def run_variable_counts_with_arguments(d):
            for c in counts:
                d['count'] = c
                r = run_test_vertices(d)
                h = 'count: {}, constant_color: {}'.format(d['count'], d['constant_color'], )
                log('{:>{w}}'.format('{} >> {} >> {:.3f} ms'.format(h, r, r.microseconds / 1000), w=w, ), 2)
            log('', 2)
        
        run_variable_counts_with_arguments(d)
        
        d['constant_color'] = (0.1, 0.2, 0.3, )
        run_variable_counts_with_arguments(d)
        
        log('{}'.format('-' * (w + 4)), 1)
        
        return {'FINISHED'}


class PCVIV3_OT_test_generator_profile(Operator):
    bl_idname = "point_cloud_visualizer.test_generator_profile"
    bl_label = "test_generator_profile"
    
    def execute(self, context):
        log(self.bl_idname)
        
        o = context.object
        n = 1000
        
        log('tests will run {} times'.format(n), 1)
        
        import cProfile
        import pstats
        import io
        pr = cProfile.Profile()
        pr.enable()
        
        d = {'target': o,
             'count': -1,
             'seed': 0,
             'colorize': None,
             'constant_color': (1.0, 0.0, 0.0),
             'use_face_area': None,
             'use_material_factors': None, }
        for i in range(n):
            p = PCVIV3FacesSampler(**d)
        
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        
        import cProfile
        import pstats
        import io
        pr = cProfile.Profile()
        pr.enable()
        
        d = {'target': o,
             'count': -1,
             'seed': 0,
             'constant_color': (0.0, 1.0, 0.0), }
        for i in range(n):
            v = PCVIV3VertsSampler(**d)
        
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        
        return {'FINISHED'}


class PCVIV3_OT_test_generator_draw(Operator):
    bl_idname = "point_cloud_visualizer.test_generator_draw"
    bl_label = "test_generator_draw"
    
    def execute(self, context):
        log(self.bl_idname)
        
        o = context.object
        
        d = {'target': o,
             'count': -1,
             'seed': 0,
             'colorize': None,
             'constant_color': (1.0, 0.0, 0.0),
             'use_face_area': None,
             'use_material_factors': None, }
        p = PCVIV3FacesSampler(**d)
        
        d = {'target': o,
             'count': -1,
             'seed': 0,
             'constant_color': (0.0, 1.0, 0.0), }
        v = PCVIV3VertsSampler(**d)
        
        vs = np.concatenate((p.vs, v.vs, ), axis=0, )
        ns = np.concatenate((p.ns, v.ns, ), axis=0, )
        cs = np.concatenate((p.cs, v.cs, ), axis=0, )
        
        from .machine import PCVControl
        c = PCVControl(o)
        c.draw(vs, ns, cs)
        
        return {'FINISHED'}


class PCVIV3_PT_tests(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "View"
    bl_label = "PCVIV3 Tests"
    bl_parent_id = "PCV_PT_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        o = context.active_object
        if(o is None):
            return False
        return True
    
    def draw(self, context):
        l = self.layout
        c = l.column()
        c.operator('point_cloud_visualizer.test_generator_speed')
        c.operator('point_cloud_visualizer.test_generator_profile')
        c.operator('point_cloud_visualizer.test_generator_draw')
