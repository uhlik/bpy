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

bl_info = {"name": "PCV Instance Visualizer 3",
           "description": "",
           "author": "Jakub Uhlik",
           "version": (0, 0, 1),
           "blender": (2, 80, 0),
           "location": "View3D > Sidebar > Point Cloud Visualizer",
           "warning": "",
           "wiki_url": "https://github.com/uhlik/bpy",
           "tracker_url": "https://github.com/uhlik/bpy/issues",
           "category": "3D View", }

import os
import uuid
import time
import datetime
import random
import numpy as np

import bpy
import bmesh
from bpy.props import PointerProperty, BoolProperty, StringProperty, FloatProperty, IntProperty, FloatVectorProperty, EnumProperty, CollectionProperty
from bpy.types import PropertyGroup, Panel, Operator
from mathutils import Matrix, Vector, Quaternion, Color
from bpy.app.handlers import persistent
import bgl
from gpu.types import GPUShader
from gpu_extras.batch import batch_for_shader


try:
    from .debug import log, debug_mode
except ImportError:
    def debug_mode():
        return True
        # return (bpy.app.debug_value != 0)
    
    def log(msg, indent=0, prefix='>', ):
        m = "{}{} {}".format("    " * indent, prefix, msg)
        if(debug_mode()):
            print(m)


shader_directory = os.path.join(os.path.dirname(__file__), 'shaders')
shader_registry = {
    'instavis3_rich': {'v': "instavis3_rich.vert", 'f': "instavis3_rich.frag", 'g': "instavis3_rich.geom", },
    'instavis3_basic': {'v': "instavis3_basic.vert", 'f': "instavis3_basic.frag", },
}


def load_shader_code(name):
    if(name not in shader_registry.keys()):
        raise TypeError("Unknown shader requested..")
    d = shader_registry[name]
    vf = d['v']
    ff = d['f']
    gf = None
    if('g' in d.keys()):
        gf = d['g']
    
    with open(os.path.join(shader_directory, vf), mode='r', encoding='utf-8') as f:
        vs = f.read()
    with open(os.path.join(shader_directory, ff), mode='r', encoding='utf-8') as f:
        fs = f.read()
    
    gs = None
    if(gf is not None):
        with open(os.path.join(shader_directory, gf), mode='r', encoding='utf-8') as f:
            gs = f.read()
    
    return vs, fs, gs


class PCVIV3FacesSampler():
    # shuffling points in mesh sampler is very slow, PCV display percentage won't work as expected if points are not shuffled, but for instavis is not that important
    sampler_shuffle = False
    # default mesh sampler point color
    sampler_constant_color = (1.0, 0.0, 1.0, )
    # used when mesh data is not available, like when face sampler is used, but mesh has no faces, or vertex sampler with empty mesh
    sampler_error_color = (1.0, 0.0, 1.0, )
    
    def __init__(self, target, count=-1, seed=0, colorize=None, constant_color=None, use_face_area=None, use_material_factors=None, ):
        if(colorize is None):
            colorize = 'CONSTANT'
        if(constant_color is None):
            constant_color = self.sampler_constant_color
        if(use_material_factors is None and colorize == 'VIEWPORT_DISPLAY_COLOR'):
            use_material_factors = False
        if(use_material_factors):
            if(colorize == 'CONSTANT'):
                use_material_factors = False
        
        me = target.data
        if(target.type not in ('MESH', )):
            self.vs = np.array(((0.0, 0.0, 0.0, ), ), dtype=np.float32, )
            self.ns = np.array(((0.0, 0.0, 1.0, ), ), dtype=np.float32, )
            self.cs = np.array((self.sampler_error_color, ), dtype=np.float32, )
            return
        elif(len(me.polygons) == 0):
            # no polygons to generate from, use origin
            self.vs = np.array(((0.0, 0.0, 0.0, ), ), dtype=np.float32, )
            self.ns = np.array(((0.0, 0.0, 1.0, ), ), dtype=np.float32, )
            self.cs = np.array((self.sampler_error_color, ), dtype=np.float32, )
            return
        if(colorize == 'VIEWPORT_DISPLAY_COLOR'):
            if(len(target.data.materials) == 0):
                # no materials, set to constant
                colorize = 'CONSTANT'
                constant_color = self.sampler_error_color
                use_material_factors = False
            materials = target.data.materials
            if(None in materials[:]):
                # if there is empty slot, abort it and set to constant
                # TODO: make some workaround empty slots, this would require check for polygons with that empty material assigned and replacing that with constant color
                colorize = 'CONSTANT'
                constant_color = self.sampler_error_color
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
        
        if(self.sampler_shuffle):
            a = np.concatenate((vs, ns, cs), axis=1, )
            np.random.shuffle(a)
            vs = a[:, :3]
            ns = a[:, 3:6]
            cs = a[:, 6:]
        
        self.vs = vs
        self.ns = ns
        self.cs = cs


class PCVIV3VertsSampler():
    # shuffling points in mesh sampler is very slow, PCV display percentage won't work as expected if points are not shuffled, but for instavis is not that important
    sampler_shuffle = False
    # default mesh sampler point color
    sampler_constant_color = (1.0, 0.0, 1.0, )
    # used when mesh data is not available, like when face sampler is used, but mesh has no faces, or vertex sampler with empty mesh
    sampler_error_color = (1.0, 0.0, 1.0, )
    
    def __init__(self, target, count=-1, seed=0, constant_color=None, ):
        # NOTE: material display color is not useable here, material is assigned to polygons. checking each vertex for its polygon (in fact there can be many of them) will be cpu intensive, so leave vertex sampler for case when user wants fast results or need to see vertices because of low poly geometry..
        if(constant_color is None):
            constant_color = self.sampler_constant_color
        
        me = target.data
        
        if(target.type not in ('MESH', )):
            self.vs = np.array(((0.0, 0.0, 0.0, ), ), dtype=np.float32, )
            self.ns = np.array(((0.0, 0.0, 1.0, ), ), dtype=np.float32, )
            self.cs = np.array((self.sampler_error_color, ), dtype=np.float32, )
            return
        elif(len(me.vertices) == 0):
            # no vertices to generate from, use origin
            self.vs = np.array(((0.0, 0.0, 0.0, ), ), dtype=np.float32, )
            self.ns = np.array(((0.0, 0.0, 1.0, ), ), dtype=np.float32, )
            self.cs = np.array((self.sampler_error_color, ), dtype=np.float32, )
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
        
        if(self.sampler_shuffle):
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
    
    # if True, properties callback erase its object from cache and force it to be rebuild
    cache_auto_update = True
    
    @classmethod
    def init(cls):
        if(cls.initialized):
            return
        log("init", prefix='>>>', )
        
        bpy.app.handlers.depsgraph_update_pre.append(cls.depsgraph_update_pre)
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
        
        prefs = bpy.context.scene.pcv_instance_visualizer3
        for k, psys in cls.registry.items():
            psys.settings.display_method = prefs.exit_psys_display_method
        for n in cls.cache.keys():
            o = bpy.data.objects.get(n)
            if(o is not None):
                o.display_type = prefs.exit_object_display_type
        
        cls.registry = {}
        cls.cache = {}
        cls.flag = False
        cls.buffer = []
        cls.handle = None
        cls.stats = 0
        
        cls._redraw_view_3d()
    
    @classmethod
    def register(cls, o, psys, ):
        pset = psys.settings
        pcviv = pset.pcv_instance_visualizer3
        if(pcviv.uuid == ''):
            # not used before..
            pcviv.uuid = str(uuid.uuid1())
        else:
            if(pcviv.uuid in cls.registry.keys()):
                # is already registered
                pass
            else:
                # was registered, then unregistered or removed, make new uuid
                pcviv.uuid = str(uuid.uuid1())
                # # just in case..
                # pset.display_method = 'RENDER'
                
        cls.registry[pcviv.uuid] = psys
        
        if(cls.initialized):
            scene = bpy.context.scene
            depsgraph = bpy.context.evaluated_depsgraph_get()
            cls.depsgraph_update_post(scene, depsgraph, )
    
    @classmethod
    def depsgraph_update_pre(cls, scene, depsgraph, ):
        rm = []
        for k, v in cls.registry.items():
            if(v.settings is None):
                rm.append(k)
        for k in rm:
            del cls.registry[k]
    
    @classmethod
    def depsgraph_update_post(cls, scene, depsgraph, ):
        # log("update!", prefix='>>>', )
        cls.update(scene, depsgraph)
    
    @classmethod
    def update(cls, scene, depsgraph, ):
        _t = time.time()
        
        prefs = scene.pcv_instance_visualizer3
        quality = prefs.quality
        
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
        
        # import cProfile
        # import pstats
        # import io
        # pr = cProfile.Profile()
        # pr.enable()
        
        registered = tuple([v for k, v in cls.registry.items()])
        
        dt = {}
        
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
                                if(co.name not in dt.keys()):
                                    dt[co.name] = (co, co.display_type, )
                                co.display_type = 'BOUNDS'
                        elif(pset.render_type == 'OBJECT'):
                            co = pset.instance_object
                            if(co.name not in dt.keys()):
                                dt[co.name] = (co, co.display_type, )
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
                        # if(pset.render_type == 'COLLECTION'):
                        #     col = pset.instance_collection
                        #     for co in col.objects:
                        #         co.display_type = 'TEXTURED'
                        # elif(pset.render_type == 'OBJECT'):
                        #     co = pset.instance_object
                        #     co.display_type = 'TEXTURED'
                        for co, t in dt.values():
                            co.display_type = t
        
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
                iscale = ipcviv.point_scale
                if(iuuid in registered_uuids):
                    m = instance.matrix_world.copy()
                    base = instance.object
                    instance_options = base.pcv_instance_visualizer3
                    
                    if(base.name not in cls.cache.keys()):
                        count = instance_options.max_points
                        color_constant = instance_options.color_constant
                        if(instance_options.source == 'VERTICES'):
                            sampler = PCVIV3VertsSampler(base,
                                                         count=count,
                                                         seed=0,
                                                         constant_color=color_constant, )
                        else:
                            sampler = PCVIV3FacesSampler(base,
                                                         count=count,
                                                         seed=0,
                                                         colorize=instance_options.color_source,
                                                         constant_color=color_constant,
                                                         use_face_area=instance_options.use_face_area,
                                                         use_material_factors=instance_options.use_material_factors, )
                        
                        vs, ns, cs = (sampler.vs, sampler.ns, sampler.cs, )
                        
                        if(quality == 'BASIC'):
                            # TODO: caching results of load_shader_code to memory, skip disk i/o, it's not a big deal, i it called once per base object
                            shader_data_vert, shader_data_frag, shader_data_geom = load_shader_code('instavis3_basic')
                            shader = GPUShader(shader_data_vert, shader_data_frag, )
                            batch = batch_for_shader(shader, 'POINTS', {"position": vs, "color": cs, }, )
                        else:
                            # TODO: caching results of load_shader_code to memory, skip disk i/o, it's not a big deal, i it called once per base object
                            shader_data_vert, shader_data_frag, shader_data_geom = load_shader_code('instavis3_rich')
                            shader = GPUShader(shader_data_vert, shader_data_frag, geocode=shader_data_geom, )
                            batch = batch_for_shader(shader, 'POINTS', {"position": vs, "normal": ns, "color": cs, }, )
                        
                        cls.cache[base.name] = (vs, ns, cs, shader, batch, )
                    else:
                        vs, ns, cs, shader, batch = cls.cache[base.name]
                    
                    if(quality == 'BASIC'):
                        draw_size = int(instance_options.point_size * iscale)
                        draw_quality = 0
                    else:
                        draw_size = instance_options.point_size_f * iscale
                        draw_quality = 1
                    
                    buffer[c] = (draw_quality, shader, batch, m, draw_size, )
                    c += 1
                    cls.stats += vs.shape[0]
        
        # i count elements, so i can slice here without expensive filtering None out..
        buffer = buffer[:c]
        cls.buffer = buffer
        
        post()
        cls.flag = False
        
        _d = datetime.timedelta(seconds=time.time() - _t)
        log("update: {}".format(_d), prefix='>>>', )
        
        # pr.disable()
        # s = io.StringIO()
        # sortby = 'cumulative'
        # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        # ps.print_stats()
        # print(s.getvalue())
    
    @classmethod
    def draw(cls):
        # import cProfile
        # import pstats
        # import io
        # pr = cProfile.Profile()
        # pr.enable()
        
        # _t = time.time()
        
        bgl.glEnable(bgl.GL_PROGRAM_POINT_SIZE)
        bgl.glEnable(bgl.GL_DEPTH_TEST)
        # bgl.glEnable(bgl.GL_BLEND)
        
        # # NOTE: i draw all points facing camera, so this won't work
        # bgl.glEnable(bgl.GL_CULL_FACE)
        # # bgl.glCullFace(bgl.GL_BACK)
        
        buffer = cls.buffer
        
        # get those just once per draw call
        perspective_matrix = bpy.context.region_data.perspective_matrix
        view_matrix = bpy.context.region_data.view_matrix
        window_matrix = bpy.context.region_data.window_matrix
        light = view_matrix.copy().inverted()
        lt = light.translation
        vp = lt
        
        '''
        matrix = view_matrix.inverted()
        distance = matrix.translation.length
        eye = matrix.to_translation()
        target = matrix @ Vector((0.0, 0.0, -distance))
        up = matrix @ Vector((0.0, 1.0, 0.0)) - eye
        
        def look_at_matrix(eye, target, up):
            z = eye - target
            x = up.cross(z)
            y = z.cross(x)
            
            x.normalize()
            y.normalize()
            z.normalize()
            
            r = Matrix()
            r[0][0] = x[0]
            r[0][1] = y[0]
            r[0][2] = z[0]
            r[0][3] = 0
            r[1][0] = x[1]
            r[1][1] = y[1]
            r[1][2] = z[1]
            r[1][3] = 0
            r[2][0] = x[2]
            r[2][1] = y[2]
            r[2][2] = z[2]
            r[2][3] = 0
            
            t = Matrix.Translation(eye)
            return t @ r
        
        eye2 = eye + Vector((-1.0, -1.0, 1.0))
        lt = look_at_matrix(eye2, target, up).translation
        '''
        
        # matrix = view_matrix.inverted()
        # eye = matrix.to_translation()
        # up = matrix @ Vector((0.0, 1.0, 0.0)) - eye
        # eye2 = eye + (up.normalized() * 3.0)
        # lt = eye2
        
        for quality, shader, batch, matrix, size in buffer:
            shader.bind()
            
            if(quality == 0):
                shader.uniform_float("perspective_matrix", perspective_matrix)
                shader.uniform_float("object_matrix", matrix)
                shader.uniform_float("size", size)
                # shader.uniform_float("alpha", 1.0)
            else:
                shader.uniform_float("model_matrix", matrix)
                shader.uniform_float("view_matrix", view_matrix)
                shader.uniform_float("window_matrix", window_matrix)
                shader.uniform_float("size", size)
                # shader.uniform_float("alpha", 1.0)
                # shader.uniform_float("light_position", light.translation)
                shader.uniform_float("light_position", lt)
                # shader.uniform_float("light_color", (0.8, 0.8, 0.8, ))
                # shader.uniform_float("view_position", light.translation)
                shader.uniform_float("view_position", vp)
                # shader.uniform_float("ambient_strength", 0.5)
                # shader.uniform_float("specular_strength", 0.5)
                # shader.uniform_float("specular_exponent", 8.0)
            
            batch.draw(shader)
        
        bgl.glDisable(bgl.GL_PROGRAM_POINT_SIZE)
        bgl.glDisable(bgl.GL_DEPTH_TEST)
        # bgl.glDisable(bgl.GL_BLEND)
        
        # bgl.glDisable(bgl.GL_CULL_FACE)
        
        # _d = datetime.timedelta(seconds=time.time() - _t)
        # log("draw: {}".format(_d), prefix='>>>', )
        
        # pr.disable()
        # s = io.StringIO()
        # sortby = 'cumulative'
        # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        # ps.print_stats()
        # print(s.getvalue())
    
    @classmethod
    def invalidate_object_cache(cls, name, ):
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
def watcher(undefined):
    PCVIV3Manager.deinit()


class PCVIV3_preferences(PropertyGroup):
    
    def _switch_shader(self, context, ):
        PCVIV3Manager.cache = {}
        
        scene = bpy.context.scene
        depsgraph = bpy.context.evaluated_depsgraph_get()
        PCVIV3Manager.depsgraph_update_post(scene, depsgraph, )
    
    quality: EnumProperty(name="Quality", items=[('BASIC', "Basic", "Basic pixel point based shader with flat colors", ),
                                                 ('RICH', "Rich", "Rich billboard shader with phong shading", ),
                                                 ], default='RICH', description="Global quality settings for all", update=_switch_shader, )
    
    exit_object_display_type: EnumProperty(name="Instanced Objects", items=[('BOUNDS', "Bounds", "", ), ('TEXTURED', "Textured", "", ), ], default='BOUNDS', description="To what set instance base objects Display Type when point cloud mode is exited", )
    exit_psys_display_method: EnumProperty(name="Particle Systems", items=[('NONE', "None", "", ), ('RENDER', "Render", "", ), ], default='RENDER', description="To what set particles system Display Method when point cloud mode is exited", )
    
    @classmethod
    def register(cls):
        bpy.types.Scene.pcv_instance_visualizer3 = PointerProperty(type=cls)
    
    @classmethod
    def unregister(cls):
        del bpy.types.Scene.pcv_instance_visualizer3


class PCVIV3_psys_properties(PropertyGroup):
    # this is going to be assigned during runtime by manager if it detects new psys creation on depsgraph update
    uuid: StringProperty(default="", options={'HIDDEN', }, )
    
    point_scale: FloatProperty(name="Point Scale", default=1.0, min=0.001, max=10.0, description="Adjust point size of all points", precision=6, )
    
    @classmethod
    def register(cls):
        bpy.types.ParticleSettings.pcv_instance_visualizer3 = PointerProperty(type=cls)
    
    @classmethod
    def unregister(cls):
        del bpy.types.ParticleSettings.pcv_instance_visualizer3


class PCVIV3_object_properties(PropertyGroup):
    
    def _invalidate_object_cache(self, context, ):
        PCVIV3Manager.invalidate_object_cache(context.object.name)
    
    source: EnumProperty(name="Source", items=[('POLYGONS', "Polygons", "Mesh Polygons (constant or material viewport display color)"),
                                               ('VERTICES', "Vertices", "Mesh Vertices (constant color only)"),
                                               ], default='POLYGONS', description="Point cloud generation source", update=_invalidate_object_cache, )
    max_points: IntProperty(name="Max. Points", default=100, min=1, max=10000, description="Maximum number of points per instance", update=_invalidate_object_cache, )
    color_source: EnumProperty(name="Color Source", items=[('CONSTANT', "Constant Color", "Use constant color value"),
                                                           ('VIEWPORT_DISPLAY_COLOR', "Material Viewport Display Color", "Use material viewport display color property"),
                                                           ], default='VIEWPORT_DISPLAY_COLOR', description="Color source for generated point cloud", update=_invalidate_object_cache, )
    color_constant: FloatVectorProperty(name="Color", description="Constant color", default=(0.7, 0.7, 0.7, ), min=0, max=1, subtype='COLOR', size=3, update=_invalidate_object_cache, )
    
    use_face_area: BoolProperty(name="Use Face Area", default=True, description="Use mesh face area as probability factor during point cloud generation", update=_invalidate_object_cache, )
    use_material_factors: BoolProperty(name="Use Material Factors", default=False, description="Use material probability factor during point cloud generation", update=_invalidate_object_cache, )
    
    point_size: IntProperty(name="Size (Basic Shader)", default=6, min=1, max=10, subtype='PIXEL', description="Point size", )
    point_size_f: FloatProperty(name="Size (Rich Shader)", default=0.02, min=0.001, max=1.0, description="Point size", precision=6, )
    
    @classmethod
    def register(cls):
        bpy.types.Object.pcv_instance_visualizer3 = PointerProperty(type=cls)
    
    @classmethod
    def unregister(cls):
        del bpy.types.Object.pcv_instance_visualizer3


class PCVIV3_material_properties(PropertyGroup):
    
    def _invalidate_object_cache(self, context, ):
        PCVIV3Manager.invalidate_object_cache(context.object.name)
    
    # this serves as material weight value for polygon point generator, higher value means that it is more likely for polygon to be used as point source
    factor: FloatProperty(name="Factor", default=0.5, min=0.001, max=1.0, precision=3, subtype='FACTOR', description="Probability factor of choosing polygon with this material", update=_invalidate_object_cache, )
    
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
        if(PCVIV3Manager.initialized):
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
        if(not PCVIV3Manager.initialized):
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
                uuid = o.particle_systems.active.settings.pcv_instance_visualizer3.uuid
                if(uuid == ""):
                    ok = True
                if(uuid in PCVIV3Manager.registry.keys()):
                    rpsys = PCVIV3Manager.registry[uuid]
                    if(rpsys == o.particle_systems.active):
                        ok = False
        return ok
    
    def execute(self, context):
        PCVIV3Manager.register(context.object, context.object.particle_systems.active)
        return {'FINISHED'}


class PCVIV3_OT_apply_generator_settings(Operator):
    bl_idname = "point_cloud_visualizer.pcviv3_apply_generator_settings"
    bl_label = "Apply Settings To Selected"
    bl_description = "Apply generator settings to all selected objects"
    
    @classmethod
    def poll(cls, context):
        if(context.object is not None):
            return True
        return False
    
    def execute(self, context):
        o = context.object
        pcviv = o.pcv_instance_visualizer3
        for so in context.selected_objects:
            if(so is o):
                continue
            ps = so.pcv_instance_visualizer3
            ps.source = pcviv.source
            ps.max_points = pcviv.max_points
            ps.color_source = pcviv.color_source
            ps.color_constant = pcviv.color_constant
            ps.use_face_area = pcviv.use_face_area
            ps.use_material_factors = pcviv.use_material_factors
            ps.point_size = pcviv.point_size
            ps.point_size_f = pcviv.point_size_f
        
        for so in context.selected_objects:
            if(so is o):
                continue
            PCVIV3Manager.invalidate_object_cache(so.name)
        
        return {'FINISHED'}


class PCVIV3_OT_invalidate_caches(Operator):
    bl_idname = "point_cloud_visualizer.pcviv3_invalidate_caches"
    bl_label = "Invalidate All Caches"
    bl_description = "Force refresh of all point caches"
    
    @classmethod
    def poll(cls, context):
        return True
    
    def execute(self, context):
        PCVIV3Manager.cache = {}
        
        scene = bpy.context.scene
        depsgraph = bpy.context.evaluated_depsgraph_get()
        PCVIV3Manager.depsgraph_update_post(scene, depsgraph, )
        
        return {'FINISHED'}


class PCVIV3_OT_reset_viewport_draw(Operator):
    bl_idname = "point_cloud_visualizer.pcviv3_reset_viewport_draw"
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


class PCVIV3_PT_base(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Point Cloud Visualizer"
    bl_label = "PCVIV3 base"
    bl_options = {'DEFAULT_CLOSED'}
    
    def prop_name(self, cls, prop, colon=False, ):
        for p in cls.bl_rna.properties:
            if(p.identifier == prop):
                if(colon):
                    return "{}:".format(p.name)
                return p.name
        return ''
    
    def third_label_two_thirds_prop(self, cls, prop, uil, ):
        f = 0.33
        r = uil.row()
        s = r.split(factor=f)
        s.label(text=self.prop_name(cls, prop, True, ))
        s = s.split(factor=1.0)
        r = s.row()
        r.prop(cls, prop, text='', )
    
    @classmethod
    def poll(cls, context):
        o = context.active_object
        if(o is None):
            return False
        return True


class PCVIV3_PT_main(PCVIV3_PT_base):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    # bl_category = "View"
    bl_category = "Point Cloud Visualizer"
    bl_label = "PCVIV3 Main"
    # bl_parent_id = "PCV_PT_panel"
    # bl_options = {'DEFAULT_CLOSED'}
    bl_options = set()
    
    @classmethod
    def poll(cls, context):
        o = context.active_object
        if(o is None):
            return False
        return True
    
    def draw(self, context):
        l = self.layout
        c = l.column()
        
        c.label(text='PCVIV3 Manager:')
        r = c.row(align=True)
        cc = r.column(align=True)
        if(not PCVIV3Manager.initialized):
            cc.alert = True
        cc.operator('point_cloud_visualizer.pcviv3_init')
        cc = r.column(align=True)
        if(PCVIV3Manager.initialized):
            cc.alert = True
        cc.operator('point_cloud_visualizer.pcviv3_deinit')
        
        n = 'n/a'
        if(context.object is not None):
            o = context.object
            if(o.particle_systems.active is not None):
                n = o.particle_systems.active.name
        
        c.label(text='Active Particle System: {}'.format(n))
        r = c.row()
        if(PCVIV3_OT_register.poll(context)):
            r.alert = True
        r.operator('point_cloud_visualizer.pcviv3_register')
        
        ok = False
        if(context.object is not None):
            o = context.object
            if(o.particle_systems.active is not None):
                ok = True
        if(ok):
            pset_pcviv = o.particle_systems.active.settings.pcv_instance_visualizer3
            c.prop(pset_pcviv, 'point_scale')


class PCVIV3_PT_generator(PCVIV3_PT_base):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    # bl_category = "View"
    bl_category = "Point Cloud Visualizer"
    bl_label = "PCVIV3 Generator Options"
    # bl_parent_id = "PCV_PT_panel"
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
        pcviv_prefs = context.scene.pcv_instance_visualizer3
        if(pcviv_prefs.quality == 'BASIC'):
            c.prop(pcviv, 'point_size')
        else:
            c.prop(pcviv, 'point_size_f')
        # c.prop(pcviv, 'source')
        self.third_label_two_thirds_prop(pcviv, 'source', c, )
        
        if(pcviv.source == 'VERTICES'):
            r = c.row()
            # r.prop(pcviv, 'color_constant')
            self.third_label_two_thirds_prop(pcviv, 'color_constant', r, )
        else:
            # c.prop(pcviv, 'color_source')
            self.third_label_two_thirds_prop(pcviv, 'color_source', c, )
            if(pcviv.color_source == 'CONSTANT'):
                r = c.row()
                # r.prop(pcviv, 'color_constant')
                self.third_label_two_thirds_prop(pcviv, 'color_constant', r, )
            else:
                c.prop(pcviv, 'use_face_area')
                c.prop(pcviv, 'use_material_factors')
        
        if(pcviv.use_material_factors):
            b = c.box()
            cc = b.column(align=True)
            for slot in context.object.material_slots:
                if(slot.material is not None):
                    cc.prop(slot.material.pcv_instance_visualizer3, 'factor', text=slot.material.name)
        
        c.operator('point_cloud_visualizer.pcviv3_apply_generator_settings')


class PCVIV3_PT_preferences(PCVIV3_PT_base):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    # bl_category = "View"
    bl_category = "Point Cloud Visualizer"
    bl_label = "PCVIV3 Preferences"
    # bl_parent_id = "PCV_PT_panel"
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
        
        pcviv_prefs = context.scene.pcv_instance_visualizer3
        c.label(text="Global Settings:")
        self.third_label_two_thirds_prop(pcviv_prefs, 'quality', c, )
        c.separator()
        c.label(text="Exit Display Settings:")
        self.third_label_two_thirds_prop(pcviv_prefs, 'exit_object_display_type', c, )
        self.third_label_two_thirds_prop(pcviv_prefs, 'exit_psys_display_method', c, )


class PCVIV3_PT_debug(PCVIV3_PT_base):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    # bl_category = "View"
    bl_category = "Point Cloud Visualizer"
    bl_label = "PCVIV3 Debug"
    # bl_parent_id = "PCV_PT_panel"
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
        
        tab = '    '
        
        b = c.box()
        b.scale_y = 0.5
        b.label(text='registry: ({})'.format(len(PCVIV3Manager.registry.keys())))
        for k, v in PCVIV3Manager.registry.items():
            b.label(text='{}{}'.format(tab, k))
        b = c.box()
        b.scale_y = 0.5
        b.label(text='cache: ({})'.format(len(PCVIV3Manager.cache.keys())))
        for k, v in PCVIV3Manager.cache.items():
            b.label(text='{}{}'.format(tab, k))
        
        def human_readable_number(num, suffix='', ):
            f = 1000.0
            for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z', ]:
                if(abs(num) < f):
                    return "{:3.1f}{}{}".format(num, unit, suffix)
                num /= f
            return "{:.1f}{}{}".format(num, 'Y', suffix)
        
        b = c.box()
        b.scale_y = 0.5
        b.label(text='points: {}'.format(human_readable_number(PCVIV3Manager.stats)))
        
        c.separator()
        c.operator('point_cloud_visualizer.pcviv3_reset_viewport_draw')
        c.operator('point_cloud_visualizer.pcviv3_invalidate_caches')
        c.separator()
        r = c.row()
        r.alert = True
        r.operator('script.reload')


classes = (
    PCVIV3_preferences, PCVIV3_psys_properties, PCVIV3_object_properties, PCVIV3_material_properties,
    PCVIV3_OT_init, PCVIV3_OT_deinit, PCVIV3_OT_register,
    PCVIV3_OT_apply_generator_settings, PCVIV3_OT_reset_viewport_draw, PCVIV3_OT_invalidate_caches,
    PCVIV3_PT_main, PCVIV3_PT_generator, PCVIV3_PT_preferences, PCVIV3_PT_debug,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    PCVIV3Manager.deinit()
    
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()