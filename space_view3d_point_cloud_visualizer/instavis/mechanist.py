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

import os
import time
import datetime
import numpy as np

import bpy
from mathutils import Matrix, Vector, Quaternion, Color
import bgl
from gpu.types import GPUShader
from gpu_extras.batch import batch_for_shader

from .debug import debug_mode, log

shader_directory = os.path.realpath(os.path.join(os.path.dirname(__file__), 'shaders'))
if(not os.path.exists(shader_directory)):
    raise OSError("Did you point me to an imaginary directory? ('{}')".format(shader_directory))
shader_registry = {'RICH': {'v': "instavis_rich.vert", 'f': "instavis_rich.frag", 'g': "instavis_rich.geom", },
                   'BASIC': {'v': "instavis_basic.vert", 'f': "instavis_basic.frag", }, }
shader_cache = {}


def load_shader_code(name):
    if(name not in shader_registry.keys()):
        raise TypeError("Unknown shader requested..")
    
    if(name in shader_cache.keys()):
        c = shader_cache[name]
        return c['v'], c['f'], c['g']
    
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
    
    shader_cache[name] = {'v': vs, 'f': fs, 'g': gs, }
    return vs, fs, gs


class PCVIVFacesSampler():
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
        if(colorize == 'OBJECT_VIEWPORT_DISPLAY_COLOR'):
            colorize = 'CONSTANT'
            constant_color = tuple(target.color[:3])
        if(use_material_factors is None and colorize == 'MATERIAL_VIEWPORT_DISPLAY_COLOR'):
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
        if(colorize == 'MATERIAL_VIEWPORT_DISPLAY_COLOR'):
            if(len(target.data.materials) == 0):
                # no materials, set to constant
                colorize = 'CONSTANT'
                constant_color = self.sampler_error_color
                use_material_factors = False
            materials = target.data.materials
            if(None in materials[:]):
                # if there is empty slot, abort it and set to constant, checking each polygon will be slow..
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
        
        choices = np.indices((l, ), dtype=np.int, )
        choices.shape = (l, )
        weights = np.zeros(l, dtype=np.float32, )
        me.polygons.foreach_get('area', weights, )
        # make it all sum to 1.0
        weights *= 1.0 / np.sum(weights)
        indices = np.random.choice(choices, size=count, replace=False, p=weights, )
        
        if(colorize == 'MATERIAL_VIEWPORT_DISPLAY_COLOR'):
            material_indices = np.zeros(l, dtype=np.int, )
            me.polygons.foreach_get('material_index', material_indices, )
            material_colors = np.zeros((len(materials), 3), dtype=np.float32, )
            material_factors = np.zeros((len(materials)), dtype=np.float32, )
            for i, m in enumerate(materials):
                mc = m.diffuse_color[:3]
                material_colors[i][0] = mc[0] ** (1 / 2.2)
                material_colors[i][1] = mc[1] ** (1 / 2.2)
                material_colors[i][2] = mc[2] ** (1 / 2.2)
                material_factors[i] = m.pcv_instavis.factor
        
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
        elif(colorize == 'MATERIAL_VIEWPORT_DISPLAY_COLOR'):
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


class PCVIVVertsSampler():
    # shuffling points in mesh sampler is very slow, PCV display percentage won't work as expected if points are not shuffled, but for instavis is not that important
    sampler_shuffle = False
    # default mesh sampler point color
    sampler_constant_color = (1.0, 0.0, 1.0, )
    # used when mesh data is not available, like when face sampler is used, but mesh has no faces, or vertex sampler with empty mesh
    sampler_error_color = (1.0, 0.0, 1.0, )
    
    def __init__(self, target, count=-1, seed=0, constant_color=None, ):
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


class PCVIVMechanist():
    initialized = False
    
    handle = None
    buffer = []
    
    cache = {}
    flag = False
    
    stats_enabled = debug_mode()
    stats_num_points = 0
    stats_num_instances = 0
    stats_num_draws = 0
    
    pre_save_state = {}
    save_active = False
    pre_viewport_render_state = {}
    viewport_render_active = False
    
    msgbus_handle = object()
    msgbus_subs = ()
    
    origins_shader = None
    origins_batch = None
    origins_quality = None
    origins_size = None
    
    @classmethod
    def init(cls):
        if(cls.initialized):
            return
        log("init", prefix='>>>', )
        
        bpy.app.handlers.load_pre.append(watcher)
        
        prefs = bpy.context.scene.pcv_instavis
        if(prefs.update_method == 'MSGBUS'):
            bpy.msgbus.clear_by_owner(cls.msgbus_handle)
            cls._generate_msgbus_subs()
            for sub in cls.msgbus_subs:
                bpy.msgbus.subscribe_rna(key=sub, owner=cls.msgbus_handle, args=(), notify=mechanist_msgbus_update, options=set(), )
        else:
            bpy.app.handlers.depsgraph_update_post.append(cls.update)
        
        bpy.app.handlers.save_pre.append(cls.save_pre)
        bpy.app.handlers.save_post.append(cls.save_post)
        bpy.app.handlers.undo_post.append(cls.update)
        bpy.app.handlers.redo_post.append(cls.update)
        
        cls.initialized = True
        
        cls.handle = bpy.types.SpaceView3D.draw_handler_add(cls.draw, (), 'WINDOW', 'POST_VIEW')
        cls.update()
        cls._redraw_view_3d()
    
    @classmethod
    def deinit(cls):
        if(not cls.initialized):
            return
        log("deinit", prefix='>>>', )
        
        if(cls.update in bpy.app.handlers.depsgraph_update_post):
            bpy.app.handlers.depsgraph_update_post.remove(cls.update)
        bpy.msgbus.clear_by_owner(cls.msgbus_handle)
        
        bpy.app.handlers.save_pre.remove(cls.save_pre)
        bpy.app.handlers.save_post.remove(cls.save_post)
        bpy.app.handlers.undo_post.remove(cls.update)
        bpy.app.handlers.redo_post.remove(cls.update)
        
        bpy.app.handlers.load_pre.remove(watcher)
        
        cls.initialized = False
        
        bpy.types.SpaceView3D.draw_handler_remove(cls.handle, 'WINDOW')
        
        scene = bpy.context.scene
        prefs = scene.pcv_instavis
        targets = set([o for o in scene.objects if o.pcv_instavis.target])
        psystems = set([p for o in targets for p in o.particle_systems])
        psettings = set([p.settings for p in psystems])
        for pset in psettings:
            pset.display_method = prefs.exit_psys_display_method
        for n in cls.cache.keys():
            o = bpy.data.objects.get(n)
            if(o is not None):
                o.display_type = prefs.exit_object_display_type
        
        cls.initialized = False
        
        cls.handle = None
        cls.buffer = []
        cls.cache = {}
        cls.flag = False
        cls.stats_enabled = debug_mode()
        cls.stats_num_points = 0
        cls.stats_num_instances = 0
        cls.stats_num_draws = 0
        
        cls.pre_save_state = {}
        cls.save_active = False
        cls.pre_viewport_render_state = {}
        cls.viewport_render_active = False
        
        cls.origins_shader = None
        cls.origins_batch = None
        cls.origins_quality = None
        cls.origins_size = None
        
        cls._redraw_view_3d()
    
    @classmethod
    def _generate_msgbus_subs(cls):
        if(len(cls.msgbus_subs) != 0):
            return cls.msgbus_subs
        
        # add classes to subscribe if MSGBUS is used for update, this is not quite elegant, but at least it is easily accesible. i expect more types to be added..
        cls.msgbus_subs += (bpy.types.ParticleSettings,
                            # bpy.types.ParticleSystems,
                            # (bpy.types.ParticleSystems, 'active', ),
                            # (bpy.types.Object, 'particle_systems', ),
                            # bpy.types.ParticleSystemModifier,
                            # bpy.types.ParticleSettingsTextureSlot,
                            bpy.types.ImageTexture,
                            bpy.types.CloudsTexture,
                            (bpy.types.View3DShading, 'type', ), )
        
        def generate_pset_subs():
            l = bpy.types.ParticleSettings.bl_rna.properties.keys()
            l.remove('render_type')
            l.remove('display_method')
            t = bpy.types.ParticleSettings
            r = tuple([(t, i, ) for i in l])
            return r
        
        cls.msgbus_subs += generate_pset_subs()
        
        from . import props
        cls.msgbus_subs += (props.PCVIV_preferences,
                            props.PCVIV_psys_properties,
                            props.PCVIV_object_properties,
                            props.PCVIV_material_properties,
                            props.PCVIV_collection_properties, )
    
    @classmethod
    def update(cls, scene=None, depsgraph=None, ):
        # disable drawing when viewport render is detected or enable back again when finished
        if(not cls.viewport_render_active):
            r = cls._all_viewports_shading_type()
            if('RENDERED' in r):
                cls.viewport_render_active = True
                cls.viewport_render_pre(scene, depsgraph, )
        else:
            r = cls._all_viewports_shading_type()
            if('RENDERED' not in r):
                cls.viewport_render_active = False
                cls.viewport_render_post(scene, depsgraph, )
        
        # flag prevents recursion because i will fire depsgraph update a few times from now on
        if(cls.flag):
            return
        cls.flag = True
        
        # import cProfile
        # import pstats
        # import io
        # pr = cProfile.Profile()
        # pr.enable()
        
        _t = time.time()
        
        if(scene is None):
            scene = bpy.context.scene
        if(depsgraph is None):
            depsgraph = bpy.context.evaluated_depsgraph_get()
        
        # collect all targets
        targets = set([o for o in scene.objects if o.pcv_instavis.target])
        # and all particle systems on targets
        psystems = set([p for o in targets for p in o.particle_systems])
        # and their settings
        psettings = set([p.settings for p in psystems])
        
        prefs = scene.pcv_instavis
        quality = prefs.quality
        
        # auto switch to origins only
        if(prefs.switch_origins_only):
            for pset in psettings:
                if(pset.count >= prefs.switch_origins_only_threshold):
                    pset.pcv_instavis.use_origins_only = True
        
        # store viewport draw settings of objects in pre() to be able to restore them in post()
        dt = {}
        
        def pre():
            # turn on invisible instances to be able to get their matrices
            for pset in psettings:
                if(pset.render_type == 'COLLECTION'):
                    col = pset.instance_collection
                    if(col is not None):
                        for co in col.objects:
                            if(co.name not in dt.keys()):
                                dt[co.name] = (co, co.display_type, )
                            co.display_type = 'BOUNDS'
                elif(pset.render_type == 'OBJECT'):
                    co = pset.instance_object
                    if(co is not None):
                        if(co.name not in dt.keys()):
                            dt[co.name] = (co, co.display_type, )
                        co.display_type = 'BOUNDS'
                pset.display_method = 'RENDER'
        
        def post():
            # hide instance back when i am finished
            for pset in psettings:
                pset.display_method = 'NONE'
                for co, t in dt.values():
                    co.display_type = t
        
        # if(not cls.viewport_render_active and not cls.render_active and not cls.save_active):
        if(not cls.viewport_render_active and not cls.save_active):
            pre()
        
        # instances are visible, update depsgraph
        depsgraph.update()
        
        # prepopulate buffer
        buffer = [None] * len(depsgraph.object_instances)
        c = 0
        
        # zero out stats
        cls.stats_num_points = 0
        cls.stats_num_instances = 0
        
        l = len(depsgraph.object_instances)
        origins_vs = np.zeros((l, 3, ), dtype=np.float32, )
        origins_ns = np.zeros((l, 3, ), dtype=np.float32, )
        origins_cs = np.zeros((l, 3, ), dtype=np.float32, )
        oc = 0
        
        # set seed to prevent changes between updates
        np.random.seed(seed=0, )
        
        # loop over all instances in scene, choose and process those originating from psys on targets
        for instance in depsgraph.object_instances:
            ipsys = instance.particle_system
            if(ipsys is None):
                continue
            ipset = ipsys.settings
            ipset_eval = ipset.evaluated_get(depsgraph)
            
            if(ipset_eval.original in psettings):
                ipcviv = ipset.pcv_instavis
                iscale = ipcviv.point_scale
                if(ipcviv.display <= 99.0):
                    if(np.random.random() > ipcviv.display / 100.0):
                        # skip all processing and drawing.. also at this stage i can determine to which psys instance belongs and i can have display percentage per psys.
                        continue
                
                if(cls.stats_enabled):
                    cls.stats_num_instances += 1
                
                m = instance.matrix_world.copy()
                base = instance.object
                instance_options = base.pcv_instavis
                
                if(base.name not in cls.cache.keys()):
                    # generate point cloud from object and store in cache, if object is changed it won't be updated until invalidate_object_cache is called, this is done with generator properties, but if user changes mesh directly, it won't be noticed..
                    count = instance_options.max_points
                    color_constant = instance_options.color_constant
                    if(instance_options.source == 'VERTICES'):
                        sampler = PCVIVVertsSampler(base, count=count, seed=0, constant_color=color_constant, )
                    else:
                        sampler = PCVIVFacesSampler(base, count=count, seed=0, colorize=instance_options.color_source, constant_color=color_constant, use_face_area=instance_options.use_face_area, use_material_factors=instance_options.use_material_factors, )
                    vs, ns, cs = (sampler.vs, sampler.ns, sampler.cs, )
                    
                    if(quality == 'BASIC'):
                        # shader_data_vert, shader_data_frag, shader_data_geom = PCVIVShaders.get_shader('BASIC')
                        shader_data_vert, shader_data_frag, shader_data_geom = load_shader_code('BASIC')
                        shader = GPUShader(shader_data_vert, shader_data_frag, )
                        batch = batch_for_shader(shader, 'POINTS', {"position": vs, "color": cs, }, )
                    else:
                        # shader_data_vert, shader_data_frag, shader_data_geom = PCVIVShaders.get_shader('RICH')
                        shader_data_vert, shader_data_frag, shader_data_geom = load_shader_code('RICH')
                        shader = GPUShader(shader_data_vert, shader_data_frag, geocode=shader_data_geom, )
                        batch = batch_for_shader(shader, 'POINTS', {"position": vs, "normal": ns, "color": cs, }, )
                    
                    cls.cache[base.name] = (vs, ns, cs, shader, batch, )
                else:
                    vs, ns, cs, shader, batch = cls.cache[base.name]
                
                # size data for draw buffer
                if(quality == 'BASIC'):
                    draw_size = int(instance_options.point_size * iscale)
                    draw_quality = 0
                else:
                    draw_size = instance_options.point_size_f * iscale
                    draw_quality = 1
                
                if(ipcviv.draw and not ipcviv.use_origins_only):
                    buffer[c] = (draw_quality, shader, batch, m, draw_size, )
                    c += 1
                    if(cls.stats_enabled):
                        cls.stats_num_points += vs.shape[0]
                
                if(ipcviv.draw and ipcviv.use_origins_only):
                    v = m.translation
                    origins_vs[oc][0] = v[0]
                    origins_vs[oc][1] = v[1]
                    origins_vs[oc][2] = v[2]
                    # q = m.to_quaternion()
                    n = Vector((0.0, 0.0, 1.0, ))
                    # n.rotate(q)
                    origins_ns[oc][0] = n[0]
                    origins_ns[oc][1] = n[1]
                    origins_ns[oc][2] = n[2]
                    origins_cs[oc] = cs[0]
                    oc += 1
        
        # i count elements, so i can slice here without expensive filtering None out..
        buffer = buffer[:c]
        cls.buffer = buffer
        
        if(oc != 0):
            origins_vs = origins_vs[:oc]
            origins_ns = origins_ns[:oc]
            origins_cs = origins_cs[:oc]
            
            if(quality == 'BASIC'):
                # shader_data_vert, shader_data_frag, shader_data_geom = PCVIVShaders.get_shader('BASIC')
                shader_data_vert, shader_data_frag, shader_data_geom = load_shader_code('BASIC')
                shader = GPUShader(shader_data_vert, shader_data_frag, )
                batch = batch_for_shader(shader, 'POINTS', {"position": origins_vs, "color": origins_cs, }, )
                
                draw_size = prefs.origins_point_size
                draw_quality = 0
            else:
                # shader_data_vert, shader_data_frag, shader_data_geom = PCVIVShaders.get_shader('RICH')
                shader_data_vert, shader_data_frag, shader_data_geom = load_shader_code('RICH')
                shader = GPUShader(shader_data_vert, shader_data_frag, geocode=shader_data_geom, )
                batch = batch_for_shader(shader, 'POINTS', {"position": origins_vs, "normal": origins_ns, "color": origins_cs, }, )
                
                draw_size = prefs.origins_point_size_f
                draw_quality = 1
            
            cls.origins_shader = shader
            cls.origins_batch = batch
            cls.origins_quality = draw_quality
            cls.origins_size = draw_size
            
            if(cls.stats_enabled):
                cls.stats_num_points += origins_vs.shape[0]
        else:
            cls.origins_shader = None
            cls.origins_batch = None
            cls.origins_quality = None
            cls.origins_size = None
        
        if(not cls.viewport_render_active and not cls.save_active):
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
        
        _t = time.time()
        
        bgl.glEnable(bgl.GL_PROGRAM_POINT_SIZE)
        bgl.glEnable(bgl.GL_DEPTH_TEST)
        # bgl.glEnable(bgl.GL_BLEND)
        
        buffer = cls.buffer
        
        cls.stats_num_draws = 0
        
        # get those just once per draw call
        perspective_matrix = bpy.context.region_data.perspective_matrix
        view_matrix = bpy.context.region_data.view_matrix
        window_matrix = bpy.context.region_data.window_matrix
        light = view_matrix.copy().inverted()
        lt = light.translation
        vp = lt
        if(not bpy.context.region_data.is_perspective):
            # "A long time ago in a galaxy far, far away...."
            lt = view_matrix.copy().inverted() @ Vector((0.0, 0.0, 10.0 ** 6))
            vp = lt
        
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
            
            if(cls.stats_enabled):
                cls.stats_num_draws += 1
        
        if(cls.origins_shader is not None):
            shader = cls.origins_shader
            batch = cls.origins_batch
            quality = cls.origins_quality
            size = cls.origins_size
            matrix = Matrix()
            
            shader.bind()
            if(quality == 0):
                shader.uniform_float("perspective_matrix", perspective_matrix)
                shader.uniform_float("object_matrix", matrix)
                shader.uniform_float("size", size)
            else:
                shader.uniform_float("model_matrix", matrix)
                shader.uniform_float("view_matrix", view_matrix)
                shader.uniform_float("window_matrix", window_matrix)
                shader.uniform_float("size", size)
                shader.uniform_float("light_position", lt)
                shader.uniform_float("view_position", vp)
            batch.draw(shader)
            
            if(cls.stats_enabled):
                cls.stats_num_draws += 1
        
        bgl.glDisable(bgl.GL_PROGRAM_POINT_SIZE)
        bgl.glDisable(bgl.GL_DEPTH_TEST)
        # bgl.glDisable(bgl.GL_BLEND)
        
        _d = datetime.timedelta(seconds=time.time() - _t)
        # log("draw: {}".format(_d), prefix='>>>', )
        
        # pr.disable()
        # s = io.StringIO()
        # sortby = 'cumulative'
        # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        # ps.print_stats()
        # print(s.getvalue())
    
    @classmethod
    def force_update(cls, with_caches=False, ):
        if(not cls.initialized):
            return
        
        if(with_caches):
            cls.cache = {}
        cls.update()
    
    @classmethod
    def invalidate_object_cache(cls, name, ):
        # force regenerating instanced object point cloud fragment
        if(name in cls.cache.keys()):
            del cls.cache[name]
            return True
        return False
    
    @classmethod
    def viewport_render_pre(cls, scene, depsgraph, ):
        # do not draw point cloud during viewport render
        log("viewport_render_pre", prefix='>>>', )
        
        if(scene is None):
            scene = bpy.context.scene
        targets = set([o for o in scene.objects if o.pcv_instavis.target])
        psystems = set([p for o in targets for p in o.particle_systems])
        psettings = set([p.settings for p in psystems])
        
        for pset in psettings:
            cls.pre_viewport_render_state[pset.name] = pset.pcv_instavis.draw
            pset.pcv_instavis.draw = False
            pset.display_method = 'RENDER'
        for n in cls.cache.keys():
            o = bpy.data.objects.get(n)
            if(o is not None):
                cls.pre_viewport_render_state[o.name] = o.display_type
                o.display_type = 'TEXTURED'
    
    @classmethod
    def viewport_render_post(cls, scene, depsgraph, ):
        # restore drawing point cloud after viewport render
        log("viewport_render_post", prefix='>>>', )
        
        if(scene is None):
            scene = bpy.context.scene
        targets = set([o for o in scene.objects if o.pcv_instavis.target])
        psystems = set([p for o in targets for p in o.particle_systems])
        psettings = set([p.settings for p in psystems])
        
        for pset in psettings:
            if(pset.name in cls.pre_viewport_render_state.keys()):
                pset.pcv_instavis.draw = cls.pre_viewport_render_state[pset.name]
                pset.display_method = 'NONE'
        for n in cls.cache.keys():
            o = bpy.data.objects.get(n)
            if(o is not None):
                if(o.name in cls.pre_viewport_render_state.keys()):
                    o.display_type = cls.pre_viewport_render_state[o.name]
        
        cls.pre_viewport_render_state = {}
    
    @classmethod
    def save_pre(cls, scene, depsgraph=None, ):
        # switch to exit display mode before file save to not get hidden instances after file is opened again
        log("save_pre", prefix='>>>', )
        cls.save_active = True
        prefs = bpy.context.scene.pcv_instavis
        
        if(scene is None):
            scene = bpy.context.scene
        targets = set([o for o in scene.objects if o.pcv_instavis.target])
        psystems = set([p for o in targets for p in o.particle_systems])
        psettings = set([p.settings for p in psystems])
        
        for pset in psettings:
            cls.pre_save_state[pset.name] = pset.display_method
            pset.display_method = prefs.exit_psys_display_method
        for n in cls.cache.keys():
            o = bpy.data.objects.get(n)
            if(o is not None):
                cls.pre_save_state[o.name] = o.display_type
                o.display_type = prefs.exit_object_display_type
    
    @classmethod
    def save_post(cls, scene, depsgraph=None, ):
        # switch back after file save
        log("save_post", prefix='>>>', )
        
        if(scene is None):
            scene = bpy.context.scene
        targets = set([o for o in scene.objects if o.pcv_instavis.target])
        psystems = set([p for o in targets for p in o.particle_systems])
        psettings = set([p.settings for p in psystems])
        
        for pset in psettings:
            if(pset.name in cls.pre_save_state.keys()):
                pset.display_method = cls.pre_save_state[pset.name]
        for n in cls.cache.keys():
            o = bpy.data.objects.get(n)
            if(o is not None):
                if(o.name in cls.pre_save_state.keys()):
                    o.display_type = cls.pre_save_state[o.name]
        
        cls.pre_save_state = {}
        cls.save_active = False
    
    @classmethod
    def _all_viewports_shading_type(cls):
        # used for viewport render detection
        r = []
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if(area.type == 'VIEW_3D'):
                    for space in area.spaces:
                        if(space.type == 'VIEW_3D'):
                            r.append(space.shading.type)
        return r
    
    @classmethod
    def _redraw_view_3d(cls):
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if(area.type == 'VIEW_3D'):
                    area.tag_redraw()


def mechanist_msgbus_update():
    PCVIVMechanist.update()


@bpy.app.handlers.persistent
def watcher(undefined):
    PCVIVMechanist.deinit()
