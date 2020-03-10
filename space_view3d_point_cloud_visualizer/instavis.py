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

bl_info = {"name": "PCV Instance Visualizer",
           "description": "",
           "author": "Jakub Uhlik",
           "version": (0, 0, 5),
           "blender": (2, 80, 0),
           "location": "View3D > Sidebar > PCVIV",
           "warning": "",
           "wiki_url": "https://github.com/uhlik/bpy",
           "tracker_url": "https://github.com/uhlik/bpy/issues",
           "category": "3D View", }

import os
import time
import datetime
import numpy as np

import bpy
from bpy.props import PointerProperty, BoolProperty, StringProperty, FloatProperty, IntProperty, FloatVectorProperty, EnumProperty, CollectionProperty
from bpy.types import PropertyGroup, Panel, Operator, UIList
from mathutils import Matrix, Vector, Quaternion, Color
import bgl
from gpu.types import GPUShader
from gpu_extras.batch import batch_for_shader


def debug_mode():
    return (bpy.app.debug_value != 0)


def log(msg, indent=0, prefix='>', ):
    m = "{}{} {}".format("    " * indent, prefix, msg)
    if(debug_mode()):
        print(m)


shader_directory = os.path.join(os.path.dirname(__file__), 'space_view3d_point_cloud_visualizer', 'shaders')
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


"""
class PCVIVShaders():
    instavis_basic_vert = '''
        in vec3 position;
        in vec3 color;
        
        uniform mat4 perspective_matrix;
        uniform mat4 object_matrix;
        uniform float size;
        
        out vec3 f_color;
        
        void main()
        {
            gl_Position = perspective_matrix * object_matrix * vec4(position, 1.0f);
            gl_PointSize = size;
            f_color = color;
        }
    '''
    instavis_basic_frag = '''
        in vec3 f_color;
        
        uniform float alpha = 1.0;
        
        out vec4 fragColor;
        
        void main()
        {
            fragColor = vec4(f_color, alpha);
        }
    '''
    instavis_rich_vert = '''
        layout (location = 0) in vec3 position;
        layout (location = 1) in vec3 normal;
        layout (location = 2) in vec3 color;
        
        uniform mat4 model_matrix;
        
        out vec3 g_position;
        out vec3 g_normal;
        out vec3 g_color;
        
        void main()
        {
            gl_Position = model_matrix * vec4(position, 1.0);
            g_position = vec3(model_matrix * vec4(position, 1.0));
            g_normal = mat3(transpose(inverse(model_matrix))) * normal;
            g_color = color;
        }
    '''
    instavis_rich_frag = '''
        layout (location = 0) out vec4 frag_color;
        
        in vec3 f_position;
        in vec3 f_normal;
        in vec3 f_color;
        
        uniform float alpha = 1.0;
        uniform vec3 light_position;
        uniform vec3 light_color = vec3(0.8, 0.8, 0.8);
        uniform vec3 view_position;
        uniform float ambient_strength = 0.5;
        uniform float specular_strength = 0.5;
        uniform float specular_exponent = 8.0;
        
        void main()
        {
            vec3 ambient = ambient_strength * light_color;
            
            vec3 nor = normalize(f_normal);
            vec3 light_direction = normalize(light_position - f_position);
            vec3 diffuse = max(dot(nor, light_direction), 0.0) * light_color;
            
            vec3 view_direction = normalize(view_position - f_position);
            vec3 reflection_direction = reflect(-light_direction, nor);
            float spec = pow(max(dot(view_direction, reflection_direction), 0.0), specular_exponent);
            vec3 specular = specular_strength * spec * light_color;
            
            vec3 col = (ambient + diffuse + specular) * f_color.rgb;
            frag_color = vec4(col, alpha);
            
            // if(!gl_FrontFacing){
            //     discard;
            // }
            
        }
    '''
    instavis_rich_geom = '''
        layout (points) in;
        layout (triangle_strip, max_vertices = 4) out;
        
        in vec3 g_position[];
        in vec3 g_normal[];
        in vec3 g_color[];
        
        uniform mat4 view_matrix;
        uniform mat4 window_matrix;
        uniform float size[];
        
        out vec3 f_position;
        out vec3 f_normal;
        out vec3 f_color;
        
        void main()
        {
            f_position = g_position[0];
            f_normal = g_normal[0];
            f_color = g_color[0];
            
            float s = size[0] / 2;
            
            vec4 pos = view_matrix * gl_in[0].gl_Position;
            vec2 xyloc = vec2(-1 * s, -1 * s);
            gl_Position = window_matrix * (pos + vec4(xyloc, 0, 0));
            EmitVertex();
            
            xyloc = vec2(1 * s, -1 * s);
            gl_Position = window_matrix * (pos + vec4(xyloc, 0, 0));
            EmitVertex();
            
            xyloc = vec2(-1 * s, 1 * s);
            gl_Position = window_matrix * (pos + vec4(xyloc, 0, 0));
            EmitVertex();
            
            xyloc = vec2(1 * s, 1 * s);
            gl_Position = window_matrix * (pos + vec4(xyloc, 0, 0));
            EmitVertex();
            
            EndPrimitive();
        }
    '''
    
    types = {'BASIC': (instavis_basic_vert, instavis_basic_frag, None, ),
             'RICH': (instavis_rich_vert, instavis_rich_frag, instavis_rich_geom, ), }
    
    @classmethod
    def get_shader(cls, type, ):
        if(type not in cls.types.keys()):
            raise Exception("Unknown shader type")
        return cls.types[type]
"""


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


class PCVIV_preferences(PropertyGroup):
    
    def _switch_shader(self, context, ):
        PCVIVMechanist.cache = {}
        PCVIVMechanist.update()
    
    # shader quality, switch between basic pixel based and rich shaded geometry based, can be changed on the fly
    quality: EnumProperty(name="Quality", items=[('BASIC', "Basic", "Basic pixel point based shader with flat colors", ),
                                                 ('RICH', "Rich", "Rich billboard shader with phong shading", ),
                                                 ], default='RICH', description="Global quality settings for all", update=_switch_shader, )
    
    # exit display settings is used for file save and when instavis is deinitialized, just to prevent viewport slowdown
    exit_object_display_type: EnumProperty(name="Instanced Objects", items=[('BOUNDS', "Bounds", "", ), ('TEXTURED', "Textured", "", ), ], default='BOUNDS', description="To what set instance base objects Display Type when point cloud mode is exited", )
    exit_psys_display_method: EnumProperty(name="Particle Systems", items=[('NONE', "None", "", ), ('RENDER', "Render", "", ), ], default='RENDER', description="To what set particles system Display Method when point cloud mode is exited", )
    
    origins_point_size: IntProperty(name="Size (Basic Shader)", default=6, min=1, max=10, subtype='PIXEL', description="Point size", )
    origins_point_size_f: FloatProperty(name="Size (Rich Shader)", default=0.05, min=0.001, max=1.0, description="Point size", precision=6, )
    
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
    point_size: IntProperty(name="Size (Basic Shader)", default=6, min=1, max=10, subtype='PIXEL', description="Point size", )
    point_size_f: FloatProperty(name="Size (Rich Shader)", default=0.02, min=0.001, max=1.0, description="Point size", precision=6, )
    
    def _target_update(self, context, ):
        pass
    
    target: BoolProperty(default=False, options={'HIDDEN', }, update=_target_update, )
    
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


class PCVIV_OT_apply_generator_settings(Operator):
    bl_idname = "point_cloud_visualizer.pcviv_apply_generator_settings"
    bl_label = "Apply Settings To Collection"
    bl_description = "Apply generator settings to all objects in current collection"
    
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


class PCVIV_PT_base(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "PCVIV"
    bl_label = "PCVIV base"
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


class PCVIV_PT_main(PCVIV_PT_base):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    # bl_category = "View"
    bl_category = "PCVIV"
    bl_label = "PCV Instance Visualizer"
    # bl_parent_id = "PCV_PT_panel"
    # bl_options = {'DEFAULT_CLOSED'}
    bl_options = set()
    
    @classmethod
    def poll(cls, context):
        # o = context.active_object
        # if(o is None):
        #     return False
        return True
    
    def draw(self, context):
        o = context.active_object
        if(not o):
            self.layout.label(text='Select an object..', icon='ERROR', )
            return
        
        l = self.layout
        c = l.column()
        
        # active object
        c.label(text="Active Object:")
        r = c.row(align=True)
        cc = r.column(align=True)
        if(not o.pcv_instavis.target):
            cc.alert = True
        cc.prop(o.pcv_instavis, 'target', text='Target', toggle=True, )
        
        # manager
        c.label(text='PCVIV Mechanist:')
        r = c.row(align=True)
        cc = r.column(align=True)
        if(not PCVIVMechanist.initialized):
            cc.alert = True
        cc.operator('point_cloud_visualizer.pcviv_init')
        cc = r.column(align=True)
        if(PCVIVMechanist.initialized):
            cc.alert = True
        cc.operator('point_cloud_visualizer.pcviv_deinit')
        r = c.row()
        r.alert = PCVIV_OT_force_update.poll(context)
        r.operator('point_cloud_visualizer.pcviv_force_update')


class PCVIV_PT_particles(PCVIV_PT_base):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    # bl_category = "View"
    bl_category = "PCVIV"
    bl_label = "Particle Systems"
    bl_parent_id = "PCVIV_PT_main"
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
        
        if(context.object is not None):
            o = context.object
            c.label(text='{}: Particle Systems:'.format(o.name))
            c.template_list("PARTICLE_UL_particle_systems", "particle_systems", o, "particle_systems", o.particle_systems, "active_index", rows=3, )
        
        # psys if there is any..
        n = 'n/a'
        if(context.object is not None):
            o = context.object
            if(o.particle_systems.active is not None):
                n = o.particle_systems.active.name
        c.label(text='Active Particle System: {}'.format(n))
        
        ok = False
        if(context.object is not None):
            o = context.object
            if(o.particle_systems.active is not None):
                ok = True
        if(ok):
            pset_pcviv = o.particle_systems.active.settings.pcv_instavis
            r = c.row()
            r.prop(pset_pcviv, 'draw', toggle=True, )
            r.scale_y = 1.5
            r = c.row()
            r.prop(pset_pcviv, 'display')
            r = c.row()
            r.prop(pset_pcviv, 'point_scale')
            
            if(pset_pcviv.use_origins_only):
                r.enabled = False
            c.prop(pset_pcviv, 'use_origins_only')
            
            cc = c.column(align=True)
            pcviv_prefs = context.scene.pcv_instavis
            if(pcviv_prefs.quality == 'BASIC'):
                cc.prop(pcviv_prefs, 'origins_point_size')
            else:
                cc.prop(pcviv_prefs, 'origins_point_size_f')
            if(not pset_pcviv.use_origins_only):
                cc.enabled = False


class PCVIV_UL_instances(UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, ):
        pcviv = context.object.pcv_instavis
        layout.label(text=item.name, icon='OBJECT_DATA', )


class PCVIV_PT_instances(PCVIV_PT_base):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    # bl_category = "View"
    bl_category = "PCVIV"
    bl_label = "Instance Options"
    bl_parent_id = "PCVIV_PT_main"
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
        
        ok = False
        if(context.object is not None):
            o = context.object
            c.label(text='{}: Particle Systems:'.format(o.name))
            c.template_list("PARTICLE_UL_particle_systems", "particle_systems", o, "particle_systems", o.particle_systems, "active_index", rows=3, )
            if(o.particle_systems.active is not None):
                pset = o.particle_systems.active.settings
                if(pset.render_type == 'COLLECTION' and pset.instance_collection is not None):
                    c.label(text='{}: Instanced Collection Objects:'.format(pset.name))
                    
                    col = pset.instance_collection
                    pcvcol = col.pcv_instavis
                    c.template_list("PCVIV_UL_instances", "", col, "objects", pcvcol, "active_index", rows=5, )
                    
                    co = col.objects[col.objects.keys()[pcvcol.active_index]]
                    pcvco = co.pcv_instavis
                    
                    c.label(text='Base Object "{}" Settings:'.format(co.name), )
                    
                    pcviv_prefs = context.scene.pcv_instavis
                    if(pcviv_prefs.quality == 'BASIC'):
                        c.prop(pcvco, 'point_size')
                    else:
                        c.prop(pcvco, 'point_size_f')
                    
                    self.third_label_two_thirds_prop(pcvco, 'source', c, )
                    c.prop(pcvco, 'max_points')
                    
                    if(pcvco.source == 'VERTICES'):
                        r = c.row()
                        self.third_label_two_thirds_prop(pcvco, 'color_constant', r, )
                    else:
                        self.third_label_two_thirds_prop(pcvco, 'color_source', c, )
                        if(pcvco.color_source == 'CONSTANT'):
                            r = c.row()
                            self.third_label_two_thirds_prop(pcvco, 'color_constant', r, )
                        else:
                            c.prop(pcvco, 'use_face_area')
                            c.prop(pcvco, 'use_material_factors')
                    
                    if(pcvco.use_material_factors):
                        b = c.box()
                        cc = b.column(align=True)
                        for slot in co.material_slots:
                            if(slot.material is not None):
                                cc.prop(slot.material.pcv_instavis, 'factor', text=slot.material.name)
                    c.operator('point_cloud_visualizer.pcviv_apply_generator_settings')
                elif(pset.render_type == 'OBJECT' and pset.instance_object is not None):
                    c.label(text='{}: Instanced Object:'.format(pset.name))
                    
                    co = pset.instance_object
                    b = c.box()
                    b.label(text=co.name, icon='OBJECT_DATA', )
                    
                    c.label(text='Base Object "{}" Settings:'.format(co.name), )
                    
                    pcvco = co.pcv_instavis
                    
                    pcviv_prefs = context.scene.pcv_instavis
                    if(pcviv_prefs.quality == 'BASIC'):
                        c.prop(pcvco, 'point_size')
                    else:
                        c.prop(pcvco, 'point_size_f')
                    
                    self.third_label_two_thirds_prop(pcvco, 'source', c, )
                    c.prop(pcvco, 'max_points')
                    
                    if(pcvco.source == 'VERTICES'):
                        r = c.row()
                        self.third_label_two_thirds_prop(pcvco, 'color_constant', r, )
                    else:
                        self.third_label_two_thirds_prop(pcvco, 'color_source', c, )
                        if(pcvco.color_source == 'CONSTANT'):
                            r = c.row()
                            self.third_label_two_thirds_prop(pcvco, 'color_constant', r, )
                        else:
                            c.prop(pcvco, 'use_face_area')
                            c.prop(pcvco, 'use_material_factors')
                    
                    if(pcvco.use_material_factors):
                        b = c.box()
                        cc = b.column(align=True)
                        for slot in co.material_slots:
                            if(slot.material is not None):
                                cc.prop(slot.material.pcv_instavis, 'factor', text=slot.material.name)
                else:
                    c.label(text="No collection/object found.", icon='ERROR', )
            else:
                c.label(text="No particle systems found.", icon='ERROR', )


class PCVIV_PT_preferences(PCVIV_PT_base):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    # bl_category = "View"
    bl_category = "PCVIV"
    bl_label = "Preferences"
    bl_parent_id = "PCVIV_PT_main"
    bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        o = context.active_object
        if(o is None):
            return False
        return True
    
    def draw(self, context):
        pcviv = context.object.pcv_instavis
        l = self.layout
        c = l.column()
        
        pcviv_prefs = context.scene.pcv_instavis
        c.label(text="Global Settings:")
        self.third_label_two_thirds_prop(pcviv_prefs, 'quality', c, )
        self.third_label_two_thirds_prop(pcviv_prefs, 'update_method', c, )
        c.separator()
        c.label(text="Exit Display Settings:")
        self.third_label_two_thirds_prop(pcviv_prefs, 'exit_object_display_type', c, )
        self.third_label_two_thirds_prop(pcviv_prefs, 'exit_psys_display_method', c, )
        c.separator()
        c.label(text="Auto Switch To Origins Only:")
        c.prop(pcviv_prefs, 'switch_origins_only', text='Enabled', )
        c.prop(pcviv_prefs, 'switch_origins_only_threshold')


class PCVIV_PT_debug(PCVIV_PT_base):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    # bl_category = "View"
    bl_category = "PCVIV"
    bl_label = "Debug"
    bl_parent_id = "PCVIV_PT_main"
    bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        o = context.active_object
        if(o is not None):
            if(debug_mode()):
                return True
        return False
    
    def draw(self, context):
        pcviv = context.object.pcv_instavis
        l = self.layout
        c = l.column()
        
        tab = '    '
        
        targets = [o for o in context.scene.objects if o.pcv_instavis.target]
        b = c.box()
        b.scale_y = 0.333
        b.label(text='targets: ({})'.format(len(targets)))
        for t in targets:
            b.label(text='{}o: {}'.format(tab, t.name))
            for p in t.particle_systems:
                b.label(text='{}ps: {}'.format(tab * 2, p.name))
        
        b = c.box()
        b.scale_y = 0.333
        b.label(text='cache: ({})'.format(len(PCVIVMechanist.cache.keys())))
        for k, v in PCVIVMechanist.cache.items():
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
        f = 0.5
        cc = b.column()
        
        def table_row(uil, ct1, ct2, fac, ):
            r = uil.row()
            s = r.split(factor=fac)
            s.label(text=ct1)
            s = s.split(factor=1.0)
            s.alignment = 'RIGHT'
            s.label(text=ct2)
        
        if(PCVIVMechanist.stats_enabled):
            table_row(cc, 'points: ', '{}'.format(human_readable_number(PCVIVMechanist.stats_num_points)), f, )
            table_row(cc, 'instances: ', '{}'.format(human_readable_number(PCVIVMechanist.stats_num_instances)), f, )
            table_row(cc, 'draws: ', '{}'.format(human_readable_number(PCVIVMechanist.stats_num_draws)), f, )
        else:
            table_row(cc, 'points: ', 'n/a', f, )
            table_row(cc, 'instances: ', 'n/a', f, )
            table_row(cc, 'draws: ', 'n/a', f, )
        
        c.separator()
        c.operator('point_cloud_visualizer.pcviv_reset_viewport_draw')
        c.operator('point_cloud_visualizer.pcviv_invalidate_caches')
        c.separator()
        r = c.row()
        r.alert = True
        r.operator('script.reload')


# add classes to subscribe if MSGBUS is used for update, this is not quite elegant, but at least it is easily accesible. i expect more types to be added..
PCVIVMechanist.msgbus_subs += (bpy.types.ParticleSettings,
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


PCVIVMechanist.msgbus_subs += generate_pset_subs()
PCVIVMechanist.msgbus_subs += (PCVIV_preferences, PCVIV_psys_properties, PCVIV_object_properties, PCVIV_material_properties, PCVIV_collection_properties, )

classes_debug = (
    PCVIV_preferences, PCVIV_psys_properties, PCVIV_object_properties, PCVIV_material_properties, PCVIV_collection_properties,
    PCVIV_OT_init, PCVIV_OT_deinit, PCVIV_OT_force_update,
    PCVIV_OT_apply_generator_settings, PCVIV_OT_reset_viewport_draw, PCVIV_OT_invalidate_caches,
    PCVIV_UL_instances, PCVIV_PT_main, PCVIV_PT_particles, PCVIV_PT_instances, PCVIV_PT_preferences, PCVIV_PT_debug,
)
classes_release = (
    PCVIV_preferences, PCVIV_psys_properties, PCVIV_object_properties, PCVIV_material_properties,
)
classes = classes_release
if(debug_mode() != 0):
    classes = classes_debug


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    PCVIVMechanist.deinit()
    
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
