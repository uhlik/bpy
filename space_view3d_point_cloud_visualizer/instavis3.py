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
from gpu.types import GPUShader, GPUBatch, GPUVertBuf, GPUVertFormat
from gpu_extras.batch import batch_for_shader

from .debug import log, debug_mode
from .machine import PCVManager, PCVControl, load_shader_code


class PCVIV3Config():
    # shuffling points in mesh sampler is very slow, PCV display percentage won't work as expected if points are not shuffled, but for instavis is not that important
    sampler_shuffle = False
    # default mesh sampler point color
    sampler_constant_color = (1.0, 0.0, 1.0, )
    # used when mesh data is not available, like when face sampler is used, but mesh has no faces, or vertex sampler with empty mesh
    sampler_error_color = (1.0, 0.0, 1.0, )


class PCVIV3Runtime():
    pass


class PCVIV3FacesSampler():
    def __init__(self, context, target, count=-1, seed=0, colorize=None, constant_color=None, use_face_area=None, use_material_factors=None, ):
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
    def __init__(self, context, target, count=-1, seed=0, constant_color=None, ):
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


def _pcv_pe_manager_update():
    PCVIV3Manager.update()
    return None


class PCVIV3Manager():
    initialized = False
    # cache = {}
    
    # delay = 0.1
    # flag = False
    # override = False
    
    registry = {}
    cache = {}
    flag = False
    buffer = []
    stats = 0
    
    @classmethod
    def init(cls):
        if(cls.initialized):
            return
        log("init", prefix='>>>', )
        
        # handlers here..
        # bpy.app.handlers.depsgraph_update_pre.append(cls.uuid_handler)
        # bpy.app.handlers.depsgraph_update_post.append(cls.uuid_handler)
        
        # cls.setup()
        
        # bpy.app.handlers.depsgraph_update_post.append(cls.update)
        bpy.app.handlers.depsgraph_update_post.append(cls.depsgraph_update_post)
        
        # bpy.app.handlers.depsgraph_update_post.append(cls.handler)
        
        bpy.app.handlers.load_pre.append(watcher)
        cls.initialized = True
        
        bpy.types.SpaceView3D.draw_handler_add(cls.draw, (), 'WINDOW', 'POST_VIEW')
        
        # scene = bpy.context.scene
        # depsgraph = bpy.context.evaluated_depsgraph_get()
        # cls.uuid_handler(scene, depsgraph, )
    
    @classmethod
    def deinit(cls):
        if(not cls.initialized):
            return
        log("deinit", prefix='>>>', )
        
        # handlers here..
        # bpy.app.handlers.depsgraph_update_pre.remove(cls.uuid_handler)
        # bpy.app.handlers.depsgraph_update_post.remove(cls.uuid_handler)
        
        # bpy.app.handlers.depsgraph_update_post.remove(cls.update)
        # bpy.app.handlers.depsgraph_update_post.remove(cls.handler)
        bpy.app.handlers.depsgraph_update_post.remove(cls.depsgraph_update_post)
        
        bpy.app.handlers.load_pre.remove(watcher)
        cls.initialized = False
    
    """
    @classmethod
    def uuid_handler(cls, scene, depsgraph, ):
        if(not cls.initialized):
            return
        # log("uuid_handler", prefix='>>>', )
        dirty = False
        ls = []
        dps = bpy.data.particles
        for ps in dps:
            if(ps.users == 0):
                continue
            
            pcviv = ps.pcv_instance_visualizer3
            
            if(pcviv.uuid == ""):
                log("uuid_handler: found psys without uuid", prefix='>>>', )
                pcviv.uuid = str(uuid.uuid1())
                cls.cache[pcviv.uuid] = {
                    'pset': ps,
                }
                
                # cls._redraw_view_3d()
                dirty = True
            else:
                if(pcviv.uuid not in cls.cache.keys()):
                    cls.cache[pcviv.uuid] = {
                        'pset': ps,
                    }
                    # cls._redraw_view_3d()
                    dirty = True
            
            ls.append(pcviv.uuid)
        
        kill = []
        for k in cls.cache.keys():
            if(k not in ls):
                kill.append(k)
        for k in kill:
            del cls.cache[k]
        
        if(len(kill) > 0):
            # cls._redraw_view_3d()
            dirty = True
        
        if(dirty):
            cls.update(scene, depsgraph, )
            cls._redraw_view_3d()
    """
    """
    @classmethod
    def update(cls, scene, depsgraph, ):
        _t = time.time()
        
        # import cProfile
        # import pstats
        # import io
        # pr = cProfile.Profile()
        # pr.enable()
        
        # for k, v in cls.cache.items():
        #     pset = v['pset']
        
        frags = []
        parent = None
        
        def apply_matrix(m, vs, ns=None, ):
            vs.shape = (-1, 3)
            vs = np.c_[vs, np.ones(vs.shape[0])]
            vs = np.dot(m, vs.T)[0:3].T.reshape((-1))
            vs.shape = (-1, 3)
            if(ns is not None):
                _, rot, _ = m.decompose()
                rmat = rot.to_matrix().to_4x4()
                ns.shape = (-1, 3)
                ns = np.c_[ns, np.ones(ns.shape[0])]
                ns = np.dot(rmat, ns.T)[0:3].T.reshape((-1))
                ns.shape = (-1, 3)
            return vs, ns
        
        for instance in depsgraph.object_instances:
            # TODO: instance.particle_system can check for the right particle system so i can skip other particle system, it is None for instancers aka duplis?
            if(instance.is_instance):
                base = instance.object
                # get matrix
                m = instance.matrix_world
                # unapply emitter matrix, instance.parent should refer to object holding particle system
                parent = instance.parent
                m = parent.matrix_world.inverted() @ m
                
                if(base.name not in cls.cache.keys()):
                    sampler = PCVIV3VertsSampler(None, base, count=100, )
                    vs, ns, cs = (sampler.vs, sampler.ns, sampler.cs, )
                    cls.cache[base.name] = (vs, ns, cs, )
                else:
                    vs, ns, cs = cls.cache[base.name]
                
                vs, ns = apply_matrix(m, vs, ns)
                frags.append((vs, ns, cs, ))
        
        if(len(frags) > 0):
            vs = np.concatenate([i[0] for i in frags], axis=0, )
            ns = np.concatenate([i[1] for i in frags], axis=0, )
            cs = np.concatenate([i[2] for i in frags], axis=0, )
            c = PCVControl(bpy.context.scene.objects[parent.name])
            c.draw(vs, ns, cs)
        
        # pr.disable()
        # s = io.StringIO()
        # sortby = 'cumulative'
        # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        # ps.print_stats()
        # print(s.getvalue())
        
        _d = datetime.timedelta(seconds=time.time() - _t)
        print(_d)
    """
    """
    @classmethod
    def update(cls, scene=None, depsgraph=None, ):
        if(not cls.initialized):
            return
        if(cls.override):
            return
        
        cls.override = True
        
        _t = time.time()
        
        # '''
        def pre():
            for o in bpy.data.objects:
                if(len(o.particle_systems) > 0):
                    for psys in o.particle_systems:
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
        # '''
        
        depsgraph = bpy.context.evaluated_depsgraph_get()
        # depsgraph.update()
        
        vs = np.zeros((len(depsgraph.object_instances), 3), dtype=np.float32, )
        c = 0
        parent = None
        for instance in depsgraph.object_instances:
            if(instance.is_instance):
                m = instance.matrix_world
                parent = instance.parent
                m = parent.matrix_world.inverted() @ m
                loc = np.array(m.translation.to_tuple(), dtype=np.float32, )
                vs[c] = loc
                c += 1
        
        vs = vs[:c]
        # print(parent)
        if(parent):
            c = PCVControl(bpy.context.scene.objects[parent.name])
            c.draw(vs, None, None)
        
        # '''
        post()
        # '''
        
        cls.override = False
        cls.flag = True
        
        _d = datetime.timedelta(seconds=time.time() - _t)
        print(_d)
    """
    """
    @classmethod
    def handler(cls, scene, depsgraph, ):
        if(not cls.initialized):
            return
        
        if(cls.override):
            return
        
        if(cls.flag):
            # FIXME: this is hackish, very hackish indeed, how to solve it? persuade some blender dev that when particle instances are hidden in viewport, depsgraph should return them.. basically, now i ignore next depsgraph update event and maybe i can miss something i should not with that..
            cls.flag = False
            return
        
        if(not bpy.app.timers.is_registered(_pcv_pe_manager_update)):
            bpy.app.timers.register(_pcv_pe_manager_update, first_interval=cls.delay, persistent=False, )
        else:
            if(bpy.app.timers.is_registered(_pcv_pe_manager_update)):
                # i've seen some 'ValueError: Error: function is not registered' here, how is that possible? no idea. so, lets check once more
                # i think it was caused by some forgotten registered handler which was not removed, when finished PCVIVManager should correctly add and remove handlers..
                # or meanwhile it run out?
                try:
                    bpy.app.timers.unregister(_pcv_pe_manager_update)
                except ValueError as e:
                    log("PCVIVManager: handler: {}".format(e))
            bpy.app.timers.register(_pcv_pe_manager_update, first_interval=cls.delay, persistent=False, )
    """
    
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
        _t = time.time()
        cls.update()
        _d = datetime.timedelta(seconds=time.time() - _t)
        if(not cls.flag):
            log("update: {}".format(_d), prefix='>>>', )
    
    @classmethod
    def update(cls):
        if(cls.flag):
            return
        cls.flag = True
        
        def pre():
            for o in bpy.data.objects:
                if(len(o.particle_systems) > 0):
                    for psys in o.particle_systems:
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
        depsgraph = bpy.context.evaluated_depsgraph_get()
        depsgraph.update()
        
        buffer = [None] * len(depsgraph.object_instances)
        c = 0
        registered_uuids = tuple(cls.registry.keys())
        cls.stats = 0
        
        # vs = np.zeros((len(depsgraph.object_instances), 3), dtype=np.float32, )
        # c = 0
        # parent = None
        for instance in depsgraph.object_instances:
            if(instance.is_instance):
                # m = instance.matrix_world
                # parent = instance.parent
                # m = parent.matrix_world.inverted() @ m
                # loc = np.array(m.translation.to_tuple(), dtype=np.float32, )
                # vs[c] = loc
                # c += 1
                ipsys = instance.particle_system
                ipset = ipsys.settings
                ipcviv = ipset.pcv_instance_visualizer3
                iuuid = ipcviv.uuid
                if(iuuid in registered_uuids):
                    m = instance.matrix_world
                    # parent = instance.parent
                    # m = parent.matrix_world.inverted() @ m
                    m = Matrix() @ m
                    
                    base = instance.object
                    if(base.name not in cls.cache.keys()):
                        sampler = PCVIV3VertsSampler(None, base, count=1000, )
                        vs, ns, cs = (sampler.vs, sampler.ns, sampler.cs, )
                        vert_shader = '''
                            in vec3 position;
                            in vec3 color;
                            uniform mat4 perspective_matrix;
                            uniform mat4 object_matrix;
                            out vec4 f_color;
                            void main()
                            {
                                gl_Position = perspective_matrix * object_matrix * vec4(position, 1.0f);
                                gl_PointSize = 6;
                                f_color = vec4(color, 1.0);
                            }
                        '''
                        frag_shader = '''
                            in vec4 f_color;
                            out vec4 fragColor;
                            void main()
                            {
                                fragColor = f_color;
                            }
                        '''
                        shader = GPUShader(vert_shader, frag_shader)
                        batch = batch_for_shader(shader, 'POINTS', {"position": vs, "color": cs, })
                        cls.cache[base.name] = (vs, ns, cs, shader, batch)
                    else:
                        vs, ns, cs, shader, batch = cls.cache[base.name]
                    # cls.buffer.append((shader, batch, m, ))
                    buffer[c] = (shader, batch, m, )
                    c += 1
                    cls.stats += len(vs)
        
        buffer = list(filter(None, buffer))
        cls.buffer = buffer
        
        post()
        cls.flag = False
    
    @classmethod
    def draw(cls):
        _t = time.time()
        
        buffer = cls.buffer
        pm = bpy.context.region_data.perspective_matrix
        for shader, batch, matrix in buffer:
            shader.bind()
            shader.uniform_float("perspective_matrix", pm)
            shader.uniform_float("object_matrix", matrix)
            batch.draw(shader)
        
        _d = datetime.timedelta(seconds=time.time() - _t)
        log("draw: {}".format(_d), prefix='>>>', )
    
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
    
    draw: BoolProperty(name="Draw", default=True, description="Draw particle instances as point cloud", update=_draw_update, )
    # this should be just safe limit, somewhere in advanced settings
    max_points: IntProperty(name="Max. Points", default=1000000, min=1, max=10000000, description="Maximum number of points per particle system", )
    
    @classmethod
    def register(cls):
        bpy.types.ParticleSettings.pcv_instance_visualizer3 = PointerProperty(type=cls)
    
    @classmethod
    def unregister(cls):
        del bpy.types.ParticleSettings.pcv_instance_visualizer3


class PCVIV3_object_properties(PropertyGroup):
    source: EnumProperty(name="Source", items=[('POLYGONS', "Polygons", "Mesh Polygons (constant or material viewport display color)"),
                                               ('VERTICES', "Vertices", "Mesh Vertices (constant color only)"),
                                               ], default='POLYGONS', description="Point cloud generation source", )
    max_points_static: IntProperty(name="Max. Static Points", default=1000, min=1, max=100000, description="Maximum number of points per instance for static drawing", )
    max_points_interactive: IntProperty(name="Max. Interactive Points", default=100, min=1, max=10000, description="Maximum number of points per instance for interactive drawing", )
    
    # NOTE: maybe don't use pixel points, they are faster, that's for sure, but in this case, billboard points give some depth sense..
    point_size: IntProperty(name="Size", default=3, min=1, max=10, subtype='PIXEL', description="Point size", )
    point_size_f: FloatProperty(name="Scale", default=1.0, min=0.0, max=10.0, description="Point scale (shader size * scale)", precision=6, )
    
    # TODO: i write it here again, solve empty material slots without material and therefore without any data to generate color from..
    color_source: EnumProperty(name="Color Source", items=[('CONSTANT', "Constant Color", "Use constant color value"),
                                                           ('VIEWPORT_DISPLAY_COLOR', "Material Viewport Display Color", "Use material viewport display color property"),
                                                           ], default='VIEWPORT_DISPLAY_COLOR', description="Color source for generated point cloud", )
    color_constant: FloatVectorProperty(name="Color", description="Constant color", default=(0.7, 0.7, 0.7, ), min=0, max=1, subtype='COLOR', size=3, )
    
    def _method_update(self, context, ):
        if(not self.use_face_area and not self.use_material_factors):
            self.use_face_area = True
    
    use_face_area: BoolProperty(name="Use Face Area", default=True, description="Use mesh face area as probability factor during point cloud generation", update=_method_update, )
    use_material_factors: BoolProperty(name="Use Material Factors", default=False, description="Use material probability factor during point cloud generation", update=_method_update, )
    
    # # helper property, draw minimal ui or draw all props
    # subpanel_opened: BoolProperty(default=False, options={'HIDDEN', }, )
    # # store info how long was last update, generate and store to cache
    # debug_update: StringProperty(default="", )
    
    @classmethod
    def register(cls):
        bpy.types.Object.pcv_instance_visualizer3 = PointerProperty(type=cls)
    
    @classmethod
    def unregister(cls):
        del bpy.types.Object.pcv_instance_visualizer3


class PCVIV3_material_properties(PropertyGroup):
    # this serves as material weight value for polygon point generator, higher value means that it is more likely for polygon to be used as point source
    factor: FloatProperty(name="Factor", default=0.5, min=0.0, max=1.0, precision=3, subtype='FACTOR', description="Probability factor of choosing polygon with this material", )
    
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

'''
class PCVIV3_OT_update(Operator):
    bl_idname = "point_cloud_visualizer.pcviv3_update"
    bl_label = "Update"
    bl_description = "Update point cloud visualization by particle system UUID"
    
    # uuid: StringProperty(name="UUID", default='', )
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        return True
    
    def execute(self, context):
        # PCVIV3Manager.update(self.uuid, )
        PCVIV3Manager.update()
        return {'FINISHED'}
'''


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
        
        # if(PCVIV3Manager.initialized):
        #     # o = context.object
        #     # for psys in o.particle_systems:
        #     #     pset = psys.settings.pcv_instance_visualizer3
        #     #     if(pset.uuid != ''):
        #     #         c.operator('point_cloud_visualizer.pcviv3_update', text='Update: {}'.format(psys.name)).uuid = pset.uuid
        #     #     else:
        #     #         raise Exception('psys without uuid')
        #     c.operator('point_cloud_visualizer.pcviv3_update')
        
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
        
        d = {'context': context, 'target': o, 'count': -1, 'seed': 0, 'colorize': None, 'constant_color': None, 'use_face_area': None, 'use_material_factors': None, }
        
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
        
        d = {'context': context, 'target': o, 'count': -1, 'seed': 0, 'constant_color': None, }
        
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
        
        d = {'context': context,
             'target': o,
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
        
        d = {'context': context,
             'target': o,
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
        
        d = {'context': context,
             'target': o,
             'count': -1,
             'seed': 0,
             'colorize': None,
             'constant_color': (1.0, 0.0, 0.0),
             'use_face_area': None,
             'use_material_factors': None, }
        p = PCVIV3FacesSampler(**d)
        
        d = {'context': context,
             'target': o,
             'count': -1,
             'seed': 0,
             'constant_color': (0.0, 1.0, 0.0), }
        v = PCVIV3VertsSampler(**d)
        
        vs = np.concatenate((p.vs, v.vs, ), axis=0, )
        ns = np.concatenate((p.ns, v.ns, ), axis=0, )
        cs = np.concatenate((p.cs, v.cs, ), axis=0, )
        
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
