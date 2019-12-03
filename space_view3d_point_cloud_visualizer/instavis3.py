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
            materials = target.data.materials
            if(None in materials[:]):
                # if there is empty slot, abort it and set to constant
                # TODO: make some workaround empty slots, this would require check for polygons with that empty material assigned and replacing that with constant color
                colorize = 'CONSTANT'
                constant_color = PCVIV3Config.sampler_error_color
        
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
                material_factors[i] = m.pcv_instance_visualizer.factor
        
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


# tests..


class PCVIV3_OT_test_generator_speed(Operator):
    bl_idname = "point_cloud_visualizer.test_generator_speed"
    bl_label = "test_generator_speed"
    
    def execute(self, context):
        log(self.bl_idname)
        
        o = context.object
        n = 100
        
        log('use all polygons/vertices', 1)
        
        d = {'context': context,
             'target': o,
             'count': -1,
             'seed': 0,
             'colorize': None,
             'constant_color': None,
             'use_face_area': None,
             'use_material_factors': None, }
        _t = time.time()
        for i in range(n):
            s = PCVIV3FacesSampler(**d)
        _d = datetime.timedelta(seconds=time.time() - _t) / n
        log('polygons: average generation time from {} runs: {}'.format(n, _d), 2)
        
        d = {'context': context,
             'target': o,
             'count': -1,
             'seed': 0,
             'constant_color': None, }
        _t = time.time()
        for i in range(n):
            s = PCVIV3VertsSampler(**d)
        _d = datetime.timedelta(seconds=time.time() - _t) / n
        log('vertices: average generation time from {} runs: {}'.format(n, _d), 2)
        
        count = 1000
        log('use max {} polygons/vertices'.format(count), 1)
        
        d = {'context': context,
             'target': o,
             'count': count,
             'seed': 0,
             'colorize': None,
             'constant_color': None,
             'use_face_area': None,
             'use_material_factors': None, }
        _t = time.time()
        for i in range(n):
            s = PCVIV3FacesSampler(**d)
        _d = datetime.timedelta(seconds=time.time() - _t) / n
        log('polygons: average generation time from {} runs: {}'.format(n, _d), 2)
        
        d = {'context': context,
             'target': o,
             'count': count,
             'seed': 0,
             'constant_color': None, }
        _t = time.time()
        for i in range(n):
            s = PCVIV3VertsSampler(**d)
        _d = datetime.timedelta(seconds=time.time() - _t) / n
        log('vertices: average generation time from {} runs: {}'.format(n, _d), 2)
        
        count = 100
        log('use max {} polygons/vertices'.format(count), 1)
        
        d = {'context': context,
             'target': o,
             'count': count,
             'seed': 0,
             'colorize': None,
             'constant_color': None,
             'use_face_area': None,
             'use_material_factors': None, }
        _t = time.time()
        for i in range(n):
            s = PCVIV3FacesSampler(**d)
        _d = datetime.timedelta(seconds=time.time() - _t) / n
        log('polygons: average generation time from {} runs: {}'.format(n, _d), 2)
        
        d = {'context': context,
             'target': o,
             'count': count,
             'seed': 0,
             'constant_color': None, }
        _t = time.time()
        for i in range(n):
            s = PCVIV3VertsSampler(**d)
        _d = datetime.timedelta(seconds=time.time() - _t) / n
        log('vertices: average generation time from {} runs: {}'.format(n, _d), 2)
        
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
