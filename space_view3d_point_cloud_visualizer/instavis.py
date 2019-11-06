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
# from . import shaders


class PCVIVSampler():
    def __init__(self, context, o, target, rnd, percentage=1.0, triangulate=True, use_modifiers=True, source=None, colorize=None, constant_color=None, vcols=None, uvtex=None, vgroup=None, ):
        log("{}:".format(self.__class__.__name__), 0)
        
        def remap(v, min1, max1, min2, max2, ):
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
            
            if(max1 - min1 == 0):
                # handle zero division when min1 = max1
                return min2
            
            r = interpolate(normalize(v, min1, max1), min2, max2)
            return r
        
        owner = None
        if(use_modifiers and target.modifiers):
            depsgraph = context.evaluated_depsgraph_get()
            owner = target.evaluated_get(depsgraph)
            me = owner.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph, )
        else:
            owner = target
            me = owner.to_mesh(preserve_all_data_layers=True, depsgraph=None, )
        
        bm = bmesh.new()
        bm.from_mesh(me)
        if(not triangulate):
            if(colorize in ('VCOLS', 'UVTEX', 'GROUP_MONO', 'GROUP_COLOR', )):
                bmesh.ops.triangulate(bm, faces=bm.faces)
        else:
            bmesh.ops.triangulate(bm, faces=bm.faces)
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        
        if(source is None):
            source = 'VERTICES'
        
        if(len(bm.verts) == 0):
            raise Exception("Mesh has no vertices")
        if(colorize in ('UVTEX', 'VCOLS', 'VIEWPORT_DISPLAY_COLOR', )):
            if(len(bm.faces) == 0):
                raise Exception("Mesh has no faces")
        if(colorize == 'VIEWPORT_DISPLAY_COLOR'):
            try:
                if(len(target.data.materials) == 0):
                    raise Exception("Cannot find any material")
                materials = target.data.materials
            except Exception as e:
                raise Exception(str(e))
        if(colorize == 'UVTEX'):
            try:
                if(target.active_material is None):
                    raise Exception("Cannot find active material")
                uvtexnode = target.active_material.node_tree.nodes.active
                if(uvtexnode is None):
                    raise Exception("Cannot find active image texture in active material")
                uvimage = uvtexnode.image
                if(uvimage is None):
                    raise Exception("Cannot find active image texture with loaded image in active material")
                uvimage.update()
                uvarray = np.asarray(uvimage.pixels)
                uvarray = uvarray.reshape((uvimage.size[1], uvimage.size[0], 4))
                uvlayer = bm.loops.layers.uv.active
                if(uvlayer is None):
                    raise Exception("Cannot find active UV layout")
            except Exception as e:
                raise Exception(str(e))
        if(colorize == 'VCOLS'):
            try:
                col_layer = bm.loops.layers.color.active
                if(col_layer is None):
                    raise Exception()
            except Exception:
                raise Exception("Cannot find active vertex colors")
        if(colorize in ('GROUP_MONO', 'GROUP_COLOR')):
            try:
                group_layer = bm.verts.layers.deform.active
                if(group_layer is None):
                    raise Exception()
                group_layer_index = target.vertex_groups.active.index
            except Exception:
                raise Exception("Cannot find active vertex group")
        
        # if(percentage < 1.0):
        #     if(source == 'FACES'):
        #         rnd_layer = bm.faces.layers.float.new('face_random')
        #         for f in bm.faces:
        #             f[rnd_layer] = rnd.random()
        #     if(source == 'VERTICES'):
        #         rnd_layer = bm.verts.layers.float.new('vertex_random')
        #         for v in bm.verts:
        #             v[rnd_layer] = rnd.random()
        
        vs = []
        ns = []
        cs = []
        
        if(source == 'FACES'):
            for f in bm.faces:
                if(percentage < 1.0):
                    # if(f[rnd_layer] > percentage):
                    #     continue
                    if(rnd.random() > percentage):
                        continue
                
                v = f.calc_center_median()
                vs.append(v.to_tuple())
                ns.append(f.normal.to_tuple())
                
                if(colorize is None):
                    cs.append((1.0, 0.0, 0.0, ))
                elif(colorize == 'CONSTANT'):
                    cs.append(constant_color)
                elif(colorize == 'VIEWPORT_DISPLAY_COLOR'):
                    c = materials[f.material_index].diffuse_color[:3]
                    c = [v ** (1 / 2.2) for v in c]
                    cs.append(c)
                elif(colorize == 'VCOLS'):
                    ws = poly_3d_calc([f.verts[0].co, f.verts[1].co, f.verts[2].co, ], v)
                    ac = f.loops[0][col_layer][:3]
                    bc = f.loops[1][col_layer][:3]
                    cc = f.loops[2][col_layer][:3]
                    r = ac[0] * ws[0] + bc[0] * ws[1] + cc[0] * ws[2]
                    g = ac[1] * ws[0] + bc[1] * ws[1] + cc[1] * ws[2]
                    b = ac[2] * ws[0] + bc[2] * ws[1] + cc[2] * ws[2]
                    cs.append((r, g, b, ))
                elif(colorize == 'UVTEX'):
                    uvtriangle = []
                    for l in f.loops:
                        uvtriangle.append(Vector(l[uvlayer].uv.to_tuple() + (0.0, )))
                    uvpoint = barycentric_transform(v, f.verts[0].co, f.verts[1].co, f.verts[2].co, *uvtriangle, )
                    w, h = uvimage.size
                    # x,y % 1.0 to wrap around if uv coordinate is outside 0.0-1.0 range
                    x = int(round(remap(uvpoint.x % 1.0, 0.0, 1.0, 0, w - 1)))
                    y = int(round(remap(uvpoint.y % 1.0, 0.0, 1.0, 0, h - 1)))
                    cs.append(tuple(uvarray[y][x][:3].tolist()))
                elif(colorize == 'GROUP_MONO'):
                    ws = poly_3d_calc([f.verts[0].co, f.verts[1].co, f.verts[2].co, ], v)
                    aw = f.verts[0][group_layer].get(group_layer_index, 0.0)
                    bw = f.verts[1][group_layer].get(group_layer_index, 0.0)
                    cw = f.verts[2][group_layer].get(group_layer_index, 0.0)
                    m = aw * ws[0] + bw * ws[1] + cw * ws[2]
                    cs.append((m, m, m, ))
                elif(colorize == 'GROUP_COLOR'):
                    ws = poly_3d_calc([f.verts[0].co, f.verts[1].co, f.verts[2].co, ], v)
                    aw = f.verts[0][group_layer].get(group_layer_index, 0.0)
                    bw = f.verts[1][group_layer].get(group_layer_index, 0.0)
                    cw = f.verts[2][group_layer].get(group_layer_index, 0.0)
                    m = aw * ws[0] + bw * ws[1] + cw * ws[2]
                    hue = remap(1.0 - m, 0.0, 1.0, 0.0, 1 / 1.5)
                    c = Color()
                    c.hsv = (hue, 1.0, 1.0, )
                    cs.append((c.r, c.g, c.b, ))
        else:
            # source == 'VERTICES'
            for v in bm.verts:
                if(percentage < 1.0):
                    # if(v[rnd_layer] > percentage):
                    #     continue
                    if(rnd.random() > percentage):
                        continue
                
                if(len(v.link_loops) == 0 and colorize in ('UVTEX', 'VCOLS', 'VIEWPORT_DISPLAY_COLOR', )):
                    # single vertex without faces, skip when faces are required for colorizing
                    continue
                
                vs.append(v.co.to_tuple())
                ns.append(v.normal.to_tuple())
                
                if(colorize is None):
                    cs.append((1.0, 0.0, 0.0, ))
                elif(colorize == 'CONSTANT'):
                    cs.append(constant_color)
                elif(colorize == 'VIEWPORT_DISPLAY_COLOR'):
                    r = 0.0
                    g = 0.0
                    b = 0.0
                    lfs = v.link_faces
                    for f in lfs:
                        c = materials[f.material_index].diffuse_color[:3]
                        c = [v ** (1 / 2.2) for v in c]
                        r += c[0]
                        g += c[1]
                        b += c[2]
                    r /= len(lfs)
                    g /= len(lfs)
                    b /= len(lfs)
                    cs.append((r, g, b, ))
                elif(colorize == 'VCOLS'):
                    ls = v.link_loops
                    r = 0.0
                    g = 0.0
                    b = 0.0
                    for l in ls:
                        c = l[col_layer][:3]
                        r += c[0]
                        g += c[1]
                        b += c[2]
                    r /= len(ls)
                    g /= len(ls)
                    b /= len(ls)
                    cs.append((r, g, b, ))
                elif(colorize == 'UVTEX'):
                    ls = v.link_loops
                    w, h = uvimage.size
                    r = 0.0
                    g = 0.0
                    b = 0.0
                    for l in ls:
                        uvloc = l[uvlayer].uv.to_tuple()
                        # x,y % 1.0 to wrap around if uv coordinate is outside 0.0-1.0 range
                        x = int(round(remap(uvloc[0] % 1.0, 0.0, 1.0, 0, w - 1)))
                        y = int(round(remap(uvloc[1] % 1.0, 0.0, 1.0, 0, h - 1)))
                        c = tuple(uvarray[y][x][:3].tolist())
                        r += c[0]
                        g += c[1]
                        b += c[2]
                    r /= len(ls)
                    g /= len(ls)
                    b /= len(ls)
                    cs.append((r, g, b, ))
                elif(colorize == 'GROUP_MONO'):
                    w = v[group_layer].get(group_layer_index, 0.0)
                    cs.append((w, w, w, ))
                elif(colorize == 'GROUP_COLOR'):
                    w = v[group_layer].get(group_layer_index, 0.0)
                    hue = remap(1.0 - w, 0.0, 1.0, 0.0, 1 / 1.5)
                    c = Color()
                    c.hsv = (hue, 1.0, 1.0, )
                    cs.append((c.r, c.g, c.b, ))
        
        # and shuffle..
        a = np.concatenate((vs, ns, cs), axis=1, )
        np.random.shuffle(a)
        vs = a[:, :3]
        ns = a[:, 3:6]
        cs = a[:, 6:]
        
        self.vs = vs[:]
        self.ns = ns[:]
        self.cs = cs[:]
        
        bm.free()
        owner.to_mesh_clear()


class PCVIVDraftSampler():
    def __init__(self, context, target, percentage=1.0, seed=0, colorize=None, constant_color=None, ):
        log("{}:".format(self.__class__.__name__), 0)
        
        if(colorize is None):
            colorize = 'CONSTANT'
        if(constant_color is None):
            constant_color = (1.0, 0.0, 0.0, )
        
        me = target.data
        
        if(len(me.polygons) == 0):
            raise Exception("Mesh has no faces")
        if(colorize in ('VIEWPORT_DISPLAY_COLOR', )):
            if(len(me.polygons) == 0):
                raise Exception("Mesh has no faces")
        if(colorize == 'VIEWPORT_DISPLAY_COLOR'):
            if(len(target.data.materials) == 0):
                raise Exception("Cannot find any material")
            materials = target.data.materials
        
        vs = []
        ns = []
        cs = []
        
        np.random.seed(seed=seed)
        rnd = np.random.uniform(low=0.0, high=1.0, size=len(me.polygons), )
        
        for i, f in enumerate(me.polygons):
            if(percentage < 1.0):
                if(rnd[i] > percentage):
                    continue
            
            v = f.center
            vs.append(v.to_tuple())
            ns.append(f.normal.to_tuple())
            
            if(colorize == 'CONSTANT'):
                cs.append(constant_color)
            elif(colorize == 'VIEWPORT_DISPLAY_COLOR'):
                c = materials[f.material_index].diffuse_color[:3]
                c = [v ** (1 / 2.2) for v in c]
                cs.append(c)
        
        # # skip normals..
        # n = len(vs)
        # ns = np.column_stack((np.full(n, 0.0, dtype=np.float32, ),
        #                       np.full(n, 0.0, dtype=np.float32, ),
        #                       np.full(n, 1.0, dtype=np.float32, ), ))
        
        # and shuffle..
        a = np.concatenate((vs, ns, cs), axis=1, )
        np.random.shuffle(a)
        vs = a[:, :3]
        ns = a[:, 3:6]
        cs = a[:, 6:]
        
        self.vs = vs[:]
        self.ns = ns[:]
        self.cs = cs[:]


class PCVIVDraftPercentageNumpySampler():
    def __init__(self, context, target, percentage=1.0, seed=0, colorize=None, constant_color=None, ):
        log("{}:".format(self.__class__.__name__), 0)
        
        if(colorize is None):
            colorize = 'CONSTANT'
        if(constant_color is None):
            constant_color = (1.0, 0.0, 0.0, )
        
        me = target.data
        
        if(len(me.polygons) == 0):
            raise Exception("Mesh has no faces")
        if(colorize in ('VIEWPORT_DISPLAY_COLOR', )):
            if(len(me.polygons) == 0):
                raise Exception("Mesh has no faces")
        if(colorize == 'VIEWPORT_DISPLAY_COLOR'):
            if(len(target.data.materials) == 0):
                raise Exception("Cannot find any material")
            materials = target.data.materials
        
        vs = []
        ns = []
        cs = []
        
        l = len(me.polygons)
        
        np.random.seed(seed=seed)
        rnd = np.random.uniform(low=0.0, high=1.0, size=l, )
        
        centers = np.zeros((l * 3), dtype=np.float32, )
        me.polygons.foreach_get('center', centers, )
        centers.shape = (l, 3)
        
        normals = np.zeros((l * 3), dtype=np.float32, )
        me.polygons.foreach_get('normal', normals, )
        normals.shape = (l, 3)
        
        rnd[rnd < percentage] = 1
        rnd[rnd < 1] = 0
        indices = []
        for i, v in enumerate(rnd):
            if(v):
                indices.append(i)
        indices = np.array(indices, dtype=np.int, )
        
        li = len(indices)
        if(colorize == 'CONSTANT'):
            colors = np.column_stack((np.full(li, constant_color[0], dtype=np.float32, ),
                                      np.full(li, constant_color[1], dtype=np.float32, ),
                                      np.full(li, constant_color[2], dtype=np.float32, ), ))
        elif(colorize == 'VIEWPORT_DISPLAY_COLOR'):
            colors = np.zeros((li, 3), dtype=np.float32, )
            for i, index in enumerate(indices):
                p = me.polygons[index]
                c = materials[p.material_index].diffuse_color[:3]
                c = [v ** (1 / 2.2) for v in c]
                colors[i][0] = c[0]
                colors[i][1] = c[1]
                colors[i][2] = c[2]
        
        vs = np.take(centers, indices, axis=0, )
        ns = np.take(normals, indices, axis=0, )
        cs = colors
        
        # NOTE: shuffle can be removed if i am not going to use all points, shuffle also slows everything down
        
        # and shuffle..
        a = np.concatenate((vs, ns, cs), axis=1, )
        np.random.shuffle(a)
        vs = a[:, :3]
        ns = a[:, 3:6]
        cs = a[:, 6:]
        
        self.vs = vs[:]
        self.ns = ns[:]
        self.cs = cs[:]


class PCVIVDraftFixedCountNumpySampler():
    def __init__(self, context, target, count=-1, seed=0, colorize=None, constant_color=None, ):
        # log("{}:".format(self.__class__.__name__), 0)
        # log("target: {}, count: {}".format(target, count, ), 1)
        
        if(colorize is None):
            colorize = 'CONSTANT'
        if(constant_color is None):
            constant_color = (1.0, 0.0, 0.0, )
        
        me = target.data
        
        if(len(me.polygons) == 0):
            raise Exception("Mesh has no faces")
        if(colorize in ('VIEWPORT_DISPLAY_COLOR', )):
            if(len(me.polygons) == 0):
                raise Exception("Mesh has no faces")
        if(colorize == 'VIEWPORT_DISPLAY_COLOR'):
            if(len(target.data.materials) == 0):
                raise Exception("Cannot find any material")
            materials = target.data.materials
        
        vs = []
        ns = []
        cs = []
        
        l = len(me.polygons)
        if(count == -1):
            count = l
        if(count > l):
            count = l
        
        np.random.seed(seed=seed)
        
        centers = np.zeros((l * 3), dtype=np.float32, )
        me.polygons.foreach_get('center', centers, )
        centers.shape = (l, 3)
        
        normals = np.zeros((l * 3), dtype=np.float32, )
        me.polygons.foreach_get('normal', normals, )
        normals.shape = (l, 3)
        
        indices = np.random.randint(0, l, count, dtype=np.int, )
        
        material_indices = np.zeros(l, dtype=np.int, )
        me.polygons.foreach_get('material_index', material_indices, )
        material_colors = np.zeros((len(materials), 3), dtype=np.float32, )
        for i, m in enumerate(materials):
            mc = m.diffuse_color[:3]
            material_colors[i][0] = mc[0] ** (1 / 2.2)
            material_colors[i][1] = mc[1] ** (1 / 2.2)
            material_colors[i][2] = mc[2] ** (1 / 2.2)
        
        li = len(indices)
        if(colorize == 'CONSTANT'):
            colors = np.column_stack((np.full(li, constant_color[0], dtype=np.float32, ),
                                      np.full(li, constant_color[1], dtype=np.float32, ),
                                      np.full(li, constant_color[2], dtype=np.float32, ), ))
        elif(colorize == 'VIEWPORT_DISPLAY_COLOR'):
            colors = np.zeros((li, 3), dtype=np.float32, )
            colors = np.take(material_colors, material_indices, axis=0,)
        
        if(l == count):
            vs = centers
            ns = normals
            cs = colors
        else:
            vs = np.take(centers, indices, axis=0, )
            ns = np.take(normals, indices, axis=0, )
            cs = np.take(colors, indices, axis=0, )
        
        # NOTE: shuffle can be removed if i am not going to use all points, shuffle also slows everything down, but display won't work as nicely as it does now..
        
        # and shuffle..
        a = np.concatenate((vs, ns, cs), axis=1, )
        np.random.shuffle(a)
        vs = a[:, :3]
        ns = a[:, 3:6]
        cs = a[:, 6:]
        
        self.vs = vs[:]
        self.ns = ns[:]
        self.cs = cs[:]


class PCVIVDraftWeightedFixedCountNumpySampler():
    def __init__(self, context, target, count=-1, seed=0, colorize=None, constant_color=None, ):
        # log("{}:".format(self.__class__.__name__), 0)
        # log("target: {}, count: {}".format(target, count, ), 1)
        
        if(colorize is None):
            colorize = 'CONSTANT'
        if(constant_color is None):
            constant_color = (1.0, 0.0, 0.0, )
        
        me = target.data
        
        if(len(me.polygons) == 0):
            raise Exception("Mesh has no faces")
        if(colorize in ('VIEWPORT_DISPLAY_COLOR', )):
            if(len(me.polygons) == 0):
                raise Exception("Mesh has no faces")
        if(colorize == 'VIEWPORT_DISPLAY_COLOR'):
            if(len(target.data.materials) == 0):
                raise Exception("Cannot find any material")
            materials = target.data.materials
        
        vs = []
        ns = []
        cs = []
        
        l = len(me.polygons)
        if(count == -1):
            count = l
        if(count > l):
            count = l
        
        np.random.seed(seed=seed)
        
        centers = np.zeros((l * 3), dtype=np.float32, )
        me.polygons.foreach_get('center', centers, )
        centers.shape = (l, 3)
        
        normals = np.zeros((l * 3), dtype=np.float32, )
        me.polygons.foreach_get('normal', normals, )
        normals.shape = (l, 3)
        
        # indices = np.random.randint(0, l, count, dtype=np.int, )
        
        choices = np.indices((l, ), dtype=np.int, )
        choices.shape = (l, )
        weights = np.zeros(l, dtype=np.float32, )
        me.polygons.foreach_get('area', weights, )
        # # normalize
        # weights *= (1.0 / np.max(weights))
        # make it all sum to 1.0
        weights *= 1.0 / np.sum(weights)
        indices = np.random.choice(choices, size=count, replace=False, p=weights, )
        
        if(colorize == 'VIEWPORT_DISPLAY_COLOR'):
            material_indices = np.zeros(l, dtype=np.int, )
            me.polygons.foreach_get('material_index', material_indices, )
            material_colors = np.zeros((len(materials), 3), dtype=np.float32, )
            for i, m in enumerate(materials):
                mc = m.diffuse_color[:3]
                material_colors[i][0] = mc[0] ** (1 / 2.2)
                material_colors[i][1] = mc[1] ** (1 / 2.2)
                material_colors[i][2] = mc[2] ** (1 / 2.2)
        
        li = len(indices)
        if(colorize == 'CONSTANT'):
            colors = np.column_stack((np.full(l, constant_color[0] ** (1 / 2.2), dtype=np.float32, ),
                                      np.full(l, constant_color[1] ** (1 / 2.2), dtype=np.float32, ),
                                      np.full(l, constant_color[2] ** (1 / 2.2), dtype=np.float32, ), ))
        elif(colorize == 'VIEWPORT_DISPLAY_COLOR'):
            colors = np.zeros((li, 3), dtype=np.float32, )
            colors = np.take(material_colors, material_indices, axis=0,)
        
        if(l == count):
            vs = centers
            ns = normals
            cs = colors
        else:
            vs = np.take(centers, indices, axis=0, )
            ns = np.take(normals, indices, axis=0, )
            cs = np.take(colors, indices, axis=0, )
        
        # NOTE: shuffle can be removed if i am not going to use all points, shuffle also slows everything down, but display won't work as nicely as it does now..
        
        # and shuffle..
        a = np.concatenate((vs, ns, cs), axis=1, )
        np.random.shuffle(a)
        vs = a[:, :3]
        ns = a[:, 3:6]
        cs = a[:, 6:]
        
        self.vs = vs[:]
        self.ns = ns[:]
        self.cs = cs[:]


class PCVIVDraftWeightedFixedCountNumpyWeightedColorsSampler():
    def __init__(self, context, target, count=-1, seed=0, colorize=None, constant_color=None, use_face_area=None, use_material_factors=None, ):
        if(colorize is None):
            colorize = 'CONSTANT'
        if(constant_color is None):
            # constant_color = (1.0, 0.0, 0.0, )
            constant_color = (1.0, 0.0, 1.0, )
        if(use_material_factors is None and colorize == 'VIEWPORT_DISPLAY_COLOR'):
            use_material_factors = False
        if(use_material_factors):
            if(colorize == 'CONSTANT'):
                use_material_factors = False
        
        me = target.data
        
        if(len(me.polygons) == 0):
            # raise Exception("Mesh has no faces")
            # no polygons to generate from, use origin
            self.vs = np.array(((0.0, 0.0, 0.0, ), ), dtype=np.float32, )
            self.ns = np.array(((0.0, 0.0, 1.0, ), ), dtype=np.float32, )
            self.cs = np.array(((1.0, 0.0, 1.0, ), ), dtype=np.float32, )
            return
        # if(colorize in ('VIEWPORT_DISPLAY_COLOR', )):
        #     if(len(me.polygons) == 0):
        #         raise Exception("Mesh has no faces")
        if(colorize == 'VIEWPORT_DISPLAY_COLOR'):
            if(len(target.data.materials) == 0):
                # raise Exception("Cannot find any material")
                # no materials, set to constant
                colorize = 'CONSTANT'
                constant_color = (1.0, 0.0, 1.0, )
            materials = target.data.materials
            if(None in materials[:]):
                # if there is empty slot, abort it and set to constant
                # TODO: make some workaround empty slots, this would require check for polygons with that empty material assigned and replacing that with constant color
                colorize = 'CONSTANT'
                constant_color = (1.0, 0.0, 1.0, )
        
        vs = []
        ns = []
        cs = []
        
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
            # material_weights *= 1.0 / np.sum(material_weights)
            material_weights *= 1.0 / np.sum(material_weights)
            # weights = material_weights
            
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
        
        # NOTE: shuffle can be removed if i am not going to use all points, shuffle also slows everything down, but display won't work as nicely as it does now..
        
        # and shuffle..
        a = np.concatenate((vs, ns, cs), axis=1, )
        np.random.shuffle(a)
        vs = a[:, :3]
        ns = a[:, 3:6]
        cs = a[:, 6:]
        
        self.vs = vs[:]
        self.ns = ns[:]
        self.cs = cs[:]


class PCVIV2Manager():
    initialized = False
    cache = {}
    
    use_extra_handlers = True
    undo_redo_catalogue = {}
    psys_existence = {}
    
    pre_save_state = {}
    
    @classmethod
    def init(cls):
        if(cls.initialized):
            return
        log("init", prefix='>>>', )
        # bpy.app.handlers.depsgraph_update_post.append(cls.handler)
        bpy.app.handlers.depsgraph_update_pre.append(cls.uuid_handler)
        
        if(cls.use_extra_handlers):
            # undo/redo handling
            bpy.app.handlers.redo_pre.append(cls.redo_pre)
            bpy.app.handlers.redo_post.append(cls.redo_post)
            bpy.app.handlers.undo_pre.append(cls.undo_pre)
            bpy.app.handlers.undo_post.append(cls.undo_post)
            # psys removal handling
            bpy.app.handlers.depsgraph_update_post.append(cls.psys_existence_post)
        
        bpy.app.handlers.save_pre.append(cls.save_handler_pre)
        bpy.app.handlers.save_post.append(cls.save_handler_post)
        
        bpy.app.handlers.load_pre.append(watcher)
        cls.initialized = True
        cls.uuid_handler(None)
    
    @classmethod
    def deinit(cls):
        if(not cls.initialized):
            return
        log("deinit", prefix='>>>', )
        # bpy.app.handlers.depsgraph_update_post.remove(cls.handler)
        bpy.app.handlers.depsgraph_update_pre.remove(cls.uuid_handler)
        
        if(cls.use_extra_handlers):
            # undo/redo handling
            bpy.app.handlers.redo_pre.remove(cls.redo_pre)
            bpy.app.handlers.redo_post.remove(cls.redo_post)
            bpy.app.handlers.undo_pre.remove(cls.undo_pre)
            bpy.app.handlers.undo_post.remove(cls.undo_post)
            # psys removal handling
            bpy.app.handlers.depsgraph_update_post.remove(cls.psys_existence_post)
        
        bpy.app.handlers.load_pre.remove(watcher)
        cls.initialized = False
    
    @classmethod
    def uuid_handler(cls, scene, ):
        if(not cls.initialized):
            return
        # log("uuid_handler", prefix='>>>', )
        dps = bpy.data.particles
        for ps in dps:
            pcviv = ps.pcv_instance_visualizer
            if(pcviv.uuid == ""):
                log("uuid_handler: found psys without uuid", 1, prefix='>>>', )
                pcviv.uuid = str(uuid.uuid1())
                # if psys is added outside of 3dview
                cls._redraw_view_3d()
    
    @classmethod
    def psys_existence_post(cls, scene, ):
        log("existence post", prefix='>>>', )
        
        # NOTE: this is run on every update that happen in blender, even on something completely unrelated to pcv, pcviv or particles, this might slow down everything
        
        ls = []
        for pset in bpy.data.particles:
            pcviv = pset.pcv_instance_visualizer
            if(pcviv.uuid != ""):
                ls.append((pcviv.uuid, pset, ))
        
        for u, pset in ls:
            if(u in cls.cache.keys()):
                # was in cache, so it should draw, unless psys was removed, or its object
                # so check for objects with psys with that pset
                ok = False
                for o in bpy.data.objects:
                    for psys in o.particle_systems:
                        if(psys.settings == pset):
                            ok = True
                if(not ok):
                    del cls.cache[u]
                    # now i should run update on object, but how to find which one was it?
                    if(u in cls.psys_existence.keys()):
                        onm = cls.psys_existence[u]
                        o = bpy.data.objects.get(onm)
                        if(o is not None):
                            # object still exist
                            cls.update_all(o)
        
        cls.psys_existence = {}
        for o in bpy.data.objects:
            for i, psys in enumerate(o.particle_systems):
                u = psys.settings.pcv_instance_visualizer.uuid
                if(u in cls.cache.keys()):
                    cls.psys_existence[u] = o.name
    
    @classmethod
    def save_handler_pre(cls, scene, ):
        # store by object name which is used as instance visualizer
        for o in bpy.data.objects:
            pcv = o.point_cloud_visualizer
            if(pcv.instance_visualizer_active_hidden_value):
                cls.pre_save_state[o.name] = True
                pcv.instance_visualizer_active_hidden_value = False
    
    @classmethod
    def save_handler_post(cls, scene, ):
        # revert pre save changes..
        for n, v in cls.pre_save_state.items():
            o = bpy.data.objects.get(n)
            if(o is not None):
                pcv = o.point_cloud_visualizer
                pcv.instance_visualizer_active_hidden_value = v
    
    @classmethod
    def undo_pre(cls, scene, ):
        log("undo/redo pre", prefix='>>>', )
        
        for o in bpy.data.objects:
            pcv = o.point_cloud_visualizer
            for i, psys in enumerate(o.particle_systems):
                pset = psys.settings
                pcviv = pset.pcv_instance_visualizer
                u = pcviv.uuid
                if(u in cls.cache.keys()):
                    if(o.name not in cls.undo_redo_catalogue.keys()):
                        cls.undo_redo_catalogue[o.name] = {}
                    cls.undo_redo_catalogue[o.name][pset.name] = u
    
    @classmethod
    def undo_post(cls, scene, ):
        log("undo/redo post", prefix='>>>', )
        
        for onm, psnms in cls.undo_redo_catalogue.items():
            o = bpy.data.objects.get(onm)
            if(o is not None):
                dirty = False
                # ok, object still exists, unless it was object rename that was undone.. huh? now what?
                for psnm, u in psnms.items():
                    pset = bpy.data.particles.get(psnm)
                    if(pset is not None):
                        found = False
                        for psys in o.particle_systems:
                            if(psys.settings == pset):
                                found = True
                                break
                        if(found):
                            # all is ok
                            pass
                        else:
                            # psys is gone, we should remove it
                            dirty = True
                    else:
                        # pset is there, but psys is gone, we should remove it
                        dirty = True
                
                if(dirty):
                    cls.update_all(o)
            else:
                # object is gone, we should remove it
                # this is handled by PCV, no object - no draw
                pass
        
        cls.undo_redo_catalogue = {}
    
    @classmethod
    def redo_pre(cls, scene, ):
        log("undo/redo pre", prefix='>>>', )
        cls.undo_pre(scene)
    
    @classmethod
    def redo_post(cls, scene, ):
        log("undo/redo post", prefix='>>>', )
        cls.undo_post(scene)
    
    @classmethod
    def reset_all(cls):
        # if(not cls.initialized):
        #     return
        
        log("reset_all", prefix='>>>', )
        cls.deinit()
        
        for o in bpy.data.objects:
            pcv = o.point_cloud_visualizer
            pcv.instance_visualizer_active_hidden_value = False
            pcv.dev_minimal_shader_variable_size_enabled = False
            pcv.pcviv_debug_draw = ''
            for i, psys in enumerate(o.particle_systems):
                pset = psys.settings
                pcviv = pset.pcv_instance_visualizer
                pcviv.uuid = ""
                pcviv.draw = True
                pcviv.debug_update = ""
                
                # NOTE: maybe store these somewhere, but these are defaults, so makes sense too
                # NOTE: switch to BOUNDS to keep viewport alive
                if(pset.render_type == 'COLLECTION'):
                    for co in pset.instance_collection.objects:
                        # co.display_type = 'TEXTURED'
                        co.display_type = 'BOUNDS'
                elif(pset.render_type == 'OBJECT'):
                    # pset.instance_object.display_type = 'TEXTURED'
                    pset.instance_object.display_type = 'BOUNDS'
                pset.display_method = 'RENDER'
            
            c = PCVIV2Control(o)
            c.reset()
        
        cls.cache = {}
    
    @classmethod
    def reset(cls, o, ):
        # if(not cls.initialized):
        #     return
        
        log("reset", prefix='>>>', )
        cls.deinit()
        
        pcv = o.point_cloud_visualizer
        pcv.instance_visualizer_active_hidden_value = False
        pcv.dev_minimal_shader_variable_size_enabled = False
        pcv.pcviv_debug_draw = ''
        keys = []
        for i, psys in enumerate(o.particle_systems):
            pset = psys.settings
            pcviv = pset.pcv_instance_visualizer
            keys.append(pcviv.uuid)
            pcviv.uuid = ""
            pcviv.draw = True
            pcviv.debug_update = ""
            
            # NOTE: maybe store these somewhere, but these are defaults, so makes sense too
            # NOTE: switch to BOUNDS to keep viewport alive
            if(pset.render_type == 'COLLECTION'):
                for co in pset.instance_collection.objects:
                    # co.display_type = 'TEXTURED'
                    co.display_type = 'BOUNDS'
            elif(pset.render_type == 'OBJECT'):
                # pset.instance_object.display_type = 'TEXTURED'
                pset.instance_object.display_type = 'BOUNDS'
            pset.display_method = 'RENDER'
        
        c = PCVIV2Control(o)
        c.reset()
        
        # delete just reseted keys from cache
        for k in keys:
            if(k in cls.cache):
                del cls.cache[k]
        # # initialize back if there are some keys in cache to keep other objects with visualizations
        # if(len(cls.cache.keys()) > 0):
        #     cls.init()
        
        # always initialize back after single object reset
        cls.init()
    
    @classmethod
    def update(cls, o, uuid, skip_render=False, ):
        if(not cls.initialized):
            return
        
        pcv = o.point_cloud_visualizer
        pcv.instance_visualizer_active_hidden_value = True
        
        log("update", prefix='>>>', )
        
        if(uuid in cls.cache.keys()):
            log("update: is cached, setting dirty..", 1, prefix='>>>', )
            cls.cache[uuid]['dirty'] = True
        else:
            found = False
            for i, psys in enumerate(o.particle_systems):
                pcviv = psys.settings.pcv_instance_visualizer
                if(pcviv.uuid == uuid):
                    found = True
                    break
            if(not found):
                raise Exception('PCVIV2Manager.update, uuid {} not found in particle systems on object: {}'.format(uuid, o.name))
            
            log("update: psys not in cache, adding..", 1, prefix='>>>', )
            # not yet in cache, we should fix that
            ci = {'uuid': uuid,
                  # we just started, so it's always dirty, yeah baby..
                  'dirty': True,
                  'draw': pcviv.draw,
                  # 'draw': True,
                  'vs': None,
                  'ns': None,
                  'cs': None,
                  'sz': None, }
            cls.cache[uuid] = ci
        
        if(not skip_render):
            cls.render(o)
    
    @classmethod
    def update_all(cls, o, ):
        if(not cls.initialized):
            return
        
        log("update_all", prefix='>>>', )
        
        ls = []
        for i, psys in enumerate(o.particle_systems):
            pcviv = psys.settings.pcv_instance_visualizer
            ls.append(pcviv.uuid)
        for u in ls:
            cls.update(o, u, True, )
        
        cls.render(o)
    
    @classmethod
    def draw_update(cls, o, uuid, do_draw, ):
        if(not cls.initialized):
            return
        
        log("draw_update", prefix='>>>', )
        
        found = False
        for k, v in cls.cache.items():
            if(k == uuid):
                found = True
                break
        
        if(not found):
            # if not found, it should be first time called..
            log("draw_update: uuid not found in cache, updating..", 1, prefix='>>>', )
            # return
            cls.update(o, uuid, )
            return
        
        ci = cls.cache[uuid]['draw'] = do_draw
        # cls.update(uuid)
        cls.render(o)
    
    @classmethod
    def generate_psys(cls, o, psys_slot, max_points, color_source, color_constant, use_face_area, use_material_factors, ):
        log("generate_psys", prefix='>>>', )
        
        # FIXME: this should be created just once and passed to next call in the same update.. possible optimization?
        depsgraph = bpy.context.evaluated_depsgraph_get()
        
        o = o.evaluated_get(depsgraph)
        # why can particle systems have the same name? WHY? NOTHING else works like that in blender
        # psys = o.particle_systems[psys.name]
        psys = o.particle_systems[psys_slot]
        settings = psys.settings
        
        if(settings.render_type == 'COLLECTION'):
            collection = settings.instance_collection
            cos = collection.objects
            fragments = []
            fragments_indices = {}
            for i, co in enumerate(cos):
                no_geometry = False
                if(co.type not in ('MESH', 'CURVE', 'SURFACE', 'FONT', )):
                    # self.report({'ERROR'}, "Object does not have geometry data.")
                    # return {'CANCELLED'}
                    # NOTE: handle no mesh objects by generating single vertex in object origin (0,0,0)
                    no_geometry = True
                if(no_geometry):
                    fragments.append((
                        np.array(((0.0, 0.0, 0.0, ), ), dtype=np.float32, ),
                        np.array(((0.0, 0.0, 1.0, ), ), dtype=np.float32, ),
                        np.array(((1.0, 0.0, 1.0, ), ), dtype=np.float32, ),
                    ), )
                    fragments_indices[co.name] = (i, co, )
                else:
                    # extract points
                    # sampler = PCVIVDraftWeightedFixedCountNumpySampler(bpy.context, co, count=max_points, colorize='VIEWPORT_DISPLAY_COLOR', )
                    # sampler = PCVIVDraftWeightedFixedCountNumpySampler(bpy.context, co, count=max_points, colorize=color_source, constant_color=color_constant, )
                    sampler = PCVIVDraftWeightedFixedCountNumpyWeightedColorsSampler(bpy.context, co, count=max_points, colorize=color_source, constant_color=color_constant, use_face_area=use_face_area, use_material_factors=use_material_factors, )
                    # store
                    fragments.append((sampler.vs, sampler.ns, sampler.cs, ))
                    # FIXME: better not to access data by object name, find something different
                    fragments_indices[co.name] = (i, co, )
            
            # process all instances, transform fragment and store
            all_frags = []
            # loop over instances
            for object_instance in depsgraph.object_instances:
                obj = object_instance.object
                # if it is from psys
                if(object_instance.particle_system == psys):
                    # and it is instance
                    if(object_instance.is_instance):
                        # get matrix
                        m = object_instance.matrix_world
                        # unapply emitter matrix
                        m = o.matrix_world.inverted() @ m
                        # get correct fragment
                        i, _ = fragments_indices[obj.name]
                        fvs, fns, fcs = fragments[i]
                        # transform
                        fvs.shape = (-1, 3)
                        fvs = np.c_[fvs, np.ones(fvs.shape[0])]
                        fvs = np.dot(m, fvs.T)[0:3].T.reshape((-1))
                        fvs.shape = (-1, 3)
                        # transform also normals
                        _, rot, _ = m.decompose()
                        rmat = rot.to_matrix().to_4x4()
                        fns.shape = (-1, 3)
                        fns = np.c_[fns, np.ones(fns.shape[0])]
                        fns = np.dot(rmat, fns.T)[0:3].T.reshape((-1))
                        fns.shape = (-1, 3)
                        # store
                        all_frags.append((fvs, fns, fcs, ))
            
            # join all frags
            if(len(all_frags) == 0):
                # vs = []
                # ns = []
                # cs = []
                vs = np.zeros(0, dtype=np.float32, )
                ns = np.zeros(0, dtype=np.float32, )
                cs = np.zeros(0, dtype=np.float32, )
            else:
                vs = np.concatenate([i[0] for i in all_frags], axis=0, )
                ns = np.concatenate([i[1] for i in all_frags], axis=0, )
                cs = np.concatenate([i[2] for i in all_frags], axis=0, )
            
        elif(settings.render_type == 'OBJECT'):
            co = settings.instance_object
            if(co.type not in ('MESH', 'CURVE', 'SURFACE', 'FONT', )):
                self.report({'ERROR'}, "Object does not have geometry data.")
                return {'CANCELLED'}
            # extract points
            # sampler = PCVIVDraftWeightedFixedCountNumpySampler(bpy.context, co, count=max_points, colorize='VIEWPORT_DISPLAY_COLOR', )
            # sampler = PCVIVDraftWeightedFixedCountNumpySampler(bpy.context, co, count=max_points, colorize=color_source, constant_color=color_constant, )
            sampler = PCVIVDraftWeightedFixedCountNumpyWeightedColorsSampler(bpy.context, co, count=max_points, colorize=color_source, constant_color=color_constant, use_face_area=use_face_area, use_material_factors=use_material_factors, )
            ofvs = sampler.vs
            ofns = sampler.ns
            ofcs = sampler.cs
            
            all_frags = []
            for object_instance in depsgraph.object_instances:
                obj = object_instance.object
                if(object_instance.particle_system == psys):
                    if(object_instance.is_instance):
                        m = object_instance.matrix_world
                        m = o.matrix_world.inverted() @ m
                        fvs = ofvs[:]
                        fvs.shape = (-1, 3)
                        fvs = np.c_[fvs, np.ones(fvs.shape[0])]
                        fvs = np.dot(m, fvs.T)[0:3].T.reshape((-1))
                        fvs.shape = (-1, 3)
                        _, rot, _ = m.decompose()
                        rmat = rot.to_matrix().to_4x4()
                        fns = ofns[:]
                        fns.shape = (-1, 3)
                        fns = np.c_[fns, np.ones(fns.shape[0])]
                        fns = np.dot(rmat, fns.T)[0:3].T.reshape((-1))
                        fns.shape = (-1, 3)
                        all_frags.append((fvs, fns, ofcs, ))
            
            if(len(all_frags) == 0):
                vs = np.zeros(0, dtype=np.float32, )
                ns = np.zeros(0, dtype=np.float32, )
                cs = np.zeros(0, dtype=np.float32, )
            else:
                vs = np.concatenate([i[0] for i in all_frags], axis=0, )
                ns = np.concatenate([i[1] for i in all_frags], axis=0, )
                cs = np.concatenate([i[2] for i in all_frags], axis=0, )
            
        else:
            # just generate pink points
            l = len(psys.particles)
            vs = np.zeros((l * 3), dtype=np.float32, )
            psys.particles.foreach_get('location', vs, )
            vs.shape = (l, 3)
            
            m = o.matrix_world.inverted()
            vs.shape = (-1, 3)
            vs = np.c_[vs, np.ones(vs.shape[0])]
            vs = np.dot(m, vs.T)[0:3].T.reshape((-1))
            vs.shape = (-1, 3)
            
            ns = np.zeros((l * 3), dtype=np.float32, )
            # NOTE: what should i consider as normal in particles? rotation? velocity? sometimes is rotation just identity quaternion, do i know nothing about basics? or just too tired? maybe both..
            psys.particles.foreach_get('velocity', ns, )
            _, rot, _ = m.decompose()
            rmat = rot.to_matrix().to_4x4()
            ns.shape = (-1, 3)
            ns = np.c_[ns, np.ones(ns.shape[0])]
            ns = np.dot(rmat, ns.T)[0:3].T.reshape((-1))
            ns.shape = (-1, 3)
            
            cs = np.column_stack((np.full(l, 1.0, dtype=np.float32, ),
                                  np.full(l, 0.0, dtype=np.float32, ),
                                  np.full(l, 1.0, dtype=np.float32, ), ))
        
        return vs, ns, cs
    
    @classmethod
    def render(cls, o, ):
        # if(not cls.initialized):
        #     return
        
        log("render", prefix='>>>', )
        
        def pre_generate(o, psys, ):
            dts = None
            dt = None
            psys_collection = None
            psys_object = None
            settings = psys.settings
            if(settings.render_type == 'COLLECTION'):
                psys_collection = settings.instance_collection
                dts = []
                for co in psys_collection.objects:
                    dts.append((co, co.display_type, ))
                    co.display_type = 'BOUNDS'
            elif(settings.render_type == 'OBJECT'):
                psys_object = settings.instance_object
                dt = psys_object.display_type
                psys_object.display_type = 'BOUNDS'
            settings.display_method = 'RENDER'
            
            mod = None
            mview = None
            for m in o.modifiers:
                if(m.type == 'PARTICLE_SYSTEM'):
                    if(m.particle_system == psys):
                        mod = m
                        mview = m.show_viewport
                        m.show_viewport = True
                        break
            
            return dts, dt, psys_collection, psys_object, mod, mview
        
        def post_generate(psys, dts, dt, psys_collection, psys_object, mod, mview, ):
            settings = psys.settings
            settings.display_method = 'NONE'
            if(settings.render_type == 'COLLECTION'):
                for co, dt in dts:
                    co.display_type = dt
            elif(settings.render_type == 'OBJECT'):
                psys_object.display_type = dt
            mod.show_viewport = mview
        
        a = []
        for i, psys in enumerate(o.particle_systems):
            _t = time.time()
            
            # pcv = o.point_cloud_visualizer
            pcviv = psys.settings.pcv_instance_visualizer
            if(pcviv.uuid not in cls.cache.keys()):
                continue
            
            ci = cls.cache[pcviv.uuid]
            if(ci['dirty']):
                log("render: psys is dirty", 1, prefix='>>>', )
                
                dts, dt, psys_collection, psys_object, mod, mview = pre_generate(o, psys, )
                vs, ns, cs = cls.generate_psys(o, i, pcviv.max_points, pcviv.color_source, pcviv.color_constant, pcviv.use_face_area, pcviv.use_material_factors, )
                post_generate(psys, dts, dt, psys_collection, psys_object, mod, mview, )
                ci['vs'] = vs
                ci['ns'] = ns
                ci['cs'] = cs
                
                # FIXME: PCV handles different shaders in a kinda messy way, would be nice to rewrite PCVManager.render to be more flexible, get rid of basic shader auto-creation, store extra shaders in the same way as default one, get rid of booleans for enabling extra shaders and make it to enum, etc.. big task, but it will make next customization much easier and simpler..
                # sz = np.full(len(vs), pcviv.point_size, dtype=np.int8, )
                sz = np.full(len(vs), pcviv.point_size, dtype=np.int, )
                ci['sz'] = sz
                
                szf = np.full(len(vs), pcviv.point_size_f, dtype=np.float32, )
                ci['szf'] = szf
                
                ci['dirty'] = False
            
            if(ci['draw']):
                log("render: psys is marked to draw", 1, prefix='>>>', )
                
                a.append((ci['vs'], ci['ns'], ci['cs'], ci['sz'], ci['szf'], ))
            
            _d = datetime.timedelta(seconds=time.time() - _t)
            pcviv.debug_update = "last update completed in {}".format(_d)
        
        _t = time.time()
        vs = []
        ns = []
        cs = []
        sz = []
        szf = []
        av = []
        an = []
        ac = []
        az = []
        azf = []
        for v, n, c, s, f in a:
            if(len(v) == 0):
                # skip systems with zero particles
                continue
            av.append(v)
            an.append(n)
            ac.append(c)
            az.append(s)
            azf.append(f)
        if(len(av) > 0):
            vs = np.concatenate(av, axis=0, )
            ns = np.concatenate(an, axis=0, )
            cs = np.concatenate(ac, axis=0, )
            sz = np.concatenate(az, axis=0, )
            szf = np.concatenate(azf, axis=0, )
        
        log("render: drawing..", 1, prefix='>>>', )
        c = PCVIV2Control(o)
        c.draw(vs, ns, cs, sz, szf, )
        
        _d = datetime.timedelta(seconds=time.time() - _t)
        pcv = o.point_cloud_visualizer
        pcv.pcviv_debug_draw = "last draw completed in {}".format(_d)
    
    @classmethod
    def _redraw_view_3d(cls):
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if(area.type == 'VIEW_3D'):
                    area.tag_redraw()


@persistent
def watcher(scene):
    PCVIV2Manager.deinit()


class PCVIV2_OT_init(Operator):
    bl_idname = "point_cloud_visualizer.pcviv_init"
    bl_label = "Initialize"
    bl_description = "Initialize Instance Visualizer"
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        return True
    
    def execute(self, context):
        PCVIV2Manager.init()
        return {'FINISHED'}


class PCVIV2_OT_deinit(Operator):
    bl_idname = "point_cloud_visualizer.pcviv_deinit"
    bl_label = "Deinitialize"
    bl_description = "Deinitialize Instance Visualizer"
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        return True
    
    def execute(self, context):
        PCVIV2Manager.deinit()
        return {'FINISHED'}


class PCVIV2_OT_reset(Operator):
    bl_idname = "point_cloud_visualizer.pcviv_reset"
    bl_label = "Reset"
    bl_description = "Reset active object particle systems visualizations"
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        return True
    
    def execute(self, context):
        PCVIV2Manager.reset(context.object, )
        return {'FINISHED'}


class PCVIV2_OT_reset_all(Operator):
    bl_idname = "point_cloud_visualizer.pcviv_reset_all"
    bl_label = "Reset All"
    bl_description = "Reset all particle systems visualizations on all objects"
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        return True
    
    def execute(self, context):
        PCVIV2Manager.reset_all()
        return {'FINISHED'}


class PCVIV2_OT_update(Operator):
    bl_idname = "point_cloud_visualizer.pcviv_update"
    bl_label = "Update"
    bl_description = "Update point cloud visualization by particle system UUID"
    
    uuid: StringProperty(name="UUID", default='', )
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        return True
    
    def execute(self, context):
        PCVIV2Manager.update(context.object, self.uuid, )
        return {'FINISHED'}


class PCVIV2_OT_update_all(Operator):
    bl_idname = "point_cloud_visualizer.pcviv_update_all"
    bl_label = "Update All"
    bl_description = "Update all point cloud visualizations for active object"
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        return True
    
    def execute(self, context):
        PCVIV2Manager.update_all(context.object, )
        return {'FINISHED'}


class PCVIV2_OT_dev_transform_normals(Operator):
    bl_idname = "point_cloud_visualizer.pcviv_dev_transform_normals"
    bl_label = "dev_transform_normals"
    bl_description = ""
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        return True
    
    def execute(self, context):
        o = context.object
        pcv = o.point_cloud_visualizer
        
        # sample target object
        to = pcv.dev_transform_normals_target_object
        sampler = PCVIVDraftWeightedFixedCountNumpySampler(bpy.context, to, count=1000, colorize='CONSTANT', constant_color=(1.0, 0.0, 0.0), )
        vs = sampler.vs
        ns = sampler.ns
        cs = sampler.cs
        
        # target and emitter matrices
        m = to.matrix_world
        m = o.matrix_world.inverted() @ m
        
        # transform vertices
        vs.shape = (-1, 3)
        vs = np.c_[vs, np.ones(vs.shape[0])]
        vs = np.dot(m, vs.T)[0:3].T.reshape((-1))
        vs.shape = (-1, 3)
        
        # transform normals
        _, r, _ = m.decompose()
        m = r.to_matrix().to_4x4()
        
        ns.shape = (-1, 3)
        ns = np.c_[ns, np.ones(ns.shape[0])]
        ns = np.dot(m, ns.T)[0:3].T.reshape((-1))
        ns.shape = (-1, 3)
        
        # for n in ns:
        #     print(sum(n ** 2) ** 0.5)
        
        # fixed sizes
        sz = np.full(len(vs), 3, dtype=np.int, )
        
        # draw
        c = PCVIV2Control(o)
        c.draw(vs, ns, cs, sz, )
        
        return {'FINISHED'}


class PCVIV2Control(PCVControl):
    def __init__(self, o, ):
        super(PCVIV2Control, self, ).__init__(o, )
        # pcv = o.point_cloud_visualizer
        # pcv.dev_minimal_shader_variable_size_enabled = True
    
    def draw(self, vs=None, ns=None, cs=None, sz=None, szf=None, ):
        o = self.o
        pcv = o.point_cloud_visualizer
        
        # pcv.dev_minimal_shader_variable_size_enabled = True
        # pcv.dev_minimal_shader_variable_size_and_depth_enabled = True
        # pcv.dev_rich_billboard_point_cloud_enabled = True
        
        # FIXME: this is also stupid
        if(pcv.dev_minimal_shader_variable_size_enabled or pcv.dev_minimal_shader_variable_size_and_depth_enabled or pcv.dev_rich_billboard_point_cloud_enabled or pcv.dev_rich_billboard_point_cloud_no_depth_enabled):
            pass
        else:
            # NOTE: this is what will be default draw type, should i put preferences for it somewhere?
            # pcv.dev_minimal_shader_variable_size_and_depth_enabled = True
            
            # lets go with the best one
            pcv.dev_rich_billboard_point_cloud_enabled = True
        
        # check if object has been used before, i.e. has uuid and uuid item is in cache
        if(pcv.uuid != "" and pcv.runtime):
            # was used or blend was saved after it was used and uuid is saved from last time, check cache
            if(pcv.uuid in PCVManager.cache):
                # cache item is found, object has been used before
                self._update(vs, ns, cs, sz, szf, )
                return
        # otherwise setup as new
        
        u = str(uuid.uuid1())
        # use that as path, some checks wants this not empty
        filepath = u
        
        # validate/prepare input data
        vs, ns, cs, points, has_normals, has_colors = self._prepare(vs, ns, cs)
        n = len(vs)
        
        # TODO: validate also sz array
        
        # build cache dict
        d = {}
        d['uuid'] = u
        d['filepath'] = filepath
        d['points'] = points
        
        # but because colors i just stored in uint8, store them also as provided to enable reload operator
        cs_orig = np.column_stack((cs[:, 0], cs[:, 1], cs[:, 2], np.ones(n), ))
        cs_orig = cs_orig.astype(np.float32)
        d['colors_original'] = cs_orig
        
        d['stats'] = n
        d['vertices'] = vs
        d['colors'] = cs
        d['normals'] = ns
        d['length'] = n
        dp = pcv.display_percent
        l = int((n / 100) * dp)
        if(dp >= 99):
            l = n
        d['display_length'] = l
        d['current_display_length'] = l
        # d['illumination'] = pcv.illumination
        d['illumination'] = False
        
        # if(pcv.dev_minimal_shader_variable_size_enabled):
        #     shader = GPUShader(PCVShaders.vertex_shader_minimal_variable_size, PCVShaders.fragment_shader_minimal_variable_size, )
        #     batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "color": cs[:l], "size": sz[:l], })
        # else:
        #     shader = GPUShader(PCVShaders.vertex_shader_simple, PCVShaders.fragment_shader_simple)
        #     batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "color": cs[:l], })
        
        # shader = GPUShader(PCVShaders.vertex_shader_simple, PCVShaders.fragment_shader_simple)
        shader_data_vert, shader_data_frag, shader_data_geom = load_shader_code('simple')
        shader = GPUShader(shader_data_vert, shader_data_frag)
        
        batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "color": cs[:l], })
        d['shader'] = shader
        d['batch'] = batch
        d['ready'] = True
        d['draw'] = False
        d['kill'] = False
        d['object'] = o
        d['name'] = o.name
        
        d['extra'] = {}
        if(pcv.dev_minimal_shader_variable_size_enabled):
            shader_data_vert, shader_data_frag, shader_data_geom = load_shader_code('minimal_variable_size')
            e_shader = GPUShader(shader_data_vert, shader_data_frag)
            
            e_batch = batch_for_shader(e_shader, 'POINTS', {"position": vs[:l], "color": cs[:l], "size": sz[:l], })
            extra = {
                'shader': e_shader,
                'batch': e_batch,
                'sizes': sz,
                'sizesf': szf,
                'length': l,
            }
            d['extra']['MINIMAL_VARIABLE_SIZE'] = extra
        if(pcv.dev_minimal_shader_variable_size_and_depth_enabled):
            # TODO: do the same for the other shader until i decide which is better..
            # e_shader = GPUShader(PCVShaders.vertex_shader_minimal_variable_size_and_depth, PCVShaders.fragment_shader_minimal_variable_size_and_depth, )
            
            shader_data_vert, shader_data_frag, shader_data_geom = load_shader_code('minimal_variable_size_and_depth')
            e_shader = GPUShader(shader_data_vert, shader_data_frag)
            
            e_batch = batch_for_shader(e_shader, 'POINTS', {"position": vs[:l], "color": cs[:l], "size": sz[:l], })
            extra = {
                'shader': e_shader,
                'batch': e_batch,
                'sizes': sz,
                'sizesf': szf,
                'length': l,
            }
            d['extra']['MINIMAL_VARIABLE_SIZE_AND_DEPTH'] = extra
        if(pcv.dev_rich_billboard_point_cloud_enabled):
            # FIXME: this is getting ridiculous
            # e_shader = GPUShader(PCVShaders.billboard_vertex_with_depth_and_size, PCVShaders.billboard_fragment_with_depth_and_size, geocode=PCVShaders.billboard_geometry_with_depth_and_size, )
            
            shader_data_vert, shader_data_frag, shader_data_geom = load_shader_code('billboard_with_depth_and_size')
            e_shader = GPUShader(shader_data_vert, shader_data_frag, geocode=shader_data_geom, )
            
            e_batch = batch_for_shader(e_shader, 'POINTS', {"position": vs[:l], "color": cs[:l], "sizef": szf[:l], })
            extra = {
                'shader': e_shader,
                'batch': e_batch,
                'sizes': sz,
                'sizesf': szf,
                'length': l,
            }
            d['extra']['RICH_BILLBOARD'] = extra
        
        if(pcv.dev_rich_billboard_point_cloud_no_depth_enabled):
            # FIXME: this is getting ridiculous
            # e_shader = GPUShader(PCVShaders.billboard_vertex_with_no_depth_and_size, PCVShaders.billboard_fragment_with_no_depth_and_size, geocode=PCVShaders.billboard_geometry_with_no_depth_and_size, )
            
            shader_data_vert, shader_data_frag, shader_data_geom = load_shader_code('billboard_with_no_depth_and_size')
            e_shader = GPUShader(shader_data_vert, shader_data_frag, geocode=shader_data_geom, )
            
            e_batch = batch_for_shader(e_shader, 'POINTS', {"position": vs[:l], "color": cs[:l], "sizef": szf[:l], })
            extra = {
                'shader': e_shader,
                'batch': e_batch,
                'sizes': sz,
                'sizesf': szf,
                'length': l,
            }
            d['extra']['RICH_BILLBOARD_NO_DEPTH'] = extra
        
        # d['extra_data'] = {
        #     'sizes': sz,
        #     'sizesf': szf,
        # }
        
        # set properties
        pcv.uuid = u
        pcv.filepath = filepath
        pcv.has_normals = has_normals
        pcv.has_vcols = has_colors
        pcv.runtime = True
        
        PCVManager.add(d)
        
        # mark to draw
        c = PCVManager.cache[pcv.uuid]
        c['draw'] = True
        
        self._redraw()
    
    def _update(self, vs, ns, cs, sz, szf, ):
        o = self.o
        pcv = o.point_cloud_visualizer
        
        # pcv.dev_minimal_shader_variable_size_enabled = True
        # pcv.dev_minimal_shader_variable_size_and_depth_enabled = True
        # pcv.dev_rich_billboard_point_cloud_enabled = True
        
        # FIXME: this is also stupid
        if(pcv.dev_minimal_shader_variable_size_enabled or pcv.dev_minimal_shader_variable_size_and_depth_enabled or pcv.dev_rich_billboard_point_cloud_enabled or pcv.dev_rich_billboard_point_cloud_no_depth_enabled):
            pass
        else:
            # NOTE: this is what will be default draw type, should i put preferences for it somewhere?
            # pcv.dev_minimal_shader_variable_size_and_depth_enabled = True
            
            # lets go with the best one
            pcv.dev_rich_billboard_point_cloud_enabled = True
        
        # validate/prepare input data
        vs, ns, cs, points, has_normals, has_colors = self._prepare(vs, ns, cs)
        n = len(vs)
        
        d = PCVManager.cache[pcv.uuid]
        d['points'] = points
        
        # kill normals, might not be no longer valid, it will be recreated later
        if('vertex_normals' in d.keys()):
            del d['vertex_normals']
        
        # but because colors i just stored in uint8, store them also as provided to enable reload operator
        cs_orig = np.column_stack((cs[:, 0], cs[:, 1], cs[:, 2], np.ones(n), ))
        cs_orig = cs_orig.astype(np.float32)
        d['colors_original'] = cs_orig
        
        d['stats'] = n
        d['vertices'] = vs
        d['colors'] = cs
        d['normals'] = ns
        d['length'] = n
        dp = pcv.display_percent
        l = int((n / 100) * dp)
        if(dp >= 99):
            l = n
        d['display_length'] = l
        d['current_display_length'] = l
        # d['illumination'] = pcv.illumination
        d['illumination'] = False
        # if(pcv.illumination):
        #     shader = GPUShader(PCVShaders.vertex_shader_illumination, PCVShaders.fragment_shader_illumination)
        #     batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "color": cs[:l], "normal": ns[:l], })
        # else:
        #     shader = GPUShader(PCVShaders.vertex_shader_simple, PCVShaders.fragment_shader_simple)
        #     batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "color": cs[:l], })
        
        # shader = GPUShader(PCVShaders.vertex_shader_simple, PCVShaders.fragment_shader_simple)
        shader_data_vert, shader_data_frag, shader_data_geom = load_shader_code('simple')
        shader = GPUShader(shader_data_vert, shader_data_frag)
        
        batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "color": cs[:l], })
        d['shader'] = shader
        d['batch'] = batch
        
        pcv.has_normals = has_normals
        pcv.has_vcols = has_colors
        
        d['extra'] = {}
        if(pcv.dev_minimal_shader_variable_size_enabled):
            # e_shader = GPUShader(PCVShaders.vertex_shader_minimal_variable_size, PCVShaders.fragment_shader_minimal_variable_size, )
            
            shader_data_vert, shader_data_frag, shader_data_geom = load_shader_code('minimal_variable_size')
            e_shader = GPUShader(shader_data_vert, shader_data_frag)
            
            e_batch = batch_for_shader(e_shader, 'POINTS', {"position": vs[:l], "color": cs[:l], "size": sz[:l], })
            extra = {
                'shader': e_shader,
                'batch': e_batch,
                'sizes': sz,
                'sizesf': szf,
                'length': l,
            }
            d['extra']['MINIMAL_VARIABLE_SIZE'] = extra
        if(pcv.dev_minimal_shader_variable_size_and_depth_enabled):
            # TODO: do the same for the other shader until i decide which is better..
            # e_shader = GPUShader(PCVShaders.vertex_shader_minimal_variable_size_and_depth, PCVShaders.fragment_shader_minimal_variable_size_and_depth, )
            
            shader_data_vert, shader_data_frag, shader_data_geom = load_shader_code('minimal_variable_size_and_depth')
            e_shader = GPUShader(shader_data_vert, shader_data_frag)
            
            e_batch = batch_for_shader(e_shader, 'POINTS', {"position": vs[:l], "color": cs[:l], "size": sz[:l], })
            extra = {
                'shader': e_shader,
                'batch': e_batch,
                'sizes': sz,
                'sizesf': szf,
                'length': l,
            }
            d['extra']['MINIMAL_VARIABLE_SIZE_AND_DEPTH'] = extra
        if(pcv.dev_rich_billboard_point_cloud_enabled):
            # FIXME: this is getting ridiculous
            # e_shader = GPUShader(PCVShaders.billboard_vertex_with_depth_and_size, PCVShaders.billboard_fragment_with_depth_and_size, geocode=PCVShaders.billboard_geometry_with_depth_and_size, )
            
            shader_data_vert, shader_data_frag, shader_data_geom = load_shader_code('billboard_with_depth_and_size')
            e_shader = GPUShader(shader_data_vert, shader_data_frag, geocode=shader_data_geom, )
            
            e_batch = batch_for_shader(e_shader, 'POINTS', {"position": vs[:l], "color": cs[:l], "sizef": szf[:l], })
            extra = {
                'shader': e_shader,
                'batch': e_batch,
                'sizes': sz,
                'sizesf': szf,
                'length': l,
            }
            d['extra']['RICH_BILLBOARD'] = extra
        
        if(pcv.dev_rich_billboard_point_cloud_no_depth_enabled):
            # FIXME: this is getting ridiculous
            # e_shader = GPUShader(PCVShaders.billboard_vertex_with_no_depth_and_size, PCVShaders.billboard_fragment_with_no_depth_and_size, geocode=PCVShaders.billboard_geometry_with_no_depth_and_size, )
            
            shader_data_vert, shader_data_frag, shader_data_geom = load_shader_code('billboard_with_no_depth_and_size')
            e_shader = GPUShader(shader_data_vert, shader_data_frag, geocode=shader_data_geom, )
            
            e_batch = batch_for_shader(e_shader, 'POINTS', {"position": vs[:l], "color": cs[:l], "sizef": szf[:l], })
            extra = {
                'shader': e_shader,
                'batch': e_batch,
                'sizes': sz,
                'sizesf': szf,
                'length': l,
            }
            d['extra']['RICH_BILLBOARD_NO_DEPTH'] = extra
        
        # d['extra_data'] = {
        #     'sizes': sz,
        #     'sizesf': szf,
        # }
        
        c = PCVManager.cache[pcv.uuid]
        c['draw'] = True
        
        self._redraw()


class PCVIV2RuntimeSettings():
    enabled = False
    # enabled = True


class PCVIV2_PT_panel(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "View"
    bl_label = "Point Cloud Instance Visualizer"
    bl_parent_id = "PCV_PT_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        # if(not debug_mode()):
        #     return False
        
        if(not PCVIV2RuntimeSettings.enabled):
            return False
        
        o = context.active_object
        if(o is None):
            return False
        
        if(o):
            pcv = o.point_cloud_visualizer
            if(pcv.edit_is_edit_mesh):
                return False
            if(pcv.edit_initialized):
                return False
        return True
    
    def draw_header(self, context):
        pcv = context.object.point_cloud_visualizer
        l = self.layout
        l.label(text='', icon='MOD_PARTICLE_INSTANCE', )
    
    def draw(self, context):
        """
        def prop_name(cls, prop, colon=False, ):
            for p in cls.bl_rna.properties:
                if(p.identifier == prop):
                    if(colon):
                        return "{}:".format(p.name)
                    return p.name
            return ''
        
        def third_label_two_thirds_prop(cls, prop, uil, ):
            f = 0.33
            r = uil.row()
            s = r.split(factor=f)
            s.label(text=prop_name(cls, prop, True, ))
            s = s.split(factor=1.0)
            r = s.row()
            r.prop(cls, prop, text='', )
        
        def third_label_two_thirds_prop_search(cls, prop, cls2, prop2, uil, ):
            f = 0.33
            r = uil.row()
            s = r.split(factor=f)
            s.label(text=prop_name(cls, prop, True, ))
            s = s.split(factor=1.0)
            r = s.row()
            r.prop_search(cls, prop, cls2, prop2, text='', )
        
        def third_label_two_thirds_prop_search_aligned(cls, prop, cls2, prop2, uil, ):
            f = 0.33
            r = uil.row(align=True)
            s = r.split(factor=f, align=True)
            s.label(text=prop_name(cls, prop, True, ))
            s = s.split(factor=1.0, align=True)
            r = s.row(align=True)
            r.prop_search(cls, prop, cls2, prop2, text='', )
        
        def third_label_two_thirds_prop_enum_expand(cls, prop, uil, ):
            f = 0.33
            r = uil.row()
            s = r.split(factor=f)
            s.label(text=prop_name(cls, prop, True, ))
            s = s.split(factor=1.0)
            r = s.row()
            r.prop(cls, prop, expand=True, )
        """
        
        o = context.object
        pcv = o.point_cloud_visualizer
        l = self.layout
        c = l.column()
        
        if(not PCVIV2Manager.initialized):
            
            r = c.row()
            r.alignment = 'CENTER'
            r.label(text='Not initialized..', icon='ERROR', )
            c.separator()
            
            r = c.row(align=True)
            r.operator('point_cloud_visualizer.pcviv_init')
            # if(not debug_mode()):
            #     if(PCVIV2Manager.initialized):
            #         r.enabled = False
            
        else:
            
            if(debug_mode()):
                r = c.row(align=True)
                r.operator('point_cloud_visualizer.pcviv_init')
                c.separator()
            
            if(len(o.particle_systems) == 0):
                b = c.box()
                b.label(text='No Particle Systems..', icon='ERROR', )
            
            for psys in o.particle_systems:
                pcviv = psys.settings.pcv_instance_visualizer
                b = c.box()
                
                if(not pcviv.subpanel_opened):
                    r = b.row()
                    rr = r.row(align=True)
                    # twice, because i want clicking on psys name to be possible
                    rr.prop(pcviv, 'subpanel_opened', icon='TRIA_DOWN' if pcviv.subpanel_opened else 'TRIA_RIGHT', icon_only=True, emboss=False, )
                    rr.prop(pcviv, 'subpanel_opened', icon='PARTICLES', icon_only=True, emboss=False, text=psys.settings.name, )
                    rr = r.row()
                    # rr.prop(pcviv, 'draw', text='', toggle=True, icon_only=True, icon='HIDE_OFF' if pcviv.draw else 'HIDE_ON', )
                    rr.prop(pcviv, 'draw', text='', toggle=True, icon_only=True, icon='RESTRICT_VIEW_OFF' if pcviv.draw else 'RESTRICT_VIEW_ON', )
                    # update, alert when dirty. in fact it's just before first draw
                    ccc = rr.column(align=True)
                    rrr = ccc.row(align=True)
                    if(pcviv.draw):
                        alert = False
                        if(pcviv.uuid in PCVIV2Manager.cache.keys()):
                            if(PCVIV2Manager.cache[pcviv.uuid]['dirty']):
                                alert = True
                        else:
                            alert = True
                        rrr.alert = alert
                    else:
                        rrr.enabled = False
                    rrr.operator('point_cloud_visualizer.pcviv_update').uuid = pcviv.uuid
                    cccc = rrr.row(align=True)
                    cccc.alert = False
                    # cccc.prop(pcviv, 'use_face_area', icon_only=True, icon='FACESEL', toggle=True, text='', )
                    cccc.prop(pcviv, 'use_face_area', icon_only=True, icon='FACE_MAPS', toggle=True, text='', )
                    cccc.prop(pcviv, 'use_material_factors', icon_only=True, icon='MATERIAL', toggle=True, text='', )
                else:
                    r = b.row(align=True)
                    # twice, because i want clicking on psys name to be possible
                    r.prop(pcviv, 'subpanel_opened', icon='TRIA_DOWN' if pcviv.subpanel_opened else 'TRIA_RIGHT', icon_only=True, emboss=False, )
                    r.prop(pcviv, 'subpanel_opened', icon='PARTICLES', icon_only=True, emboss=False, text=psys.settings.name, )
                    # options
                    cc = b.column(align=True)
                    cc.prop(pcviv, 'max_points')
                    if(pcv.dev_rich_billboard_point_cloud_enabled or pcv.dev_rich_billboard_point_cloud_no_depth_enabled):
                        cc.prop(pcviv, 'point_size_f')
                    else:
                        cc.prop(pcviv, 'point_size')
                    _r = cc.row(align=True)
                    if(pcviv.color_source == 'CONSTANT'):
                        _s = _r.split(factor=0.75, align=True, )
                        _s.prop(pcviv, 'color_source', )
                        _s = _s.split(factor=1.0, align=True, )
                        _s.prop(pcviv, 'color_constant', text='', )
                        
                    else:
                        _r.prop(pcviv, 'color_source', )
                    # update
                    cc = b.column()
                    r = cc.row()
                    # r.prop(pcviv, 'draw', text='', toggle=True, icon_only=True, icon='HIDE_OFF' if pcviv.draw else 'HIDE_ON', )
                    r.prop(pcviv, 'draw', text='', toggle=True, icon_only=True, icon='RESTRICT_VIEW_OFF' if pcviv.draw else 'RESTRICT_VIEW_ON', )
                    ccc = r.column(align=True)
                    rrr = ccc.row(align=True)
                    if(pcviv.draw):
                        alert = False
                        if(pcviv.uuid in PCVIV2Manager.cache.keys()):
                            if(PCVIV2Manager.cache[pcviv.uuid]['dirty']):
                                # if in cache and is dirty
                                alert = True
                        else:
                            # or not in cache at all, ie. not processed yet
                            alert = True
                        rrr.alert = alert
                    else:
                        rrr.enabled = False
                    rrr.operator('point_cloud_visualizer.pcviv_update').uuid = pcviv.uuid
                    cccc = rrr.row(align=True)
                    cccc.alert = False
                    # cccc.prop(pcviv, 'use_face_area', icon_only=True, icon='FACESEL', toggle=True, text='', )
                    cccc.prop(pcviv, 'use_face_area', icon_only=True, icon='FACE_MAPS', toggle=True, text='', )
                    cccc.prop(pcviv, 'use_material_factors', icon_only=True, icon='MATERIAL', toggle=True, text='', )
                    # if(debug_mode()):
                    #     if(pcviv.debug_update == ''):
                    #         cc.label(text='(debug: {})'.format('n/a', ))
                    #     else:
                    #         cc.label(text='(debug: {})'.format(pcviv.debug_update, ))
                
                if(debug_mode()):
                    if(pcviv.debug_update == ''):
                        b.label(text='(debug: {})'.format('n/a', ))
                    else:
                        b.label(text='(debug: {})'.format(pcviv.debug_update, ))
            
            c.separator()
            r = c.row(align=True)
            r.operator('point_cloud_visualizer.pcviv_update_all')
            r.operator('point_cloud_visualizer.pcviv_reset')
            
            c.separator()
            r = c.row()
            r.alignment = 'RIGHT'
            r.label(text='Powered by: Point Cloud Visualizer')


class PCVIV2_UL_materials(UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, ):
        pcvivgp = item.pcv_instance_visualizer
        
        r = layout.row(align=True)
        s = r.split(factor=0.5, align=True, )
        s.label(text=item.name, icon='MATERIAL', )
        s = s.split(factor=0.8, align=True, )
        s.prop(pcvivgp, 'factor', text='', )
        s = s.split(factor=1.0, align=True, )
        s.prop(item, 'diffuse_color', text='', )


class PCVIV2_PT_generator(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "View"
    bl_label = "Generator Options"
    bl_parent_id = "PCVIV2_PT_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        o = context.active_object
        if(o is None):
            return False
        
        if(o):
            pcv = o.point_cloud_visualizer
            if(pcv.edit_is_edit_mesh):
                return False
            if(pcv.edit_initialized):
                return False
        
        if(not PCVIV2Manager.initialized):
            return False
        
        return True
    
    def draw(self, context):
        o = context.object
        pcv = o.point_cloud_visualizer
        l = self.layout
        c = l.column()
        # c.label(text="Material Point Probability:")
        c.label(text="Point probability per material:")
        c.template_list("PCVIV2_UL_materials", "", bpy.data, "materials", pcv, "pcviv_material_list_active_index")


class PCVIV2_PT_display(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "View"
    bl_label = "Display Options"
    bl_parent_id = "PCVIV2_PT_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        o = context.active_object
        if(o is None):
            return False
        
        if(o):
            pcv = o.point_cloud_visualizer
            if(pcv.edit_is_edit_mesh):
                return False
            if(pcv.edit_initialized):
                return False
        
        if(not PCVIV2Manager.initialized):
            return False
        
        return True
    
    def draw(self, context):
        o = context.object
        pcv = o.point_cloud_visualizer
        l = self.layout
        c = l.column()
        c.label(text="Shader:")
        r = c.row(align=True)
        r.prop(pcv, 'dev_minimal_shader_variable_size_enabled', toggle=True, text="Basic", )
        r.prop(pcv, 'dev_minimal_shader_variable_size_and_depth_enabled', toggle=True, text="Depth", )
        r.prop(pcv, 'dev_rich_billboard_point_cloud_no_depth_enabled', toggle=True, text="Billboard", )
        r.prop(pcv, 'dev_rich_billboard_point_cloud_enabled', toggle=True, text="Depth Billboard", )
        if(pcv.dev_minimal_shader_variable_size_and_depth_enabled):
            cc = c.column(align=True)
            cc.prop(pcv, 'dev_minimal_shader_variable_size_and_depth_brightness')
            cc.prop(pcv, 'dev_minimal_shader_variable_size_and_depth_contrast')
            cc.prop(pcv, 'dev_minimal_shader_variable_size_and_depth_blend')
        if(pcv.dev_rich_billboard_point_cloud_no_depth_enabled):
            cc = c.column(align=True)
            cc.prop(pcv, 'dev_rich_billboard_point_cloud_size')
        if(pcv.dev_rich_billboard_point_cloud_enabled):
            cc = c.column(align=True)
            cc.prop(pcv, 'dev_rich_billboard_point_cloud_size')
            cc = c.column(align=True)
            cc.prop(pcv, 'dev_rich_billboard_depth_brightness')
            cc.prop(pcv, 'dev_rich_billboard_depth_contrast')
            cc.prop(pcv, 'dev_rich_billboard_depth_blend')


class PCVIV2_PT_debug(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "View"
    bl_label = "PCVIV Debug"
    bl_parent_id = "PCVIV2_PT_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        o = context.active_object
        if(o is None):
            return False
        
        if(not PCVIV2Manager.initialized):
            return False
        
        if(debug_mode()):
            return True
        
        return False
    
    def draw_header(self, context):
        pcv = context.object.point_cloud_visualizer
        l = self.layout
        l.label(text='', icon='SETTINGS', )
    
    def draw(self, context):
        o = context.object
        pcv = o.point_cloud_visualizer
        l = self.layout
        c = l.column()
        
        b = c.box()
        c = b.column()
        
        c.separator()
        if(pcv.pcviv_debug_draw != ''):
            c.label(text='(debug: {})'.format(pcv.pcviv_debug_draw, ))
            c.separator()
        
        r = c.row(align=True)
        # r.operator('point_cloud_visualizer.pcviv_reset')
        r.operator('point_cloud_visualizer.pcviv_deinit')
        r.operator('point_cloud_visualizer.pcviv_reset_all')
        c.separator()
        
        r = c.row()
        r.prop(pcv, 'pcviv_debug_panel_show_info', icon='TRIA_DOWN' if pcv.pcviv_debug_panel_show_info else 'TRIA_RIGHT', icon_only=True, emboss=False, )
        r.label(text="Debug Info")
        if(pcv.pcviv_debug_panel_show_info):
            cc = c.column()
            cc.label(text="object: '{}'".format(o.name))
            cc.label(text="psystem(s): {}".format(len(o.particle_systems)))
            cc.scale_y = 0.5
            
            c.separator()
            
            tab = '        '
            
            cc = c.column()
            cc.label(text="PCVIV2Manager:")
            cc.label(text="{}initialized: {}".format(tab, PCVIV2Manager.initialized))
            cc.label(text="{}cache: {} item(s)".format(tab, len(PCVIV2Manager.cache.keys())))
            cc.scale_y = 0.5
            
            c.separator()
            tab = '    '
            ci = 0
            for k, v in PCVIV2Manager.cache.items():
                b = c.box()
                cc = b.column()
                cc.scale_y = 0.5
                cc.label(text='item: {}'.format(ci))
                ci += 1
                for l, w in v.items():
                    if(type(w) == dict):
                        cc.label(text='{}{}: {} item(s)'.format(tab, l, len(w.keys())))
                    elif(type(w) == np.ndarray or type(w) == list):
                        cc.label(text='{}{}: {} item(s)'.format(tab, l, len(w)))
                    else:
                        cc.label(text='{}{}: {}'.format(tab, l, w))
        
        # and some development shortcuts..
        c.separator()
        c.operator('script.reload', text='debug: reload scripts', )


class PCVIV2_properties(PropertyGroup):
    # to identify object, key for storing cloud in cache, etc.
    uuid: StringProperty(default="", options={'HIDDEN', }, )
    
    def _draw_update(self, context, ):
        PCVIV2Manager.draw_update(context.object, self.uuid, self.draw, )
    
    # draw cloud enabled/disabled
    draw: BoolProperty(name="Draw", default=True, description="Draw particle instances as point cloud", update=_draw_update, )
    # user can set maximum number of points drawn per instance
    max_points: IntProperty(name="Max. Points Per Instance", default=1000, min=1, max=1000000, description="Maximum number of points per instance", )
    # user can set size of points, but it will be only used when minimal shader is active
    point_size: IntProperty(name="Size", default=3, min=1, max=10, subtype='PIXEL', description="Point size", )
    # rich billboard shader size
    point_size_f: FloatProperty(name="Scale", default=1.0, min=0.0, max=10.0, description="Point scale (shader size * scale)", precision=6, )
    
    color_source: EnumProperty(name="Color", items=[('CONSTANT', "Constant Color", "Use constant color value"),
                                                    ('VIEWPORT_DISPLAY_COLOR', "Material Viewport Display Color", "Use material viewport display color property"),
                                                    ], default='VIEWPORT_DISPLAY_COLOR', description="Color source for generated point cloud", )
    color_constant: FloatVectorProperty(name="Color", description="Constant color", default=(0.7, 0.7, 0.7, ), min=0, max=1, subtype='COLOR', size=3, )
    
    def _method_update(self, context, ):
        if(not self.use_face_area and not self.use_material_factors):
            self.use_face_area = True
    
    use_face_area: BoolProperty(name="Use Face Area", default=True, description="Use mesh face area as probability factor during point cloud generation", update=_method_update, )
    use_material_factors: BoolProperty(name="Use Material Factors", default=False, description="Use material probability factor during point cloud generation", update=_method_update, )
    
    # helper property, draw minimal ui or draw all props
    subpanel_opened: BoolProperty(default=False, options={'HIDDEN', }, )
    
    # store info how long was last update, generate and store to cache
    debug_update: StringProperty(default="", )
    
    @classmethod
    def register(cls):
        bpy.types.ParticleSettings.pcv_instance_visualizer = PointerProperty(type=cls)
    
    @classmethod
    def unregister(cls):
        del bpy.types.ParticleSettings.pcv_instance_visualizer


class PCVIV2_generator_properties(PropertyGroup):
    factor: FloatProperty(name="Factor", default=0.5, min=0.0, max=1.0, precision=3, subtype='FACTOR', description="Probability factor of choosing polygon with this material", )
    
    @classmethod
    def register(cls):
        bpy.types.Material.pcv_instance_visualizer = PointerProperty(type=cls)
    
    @classmethod
    def unregister(cls):
        del bpy.types.Material.pcv_instance_visualizer
