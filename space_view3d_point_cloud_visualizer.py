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

bl_info = {"name": "Point Cloud Visualizer",
           "description": "Display, edit, filter, render, convert, generate and export colored point cloud PLY files.",
           "author": "Jakub Uhlik",
           "version": (0, 9, 30),
           "blender": (2, 81, 0),
           "location": "View3D > Sidebar > Point Cloud Visualizer",
           "warning": "",
           "wiki_url": "https://github.com/uhlik/bpy",
           "tracker_url": "https://github.com/uhlik/bpy/issues",
           "category": "3D View", }


import os
import struct
import uuid
import time
import datetime
import math
import numpy as np
import re
import shutil
import sys
import random
import statistics

import bpy
import bmesh
from bpy.props import PointerProperty, BoolProperty, StringProperty, FloatProperty, IntProperty, FloatVectorProperty, EnumProperty, CollectionProperty
from bpy.types import PropertyGroup, Panel, Operator, AddonPreferences, UIList
import gpu
from gpu.types import GPUOffScreen, GPUShader, GPUBatch, GPUVertBuf, GPUVertFormat
from gpu_extras.batch import batch_for_shader
from bpy.app.handlers import persistent
import bgl
from mathutils import Matrix, Vector, Quaternion, Color
from bpy_extras.object_utils import world_to_camera_view
from bpy_extras.io_utils import axis_conversion, ExportHelper
from mathutils.kdtree import KDTree
from mathutils.geometry import barycentric_transform
from mathutils.interpolate import poly_3d_calc
from mathutils.bvhtree import BVHTree
import mathutils.geometry


# NOTE $ pycodestyle --ignore=W293,E501,E741,E402 --exclude='io_mesh_fast_obj/blender' .


def log(msg, indent=0, prefix='>', ):
    m = "{}{} {}".format("    " * indent, prefix, msg)
    if(debug_mode()):
        print(m)


def debug_mode():
    # return True
    return (bpy.app.debug_value != 0)


class Progress():
    def __init__(self, total, indent=0, prefix="> ", ):
        self.current = 0
        self.percent = -1
        self.last = -1
        self.total = total
        self.prefix = prefix
        self.indent = indent
        self.t = "    "
        self.r = "\r"
        self.n = "\n"
    
    def step(self, numdone=1):
        if(not debug_mode()):
            return
        self.current += numdone
        self.percent = int(self.current / (self.total / 100))
        if(self.percent > self.last):
            sys.stdout.write(self.r)
            sys.stdout.write("{0}{1}{2}%".format(self.t * self.indent, self.prefix, self.percent))
            self.last = self.percent
        if(self.percent >= 100 or self.total == self.current):
            sys.stdout.write(self.r)
            sys.stdout.write("{0}{1}{2}%{3}".format(self.t * self.indent, self.prefix, 100, self.n))


class PCMeshInstancerMeshGenerator():
    def __init__(self, mesh_type='VERTEX', length=1.0, radius=1.0, subdivision=2, ):
        if(mesh_type not in ('VERTEX', 'TRIANGLE', 'TETRAHEDRON', 'CUBE', 'ICOSPHERE', )):
            raise TypeError("Unknown mesh type: {}".format(mesh_type))
        
        self.mesh_type = mesh_type
        
        if(length <= 0):
            log("length is (or less than) 0, which is ridiculous. setting to 0.001..", 1)
            length = 0.001
        self.length = length
        
        if(radius <= 0):
            log("radius is (or less than) 0, which is ridiculous. setting to 0.001..", 1)
            radius = 0.001
        self.radius = radius
        
        subdivision = int(subdivision)
        if(not (0 < subdivision <= 2)):
            log("subdivision 1 or 2 allowed, not {}, setting to 1".format(subdivision), 1)
            subdivision = 1
        self.subdivision = subdivision
        
        self.def_verts, self.def_edges, self.def_faces = self.generate()
    
    def generate(self):
        def circle2d_coords(radius, steps, offset, ox, oy):
            r = []
            angstep = 2 * math.pi / steps
            for i in range(steps):
                x = math.sin(i * angstep + offset) * radius + ox
                y = math.cos(i * angstep + offset) * radius + oy
                r.append((x, y))
            return r
        
        if(self.mesh_type == 'VERTEX'):
            return [(0, 0, 0, ), ], [], []
        elif(self.mesh_type == 'TRIANGLE'):
            offset = 0.0
            r = math.sqrt(3) / 3 * self.length
            c = circle2d_coords(r, 3, offset, 0, 0)
            dv = []
            for i in c:
                dv.append((i[0], i[1], 0, ))
            df = [(0, 2, 1, ), ]
            return dv, [], df
        elif(self.mesh_type == 'TETRAHEDRON'):
            l = self.length
            excircle_radius = math.sqrt(3) / 3 * l
            c = circle2d_coords(excircle_radius, 3, 0, 0, 0)
            h = l / 3 * math.sqrt(6)
            dv = [(c[0][0], c[0][1], 0, ), (c[1][0], c[1][1], 0, ), (c[2][0], c[2][1], 0, ), (0, 0, h, ), ]
            df = ([(0, 1, 2), (3, 2, 1), (3, 1, 0), (3, 0, 2), ])
            return dv, [], df
        elif(self.mesh_type == 'CUBE'):
            l = self.length / 2
            dv = [(+l, +l, -l), (+l, -l, -l), (-l, -l, -l), (-l, +l, -l), (+l, +l, +l), (+l, -l, +l), (-l, -l, +l), (-l, +l, +l), ]
            df = [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (4, 0, 3, 7), ]
            return dv, [], df
        elif(self.mesh_type == 'ICOSPHERE'):
            if(self.subdivision == 1):
                dv = [(0.0, 0.0, -0.5), (0.3617999851703644, -0.2628600001335144, -0.22360749542713165), (-0.13819250464439392, -0.42531999945640564, -0.22360749542713165), (-0.44721248745918274, 0.0, -0.22360749542713165), (-0.13819250464439392, 0.42531999945640564, -0.22360749542713165), (0.3617999851703644, 0.2628600001335144, -0.22360749542713165), (0.13819250464439392, -0.42531999945640564, 0.22360749542713165), (-0.3617999851703644, -0.2628600001335144, 0.22360749542713165), (-0.3617999851703644, 0.2628600001335144, 0.22360749542713165), (0.13819250464439392, 0.42531999945640564, 0.22360749542713165), (0.44721248745918274, 0.0, 0.22360749542713165), (0.0, 0.0, 0.5)]
                df = [(0, 1, 2), (1, 0, 5), (0, 2, 3), (0, 3, 4), (0, 4, 5), (1, 5, 10), (2, 1, 6), (3, 2, 7), (4, 3, 8), (5, 4, 9), (1, 10, 6), (2, 6, 7), (3, 7, 8), (4, 8, 9), (5, 9, 10), (6, 10, 11), (7, 6, 11), (8, 7, 11), (9, 8, 11), (10, 9, 11)]
            elif(self.subdivision == 2):
                dv = [(0.0, 0.0, -0.5), (0.36180365085601807, -0.2628626525402069, -0.22360976040363312), (-0.1381940096616745, -0.42532461881637573, -0.22360992431640625), (-0.4472131133079529, 0.0, -0.22360780835151672), (-0.1381940096616745, 0.42532461881637573, -0.22360992431640625), (0.36180365085601807, 0.2628626525402069, -0.22360976040363312), (0.1381940096616745, -0.42532461881637573, 0.22360992431640625), (-0.36180365085601807, -0.2628626525402069, 0.22360976040363312), (-0.36180365085601807, 0.2628626525402069, 0.22360976040363312), (0.1381940096616745, 0.42532461881637573, 0.22360992431640625), (0.4472131133079529, 0.0, 0.22360780835151672), (0.0, 0.0, 0.5), (-0.08122777938842773, -0.24999763071537018, -0.42532721161842346), (0.21266134083271027, -0.15450569987297058, -0.4253270924091339), (0.13143441081047058, -0.40450581908226013, -0.26286882162094116), (0.4253239333629608, 0.0, -0.2628679573535919), (0.21266134083271027, 0.15450569987297058, -0.4253270924091339), (-0.262864887714386, 0.0, -0.42532584071159363), (-0.3440946936607361, -0.24999846518039703, -0.26286810636520386), (-0.08122777938842773, 0.24999763071537018, -0.42532721161842346), (-0.3440946936607361, 0.24999846518039703, -0.26286810636520386), (0.13143441081047058, 0.40450581908226013, -0.26286882162094116), (0.47552892565727234, -0.15450631082057953, 0.0), (0.47552892565727234, 0.15450631082057953, 0.0), (0.0, -0.4999999701976776, 0.0), (0.2938928008079529, -0.4045083522796631, 0.0), (-0.47552892565727234, -0.15450631082057953, 0.0), (-0.2938928008079529, -0.4045083522796631, 0.0), (-0.2938928008079529, 0.4045083522796631, 0.0), (-0.47552892565727234, 0.15450631082057953, 0.0), (0.2938928008079529, 0.4045083522796631, 0.0), (0.0, 0.4999999701976776, 0.0), (0.3440946936607361, -0.24999846518039703, 0.26286810636520386), (-0.13143441081047058, -0.40450581908226013, 0.26286882162094116), (-0.4253239333629608, 0.0, 0.2628679573535919), (-0.13143441081047058, 0.40450581908226013, 0.26286882162094116), (0.3440946936607361, 0.24999846518039703, 0.26286810636520386), (0.08122777938842773, -0.24999763071537018, 0.4253271818161011), (0.262864887714386, 0.0, 0.42532584071159363), (-0.21266134083271027, -0.15450569987297058, 0.4253270924091339), (-0.21266134083271027, 0.15450569987297058, 0.4253270924091339), (0.08122777938842773, 0.24999763071537018, 0.4253271818161011)]
                df = [(0, 13, 12), (1, 13, 15), (0, 12, 17), (0, 17, 19), (0, 19, 16), (1, 15, 22), (2, 14, 24), (3, 18, 26), (4, 20, 28), (5, 21, 30), (1, 22, 25), (2, 24, 27), (3, 26, 29), (4, 28, 31), (5, 30, 23), (6, 32, 37), (7, 33, 39), (8, 34, 40), (9, 35, 41), (10, 36, 38), (38, 41, 11), (38, 36, 41), (36, 9, 41), (41, 40, 11), (41, 35, 40), (35, 8, 40), (40, 39, 11), (40, 34, 39), (34, 7, 39), (39, 37, 11), (39, 33, 37), (33, 6, 37), (37, 38, 11), (37, 32, 38), (32, 10, 38), (23, 36, 10), (23, 30, 36), (30, 9, 36), (31, 35, 9), (31, 28, 35), (28, 8, 35), (29, 34, 8), (29, 26, 34), (26, 7, 34), (27, 33, 7), (27, 24, 33), (24, 6, 33), (25, 32, 6), (25, 22, 32), (22, 10, 32), (30, 31, 9), (30, 21, 31), (21, 4, 31), (28, 29, 8), (28, 20, 29), (20, 3, 29), (26, 27, 7), (26, 18, 27), (18, 2, 27), (24, 25, 6), (24, 14, 25), (14, 1, 25), (22, 23, 10), (22, 15, 23), (15, 5, 23), (16, 21, 5), (16, 19, 21), (19, 4, 21), (19, 20, 4), (19, 17, 20), (17, 3, 20), (17, 18, 3), (17, 12, 18), (12, 2, 18), (15, 16, 5), (15, 13, 16), (13, 0, 16), (12, 14, 2), (12, 13, 14), (13, 1, 14)]
            else:
                raise ValueError("unsupported subdivision: {}".format(self.subdivision))
            return dv, [], df
        else:
            return [(0, 0, 0, ), ], [], []


class PCMeshInstancer():
    def __init__(self, name, points, generator=None, matrix=None, size=0.01, normal_align=False, vcols=False, ):
        log("{}:".format(self.__class__.__name__), 0, )
        
        self.name = name
        self.points = points
        if(generator is None):
            generator = PCMeshInstancerMeshGenerator()
        self.generator = generator
        if(matrix is None):
            matrix = Matrix()
        self.matrix = matrix
        self.size = size
        self.normal_align = normal_align
        self.vcols = vcols
        
        self.uuid = uuid.uuid1()
        
        log("calculating matrices..", 1)
        self.calc_matrices()
        
        log("calculating mesh..", 1)
        self.calc_mesh_data()
        
        log("creating mesh..", 1)
        self.mesh = bpy.data.meshes.new(self.name)
        self.mesh.from_pydata(self.verts, self.edges, self.faces)
        self.object = self.add_object(self.name, self.mesh)
        self.object.matrix_world = self.matrix
        self.activate_object(self.object)
        
        if(self.vcols):
            log("making vertex colors..", 1)
            self.make_vcols()
        
        log("cleanup..", 1)
        
        context = bpy.context
        view_layer = context.view_layer
        collection = view_layer.active_layer_collection.collection
        collection.objects.unlink(self.def_object)
        bpy.data.objects.remove(self.def_object)
        bpy.data.meshes.remove(self.def_mesh)
        
        log("done.", 1)
    
    def add_object(self, name, data, ):
        so = bpy.context.scene.objects
        for i in so:
            i.select_set(False)
        o = bpy.data.objects.new(name, data)
        context = bpy.context
        view_layer = context.view_layer
        collection = view_layer.active_layer_collection.collection
        collection.objects.link(o)
        o.select_set(True)
        view_layer.objects.active = o
        return o
    
    def activate_object(self, obj, ):
        bpy.ops.object.select_all(action='DESELECT')
        context = bpy.context
        view_layer = context.view_layer
        obj.select_set(True)
        view_layer.objects.active = obj
    
    def calc_matrices(self):
        def split(p):
            co = (p[0], p[1], p[2])
            no = (p[3], p[4], p[5])
            rgb = (p[6], p[7], p[8])
            return co, no, rgb
        
        def rotation_to(a, b):
            # http://stackoverflow.com/questions/1171849/finding-quaternion-representing-the-rotation-from-one-vector-to-another
            # https://github.com/toji/gl-matrix/blob/f0583ef53e94bc7e78b78c8a24f09ed5e2f7a20c/src/gl-matrix/quat.js#L54
            
            a = a.normalized()
            b = b.normalized()
            q = Quaternion()
            
            tmpvec3 = Vector()
            xUnitVec3 = Vector((1, 0, 0))
            yUnitVec3 = Vector((0, 1, 0))
            
            dot = a.dot(b)
            if(dot < -0.999999):
                tmpvec3 = xUnitVec3.cross(a)
                if(tmpvec3.length < 0.000001):
                    tmpvec3 = yUnitVec3.cross(a)
                tmpvec3.normalize()
                q = Quaternion(tmpvec3, math.pi)
            elif(dot > 0.999999):
                q.x = 0
                q.y = 0
                q.z = 0
                q.w = 1
            else:
                tmpvec3 = a.cross(b)
                q.x = tmpvec3[0]
                q.y = tmpvec3[1]
                q.z = tmpvec3[2]
                q.w = 1 + dot
                q.normalize()
            return q
        
        _, _, osv = self.matrix.decompose()
        osm = Matrix(((osv.x, 0.0, 0.0, 0.0), (0.0, osv.y, 0.0, 0.0), (0.0, 0.0, osv.z, 0.0), (0.0, 0.0, 0.0, 1.0))).inverted()
        
        # calculate instance matrices from points..
        self.matrices = []
        for i, p in enumerate(self.points):
            co, no, rgb = split(p)
            # location
            ml = Matrix.Translation(co).to_4x4()
            if(self.normal_align):
                # rotation from normal
                quat = rotation_to(Vector((0, 0, 1)), Vector(no))
                mr = quat.to_matrix().to_4x4()
            else:
                mr = Matrix.Rotation(0.0, 4, 'Z')
            # scale
            s = self.size
            ms = Matrix(((s, 0.0, 0.0, 0.0), (0.0, s, 0.0, 0.0), (0.0, 0.0, s, 0.0), (0.0, 0.0, 0.0, 1.0)))
            # combine
            m = ml @ mr @ ms @ osm
            self.matrices.append(m)
    
    def calc_mesh_data(self):
        # initialize lists
        l = len(self.matrices)
        self.verts = [(0, 0, 0)] * (l * len(self.generator.def_verts))
        self.edges = [(0, 0)] * (l * len(self.generator.def_edges))
        self.faces = [(0)] * (l * len(self.generator.def_faces))
        self.colors = [None] * l
        
        # generator data
        v, e, f = self.generator.generate()
        self.def_verts = v
        self.def_edges = e
        self.def_faces = f
        
        # def object
        self.def_mesh = bpy.data.meshes.new("PCInstancer-def_mesh-{}".format(self.uuid))
        self.def_mesh.from_pydata(v, e, f)
        self.def_object = self.add_object("PCInstancer-def_object-{}".format(self.uuid), self.def_mesh)
        
        # loop over matrices
        for i, m in enumerate(self.matrices):
            # transform mesh
            self.def_mesh.transform(m)
            # store
            self.write_pydata_chunk(i)
            # reset mesh
            for j, v in enumerate(self.def_object.data.vertices):
                v.co = Vector(self.def_verts[j])
    
    def write_pydata_chunk(self, i, ):
        # exponents
        ev = len(self.generator.def_verts)
        ee = len(self.generator.def_edges)
        ef = len(self.generator.def_faces)
        # vertices
        for j in range(ev):
            self.verts[(i * ev) + j] = self.def_mesh.vertices[j].co.to_tuple()
        # edges
        if(len(self.def_edges) is not 0):
            for j in range(ee):
                self.edges[(i * ee) + j] = ((i * ev) + self.def_edges[j][0],
                                            (i * ev) + self.def_edges[j][1], )
        # faces
        if(len(self.def_faces) is not 0):
            for j in range(ef):
                # tris
                if(len(self.def_faces[j]) == 3):
                    self.faces[(i * ef) + j] = ((i * ev) + self.def_faces[j][0],
                                                (i * ev) + self.def_faces[j][1],
                                                (i * ev) + self.def_faces[j][2], )
                # quads
                elif(len(self.def_faces[j]) == 4):
                    self.faces[(i * ef) + j] = ((i * ev) + self.def_faces[j][0],
                                                (i * ev) + self.def_faces[j][1],
                                                (i * ev) + self.def_faces[j][2],
                                                (i * ev) + self.def_faces[j][3], )
                # ngons
                else:
                    ngon = []
                    for a in range(len(self.def_faces[j])):
                        ngon.append((i * ev) + self.def_faces[j][a])
                    self.faces[(i * ef) + j] = tuple(ngon)
    
    def make_vcols(self):
        if(len(self.mesh.loops) != 0):
            colors = []
            for i, v in enumerate(self.points):
                rgb = (v[6], v[7], v[8])
                col = (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)
                colors.append(col)
            
            num = len(self.def_verts)
            vc = self.mesh.vertex_colors.new()
            for l in self.mesh.loops:
                vi = l.vertex_index
                li = l.index
                c = colors[int(vi / num)]
                vc.data[li].color = c + (1.0, )
        else:
            log("no mesh loops in mesh", 2, )


class PCMeshInstancer2():
    def __init__(self, name, vs, ns=None, cs=None, generator=None, matrix=None, size=0.01, with_normal_align=False, with_vertex_colors=False, ):
        log("{}:".format(self.__class__.__name__), 0, )
        
        self.name = name
        
        self.vs = vs
        self.has_normals = True
        if(ns is None):
            self.has_normals = False
            with_normal_align = False
        self.ns = ns
        self.has_colors = True
        if(cs is None):
            self.has_colors = False
            with_vertex_colors = False
        self.cs = cs
        
        if(generator is None):
            generator = PCMeshInstancerMeshGenerator()
        self.generator = generator
        if(matrix is None):
            matrix = Matrix()
        self.matrix = matrix
        self.size = size
        self.with_normal_align = with_normal_align
        self.with_vertex_colors = with_vertex_colors
        
        self.uuid = uuid.uuid1()
        
        log("matrices..", 1)
        self.calc_matrices()
        log("mesh data..", 1)
        self.calc_mesh()
        log("make mesh..", 1)
        self.make_mesh()
        if(self.with_vertex_colors):
            log("make colors..", 1)
            self.make_colors()
        log("done.", 1)
    
    def calc_matrices(self):
        def rotation_to(a, b):
            # http://stackoverflow.com/questions/1171849/finding-quaternion-representing-the-rotation-from-one-vector-to-another
            # https://github.com/toji/gl-matrix/blob/f0583ef53e94bc7e78b78c8a24f09ed5e2f7a20c/src/gl-matrix/quat.js#L54
            
            a = a.normalized()
            b = b.normalized()
            q = Quaternion()
            
            tmpvec3 = Vector()
            xUnitVec3 = Vector((1, 0, 0))
            yUnitVec3 = Vector((0, 1, 0))
            
            dot = a.dot(b)
            if(dot < -0.999999):
                tmpvec3 = xUnitVec3.cross(a)
                if(tmpvec3.length < 0.000001):
                    tmpvec3 = yUnitVec3.cross(a)
                tmpvec3.normalize()
                q = Quaternion(tmpvec3, math.pi)
            elif(dot > 0.999999):
                q.x = 0
                q.y = 0
                q.z = 0
                q.w = 1
            else:
                tmpvec3 = a.cross(b)
                q.x = tmpvec3[0]
                q.y = tmpvec3[1]
                q.z = tmpvec3[2]
                q.w = 1 + dot
                q.normalize()
            return q
        
        _, _, osv = self.matrix.decompose()
        osm = Matrix(((osv.x, 0.0, 0.0, 0.0), (0.0, osv.y, 0.0, 0.0), (0.0, 0.0, osv.z, 0.0), (0.0, 0.0, 0.0, 1.0))).inverted()
        
        vs = self.vs
        ns = self.ns
        s = self.size
        with_normal_align = self.with_normal_align
        
        matrices = np.zeros((len(vs), 4, 4), dtype=np.float, )
        for i, co in enumerate(vs):
            no = ns[i]
            ml = Matrix.Translation(co).to_4x4()
            if(with_normal_align):
                q = rotation_to(Vector((0, 0, 1)), Vector(no))
                mr = q.to_matrix().to_4x4()
            else:
                mr = Matrix.Rotation(0.0, 4, 'Z')
            ms = Matrix(((s, 0.0, 0.0, 0.0), (0.0, s, 0.0, 0.0), (0.0, 0.0, s, 0.0), (0.0, 0.0, 0.0, 1.0)))
            m = ml @ mr @ ms @ osm
            matrices[i] = np.array(m, dtype=np.float, )
        
        self.matrices = matrices
    
    def calc_mesh(self):
        matrices = self.matrices
        
        gv, _, gf = self.generator.generate()
        me = bpy.data.meshes.new('tmp-{}'.format(self.uuid))
        me.from_pydata(gv, [], gf)
        bm = bmesh.new()
        bm.from_mesh(me)
        bmesh.ops.triangulate(bm, faces=bm.faces)
        bm.to_mesh(me)
        bm.free()
        
        gv = np.zeros((len(me.vertices) * 3), dtype=np.float, )
        me.vertices.foreach_get("co", gv, )
        gv.shape = (len(me.vertices), 3, )
        gf = np.zeros((len(me.polygons) * 3), dtype=np.int, )
        me.polygons.foreach_get("vertices", gf, )
        gf.shape = (len(me.polygons), 3, )
        
        self.g_vs_chunk_length = len(gv)
        self.g_fs_chunk_length = len(gf)
        
        zgv = gv[:]
        zgv.shape = (-1, )
        
        vs = np.zeros((len(matrices) * len(gv) * 3), dtype=np.float, )
        fs = np.zeros((len(matrices) * len(gf) * 3), dtype=np.int, )
        
        cos = np.zeros((len(me.vertices) * 3), dtype=np.float, )
        fis = np.zeros((len(me.polygons) * 3), dtype=np.int, )
        for i, m in enumerate(matrices):
            # doesn't work when passing ndarray directly
            me.transform(Matrix(m))
            
            me.vertices.foreach_get("co", cos)
            vs[i * len(cos):(i * len(cos)) + len(cos)] = cos
            me.polygons.foreach_get("vertices", fis, )
            fis += (i * len(me.vertices), )
            fs[i * len(fis):(i * len(fis)) + len(fis)] = fis
            
            me.vertices.foreach_set("co", zgv)
        
        self.g_vs = vs
        self.g_fs = fs
        
        bpy.data.meshes.remove(me)
    
    def make_mesh(self):
        me = bpy.data.meshes.new(self.name)
        
        # NOTE: remember that in future: from_pydata does NOT accept numpy arrays, https://developer.blender.org/T51585
        # NOTE: so passing ndarray.tolist() to from_pydata is VERY slow, so recreate interesting bits from from_pydata implementation without using itertools
        
        vertices = self.g_vs
        faces = self.g_fs
        
        vl = int(len(vertices) / 3)
        fl = int(len(faces) / 3)
        me.vertices.add(vl)
        me.loops.add(fl * 3)
        me.polygons.add(fl)
        
        face_lengths = np.full(fl, 3, dtype=np.int, )
        loop_starts = np.arange(0, fl * 3, 3, dtype=np.int, )
        
        me.vertices.foreach_set("co", vertices)
        me.polygons.foreach_set("loop_total", face_lengths)
        me.polygons.foreach_set("loop_start", loop_starts)
        me.polygons.foreach_set("vertices", faces)
        
        me.validate()
        
        so = bpy.context.scene.objects
        for i in so:
            i.select_set(False)
        o = bpy.data.objects.new(self.name, me)
        
        view_layer = bpy.context.view_layer
        collection = view_layer.active_layer_collection.collection
        collection.objects.link(o)
        o.select_set(True)
        view_layer.objects.active = o
        
        o.matrix_world = self.matrix
        
        self.mesh = me
        self.object = o
    
    def make_colors(self):
        fl = self.g_fs_chunk_length
        me = self.mesh
        cs = self.cs
        
        indexes = np.indices((len(cs), ), dtype=np.int, )
        cols = np.take(cs, indexes, axis=0, )
        cols = cols.repeat(fl * 3, axis=1, )
        cols.shape = (-1, )
        
        vc = me.vertex_colors.new()
        vc.data.foreach_set("color", cols)


class PCInstancer():
    def __init__(self, o, mesh_size, base_sphere_subdivisions, ):
        log("{}:".format(self.__class__.__name__), 0, )
        _t = time.time()
        
        base_sphere_radius = mesh_size / 2
        
        # make uv layout of neatly packed triangles to bake vertex colors
        log("calculating uv layout..", 1)
        num_tri = len(o.data.polygons)
        num_sq = math.ceil(num_tri / 4)
        sq_per_uv_side = math.ceil(math.sqrt(num_sq))
        sq_size = 1 / sq_per_uv_side
        
        bm = bmesh.new()
        bm.from_mesh(o.data)
        uv_layer = bm.loops.layers.uv.new("PCParticlesUVMap")
        
        def tri_uv(i, x, y, q):
            # *---------------*
            # |  \    3    /  |
            # |    \     /    |
            # | 4     *    2  |
            # |    /     \    |
            # |  /    1    \  |
            # *---------------*
            # 1  0.0,0.0  1.0,0.0  0.5,0.5
            # 2  1.0,0.0  1.0,1.0  0.5,0.5
            # 3  1.0,1.0  0.0,1.0  0.5,0.5
            # 4  0.0,1.0  0.0,0.0  0.5,0.5
            
            if(i == 0):
                return ((x + (0.0 * q), y + (0.0 * q), ),
                        (x + (1.0 * q), y + (0.0 * q), ),
                        (x + (0.5 * q), y + (0.5 * q), ), )
            elif(i == 1):
                return ((x + (1.0 * q), y + (0.0 * q), ),
                        (x + (1.0 * q), y + (1.0 * q), ),
                        (x + (0.5 * q), y + (0.5 * q), ), )
            elif(i == 2):
                return ((x + (1.0 * q), y + (1.0 * q), ),
                        (x + (0.0 * q), y + (1.0 * q), ),
                        (x + (0.5 * q), y + (0.5 * q), ), )
            elif(i == 3):
                return ((x + (0.0 * q), y + (1.0 * q), ),
                        (x + (0.0 * q), y + (0.0 * q), ),
                        (x + (0.5 * q), y + (0.5 * q), ), )
            else:
                raise Exception("You're not supposed to do that..")
        
        sq_c = 0
        xsqn = 0
        ysqn = 0
        for face in bm.faces:
            co = tri_uv(sq_c, xsqn * sq_size, ysqn * sq_size, sq_size)
            coi = 0
            for loop in face.loops:
                loop[uv_layer].uv = co[coi]
                coi += 1
            sq_c += 1
            if(sq_c == 4):
                sq_c = 0
                xsqn += 1
                if(xsqn == sq_per_uv_side):
                    xsqn = 0
                    ysqn += 1
        
        bm.to_mesh(o.data)
        bm.free()
        
        # bake vertex colors
        log("baking vertex colors..", 1)
        _engine = bpy.context.scene.render.engine
        bpy.context.scene.render.engine = 'CYCLES'
        
        tex_size = sq_per_uv_side * 8
        img = bpy.data.images.new("PCParticlesBakedColors", tex_size, tex_size, )
        
        mat = bpy.data.materials.new('PCParticlesBakeMaterial')
        o.data.materials.append(mat)
        mat.use_nodes = True
        # remove all nodes
        nodes = mat.node_tree.nodes
        for node in nodes:
            nodes.remove(node)
        # make new nodes
        links = mat.node_tree.links
        node_attr = nodes.new(type='ShaderNodeAttribute')
        if(o.data.vertex_colors.active is not None):
            node_attr.attribute_name = o.data.vertex_colors.active.name
        node_diff = nodes.new(type='ShaderNodeBsdfDiffuse')
        link = links.new(node_attr.outputs[0], node_diff.inputs[0])
        node_output = nodes.new(type='ShaderNodeOutputMaterial')
        link = links.new(node_diff.outputs[0], node_output.inputs[0])
        node_tcoord = nodes.new(type='ShaderNodeTexCoord')
        node_tex = nodes.new(type='ShaderNodeTexImage')
        node_tex.image = img
        link = links.new(node_tcoord.outputs[2], node_tex.inputs[0])
        # set image texture selected and active
        node_tex.select = True
        nodes.active = node_tex
        
        # do the baking
        scene = bpy.context.scene
        _bake_type = scene.cycles.bake_type
        scene.cycles.bake_type = 'DIFFUSE'
        _samples = scene.cycles.samples
        scene.cycles.samples = 32
        bake = scene.render.bake
        _use_pass_direct = bake.use_pass_direct
        _use_pass_indirect = bake.use_pass_indirect
        _use_pass_color = bake.use_pass_color
        _use_pass_color = bake.use_pass_color
        bake.use_pass_direct = False
        bake.use_pass_indirect = False
        bake.use_pass_color = False
        bake.use_pass_color = True
        _margin = bake.margin
        bake.margin = 0
        bpy.ops.object.bake(type='DIFFUSE', )
        
        # cleanup, return original values to all changed bake settings
        scene.cycles.bake_type = _bake_type
        scene.cycles.samples = _samples
        bake.use_pass_direct = _use_pass_direct
        bake.use_pass_indirect = _use_pass_indirect
        bake.use_pass_color = _use_pass_color
        bake.use_pass_color = _use_pass_color
        bake.margin = _margin
        bpy.context.scene.render.engine = _engine
        
        # make instance
        log("making ico sphere instance mesh with material..", 1)
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=base_sphere_subdivisions, radius=base_sphere_radius, location=(0, 0, 0), )
        sphere = bpy.context.active_object
        sphere.parent = o
        
        # make material for sphere instances
        mat = bpy.data.materials.new('PCParticlesMaterial')
        sphere.data.materials.append(mat)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        for node in nodes:
            nodes.remove(node)
        links = mat.node_tree.links
        node_tcoord = nodes.new(type='ShaderNodeTexCoord')
        node_tcoord.from_instancer = True
        node_tex = nodes.new(type='ShaderNodeTexImage')
        node_tex.image = img
        link = links.new(node_tcoord.outputs[2], node_tex.inputs[0])
        node_diff = nodes.new(type='ShaderNodeBsdfDiffuse')
        link = links.new(node_tex.outputs[0], node_diff.inputs[0])
        node_output = nodes.new(type='ShaderNodeOutputMaterial')
        link = links.new(node_diff.outputs[0], node_output.inputs[0])
        
        # make instancer
        o.instance_type = 'FACES'
        o.show_instancer_for_render = False
        
        _d = datetime.timedelta(seconds=time.time() - _t)
        log("completed in {}.".format(_d), 1)


class PCParticles():
    def __init__(self, o, mesh_size, base_sphere_subdivisions, ):
        log("{}:".format(self.__class__.__name__), 0, )
        _t = time.time()
        
        base_sphere_radius = mesh_size / 2
        
        # make uv layout of neatly packed triangles to bake vertex colors
        log("calculating uv layout..", 1)
        num_tri = len(o.data.polygons)
        num_sq = math.ceil(num_tri / 4)
        sq_per_uv_side = math.ceil(math.sqrt(num_sq))
        sq_size = 1 / sq_per_uv_side
        
        bm = bmesh.new()
        bm.from_mesh(o.data)
        uv_layer = bm.loops.layers.uv.new("PCParticlesUVMap")
        
        def tri_uv(i, x, y, q):
            # *---------------*
            # |  \    3    /  |
            # |    \     /    |
            # | 4     *    2  |
            # |    /     \    |
            # |  /    1    \  |
            # *---------------*
            # 1  0.0,0.0  1.0,0.0  0.5,0.5
            # 2  1.0,0.0  1.0,1.0  0.5,0.5
            # 3  1.0,1.0  0.0,1.0  0.5,0.5
            # 4  0.0,1.0  0.0,0.0  0.5,0.5
            
            if(i == 0):
                return ((x + (0.0 * q), y + (0.0 * q), ),
                        (x + (1.0 * q), y + (0.0 * q), ),
                        (x + (0.5 * q), y + (0.5 * q), ), )
            elif(i == 1):
                return ((x + (1.0 * q), y + (0.0 * q), ),
                        (x + (1.0 * q), y + (1.0 * q), ),
                        (x + (0.5 * q), y + (0.5 * q), ), )
            elif(i == 2):
                return ((x + (1.0 * q), y + (1.0 * q), ),
                        (x + (0.0 * q), y + (1.0 * q), ),
                        (x + (0.5 * q), y + (0.5 * q), ), )
            elif(i == 3):
                return ((x + (0.0 * q), y + (1.0 * q), ),
                        (x + (0.0 * q), y + (0.0 * q), ),
                        (x + (0.5 * q), y + (0.5 * q), ), )
            else:
                raise Exception("You're not supposed to do that..")
        
        sq_c = 0
        xsqn = 0
        ysqn = 0
        for face in bm.faces:
            co = tri_uv(sq_c, xsqn * sq_size, ysqn * sq_size, sq_size)
            coi = 0
            for loop in face.loops:
                loop[uv_layer].uv = co[coi]
                coi += 1
            sq_c += 1
            if(sq_c == 4):
                sq_c = 0
                xsqn += 1
                if(xsqn == sq_per_uv_side):
                    xsqn = 0
                    ysqn += 1
        
        bm.to_mesh(o.data)
        bm.free()
        
        # bake vertex colors
        log("baking vertex colors..", 1)
        _engine = bpy.context.scene.render.engine
        bpy.context.scene.render.engine = 'CYCLES'
        
        tex_size = sq_per_uv_side * 8
        img = bpy.data.images.new("PCParticlesBakedColors", tex_size, tex_size, )
        
        mat = bpy.data.materials.new('PCParticlesBakeMaterial')
        o.data.materials.append(mat)
        mat.use_nodes = True
        # remove all nodes
        nodes = mat.node_tree.nodes
        for node in nodes:
            nodes.remove(node)
        # make new nodes
        links = mat.node_tree.links
        node_attr = nodes.new(type='ShaderNodeAttribute')
        if(o.data.vertex_colors.active is not None):
            node_attr.attribute_name = o.data.vertex_colors.active.name
        node_diff = nodes.new(type='ShaderNodeBsdfDiffuse')
        link = links.new(node_attr.outputs[0], node_diff.inputs[0])
        node_output = nodes.new(type='ShaderNodeOutputMaterial')
        link = links.new(node_diff.outputs[0], node_output.inputs[0])
        node_tcoord = nodes.new(type='ShaderNodeTexCoord')
        node_tex = nodes.new(type='ShaderNodeTexImage')
        node_tex.image = img
        link = links.new(node_tcoord.outputs[2], node_tex.inputs[0])
        # set image texture selected and active
        node_tex.select = True
        nodes.active = node_tex
        
        # do the baking
        scene = bpy.context.scene
        _bake_type = scene.cycles.bake_type
        scene.cycles.bake_type = 'DIFFUSE'
        _samples = scene.cycles.samples
        scene.cycles.samples = 32
        bake = scene.render.bake
        _use_pass_direct = bake.use_pass_direct
        _use_pass_indirect = bake.use_pass_indirect
        _use_pass_color = bake.use_pass_color
        _use_pass_color = bake.use_pass_color
        bake.use_pass_direct = False
        bake.use_pass_indirect = False
        bake.use_pass_color = False
        bake.use_pass_color = True
        _margin = bake.margin
        bake.margin = 0
        bpy.ops.object.bake(type='DIFFUSE', )
        
        # cleanup, return original values to all changed bake settings
        scene.cycles.bake_type = _bake_type
        scene.cycles.samples = _samples
        bake.use_pass_direct = _use_pass_direct
        bake.use_pass_indirect = _use_pass_indirect
        bake.use_pass_color = _use_pass_color
        bake.use_pass_color = _use_pass_color
        bake.margin = _margin
        bpy.context.scene.render.engine = _engine
        
        # make particles
        log("setting up particle system..", 1)
        pmod = o.modifiers.new('PCParticleSystem', type='PARTICLE_SYSTEM', )
        settings = pmod.particle_system.settings
        settings.count = num_tri
        settings.frame_end = 1
        settings.normal_factor = 0
        settings.emit_from = 'FACE'
        settings.use_emit_random = False
        settings.use_even_distribution = False
        settings.userjit = 1
        settings.render_type = 'OBJECT'
        settings.particle_size = 1
        settings.display_method = 'DOT'
        settings.display_size = base_sphere_radius
        settings.physics_type = 'NO'
        
        o.show_instancer_for_render = False
        
        # make instance
        log("making ico sphere instance mesh with material..", 1)
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=base_sphere_subdivisions, radius=base_sphere_radius, location=(0, 0, 0), )
        sphere = bpy.context.active_object
        sphere.parent = o
        
        # make material for sphere instances
        mat = bpy.data.materials.new('PCParticlesMaterial')
        sphere.data.materials.append(mat)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        for node in nodes:
            nodes.remove(node)
        links = mat.node_tree.links
        node_tcoord = nodes.new(type='ShaderNodeTexCoord')
        node_tcoord.from_instancer = True
        node_tex = nodes.new(type='ShaderNodeTexImage')
        node_tex.image = img
        link = links.new(node_tcoord.outputs[2], node_tex.inputs[0])
        node_diff = nodes.new(type='ShaderNodeBsdfDiffuse')
        link = links.new(node_tex.outputs[0], node_diff.inputs[0])
        node_output = nodes.new(type='ShaderNodeOutputMaterial')
        link = links.new(node_diff.outputs[0], node_output.inputs[0])
        
        # assign sphere to particles
        settings.instance_object = sphere
        
        _d = datetime.timedelta(seconds=time.time() - _t)
        log("completed in {}.".format(_d), 1)


class BinPlyPointCloudReader():
    def __init__(self, path, ):
        log("{}:".format(self.__class__.__name__), 0)
        if(os.path.exists(path) is False or os.path.isdir(path) is True):
            raise OSError("did you point me to an imaginary file? ('{0}')".format(path))
        
        self.path = path
        self._stream = open(self.path, "rb")
        log("reading header..", 1)
        self._header()
        log("reading data:", 1)
        self._data_np()
        self._stream.close()
        self.points = self.data['vertex']
        log("done.", 1)
    
    def _header(self):
        raw = []
        h = []
        for l in self._stream:
            raw.append(l)
            t = l.decode('ascii')
            h.append(t.rstrip())
            if(t == "end_header\n"):
                break
        
        self._header_length = sum([len(i) for i in raw])
        
        _supported_version = '1.0'
        _byte_order = {'binary_little_endian': '<',
                       'binary_big_endian': '>',
                       'ascii': None, }
        _types = {'char': 'c',
                  'uchar': 'B',
                  'short': 'h',
                  'ushort': 'H',
                  'int': 'i',
                  'uint': 'I',
                  'float': 'f',
                  'double': 'd', }
        
        _ply = False
        _format = None
        _endianness = None
        _version = None
        _comments = []
        _elements = []
        _current_element = None
        
        for i, l in enumerate(h):
            if(i == 0 and l == 'ply'):
                _ply = True
                continue
            if(l.startswith('format ')):
                _format = l[7:]
                a = _format.split(' ')
                _endianness = _byte_order[a[0]]
                _version = a[1]
            if(l.startswith('comment ')):
                _comments.append(l[8:])
            if(l.startswith('element ')):
                a = l.split(' ')
                _elements.append({'name': a[1], 'properties': [], 'count': int(a[2]), })
                _current_element = len(_elements) - 1
            if(l.startswith('property ')):
                a = l[9:].split(' ')
                if(a[0] != 'list'):
                    _elements[_current_element]['properties'].append((a[1], _types[a[0]]))
                else:
                    c = _types[a[2]]
                    t = _types[a[2]]
                    n = a[3]
                    _elements[_current_element]['properties'].append((n, c, t))
            if(i == len(h) - 1 and l == 'end_header'):
                continue
        
        if(not _ply):
            raise ValueError("not a ply file")
        if(_version != _supported_version):
            raise ValueError("unsupported ply file version")
        if(_endianness is None):
            raise ValueError("ascii ply files are not supported")
        
        self._endianness = _endianness
        self._elements = _elements
    
    def _data_np(self):
        self.data = {}
        for i, d in enumerate(self._elements):
            nm = d['name']
            if(nm != 'vertex'):
                # read only vertices
                continue
            props = d['properties']
            dtp = [None] * len(props)
            e = self._endianness
            for i, p in enumerate(props):
                n, t = p
                dtp[i] = (n, '{}{}'.format(e, t))
            dt = np.dtype(dtp)
            self._stream.seek(self._header_length)
            c = d['count']
            log("reading {} {} elements..".format(c, nm), 2)
            a = np.fromfile(self._stream, dtype=dt, count=c, )
            self.data[nm] = a


class PlyPointCloudReader():
    _supported_formats = ('binary_little_endian', 'binary_big_endian', 'ascii', )
    _supported_versions = ('1.0', )
    _byte_order = {'binary_little_endian': '<', 'binary_big_endian': '>', 'ascii': None, }
    # _types = {'char': 'c', 'uchar': 'B', 'short': 'h', 'ushort': 'H', 'int': 'i', 'uint': 'I', 'float': 'f', 'double': 'd', }
    _types = {
        'char': 'b',
        'uchar': 'B',
        'int8': 'b',
        'uint8': 'B',
        'int16': 'h',
        'uint16': 'H',
        'short': 'h',
        'ushort': 'H',
        'int': 'i',
        'int32': 'i',
        'uint': 'I',
        'uint32': 'I',
        'float': 'f',
        'float32': 'f',
        'float64': 'd',
        'double': 'd',
        'string': 's',
    }
    
    def __init__(self, path, ):
        log("{}:".format(self.__class__.__name__), 0)
        if(os.path.exists(path) is False or os.path.isdir(path) is True):
            raise OSError("did you point me to an imaginary file? ('{}')".format(path))
        
        self.path = path
        log("will read file at: '{}'".format(self.path), 1)
        log("reading header..", 1)
        self._header()
        log("reading data..", 1)
        if(self._ply_format == 'ascii'):
            self._data_ascii()
        else:
            self._data_binary()
        log("loaded {} vertices".format(len(self.points)), 1)
        
        # remove alpha if present (meshlab adds it)
        self.points = self.points[[b for b in list(self.points.dtype.names) if b != 'alpha']]
        
        # rename diffuse_rgb to rgb, if present
        user_rgb = ('diffuse_red', 'diffuse_green', 'diffuse_blue', )
        names = self.points.dtype.names
        ls = list(names)
        if(set(user_rgb).issubset(names)):
            for ci, uc in enumerate(user_rgb):
                for i, v in enumerate(ls):
                    if(v == uc):
                        ls[i] = ls[i].replace('diffuse_', '', )
        self.points.dtype.names = tuple(ls)
        
        # remove anything that is not (x, y, z, nx, ny, nz, red, green, blue) to prevent problems later
        self.points = self.points[[b for b in list(self.points.dtype.names) if b in ('x', 'y', 'z', 'nx', 'ny', 'nz', 'red', 'green', 'blue', )]]
        
        # some info
        nms = self.points.dtype.names
        self.has_vertices = True
        self.has_normals = True
        self.has_colors = True
        if(not set(('x', 'y', 'z')).issubset(nms)):
            self.has_vertices = False
        if(not set(('nx', 'ny', 'nz')).issubset(nms)):
            self.has_normals = False
        if(not set(('red', 'green', 'blue')).issubset(nms)):
            self.has_colors = False
        log('has_vertices: {}'.format(self.has_vertices), 2)
        log('has_normals: {}'.format(self.has_normals), 2)
        log('has_colors: {}'.format(self.has_colors), 2)
        
        log("done.", 1)
    
    def _header(self):
        raw = []
        h = []
        with open(self.path, mode='rb') as f:
            for l in f:
                raw.append(l)
                a = l.decode('ascii').rstrip()
                h.append(a)
                if(a == "end_header"):
                    break
        
        if(h[0] != 'ply'):
            raise TypeError("not a ply file")
        for i, l in enumerate(h):
            if(l.startswith('format')):
                _, f, v = l.split(' ')
                if(f not in self._supported_formats):
                    raise TypeError("unsupported ply format")
                if(v not in self._supported_versions):
                    raise TypeError("unsupported ply file version")
                self._ply_format = f
                self._ply_version = v
                if(self._ply_format != 'ascii'):
                    self._endianness = self._byte_order[self._ply_format]
        
        self._elements = []
        current_element = None
        for i, l in enumerate(h):
            if(l.startswith('ply')):
                pass
            elif(l.startswith('format')):
                pass
            elif(l.startswith('comment')):
                pass
            elif(l.startswith('element')):
                _, t, c = l.split(' ')
                a = {'type': t, 'count': int(c), 'props': [], }
                self._elements.append(a)
                current_element = a
            elif(l.startswith('property')):
                if(l.startswith('property list')):
                    _, _, c, t, n = l.split(' ')
                    if(self._ply_format == 'ascii'):
                        current_element['props'].append((n, self._types[c], self._types[t], ))
                    else:
                        current_element['props'].append((n, self._types[c], self._types[t], ))
                else:
                    _, t, n = l.split(' ')
                    if(self._ply_format == 'ascii'):
                        current_element['props'].append((n, self._types[t]))
                    else:
                        current_element['props'].append((n, self._types[t]))
            elif(l.startswith('end_header')):
                pass
            else:
                log('unknown header line: {}'.format(l))
        
        if(self._ply_format == 'ascii'):
            skip = False
            flen = 0
            hlen = 0
            with open(self.path, mode='r', encoding='utf-8') as f:
                for i, l in enumerate(f):
                    flen += 1
                    if(skip):
                        continue
                    hlen += 1
                    if(l.rstrip() == 'end_header'):
                        skip = True
            self._header_length = hlen
            self._file_length = flen
        else:
            self._header_length = sum([len(i) for i in raw])
    
    def _data_binary(self):
        self.points = []
        
        read_from = self._header_length
        for ie, element in enumerate(self._elements):
            if(element['type'] != 'vertex'):
                continue
            
            dtp = []
            for i, p in enumerate(element['props']):
                n, t = p
                dtp.append((n, '{}{}'.format(self._endianness, t), ))
            dt = np.dtype(dtp)
            with open(self.path, mode='rb') as f:
                f.seek(read_from)
                a = np.fromfile(f, dtype=dt, count=element['count'], )
            
            self.points = a
            read_from += element['count']
    
    def _data_ascii(self):
        self.points = []
        
        skip_header = self._header_length
        skip_footer = self._file_length - self._header_length
        for ie, element in enumerate(self._elements):
            if(element['type'] != 'vertex'):
                continue
            
            skip_footer = skip_footer - element['count']
            with open(self.path, mode='r', encoding='utf-8') as f:
                a = np.genfromtxt(f, dtype=np.dtype(element['props']), skip_header=skip_header, skip_footer=skip_footer, )
            self.points = a
            skip_header += element['count']


class BinPlyPointCloudWriter():
    """Save binary ply file from data numpy array
    
    Args:
        path: path to ply file
        points: strucured array of points as (x, y, z, nx, ny, nz, red, green, blue) (normals and colors are optional)
    
    Attributes:
        path (str): real path to ply file
    
    """
    
    _types = {'c': 'char', 'B': 'uchar', 'h': 'short', 'H': 'ushort', 'i': 'int', 'I': 'uint', 'f': 'float', 'd': 'double', }
    _byte_order = {'little': 'binary_little_endian', 'big': 'binary_big_endian', }
    _comment = "created with Point Cloud Visualizer"
    
    def __init__(self, path, points, ):
        log("{}:".format(self.__class__.__name__), 0)
        self.path = os.path.realpath(path)
        
        # write
        log("will write to: {}".format(self.path), 1)
        # write to temp file first
        n = os.path.splitext(os.path.split(self.path)[1])[0]
        t = "{}.temp.ply".format(n)
        p = os.path.join(os.path.dirname(self.path), t)
        
        l = len(points)
        
        with open(p, 'wb') as f:
            # write header
            log("writing header..", 2)
            dt = points.dtype
            h = "ply\n"
            # x should be a float of some kind, therefore we can get endianess
            bo = dt['x'].byteorder
            if(bo != '='):
                # not native byteorder
                if(bo == '>'):
                    h += "format {} 1.0\n".format(self._byte_order['big'])
                else:
                    h += "format {} 1.0\n".format(self._byte_order['little'])
            else:
                # byteorder was native, use what sys.byteorder says..
                h += "format {} 1.0\n".format(self._byte_order[sys.byteorder])
            h += "element vertex {}\n".format(l)
            # construct header from data names/types in points array
            for n in dt.names:
                t = self._types[dt[n].char]
                h += "property {} {}\n".format(t, n)
            h += "comment {}\n".format(self._comment)
            h += "end_header\n"
            f.write(h.encode('ascii'))
            
            # write data
            log("writing data.. ({} points)".format(l), 2)
            f.write(points.tobytes())
        
        # remove original file (if needed) and rename temp
        if(os.path.exists(self.path)):
            os.remove(self.path)
        shutil.move(p, self.path)
        
        log("done.", 1)


class PCVShaders():
    vertex_shader_illumination = '''
        in vec3 position;
        in vec3 normal;
        in vec4 color;
        
        // uniform float show_illumination;
        uniform vec3 light_direction;
        uniform vec3 light_intensity;
        uniform vec3 shadow_direction;
        uniform vec3 shadow_intensity;
        // uniform float show_normals;
        
        uniform mat4 perspective_matrix;
        uniform mat4 object_matrix;
        uniform float point_size;
        uniform float alpha_radius;
        uniform float global_alpha;
        
        out vec4 f_color;
        out float f_alpha_radius;
        out vec3 f_normal;
        
        out vec3 f_light_direction;
        out vec3 f_light_intensity;
        out vec3 f_shadow_direction;
        out vec3 f_shadow_intensity;
        // out float f_show_normals;
        // out float f_show_illumination;
        
        void main()
        {
            gl_Position = perspective_matrix * object_matrix * vec4(position, 1.0f);
            gl_PointSize = point_size;
            f_normal = normal;
            // f_color = color;
            f_color = vec4(color[0], color[1], color[2], global_alpha);
            f_alpha_radius = alpha_radius;
            
            // f_light_direction = normalize(vec3(inverse(object_matrix) * vec4(light_direction, 1.0)));
            f_light_direction = light_direction;
            f_light_intensity = light_intensity;
            // f_shadow_direction = normalize(vec3(inverse(object_matrix) * vec4(shadow_direction, 1.0)));
            f_shadow_direction = shadow_direction;
            f_shadow_intensity = shadow_intensity;
            // f_show_normals = show_normals;
            // f_show_illumination = show_illumination;
        }
    '''
    fragment_shader_illumination = '''
        in vec4 f_color;
        in vec3 f_normal;
        in float f_alpha_radius;
        
        in vec3 f_light_direction;
        in vec3 f_light_intensity;
        in vec3 f_shadow_direction;
        in vec3 f_shadow_intensity;
        // in float f_show_normals;
        // in float f_show_illumination;
        
        out vec4 fragColor;
        
        void main()
        {
            float r = 0.0f;
            float a = 1.0f;
            vec2 cxy = 2.0f * gl_PointCoord - 1.0f;
            r = dot(cxy, cxy);
            if(r > f_alpha_radius){
                discard;
            }
            // fragColor = f_color * a;
            
            vec4 col;
            
            // if(f_show_normals > 0.5){
            //     col = vec4(f_normal, 1.0) * a;
            // }else if(f_show_illumination > 0.5){
            
            // if(f_show_illumination > 0.5){
            //     vec4 light = vec4(max(dot(f_light_direction, -f_normal), 0) * f_light_intensity, 1);
            //     vec4 shadow = vec4(max(dot(f_shadow_direction, -f_normal), 0) * f_shadow_intensity, 1);
            //     col = (f_color + light - shadow) * a;
            // }else{
            //     col = f_color * a;
            // }
            
            vec4 light = vec4(max(dot(f_light_direction, -f_normal), 0) * f_light_intensity, 1);
            vec4 shadow = vec4(max(dot(f_shadow_direction, -f_normal), 0) * f_shadow_intensity, 1);
            col = (f_color + light - shadow) * a;
            
            fragColor = col;
        }
    '''
    
    vertex_shader_simple = '''
        in vec3 position;
        in vec4 color;
        uniform mat4 perspective_matrix;
        uniform mat4 object_matrix;
        uniform float point_size;
        uniform float alpha_radius;
        uniform float global_alpha;
        out vec4 f_color;
        out float f_alpha_radius;
        void main()
        {
            gl_Position = perspective_matrix * object_matrix * vec4(position, 1.0f);
            gl_PointSize = point_size;
            // f_color = color;
            f_color = vec4(color[0], color[1], color[2], global_alpha);
            f_alpha_radius = alpha_radius;
        }
    '''
    fragment_shader_simple = '''
        in vec4 f_color;
        in float f_alpha_radius;
        out vec4 fragColor;
        void main()
        {
            float r = 0.0f;
            float a = 1.0f;
            vec2 cxy = 2.0f * gl_PointCoord - 1.0f;
            r = dot(cxy, cxy);
            if(r > f_alpha_radius){
                discard;
            }
            fragColor = f_color * a;
        }
    '''
    
    normals_vertex_shader = '''
        layout(location = 0) in vec3 position;
        layout(location = 1) in vec3 normal;
        
        out vec3 vertex_normal;
        
        void main()
        {
            vertex_normal = normal;
            gl_Position = vec4(position, 1.0);
        }
    '''
    normals_fragment_shader = '''
        layout(location = 0) out vec4 frag_color;
        
        uniform float global_alpha;
        in vec4 vertex_color;
        
        void main()
        {
            // frag_color = vertex_color;
            frag_color = vec4(vertex_color[0], vertex_color[1], vertex_color[2], vertex_color[3] * global_alpha);
        }
    '''
    normals_geometry_shader = '''
        layout(points) in;
        layout(line_strip, max_vertices = 2) out;
        
        uniform mat4 perspective_matrix;
        uniform mat4 object_matrix;
        uniform float length = 1.0;
        uniform vec4 color = vec4(1.0, 1.0, 1.0, 1.0);
        
        in vec3 vertex_normal[];
        
        out vec4 vertex_color;
        
        void main()
        {
            vec3 normal = vertex_normal[0];
            
            vertex_color = color;
            
            vec4 v0 = gl_in[0].gl_Position;
            gl_Position = perspective_matrix * object_matrix * v0;
            EmitVertex();
            
            vec4 v1 = v0 + vec4(normal * length, 0.0);
            gl_Position = perspective_matrix * object_matrix * v1;
            EmitVertex();
            
            EndPrimitive();
        }
    '''
    
    depth_vertex_shader_simple = '''
        in vec3 position;
        uniform mat4 perspective_matrix;
        uniform mat4 object_matrix;
        uniform vec3 center;
        uniform float point_size;
        uniform float maxdist;
        out float f_depth;
        void main()
        {
            gl_Position = perspective_matrix * object_matrix * vec4(position, 1.0);
            gl_PointSize = point_size;
            vec4 pp = perspective_matrix * object_matrix * vec4(position, 1.0);
            vec4 op = perspective_matrix * object_matrix * vec4(center, 1.0);
            float d = op.z - pp.z;
            f_depth = ((d - (-maxdist)) / (maxdist - d)) / 2;
        }
    '''
    depth_fragment_shader_simple = '''
        in float f_depth;
        uniform float brightness;
        uniform float contrast;
        uniform float alpha_radius;
        uniform float global_alpha;
        out vec4 fragColor;
        void main()
        {
            float r = 0.0f;
            float a = 1.0f;
            vec2 cxy = 2.0f * gl_PointCoord - 1.0f;
            r = dot(cxy, cxy);
            if(r > alpha_radius){
                discard;
            }
            vec3 color = vec3(f_depth, f_depth, f_depth);
            color = (color - 0.5) * contrast + 0.5 + brightness;
            fragColor = vec4(color, global_alpha) * a;
        }
    '''
    
    selection_vertex_shader = '''
        in vec3 position;
        uniform mat4 perspective_matrix;
        uniform mat4 object_matrix;
        uniform float point_size;
        void main()
        {
            gl_Position = perspective_matrix * object_matrix * vec4(position, 1.0f);
            gl_PointSize = point_size;
        }
    '''
    selection_fragment_shader = '''
        uniform vec4 color;
        uniform float alpha_radius;
        out vec4 fragColor;
        void main()
        {
            float r = 0.0f;
            float a = 1.0f;
            vec2 cxy = 2.0f * gl_PointCoord - 1.0f;
            r = dot(cxy, cxy);
            if(r > alpha_radius){
                discard;
            }
            fragColor = color * a;
        }
    '''
    
    normal_colors_vertex_shader = '''
        in vec3 position;
        in vec3 normal;
        uniform mat4 perspective_matrix;
        uniform mat4 object_matrix;
        uniform float point_size;
        out vec3 f_color;
        void main()
        {
            gl_Position = perspective_matrix * object_matrix * vec4(position, 1.0f);
            gl_PointSize = point_size;
            f_color = normal * 0.5 + 0.5;
            // f_color = normal;
        }
    '''
    normal_colors_fragment_shader = '''
        // uniform vec4 color;
        in vec3 f_color;
        uniform float alpha_radius;
        uniform float global_alpha;
        out vec4 fragColor;
        void main()
        {
            float r = 0.0f;
            float a = 1.0f;
            vec2 cxy = 2.0f * gl_PointCoord - 1.0f;
            r = dot(cxy, cxy);
            if(r > alpha_radius){
                discard;
            }
            fragColor = vec4(f_color, global_alpha) * a;
        }
    '''
    
    depth_vertex_shader_illumination = '''
        in vec3 position;
        in vec3 normal;
        uniform mat4 perspective_matrix;
        uniform mat4 object_matrix;
        uniform vec3 center;
        uniform float point_size;
        uniform float maxdist;
        uniform vec3 light_direction;
        uniform vec3 light_intensity;
        uniform vec3 shadow_direction;
        uniform vec3 shadow_intensity;
        out float f_depth;
        out vec3 f_light_direction;
        out vec3 f_light_intensity;
        out vec3 f_shadow_direction;
        out vec3 f_shadow_intensity;
        out vec3 f_normal;
        void main()
        {
            gl_Position = perspective_matrix * object_matrix * vec4(position, 1.0);
            gl_PointSize = point_size;
            vec4 pp = perspective_matrix * object_matrix * vec4(position, 1.0);
            vec4 op = perspective_matrix * object_matrix * vec4(center, 1.0);
            float d = op.z - pp.z;
            f_depth = ((d - (-maxdist)) / (maxdist - d)) / 2;
            f_normal = normal;
            f_light_direction = light_direction;
            f_light_intensity = light_intensity;
            f_shadow_direction = shadow_direction;
            f_shadow_intensity = shadow_intensity;
        }
    '''
    depth_fragment_shader_illumination = '''
        in float f_depth;
        in vec3 f_normal;
        in vec3 f_light_direction;
        in vec3 f_light_intensity;
        in vec3 f_shadow_direction;
        in vec3 f_shadow_intensity;
        uniform float alpha_radius;
        uniform float global_alpha;
        uniform float brightness;
        uniform float contrast;
        uniform vec3 color_a;
        uniform vec3 color_b;
        out vec4 fragColor;
        void main()
        {
            float r = 0.0f;
            float a = 1.0f;
            vec2 cxy = 2.0f * gl_PointCoord - 1.0f;
            r = dot(cxy, cxy);
            if(r > alpha_radius){
                discard;
            }
            vec3 l = vec3(max(dot(f_light_direction, -f_normal), 0) * f_light_intensity);
            vec3 s = vec3(max(dot(f_shadow_direction, -f_normal), 0) * f_shadow_intensity);
            vec3 color = mix(color_b, color_a, f_depth);
            // brightness/contrast after illumination
            // vec3 c = color + l - s;
            // vec3 cc = (c - 0.5) * contrast + 0.5 + brightness;
            // fragColor = vec4(cc, global_alpha) * a;
            
            // brightness/contrast before illumination
            vec3 cc = (color - 0.5) * contrast + 0.5 + brightness;
            vec3 c = cc + l - s;
            fragColor = vec4(c, global_alpha) * a;
        }
    '''
    depth_vertex_shader_false_colors = '''
        in vec3 position;
        uniform mat4 perspective_matrix;
        uniform mat4 object_matrix;
        uniform vec3 center;
        uniform float point_size;
        uniform float maxdist;
        out float f_depth;
        void main()
        {
            gl_Position = perspective_matrix * object_matrix * vec4(position, 1.0);
            gl_PointSize = point_size;
            vec4 pp = perspective_matrix * object_matrix * vec4(position, 1.0);
            vec4 op = perspective_matrix * object_matrix * vec4(center, 1.0);
            float d = op.z - pp.z;
            f_depth = ((d - (-maxdist)) / (maxdist - d)) / 2;
        }
    '''
    depth_fragment_shader_false_colors = '''
        in float f_depth;
        uniform float alpha_radius;
        uniform float global_alpha;
        uniform float brightness;
        uniform float contrast;
        uniform vec3 color_a;
        uniform vec3 color_b;
        out vec4 fragColor;
        void main()
        {
            float r = 0.0f;
            float a = 1.0f;
            vec2 cxy = 2.0f * gl_PointCoord - 1.0f;
            r = dot(cxy, cxy);
            if(r > alpha_radius){
                discard;
            }
            vec3 color = mix(color_b, color_a, f_depth);
            color = (color - 0.5) * contrast + 0.5 + brightness;
            fragColor = vec4(color, global_alpha) * a;
        }
    '''
    
    position_colors_vertex_shader = '''
        in vec3 position;
        uniform mat4 perspective_matrix;
        uniform mat4 object_matrix;
        uniform float point_size;
        out vec3 f_color;
        void main()
        {
            gl_Position = perspective_matrix * object_matrix * vec4(position, 1.0f);
            gl_PointSize = point_size;
            // f_color = position * 0.5 + 0.5;
            f_color = position;
        }
    '''
    position_colors_fragment_shader = '''
        in vec3 f_color;
        uniform float alpha_radius;
        uniform float global_alpha;
        out vec4 fragColor;
        void main()
        {
            float r = 0.0f;
            float a = 1.0f;
            vec2 cxy = 2.0f * gl_PointCoord - 1.0f;
            r = dot(cxy, cxy);
            if(r > alpha_radius){
                discard;
            }
            fragColor = vec4(mod(f_color, 1.0), global_alpha) * a;
        }
    '''
    
    bbox_vertex_shader = '''
        layout(location = 0) in vec3 position;
        
        void main()
        {
            gl_Position = vec4(position, 1.0);
        }
    '''
    bbox_fragment_shader = '''
        layout(location = 0) out vec4 frag_color;
        
        uniform float global_alpha;
        in vec4 vertex_color;
        
        void main()
        {
            frag_color = vec4(vertex_color.rgb, vertex_color[3] * global_alpha);
        }
    '''
    bbox_geometry_shader = '''
        layout(points) in;
        layout(line_strip, max_vertices = 256) out;
        
        uniform mat4 perspective_matrix;
        uniform mat4 object_matrix;
        uniform vec4 color = vec4(1.0, 1.0, 1.0, 1.0);
        
        uniform float length = 0.1;
        uniform vec3 center = vec3(0.0, 0.0, 0.0);
        uniform vec3 dimensions = vec3(1.0, 1.0, 1.0);
        
        out vec4 vertex_color;
        
        void line();
        
        void line(vec4 o, vec3 a, vec3 b)
        {
            gl_Position = perspective_matrix * object_matrix * (o + vec4(a, 0.0));
            EmitVertex();
            gl_Position = perspective_matrix * object_matrix * (o + vec4(b, 0.0));
            EmitVertex();
            EndPrimitive();
        }
        
        void main()
        {
            vertex_color = color;
            
            //vec4 o = gl_in[0].gl_Position;
            vec4 o = vec4(center, 1.0);
            
            float w = dimensions[0] / 2;
            float h = dimensions[1] / 2;
            float d = dimensions[2] / 2;
            float l = length;
            
            vec3 p00 = vec3(-(w - l),       -h,       -d);
            vec3 p01 = vec3(      -w,       -h,       -d);
            vec3 p02 = vec3(      -w,       -h, -(d - l));
            vec3 p03 = vec3(      -w, -(h - l),       -d);
            vec3 p04 = vec3(-(w - l),       -h,        d);
            vec3 p05 = vec3(      -w,       -h,        d);
            vec3 p06 = vec3(      -w, -(h - l),        d);
            vec3 p07 = vec3(      -w,       -h,  (d - l));
            vec3 p08 = vec3(      -w,  (h - l),       -d);
            vec3 p09 = vec3(      -w,        h,       -d);
            vec3 p10 = vec3(      -w,        h, -(d - l));
            vec3 p11 = vec3(-(w - l),        h,       -d);
            vec3 p12 = vec3(-(w - l),        h,        d);
            vec3 p13 = vec3(      -w,        h,        d);
            vec3 p14 = vec3(      -w,        h,  (d - l));
            vec3 p15 = vec3(      -w,  (h - l),        d);
            vec3 p16 = vec3(       w, -(h - l),       -d);
            vec3 p17 = vec3(       w,       -h,       -d);
            vec3 p18 = vec3(       w,       -h, -(d - l));
            vec3 p19 = vec3( (w - l),       -h,       -d);
            vec3 p20 = vec3( (w - l),       -h,        d);
            vec3 p21 = vec3(       w,       -h,        d);
            vec3 p22 = vec3(       w,       -h,  (d - l));
            vec3 p23 = vec3(       w, -(h - l),        d);
            vec3 p24 = vec3( (w - l),        h,       -d);
            vec3 p25 = vec3(       w,        h,       -d);
            vec3 p26 = vec3(       w,        h, -(d - l));
            vec3 p27 = vec3(       w,  (h - l),       -d);
            vec3 p28 = vec3(       w,  (h - l),        d);
            vec3 p29 = vec3(       w,        h,        d);
            vec3 p30 = vec3(       w,        h,  (d - l));
            vec3 p31 = vec3( (w - l),        h,        d);
            
            line(o, p00, p01);
            line(o, p01, p03);
            line(o, p02, p01);
            line(o, p04, p05);
            line(o, p05, p07);
            line(o, p06, p05);
            line(o, p08, p09);
            line(o, p09, p11);
            line(o, p10, p09);
            line(o, p12, p13);
            line(o, p13, p15);
            line(o, p14, p13);
            line(o, p16, p17);
            line(o, p17, p19);
            line(o, p18, p17);
            line(o, p20, p21);
            line(o, p21, p23);
            line(o, p22, p21);
            line(o, p24, p25);
            line(o, p25, p27);
            line(o, p26, p25);
            line(o, p28, p29);
            line(o, p29, p31);
            line(o, p30, p29);
            
        }
    '''
    
    vertex_shader_color_adjustment = '''
        in vec3 position;
        in vec4 color;
        uniform mat4 perspective_matrix;
        uniform mat4 object_matrix;
        uniform float point_size;
        uniform float alpha_radius;
        uniform float global_alpha;
        
        uniform float exposure;
        uniform float gamma;
        uniform float brightness;
        uniform float contrast;
        uniform float hue;
        uniform float saturation;
        uniform float value;
        uniform float invert;
        
        out vec4 f_color;
        out float f_alpha_radius;
        
        out float f_exposure;
        out float f_gamma;
        out float f_brightness;
        out float f_contrast;
        out float f_hue;
        out float f_saturation;
        out float f_value;
        out float f_invert;
        
        void main()
        {
            gl_Position = perspective_matrix * object_matrix * vec4(position, 1.0f);
            gl_PointSize = point_size;
            f_color = vec4(color[0], color[1], color[2], global_alpha);
            f_alpha_radius = alpha_radius;
            
            f_exposure = exposure;
            f_gamma = gamma;
            f_brightness = brightness;
            f_contrast = contrast;
            f_hue = hue;
            f_saturation = saturation;
            f_value = value;
            f_invert = invert;
        }
    '''
    fragment_shader_color_adjustment = '''
        in vec4 f_color;
        in float f_alpha_radius;
        out vec4 fragColor;
        
        in float f_exposure;
        in float f_gamma;
        in float f_brightness;
        in float f_contrast;
        in float f_hue;
        in float f_saturation;
        in float f_value;
        in float f_invert;
        
        // https://stackoverflow.com/questions/15095909/from-rgb-to-hsv-in-opengl-glsl
        vec3 rgb2hsv(vec3 c)
        {
            vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
            vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
            vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));

            float d = q.x - min(q.w, q.y);
            float e = 1.0e-10;
            return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
        }
        vec3 hsv2rgb(vec3 c)
        {
            vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
            vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
            return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
        }
        
        void main()
        {
            float r = 0.0f;
            float a = 1.0f;
            vec2 cxy = 2.0f * gl_PointCoord - 1.0f;
            r = dot(cxy, cxy);
            if(r > f_alpha_radius){
                discard;
            }
            fragColor = f_color * a;
            
            // adjustments
            vec3 rgb = fragColor.rgb;
            float alpha = fragColor.a;
            vec3 color = rgb;
            
            // exposure
            color = clamp(color * pow(2, f_exposure), 0.0, 1.0);
            // gamma
            color = clamp(vec3(pow(color[0], 1 / f_gamma), pow(color[1], 1 / f_gamma), pow(color[2], 1 / f_gamma)), 0.0, 1.0);
            
            // brightness/contrast
            color = clamp((color - 0.5) * f_contrast + 0.5 + f_brightness, 0.0, 1.0);
            
            // hue/saturation/value
            vec3 hsv = rgb2hsv(color);
            float hue = f_hue;
            if(hue > 1.0){
                hue = mod(hue, 1.0);
            }
            hsv[0] = mod((hsv[0] + hue), 1.0);
            hsv[1] += f_saturation;
            hsv[2] += f_value;
            hsv = clamp(hsv, 0.0, 1.0);
            color = hsv2rgb(hsv);
            
            if(f_invert > 0.0){
                color = vec3(1.0 - color[0], 1.0 - color[1], 1.0 - color[2]);
            }
            
            fragColor = vec4(color, alpha);
            
        }
    '''
    
    vertex_shader_simple_render_smooth = '''
        in vec3 position;
        in vec4 color;
        uniform mat4 perspective_matrix;
        uniform mat4 object_matrix;
        uniform float point_size;
        uniform float alpha_radius;
        uniform float global_alpha;
        out vec4 f_color;
        out float f_alpha_radius;
        void main()
        {
            gl_Position = perspective_matrix * object_matrix * vec4(position, 1.0f);
            gl_PointSize = point_size;
            f_color = vec4(color[0], color[1], color[2], global_alpha);
            f_alpha_radius = alpha_radius;
        }
    '''
    fragment_shader_simple_render_smooth = '''
        in vec4 f_color;
        in float f_alpha_radius;
        out vec4 fragColor;
        void main()
        {
            float r = 0.0f;
            float d = 0.0f;
            float a = 1.0f;
            vec2 cxy = 2.0f * gl_PointCoord - 1.0f;
            r = dot(cxy, cxy);
            d = fwidth(r);
            a = 1.0 - smoothstep(1.0 - (d / 2), 1.0 + (d / 2), r);
            //fragColor = f_color * a;
            fragColor = vec4(f_color.rgb, f_color.a * a);
        }
    '''
    
    vertex_shader_illumination_render_smooth = '''
        in vec3 position;
        in vec3 normal;
        in vec4 color;
        
        // uniform float show_illumination;
        uniform vec3 light_direction;
        uniform vec3 light_intensity;
        uniform vec3 shadow_direction;
        uniform vec3 shadow_intensity;
        // uniform float show_normals;
        
        uniform mat4 perspective_matrix;
        uniform mat4 object_matrix;
        uniform float point_size;
        uniform float alpha_radius;
        uniform float global_alpha;
        
        out vec4 f_color;
        out float f_alpha_radius;
        out vec3 f_normal;
        
        out vec3 f_light_direction;
        out vec3 f_light_intensity;
        out vec3 f_shadow_direction;
        out vec3 f_shadow_intensity;
        // out float f_show_normals;
        // out float f_show_illumination;
        
        void main()
        {
            gl_Position = perspective_matrix * object_matrix * vec4(position, 1.0f);
            gl_PointSize = point_size;
            f_normal = normal;
            // f_color = color;
            f_color = vec4(color[0], color[1], color[2], global_alpha);
            f_alpha_radius = alpha_radius;
            
            // f_light_direction = normalize(vec3(inverse(object_matrix) * vec4(light_direction, 1.0)));
            f_light_direction = light_direction;
            f_light_intensity = light_intensity;
            // f_shadow_direction = normalize(vec3(inverse(object_matrix) * vec4(shadow_direction, 1.0)));
            f_shadow_direction = shadow_direction;
            f_shadow_intensity = shadow_intensity;
            // f_show_normals = show_normals;
            // f_show_illumination = show_illumination;
        }
    '''
    fragment_shader_illumination_render_smooth = '''
        in vec4 f_color;
        in vec3 f_normal;
        in float f_alpha_radius;
        
        in vec3 f_light_direction;
        in vec3 f_light_intensity;
        in vec3 f_shadow_direction;
        in vec3 f_shadow_intensity;
        // in float f_show_normals;
        // in float f_show_illumination;
        
        out vec4 fragColor;
        
        void main()
        {
            // float r = 0.0f;
            // float a = 1.0f;
            // vec2 cxy = 2.0f * gl_PointCoord - 1.0f;
            // r = dot(cxy, cxy);
            // if(r > f_alpha_radius){
            //     discard;
            // }
            // // fragColor = f_color * a;
            //
            // vec4 col;
            //
            // // if(f_show_normals > 0.5){
            // //     col = vec4(f_normal, 1.0) * a;
            // // }else if(f_show_illumination > 0.5){
            //
            // // if(f_show_illumination > 0.5){
            // //     vec4 light = vec4(max(dot(f_light_direction, -f_normal), 0) * f_light_intensity, 1);
            // //     vec4 shadow = vec4(max(dot(f_shadow_direction, -f_normal), 0) * f_shadow_intensity, 1);
            // //     col = (f_color + light - shadow) * a;
            // // }else{
            // //     col = f_color * a;
            // // }
            //
            // vec4 light = vec4(max(dot(f_light_direction, -f_normal), 0) * f_light_intensity, 1);
            // vec4 shadow = vec4(max(dot(f_shadow_direction, -f_normal), 0) * f_shadow_intensity, 1);
            // col = (f_color + light - shadow) * a;
            //
            // fragColor = col;
            
            float r = 0.0f;
            float d = 0.0f;
            float a = 1.0f;
            vec2 cxy = 2.0f * gl_PointCoord - 1.0f;
            r = dot(cxy, cxy);
            d = fwidth(r);
            a = 1.0 - smoothstep(1.0 - (d / 2), 1.0 + (d / 2), r);
            //fragColor = vec4(f_color.rgb, f_color.a * a);
            
            vec4 col;
            vec4 light = vec4(max(dot(f_light_direction, -f_normal), 0) * f_light_intensity, 1);
            vec4 shadow = vec4(max(dot(f_shadow_direction, -f_normal), 0) * f_shadow_intensity, 1);
            //col = (f_color + light - shadow) * a;
            col = (f_color + light - shadow) * 1.0;
            //fragColor = col;
            fragColor = vec4(col.rgb, f_color.a * a);
            
        }
    '''
    
    vertex_shader_minimal = '''
        in vec3 position;
        in vec4 color;
        uniform mat4 perspective_matrix;
        uniform mat4 object_matrix;
        uniform float point_size;
        uniform float global_alpha;
        out vec3 f_color;
        out float f_alpha;
        void main()
        {
            gl_Position = perspective_matrix * object_matrix * vec4(position, 1.0f);
            gl_PointSize = point_size;
            f_color = color.rgb;
            f_alpha = global_alpha;
        }
    '''
    fragment_shader_minimal = '''
        in vec3 f_color;
        in float f_alpha;
        out vec4 fragColor;
        void main()
        {
            fragColor = vec4(f_color, f_alpha);
        }
    '''
    
    vertex_shader_minimal_variable_size = '''
        in vec3 position;
        in vec4 color;
        // in float size;
        in int size;
        uniform mat4 perspective_matrix;
        uniform mat4 object_matrix;
        uniform float global_alpha;
        out vec3 f_color;
        out float f_alpha;
        void main()
        {
            gl_Position = perspective_matrix * object_matrix * vec4(position, 1.0f);
            gl_PointSize = size;
            f_color = color.rgb;
            f_alpha = global_alpha;
        }
    '''
    fragment_shader_minimal_variable_size = '''
        in vec3 f_color;
        in float f_alpha;
        out vec4 fragColor;
        void main()
        {
            fragColor = vec4(f_color, f_alpha);
        }
    '''
    
    vertex_shader_minimal_variable_size_and_depth = '''
        in vec3 position;
        in vec4 color;
        in int size;
        uniform mat4 perspective_matrix;
        uniform mat4 object_matrix;
        uniform float global_alpha;
        
        uniform vec3 center;
        uniform float maxdist;
        
        out vec3 f_color;
        out float f_alpha;
        
        out float f_depth;
        
        void main()
        {
            gl_Position = perspective_matrix * object_matrix * vec4(position, 1.0f);
            gl_PointSize = size;
            f_color = color.rgb;
            f_alpha = global_alpha;
            
            vec4 pp = perspective_matrix * object_matrix * vec4(position, 1.0);
            vec4 op = perspective_matrix * object_matrix * vec4(center, 1.0);
            float d = op.z - pp.z;
            f_depth = ((d - (-maxdist)) / (maxdist - d)) / 2;
        }
    '''
    fragment_shader_minimal_variable_size_and_depth = '''
        in vec3 f_color;
        in float f_alpha;
        
        in float f_depth;
        uniform float brightness;
        uniform float contrast;
        uniform float blend;
        
        out vec4 fragColor;
        void main()
        {
            // fragColor = vec4(f_color, f_alpha);
            
            vec3 depth_color = vec3(f_depth, f_depth, f_depth);
            depth_color = (depth_color - 0.5) * contrast + 0.5 + brightness;
            // fragColor = vec4(depth_color, global_alpha) * a;
            
            depth_color = mix(depth_color, vec3(1.0, 1.0, 1.0), blend);
            
            fragColor = vec4(f_color * depth_color, f_alpha);
            
        }
    '''
    
    billboard_vertex = '''
        layout(location = 0) in vec3 position;
        layout(location = 1) in vec4 color;
        
        uniform mat4 object_matrix;
        uniform float alpha;
        
        out vec4 vcolor;
        out float valpha;
        
        void main()
        {
            gl_Position = object_matrix * vec4(position, 1.0);
            vcolor = color;
            valpha = alpha;
        }
    '''
    billboard_fragment = '''
        layout(location = 0) out vec4 frag_color;
        
        in vec4 fcolor;
        in float falpha;
        
        void main()
        {
            frag_color = vec4(fcolor.rgb, falpha);
        }
    '''
    billboard_geometry = '''
        layout (points) in;
        layout (triangle_strip, max_vertices = 4) out;
        
        in vec4 vcolor[];
        in float valpha[];
        
        uniform mat4 view_matrix;
        uniform mat4 window_matrix;
        
        uniform float size[];
        
        out vec4 fcolor;
        out float falpha;
        
        void main()
        {
            fcolor = vcolor[0];
            falpha = valpha[0];
            // value is diameter, i need radius
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
    
    billboard_geometry_disc = '''
        layout (points) in;
        // 3 * 16 = 48
        layout (triangle_strip, max_vertices = 48) out;
        
        in vec4 vcolor[];
        in float valpha[];
        
        uniform mat4 view_matrix;
        uniform mat4 window_matrix;
        
        uniform float size[];
        
        out vec4 fcolor;
        out float falpha;
        
        vec2 disc_coords(float radius, int step, int steps)
        {
            const float PI = 3.1415926535897932384626433832795;
            float angstep = 2 * PI / steps;
            float x = sin(step * angstep) * radius;
            float y = cos(step * angstep) * radius;
            return vec2(x, y);
        }
        
        void main()
        {
            fcolor = vcolor[0];
            falpha = valpha[0];
            float s = size[0];
            
            vec4 pos = view_matrix * gl_in[0].gl_Position;
            float r = s / 2;
            int steps = 16;
            
            for(int i = 0; i < steps; i++)
            {
                
                gl_Position = window_matrix * (pos);
                EmitVertex();
                
                vec2 xyloc = disc_coords(r, i, steps);
                gl_Position = window_matrix * (pos + vec4(xyloc, 0, 0));
                EmitVertex();
                
                xyloc = disc_coords(r, i + 1, steps);
                gl_Position = window_matrix * (pos + vec4(xyloc, 0, 0));
                EmitVertex();
                
                EndPrimitive();
            }
            
        }
    '''
    
    billboard_vertex_with_depth_and_size = '''
        layout(location = 0) in vec3 position;
        layout(location = 1) in vec4 color;
        layout(location = 2) in float sizef;
        
        uniform mat4 object_matrix;
        uniform mat4 perspective_matrix;
        
        uniform float alpha;
        uniform vec3 center;
        uniform float maxdist;
        
        out vec4 vcolor;
        out float valpha;
        out float vsizef;
        out float vdepth;
        
        void main()
        {
            gl_Position = object_matrix * vec4(position, 1.0);
            vcolor = color;
            valpha = alpha;
            vsizef = sizef;
            
            vec4 pp = perspective_matrix * object_matrix * vec4(position, 1.0);
            vec4 op = perspective_matrix * object_matrix * vec4(center, 1.0);
            float d = op.z - pp.z;
            vdepth = ((d - (-maxdist)) / (maxdist - d)) / 2;
        }
    '''
    billboard_fragment_with_depth_and_size = '''
        layout(location = 0) out vec4 frag_color;
        
        in vec4 fcolor;
        in float falpha;
        
        in float fdepth;
        uniform float brightness;
        uniform float contrast;
        uniform float blend;
        
        void main()
        {
            vec3 depth_color = vec3(fdepth, fdepth, fdepth);
            depth_color = (depth_color - 0.5) * contrast + 0.5 + brightness;
            depth_color = mix(depth_color, vec3(1.0, 1.0, 1.0), blend);
            frag_color = vec4(fcolor.rgb * depth_color, falpha);
        }
    '''
    billboard_geometry_with_depth_and_size = '''
        layout (points) in;
        layout (triangle_strip, max_vertices = 4) out;
        
        in vec4 vcolor[];
        in float valpha[];
        in float vsizef[];
        in float vdepth[];
        
        uniform mat4 view_matrix;
        uniform mat4 window_matrix;
        
        uniform float size[];
        
        out vec4 fcolor;
        out float falpha;
        out float fdepth;
        
        void main()
        {
            fcolor = vcolor[0];
            falpha = valpha[0];
            fdepth = vdepth[0];
            
            // value is diameter, i need radius, then multiply by individual point size
            float s = (size[0] / 2) * vsizef[0];
            
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
    
    phong_vs = '''
        layout (location = 0) in vec3 position;
        layout (location = 1) in vec3 normal;
        layout (location = 2) in vec4 color;
        
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        uniform float point_size;
        uniform float alpha_radius;
        
        out vec3 f_position;
        out vec3 f_normal;
        out vec4 f_color;
        out float f_alpha_radius;
        
        void main()
        {
            gl_Position = projection * view * model * vec4(position, 1.0);
            gl_PointSize = point_size;
            f_position = vec3(model * vec4(position, 1.0));
            f_normal = mat3(transpose(inverse(model))) * normal;
            f_color = color;
            f_alpha_radius = alpha_radius;
        }
    '''
    phong_fs = '''
        in vec3 f_position;
        in vec3 f_normal;
        in vec4 f_color;
        in float f_alpha_radius;
        
        uniform float alpha;
        uniform vec3 light_position;
        uniform vec3 light_color;
        uniform vec3 view_position;
        uniform float ambient_strength;
        uniform float specular_strength;
        uniform float specular_exponent;
        
        out vec4 frag_color;
        
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
            
            float r = 0.0f;
            float a = 1.0f;
            vec2 cxy = 2.0f * gl_PointCoord - 1.0f;
            r = dot(cxy, cxy);
            if(r > f_alpha_radius){
                discard;
            }
            
            frag_color = vec4(col, alpha) * a;
        }
    '''
    
    billboard_vertex_with_no_depth_and_size = '''
        layout(location = 0) in vec3 position;
        layout(location = 1) in vec4 color;
        layout(location = 2) in float sizef;
        
        uniform mat4 object_matrix;
        
        uniform float alpha;
        
        out vec4 vcolor;
        out float valpha;
        out float vsizef;
        
        void main()
        {
            gl_Position = object_matrix * vec4(position, 1.0);
            vcolor = color;
            valpha = alpha;
            vsizef = sizef;
        }
    '''
    billboard_fragment_with_no_depth_and_size = '''
        layout(location = 0) out vec4 frag_color;
        
        in vec4 fcolor;
        in float falpha;
        
        void main()
        {
            frag_color = vec4(fcolor.rgb, falpha);
        }
    '''
    billboard_geometry_with_no_depth_and_size = '''
        layout (points) in;
        layout (triangle_strip, max_vertices = 4) out;
        
        in vec4 vcolor[];
        in float valpha[];
        in float vsizef[];
        
        uniform mat4 view_matrix;
        uniform mat4 window_matrix;
        
        uniform float size[];
        
        out vec4 fcolor;
        out float falpha;
        
        void main()
        {
            fcolor = vcolor[0];
            falpha = valpha[0];
            
            // value is diameter, i need radius, then multiply by individual point size
            float s = (size[0] / 2) * vsizef[0];
            
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
    
    vertex_shader_simple_clip = '''
        in vec3 position;
        in vec4 color;
        uniform mat4 perspective_matrix;
        uniform mat4 object_matrix;
        uniform float point_size;
        uniform float alpha_radius;
        uniform float global_alpha;
        out vec4 f_color;
        out float f_alpha_radius;
        
        uniform vec4 clip_plane0;
        uniform vec4 clip_plane1;
        uniform vec4 clip_plane2;
        uniform vec4 clip_plane3;
        uniform vec4 clip_plane4;
        uniform vec4 clip_plane5;
        
        void main()
        {
            gl_Position = perspective_matrix * object_matrix * vec4(position, 1.0f);
            gl_PointSize = point_size;
            f_color = vec4(color[0], color[1], color[2], global_alpha);
            f_alpha_radius = alpha_radius;
            
            vec4 pos = vec4(position, 1.0f);
            gl_ClipDistance[0] = dot(clip_plane0, pos);
            gl_ClipDistance[1] = dot(clip_plane1, pos);
            gl_ClipDistance[2] = dot(clip_plane2, pos);
            gl_ClipDistance[3] = dot(clip_plane3, pos);
            gl_ClipDistance[4] = dot(clip_plane4, pos);
            gl_ClipDistance[5] = dot(clip_plane5, pos);
        }
    '''
    fragment_shader_simple_clip = '''
        in vec4 f_color;
        in float f_alpha_radius;
        out vec4 fragColor;
        void main()
        {
            float r = 0.0f;
            float a = 1.0f;
            vec2 cxy = 2.0f * gl_PointCoord - 1.0f;
            r = dot(cxy, cxy);
            if(r > f_alpha_radius){
                discard;
            }
            fragColor = f_color * a;
        }
    '''
    
    billboard_phong_vs = '''
        layout (location = 0) in vec3 position;
        layout (location = 1) in vec3 normal;
        layout (location = 2) in vec4 color;
        
        uniform mat4 model;
        out vec3 g_position;
        out vec3 g_normal;
        out vec4 g_color;
        
        void main()
        {
            gl_Position = model * vec4(position, 1.0);
            g_position = vec3(model * vec4(position, 1.0));
            g_normal = mat3(transpose(inverse(model))) * normal;
            g_color = color;
        }
    '''
    billboard_phong_circles_gs = '''
        layout (points) in;
        layout (triangle_strip, max_vertices = 48) out;
        
        in vec3 g_position[];
        in vec3 g_normal[];
        in vec4 g_color[];
        uniform mat4 view_matrix;
        uniform mat4 window_matrix;
        uniform float size[];
        out vec3 f_position;
        out vec3 f_normal;
        out vec4 f_color;
        
        vec2 disc_coords(float radius, int step, int steps)
        {
            const float PI = 3.1415926535897932384626433832795;
            float angstep = 2 * PI / steps;
            float x = sin(step * angstep) * radius;
            float y = cos(step * angstep) * radius;
            return vec2(x, y);
        }
        
        void main()
        {
            f_position = g_position[0];
            f_normal = g_normal[0];
            f_color = g_color[0];
            
            float s = size[0];
            vec4 pos = view_matrix * gl_in[0].gl_Position;
            float r = s / 2;
            // 3 * 16 = max_vertices 48
            int steps = 16;
            for (int i = 0; i < steps; i++)
            {
                gl_Position = window_matrix * (pos);
                EmitVertex();
                
                vec2 xyloc = disc_coords(r, i, steps);
                gl_Position = window_matrix * (pos + vec4(xyloc, 0, 0));
                EmitVertex();
                
                xyloc = disc_coords(r, i + 1, steps);
                gl_Position = window_matrix * (pos + vec4(xyloc, 0, 0));
                EmitVertex();
                
                EndPrimitive();
            }
        }
    '''
    billboard_phong_fast_gs = '''
        layout (points) in;
        layout (triangle_strip, max_vertices = 4) out;
        
        in vec3 g_position[];
        in vec3 g_normal[];
        in vec4 g_color[];
        uniform mat4 view_matrix;
        uniform mat4 window_matrix;
        uniform float size[];
        out vec3 f_position;
        out vec3 f_normal;
        out vec4 f_color;
        
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
    billboard_phong_fs = '''
        layout (location = 0) out vec4 frag_color;
        
        in vec3 f_position;
        in vec3 f_normal;
        in vec4 f_color;
        uniform float alpha;
        uniform vec3 light_position;
        uniform vec3 light_color;
        uniform vec3 view_position;
        uniform float ambient_strength;
        uniform float specular_strength;
        uniform float specular_exponent;
        
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
        }
    '''
    
    vertex_shader_simple_skip_point_vertices = '''
        in vec3 position;
        in vec4 color;
        in int index;
        uniform mat4 perspective_matrix;
        uniform mat4 object_matrix;
        uniform float point_size;
        uniform float alpha_radius;
        uniform float global_alpha;
        uniform float skip_index;
        out vec4 f_color;
        out float f_alpha_radius;
        void main()
        {
            gl_Position = perspective_matrix * object_matrix * vec4(position, 1.0f);
            
            if(skip_index <= index){
                gl_Position = vec4(2.0, 0.0, 0.0, 1.0);
            }
            
            gl_PointSize = point_size;
            f_color = vec4(color[0], color[1], color[2], global_alpha);
            f_alpha_radius = alpha_radius;
        }
    '''
    fragment_shader_simple_skip_point_vertices = '''
        in vec4 f_color;
        in float f_alpha_radius;
        out vec4 fragColor;
        void main()
        {
            float r = 0.0f;
            float a = 1.0f;
            vec2 cxy = 2.0f * gl_PointCoord - 1.0f;
            r = dot(cxy, cxy);
            if(r > f_alpha_radius){
                discard;
            }
            fragColor = f_color * a;
        }
    '''


class PCVManager():
    cache = {}
    handle = None
    initialized = False
    
    '''
    @classmethod
    def points_batch_for_shader(cls, shader, content, ):
        for k, v in content.items():
            vbo_len = len(v)
            break
        vbo_format = shader.format_calc()
        vbo = GPUVertBuf(vbo_format, vbo_len, )
        for k, v in content.items():
            vbo.attr_fill(k, v, )
        batch = GPUBatch(type='POINTS', buf=vbo, )
        return vbo, batch
    '''
    
    @classmethod
    def load_ply_to_cache(cls, operator, context, ):
        pcv = context.object.point_cloud_visualizer
        filepath = pcv.filepath
        
        __t = time.time()
        
        log('load data..')
        _t = time.time()
        
        # FIXME ply loading might not work with all ply files, for example, file spec seems does not forbid having two or more blocks of vertices with different props, currently i load only first block of vertices. maybe construct some messed up ply and test how for example meshlab behaves
        points = []
        try:
            # points = BinPlyPointCloudReader(filepath).points
            points = PlyPointCloudReader(filepath).points
        except Exception as e:
            if(operator is not None):
                operator.report({'ERROR'}, str(e))
            else:
                raise e
        if(len(points) == 0):
            operator.report({'ERROR'}, "No vertices loaded from file at {}".format(filepath))
            return False
        
        _d = datetime.timedelta(seconds=time.time() - _t)
        log("completed in {}.".format(_d))
        
        log('shuffle data..')
        _t = time.time()
        
        preferences = bpy.context.preferences
        addon_prefs = preferences.addons[__name__].preferences
        if(addon_prefs.shuffle_points):
            np.random.shuffle(points)
        
        _d = datetime.timedelta(seconds=time.time() - _t)
        log("completed in {}.".format(_d))
        
        log('process data..')
        _t = time.time()
        
        if(not set(('x', 'y', 'z')).issubset(points.dtype.names)):
            # this is very unlikely..
            operator.report({'ERROR'}, "Loaded data seems to miss vertex locations.")
            return False
        
        # FIXME checking for normals/colors in points is kinda scattered all over.. chceck should be upon loading / setting from external script
        normals = True
        if(not set(('nx', 'ny', 'nz')).issubset(points.dtype.names)):
            normals = False
        pcv.has_normals = normals
        if(not pcv.has_normals):
            pcv.illumination = False
        vcols = True
        if(not set(('red', 'green', 'blue')).issubset(points.dtype.names)):
            vcols = False
        pcv.has_vcols = vcols
        
        vs = np.column_stack((points['x'], points['y'], points['z'], ))
        if(vs.dtype != np.float32):
            vs = vs.astype(np.float32)
        
        if(normals):
            ns = np.column_stack((points['nx'], points['ny'], points['nz'], ))
            if(ns.dtype != np.float32):
                ns = ns.astype(np.float32)
            
        else:
            n = len(points)
            ns = np.column_stack((np.full(n, 0.0, dtype=np.float32, ),
                                  np.full(n, 0.0, dtype=np.float32, ),
                                  np.full(n, 1.0, dtype=np.float32, ), ))
        
        if(vcols):
            preferences = bpy.context.preferences
            addon_prefs = preferences.addons[__name__].preferences
            if(addon_prefs.convert_16bit_colors and points['red'].dtype == 'uint16'):
                r8 = (points['red'] / 256).astype('uint8')
                g8 = (points['green'] / 256).astype('uint8')
                b8 = (points['blue'] / 256).astype('uint8')
                if(addon_prefs.gamma_correct_16bit_colors):
                    cs = np.column_stack(((r8 / 255) ** (1 / 2.2),
                                          (g8 / 255) ** (1 / 2.2),
                                          (b8 / 255) ** (1 / 2.2),
                                          np.ones(len(points), dtype=float, ), ))
                else:
                    cs = np.column_stack((r8 / 255, g8 / 255, b8 / 255, np.ones(len(points), dtype=float, ), ))
                cs = cs.astype(np.float32)
            else:
                # 'uint8'
                cs = np.column_stack((points['red'] / 255, points['green'] / 255, points['blue'] / 255, np.ones(len(points), dtype=float, ), ))
                cs = cs.astype(np.float32)
        else:
            n = len(points)
            preferences = bpy.context.preferences
            addon_prefs = preferences.addons[__name__].preferences
            col = addon_prefs.default_vertex_color[:]
            col = tuple([c ** (1 / 2.2) for c in col]) + (1.0, )
            cs = np.column_stack((np.full(n, col[0], dtype=np.float32, ),
                                  np.full(n, col[1], dtype=np.float32, ),
                                  np.full(n, col[2], dtype=np.float32, ),
                                  np.ones(n, dtype=np.float32, ), ))
        
        u = str(uuid.uuid1())
        o = context.object
        
        pcv.uuid = u
        
        d = PCVManager.new()
        d['filepath'] = filepath
        
        d['points'] = points
        
        d['uuid'] = u
        d['stats'] = len(vs)
        d['vertices'] = vs
        d['colors'] = cs
        d['normals'] = ns
        
        d['length'] = len(vs)
        dp = pcv.display_percent
        l = int((len(vs) / 100) * dp)
        if(dp >= 99):
            l = len(vs)
        d['display_length'] = l
        d['current_display_length'] = l
        
        ienabled = pcv.illumination
        d['illumination'] = ienabled
        if(ienabled):
            shader = GPUShader(PCVShaders.vertex_shader_illumination, PCVShaders.fragment_shader_illumination)
            batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "color": cs[:l], "normal": ns[:l], })
        else:
            shader = GPUShader(PCVShaders.vertex_shader_simple, PCVShaders.fragment_shader_simple)
            batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "color": cs[:l], })
        
        d['shader'] = shader
        d['batch'] = batch
        d['ready'] = True
        d['object'] = o
        d['name'] = o.name
        
        PCVManager.add(d)
        
        _d = datetime.timedelta(seconds=time.time() - _t)
        log("completed in {}.".format(_d))
        
        log("-" * 50)
        __d = datetime.timedelta(seconds=time.time() - __t)
        log("load and process completed in {}.".format(__d))
        log("-" * 50)
        
        # with new file browser in 2.81, screen is not redrawn, so i have to do it manually..
        cls._redraw()
        
        return True
    
    @classmethod
    def render(cls, uuid, ):
        bgl.glEnable(bgl.GL_PROGRAM_POINT_SIZE)
        bgl.glEnable(bgl.GL_DEPTH_TEST)
        bgl.glEnable(bgl.GL_BLEND)
        
        # TODO: replace all 'batch_for_shader' (2.80/scripts/modules/gpu_extras/batch.py) calls with something custom made and keep buffer cached. faster shader switching, less memory used, etc..
        
        ci = PCVManager.cache[uuid]
        
        shader = ci['shader']
        batch = ci['batch']
        
        if(ci['current_display_length'] != ci['display_length']):
            l = ci['display_length']
            ci['current_display_length'] = l
            vs = ci['vertices']
            cs = ci['colors']
            ns = ci['normals']
            if(ci['illumination']):
                batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "color": cs[:l], "normal": ns[:l], })
            else:
                batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "color": cs[:l], })
            ci['batch'] = batch
        
        o = ci['object']
        try:
            pcv = o.point_cloud_visualizer
        except ReferenceError:
            # FIXME undo still doesn't work in some cases, from what i've seen, only when i am undoing operations on parent object, especially when you undo/redo e.g. transforms around load/draw operators, filepath property gets reset and the whole thing is drawn, but ui looks like loding never happened, i've added a quick fix storing path in cache, but it all depends on object name and this is bad.
            # NOTE parent object reference check should be before drawing, not in the middle, it's not that bad, it's pretty early, but it's still messy, this will require rewrite of handler and render functions in manager.. so don't touch until broken
            log("PCVManager.render: ReferenceError (possibly after undo/redo?)")
            # blender on undo/redo swaps whole scene to different one stored in memory and therefore stored object references are no longer valid
            # so find object with the same name, not the best solution, but lets see how it goes..
            o = bpy.data.objects[ci['name']]
            # update stored reference
            ci['object'] = o
            pcv = o.point_cloud_visualizer
            # push back correct uuid, since undo changed it, why? WHY? why do i even bother?
            pcv.uuid = uuid
            # push back filepath, it might get lost during undo/redo
            pcv.filepath = ci['filepath']
        
        if(not o.visible_get()):
            # if parent object is not visible, skip drawing
            # this should checked earlier, but until now i can't be sure i have correct object reference
            
            # NOTE: use bpy.context.view_layer.objects.active instead of context.active_object and add option to not hide cloud when parent object is hidden? seems like this is set when object is clicked in outliner even when hidden, at least properties buttons are changed.. if i unhide and delete the object, props buttons are not drawn, if i click on another already hidden object, correct buttons are back, so i need to check if there is something active.. also this would require rewriting all panels polls, now they check for context.active_object and if None, which is when object is hidden, panel is not drawn..
            
            bgl.glDisable(bgl.GL_PROGRAM_POINT_SIZE)
            bgl.glDisable(bgl.GL_DEPTH_TEST)
            bgl.glDisable(bgl.GL_BLEND)
            return
        
        if(ci['illumination'] != pcv.illumination):
            vs = ci['vertices']
            cs = ci['colors']
            ns = ci['normals']
            l = ci['current_display_length']
            if(pcv.illumination):
                shader = GPUShader(PCVShaders.vertex_shader_illumination, PCVShaders.fragment_shader_illumination)
                batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "color": cs[:l], "normal": ns[:l], })
                ci['illumination'] = True
            else:
                shader = GPUShader(PCVShaders.vertex_shader_simple, PCVShaders.fragment_shader_simple)
                batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "color": cs[:l], })
                ci['illumination'] = False
            ci['shader'] = shader
            ci['batch'] = batch
        
        shader.bind()
        pm = bpy.context.region_data.perspective_matrix
        shader.uniform_float("perspective_matrix", pm)
        shader.uniform_float("object_matrix", o.matrix_world)
        shader.uniform_float("point_size", pcv.point_size)
        shader.uniform_float("alpha_radius", pcv.alpha_radius)
        shader.uniform_float("global_alpha", pcv.global_alpha)
        
        if(pcv.illumination and pcv.has_normals and ci['illumination']):
            cm = Matrix(((-1.0, 0.0, 0.0, 0.0, ), (0.0, -0.0, 1.0, 0.0, ), (0.0, -1.0, -0.0, 0.0, ), (0.0, 0.0, 0.0, 1.0, ), ))
            _, obrot, _ = o.matrix_world.decompose()
            mr = obrot.to_matrix().to_4x4()
            mr.invert()
            direction = cm @ pcv.light_direction
            direction = mr @ direction
            shader.uniform_float("light_direction", direction)
            
            # def get_space3dview():
            #     for a in bpy.context.screen.areas:
            #         if(a.type == "VIEW_3D"):
            #             return a.spaces[0]
            #     return None
            #
            # s3dv = get_space3dview()
            # region3d = s3dv.region_3d
            # eye = region3d.view_matrix[2][:3]
            #
            # # shader.uniform_float("light_direction", Vector(eye) * -1)
            # shader.uniform_float("light_direction", Vector(eye))
            
            inverted_direction = direction.copy()
            inverted_direction.negate()
            
            c = pcv.light_intensity
            shader.uniform_float("light_intensity", (c, c, c, ))
            shader.uniform_float("shadow_direction", inverted_direction)
            c = pcv.shadow_intensity
            shader.uniform_float("shadow_intensity", (c, c, c, ))
            # shader.uniform_float("show_normals", float(pcv.show_normals))
            # shader.uniform_float("show_illumination", float(pcv.illumination))
        else:
            pass
        
        if(not pcv.override_default_shader):
            # NOTE: just don't draw default shader, quick and easy solution, other shader will be drawn instead, would better to not create it..
            batch.draw(shader)
            
            # # remove extra if present, will be recreated if needed and if left stored it might cause problems
            # if('extra' in ci.keys()):
            #     del ci['extra']
        
        if(pcv.vertex_normals and pcv.has_normals):
            def make(ci):
                l = ci['current_display_length']
                vs = ci['vertices'][:l]
                ns = ci['normals'][:l]
                
                shader = GPUShader(PCVShaders.normals_vertex_shader, PCVShaders.normals_fragment_shader, geocode=PCVShaders.normals_geometry_shader, )
                batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "normal": ns[:l], }, )
                
                d = {'shader': shader,
                     'batch': batch,
                     'position': vs,
                     'normal': ns,
                     'current_display_length': l, }
                ci['vertex_normals'] = d
                
                return shader, batch
            
            if("vertex_normals" not in ci.keys()):
                shader, batch = make(ci)
            else:
                d = ci['vertex_normals']
                shader = d['shader']
                batch = d['batch']
                ok = True
                if(ci['current_display_length'] != d['current_display_length']):
                    ok = False
                if(not ok):
                    shader, batch = make(ci)
            
            shader.bind()
            pm = bpy.context.region_data.perspective_matrix
            shader.uniform_float("perspective_matrix", pm)
            shader.uniform_float("object_matrix", o.matrix_world)
            col = bpy.context.preferences.addons[__name__].preferences.normal_color[:]
            col = tuple([c ** (1 / 2.2) for c in col]) + (pcv.vertex_normals_alpha, )
            shader.uniform_float("color", col, )
            shader.uniform_float("length", pcv.vertex_normals_size, )
            shader.uniform_float("global_alpha", pcv.global_alpha)
            batch.draw(shader)
        
        if(pcv.dev_depth_enabled):
            
            # if(debug_mode()):
            #     import cProfile
            #     import pstats
            #     import io
            #     pr = cProfile.Profile()
            #     pr.enable()
            
            vs = ci['vertices']
            ns = ci['normals']
            l = ci['current_display_length']
            
            use_stored = False
            if('extra' in ci.keys()):
                t = 'DEPTH'
                for k, v in ci['extra'].items():
                    if(k == t):
                        if(v['length'] == l):
                            if(v['illumination'] == pcv.illumination and v['false_colors'] == pcv.dev_depth_false_colors):
                                use_stored = True
                                batch = v['batch']
                                shader = v['shader']
                                break
            
            if(not use_stored):
                if(pcv.illumination):
                    shader = GPUShader(PCVShaders.depth_vertex_shader_illumination, PCVShaders.depth_fragment_shader_illumination, )
                    batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "normal": ns[:l], })
                elif(pcv.dev_depth_false_colors):
                    shader = GPUShader(PCVShaders.depth_vertex_shader_false_colors, PCVShaders.depth_fragment_shader_false_colors, )
                    batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], })
                else:
                    shader = GPUShader(PCVShaders.depth_vertex_shader_simple, PCVShaders.depth_fragment_shader_simple, )
                    batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], })
                
                if('extra' not in ci.keys()):
                    ci['extra'] = {}
                d = {'shader': shader,
                     'batch': batch,
                     'illumination': pcv.illumination,
                     'false_colors': pcv.dev_depth_false_colors,
                     'length': l, }
                ci['extra']['DEPTH'] = d
            
            shader.bind()
            pm = bpy.context.region_data.perspective_matrix
            shader.uniform_float("perspective_matrix", pm)
            shader.uniform_float("object_matrix", o.matrix_world)
            
            # NOTE: precalculating and storing following should speed up things a bit, but then it won't reflect edits..
            # TODO: calculating center of all points is not quite correct, visually it works, but (as i've seen in bounding box shader) it's not working when distribution of points is uneven, so have a check if it might be a bit better..
            cx = np.sum(vs[:, 0]) / len(vs)
            cy = np.sum(vs[:, 1]) / len(vs)
            cz = np.sum(vs[:, 2]) / len(vs)
            _, _, s = o.matrix_world.decompose()
            l = s.length
            maxd = abs(np.max(vs))
            mind = abs(np.min(vs))
            maxdist = maxd
            if(mind > maxd):
                maxdist = mind
            shader.uniform_float("maxdist", float(maxdist) * l)
            shader.uniform_float("center", (cx, cy, cz, ))
            
            shader.uniform_float("brightness", pcv.dev_depth_brightness)
            shader.uniform_float("contrast", pcv.dev_depth_contrast)
            
            shader.uniform_float("point_size", pcv.point_size)
            shader.uniform_float("alpha_radius", pcv.alpha_radius)
            shader.uniform_float("global_alpha", pcv.global_alpha)
            
            if(pcv.illumination):
                cm = Matrix(((-1.0, 0.0, 0.0, 0.0, ), (0.0, -0.0, 1.0, 0.0, ), (0.0, -1.0, -0.0, 0.0, ), (0.0, 0.0, 0.0, 1.0, ), ))
                _, obrot, _ = o.matrix_world.decompose()
                mr = obrot.to_matrix().to_4x4()
                mr.invert()
                direction = cm @ pcv.light_direction
                direction = mr @ direction
                shader.uniform_float("light_direction", direction)
                inverted_direction = direction.copy()
                inverted_direction.negate()
                c = pcv.light_intensity
                shader.uniform_float("light_intensity", (c, c, c, ))
                shader.uniform_float("shadow_direction", inverted_direction)
                c = pcv.shadow_intensity
                shader.uniform_float("shadow_intensity", (c, c, c, ))
                if(pcv.dev_depth_false_colors):
                    shader.uniform_float("color_a", pcv.dev_depth_color_a)
                    shader.uniform_float("color_b", pcv.dev_depth_color_b)
                else:
                    shader.uniform_float("color_a", (1.0, 1.0, 1.0))
                    shader.uniform_float("color_b", (0.0, 0.0, 0.0))
            else:
                if(pcv.dev_depth_false_colors):
                    shader.uniform_float("color_a", pcv.dev_depth_color_a)
                    shader.uniform_float("color_b", pcv.dev_depth_color_b)
            
            batch.draw(shader)
            
            # if(debug_mode()):
            #     pr.disable()
            #     s = io.StringIO()
            #     sortby = 'cumulative'
            #     ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            #     ps.print_stats()
            #     print(s.getvalue())
        
        if(pcv.dev_normal_colors_enabled):
            
            vs = ci['vertices']
            ns = ci['normals']
            l = ci['current_display_length']
            
            use_stored = False
            if('extra' in ci.keys()):
                t = 'NORMAL'
                for k, v in ci['extra'].items():
                    if(k == t):
                        if(v['length'] == l):
                            use_stored = True
                            batch = v['batch']
                            shader = v['shader']
                            break
            
            if(not use_stored):
                shader = GPUShader(PCVShaders.normal_colors_vertex_shader, PCVShaders.normal_colors_fragment_shader, )
                batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "normal": ns[:l], })
                
                if('extra' not in ci.keys()):
                    ci['extra'] = {}
                d = {'shader': shader,
                     'batch': batch,
                     'length': l, }
                ci['extra']['NORMAL'] = d
            
            # shader = GPUShader(PCVShaders.normal_colors_vertex_shader, PCVShaders.normal_colors_fragment_shader, )
            # batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "normal": ns[:l], })
            shader.bind()
            pm = bpy.context.region_data.perspective_matrix
            shader.uniform_float("perspective_matrix", pm)
            shader.uniform_float("object_matrix", o.matrix_world)
            shader.uniform_float("point_size", pcv.point_size)
            shader.uniform_float("alpha_radius", pcv.alpha_radius)
            shader.uniform_float("global_alpha", pcv.global_alpha)
            batch.draw(shader)
        
        if(pcv.dev_position_colors_enabled):
            
            vs = ci['vertices']
            l = ci['current_display_length']
            
            use_stored = False
            if('extra' in ci.keys()):
                t = 'POSITION'
                for k, v in ci['extra'].items():
                    if(k == t):
                        if(v['length'] == l):
                            use_stored = True
                            batch = v['batch']
                            shader = v['shader']
                            break
            
            if(not use_stored):
                shader = GPUShader(PCVShaders.position_colors_vertex_shader, PCVShaders.position_colors_fragment_shader, )
                batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], })
                
                if('extra' not in ci.keys()):
                    ci['extra'] = {}
                d = {'shader': shader,
                     'batch': batch,
                     'length': l, }
                ci['extra']['POSITION'] = d
            
            # shader = GPUShader(PCVShaders.normal_colors_vertex_shader, PCVShaders.normal_colors_fragment_shader, )
            # batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "normal": ns[:l], })
            shader.bind()
            pm = bpy.context.region_data.perspective_matrix
            shader.uniform_float("perspective_matrix", pm)
            shader.uniform_float("object_matrix", o.matrix_world)
            shader.uniform_float("point_size", pcv.point_size)
            shader.uniform_float("alpha_radius", pcv.alpha_radius)
            shader.uniform_float("global_alpha", pcv.global_alpha)
            batch.draw(shader)
        
        if(pcv.dev_selection_shader_display):
            vs = ci['vertices']
            l = ci['current_display_length']
            shader = GPUShader(PCVShaders.selection_vertex_shader, PCVShaders.selection_fragment_shader, )
            batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], })
            shader.bind()
            pm = bpy.context.region_data.perspective_matrix
            shader.uniform_float("perspective_matrix", pm)
            shader.uniform_float("object_matrix", o.matrix_world)
            shader.uniform_float("color", pcv.dev_selection_shader_color)
            shader.uniform_float("point_size", pcv.point_size)
            shader.uniform_float("alpha_radius", pcv.alpha_radius)
            bgl.glClear(bgl.GL_DEPTH_BUFFER_BIT)
            batch.draw(shader)
        
        if(pcv.color_adjustment_shader_enabled):
            vs = ci['vertices']
            cs = ci['colors']
            l = ci['current_display_length']
            
            use_stored = False
            if('extra' in ci.keys()):
                t = 'COLOR_ADJUSTMENT'
                for k, v in ci['extra'].items():
                    if(k == t):
                        if(v['length'] == l):
                            use_stored = True
                            batch = v['batch']
                            shader = v['shader']
                            break
            
            if(not use_stored):
                shader = GPUShader(PCVShaders.vertex_shader_color_adjustment, PCVShaders.fragment_shader_color_adjustment, )
                batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "color": cs[:l], })
                
                if('extra' not in ci.keys()):
                    ci['extra'] = {}
                d = {'shader': shader,
                     'batch': batch,
                     'length': l, }
                ci['extra']['COLOR_ADJUSTMENT'] = d
            
            shader.bind()
            pm = bpy.context.region_data.perspective_matrix
            shader.uniform_float("perspective_matrix", pm)
            shader.uniform_float("object_matrix", o.matrix_world)
            shader.uniform_float("point_size", pcv.point_size)
            shader.uniform_float("alpha_radius", pcv.alpha_radius)
            shader.uniform_float("global_alpha", pcv.global_alpha)
            
            shader.uniform_float("exposure", pcv.color_adjustment_shader_exposure)
            shader.uniform_float("gamma", pcv.color_adjustment_shader_gamma)
            shader.uniform_float("brightness", pcv.color_adjustment_shader_brightness)
            shader.uniform_float("contrast", pcv.color_adjustment_shader_contrast)
            shader.uniform_float("hue", pcv.color_adjustment_shader_hue)
            shader.uniform_float("saturation", pcv.color_adjustment_shader_saturation)
            shader.uniform_float("value", pcv.color_adjustment_shader_value)
            shader.uniform_float("invert", pcv.color_adjustment_shader_invert)
            
            batch.draw(shader)
        
        # dev
        if(pcv.dev_bbox_enabled):
            vs = ci['vertices']
            
            use_stored = False
            if('extra' in ci.keys()):
                t = 'BOUNDING_BOX'
                for k, v in ci['extra'].items():
                    if(k == t):
                        use_stored = True
                        batch = v['batch']
                        shader = v['shader']
                        break
            
            if(not use_stored):
                shader = GPUShader(PCVShaders.bbox_vertex_shader, PCVShaders.bbox_fragment_shader, geocode=PCVShaders.bbox_geometry_shader, )
                batch = batch_for_shader(shader, 'POINTS', {"position": [(0.0, 0.0, 0.0, )], }, )
                
                if('extra' not in ci.keys()):
                    ci['extra'] = {}
                d = {'shader': shader,
                     'batch': batch, }
                ci['extra']['BOUNDING_BOX'] = d
            
            shader.bind()
            pm = bpy.context.region_data.perspective_matrix
            shader.uniform_float("perspective_matrix", pm)
            shader.uniform_float("object_matrix", o.matrix_world)
            # col = bpy.context.preferences.addons[__name__].preferences.normal_color[:]
            # col = tuple([c ** (1 / 2.2) for c in col]) + (pcv.vertex_normals_alpha, )
            
            # col = pcv.dev_bbox_color
            # col = tuple([c ** (1 / 2.2) for c in col]) + (1.0, )
            col = tuple(pcv.dev_bbox_color) + (pcv.dev_bbox_alpha, )
            
            shader.uniform_float("color", col, )
            # shader.uniform_float("length", pcv.vertex_normals_size, )
            shader.uniform_float("global_alpha", pcv.global_alpha)
            
            # # cx = np.sum(vs[:, 0]) / len(vs)
            # # cy = np.sum(vs[:, 1]) / len(vs)
            # # cz = np.sum(vs[:, 2]) / len(vs)
            # cx = np.median(vs[:, 0])
            # cy = np.median(vs[:, 1])
            # cz = np.median(vs[:, 2])
            # center = [cx, cy, cz]
            # # center = [0.0, 0.0, 0.0]
            # # print(center)
            # shader.uniform_float("center", center)
            
            # TODO: store values somewhere, might be slow if calculated every frame
            
            minx = np.min(vs[:, 0])
            miny = np.min(vs[:, 1])
            minz = np.min(vs[:, 2])
            maxx = np.max(vs[:, 0])
            maxy = np.max(vs[:, 1])
            maxz = np.max(vs[:, 2])
            
            def calc(mini, maxi):
                if(mini <= 0.0 and maxi <= 0.0):
                    return abs(mini) - abs(maxi)
                elif(mini <= 0.0 and maxi >= 0.0):
                    return abs(mini) + maxi
                else:
                    return maxi - mini
            
            dimensions = [calc(minx, maxx), calc(miny, maxy), calc(minz, maxz)]
            shader.uniform_float("dimensions", dimensions)
            
            center = [(minx + maxx) / 2, (miny + maxy) / 2, (minz + maxz) / 2]
            shader.uniform_float("center", center)
            
            mindim = abs(min(dimensions)) / 2 * pcv.dev_bbox_size
            shader.uniform_float("length", mindim)
            
            batch.draw(shader)
        
        # dev
        if(pcv.dev_minimal_shader_enabled):
            vs = ci['vertices']
            cs = ci['colors']
            l = ci['current_display_length']
            
            use_stored = False
            if('extra' in ci.keys()):
                t = 'MINIMAL'
                for k, v in ci['extra'].items():
                    if(k == t):
                        if(v['length'] == l):
                            use_stored = True
                            batch = v['batch']
                            shader = v['shader']
                            break
            
            if(not use_stored):
                shader = GPUShader(PCVShaders.vertex_shader_minimal, PCVShaders.fragment_shader_minimal, )
                batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "color": cs[:l], })
                
                if('extra' not in ci.keys()):
                    ci['extra'] = {}
                d = {'shader': shader,
                     'batch': batch,
                     'length': l, }
                ci['extra']['MINIMAL'] = d
            
            shader.bind()
            pm = bpy.context.region_data.perspective_matrix
            shader.uniform_float("perspective_matrix", pm)
            shader.uniform_float("object_matrix", o.matrix_world)
            shader.uniform_float("point_size", pcv.point_size)
            shader.uniform_float("global_alpha", pcv.global_alpha)
            batch.draw(shader)
        
        # dev
        if(pcv.dev_minimal_shader_variable_size_enabled):
            vs = ci['vertices']
            cs = ci['colors']
            l = ci['current_display_length']
            
            use_stored = False
            if('extra' in ci.keys()):
                t = 'MINIMAL_VARIABLE_SIZE'
                for k, v in ci['extra'].items():
                    if(k == t):
                        if(v['length'] == l):
                            use_stored = True
                            batch = v['batch']
                            shader = v['shader']
                            sizes = v['sizes']
                            break
            
            if(not use_stored):
                # # generate something to test it, later implement how to set it
                # sizes = np.random.randint(low=1, high=10, size=len(vs), )
                
                if('extra' in ci.keys()):
                    if('MINIMAL_VARIABLE_SIZE' in ci['extra'].keys()):
                        sizes = ci['extra']['MINIMAL_VARIABLE_SIZE']['sizes']
                    else:
                        sizes = np.random.randint(low=1, high=10, size=len(vs), )
                else:
                    sizes = np.random.randint(low=1, high=10, size=len(vs), )
                
                if('extra' in ci.keys()):
                    for k, v in ci['extra'].items():
                        if(k in ('MINIMAL_VARIABLE_SIZE', 'MINIMAL_VARIABLE_SIZE_AND_DEPTH', 'RICH_BILLBOARD', 'RICH_BILLBOARD_NO_DEPTH', )):
                            # FIXME: it was recently switched, try to recover already generated data, both arrays are the same, so thay are in memory just once, no problem here, the problem is with storing reference to it, this should be fixed with something, for example, shader and batch should be generated on one spot, etc.. but have to be done together with all new PCVManager.render and unified shader management. this is just mediocre workaround..
                            sizes = ci['extra'][k]['sizes']
                            sizesf = ci['extra'][k]['sizesf']
                            break
                
                shader = GPUShader(PCVShaders.vertex_shader_minimal_variable_size, PCVShaders.fragment_shader_minimal_variable_size, )
                batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "color": cs[:l], "size": sizes[:l], })
                # batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "color": cs[:l], })
                
                if('extra' not in ci.keys()):
                    ci['extra'] = {}
                
                d = {'shader': shader,
                     'batch': batch,
                     'sizes': sizes,
                     'length': l, }
                ci['extra']['MINIMAL_VARIABLE_SIZE'] = d
            
            shader.bind()
            pm = bpy.context.region_data.perspective_matrix
            shader.uniform_float("perspective_matrix", pm)
            shader.uniform_float("object_matrix", o.matrix_world)
            # shader.uniform_float("point_size", pcv.point_size)
            shader.uniform_float("global_alpha", pcv.global_alpha)
            batch.draw(shader)
        
        # dev
        if(pcv.dev_minimal_shader_variable_size_and_depth_enabled):
            vs = ci['vertices']
            cs = ci['colors']
            l = ci['current_display_length']
            
            use_stored = False
            if('extra' in ci.keys()):
                t = 'MINIMAL_VARIABLE_SIZE_AND_DEPTH'
                for k, v in ci['extra'].items():
                    if(k == t):
                        if(v['length'] == l):
                            use_stored = True
                            batch = v['batch']
                            shader = v['shader']
                            sizes = v['sizes']
                            break
            
            if(not use_stored):
                # # generate something to test it, later implement how to set it
                # sizes = np.random.randint(low=1, high=10, size=len(vs), )
                
                if('extra' in ci.keys()):
                    if('MINIMAL_VARIABLE_SIZE_AND_DEPTH' in ci['extra'].keys()):
                        sizes = ci['extra']['MINIMAL_VARIABLE_SIZE_AND_DEPTH']['sizes']
                    else:
                        sizes = np.random.randint(low=1, high=10, size=len(vs), )
                else:
                    sizes = np.random.randint(low=1, high=10, size=len(vs), )
                
                if('extra' in ci.keys()):
                    for k, v in ci['extra'].items():
                        if(k in ('MINIMAL_VARIABLE_SIZE', 'MINIMAL_VARIABLE_SIZE_AND_DEPTH', 'RICH_BILLBOARD', 'RICH_BILLBOARD_NO_DEPTH', )):
                            # FIXME: it was recently switched, try to recover already generated data, both arrays are the same, so thay are in memory just once, no problem here, the problem is with storing reference to it, this should be fixed with something, for example, shader and batch should be generated on one spot, etc.. but have to be done together with all new PCVManager.render and unified shader management. this is just mediocre workaround..
                            sizes = ci['extra'][k]['sizes']
                            sizesf = ci['extra'][k]['sizesf']
                            break
                
                shader = GPUShader(PCVShaders.vertex_shader_minimal_variable_size_and_depth, PCVShaders.fragment_shader_minimal_variable_size_and_depth, )
                batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "color": cs[:l], "size": sizes[:l], })
                
                if('extra' not in ci.keys()):
                    ci['extra'] = {}
                
                d = {'shader': shader,
                     'batch': batch,
                     'sizes': sizes,
                     'length': l, }
                ci['extra']['MINIMAL_VARIABLE_SIZE_AND_DEPTH'] = d
            
            shader.bind()
            pm = bpy.context.region_data.perspective_matrix
            shader.uniform_float("perspective_matrix", pm)
            shader.uniform_float("object_matrix", o.matrix_world)
            shader.uniform_float("global_alpha", pcv.global_alpha)
            
            if(len(vs) == 0):
                maxdist = 1.0
                cx = 0.0
                cy = 0.0
                cz = 0.0
            else:
                # NOTE: precalculating and storing following should speed up things a bit, but then it won't reflect edits..
                # TODO: calculating center of all points is not quite correct, visually it works, but (as i've seen in bounding box shader) it's not working when distribution of points is uneven, so have a check if it might be a bit better..
                cx = np.sum(vs[:, 0]) / len(vs)
                cy = np.sum(vs[:, 1]) / len(vs)
                cz = np.sum(vs[:, 2]) / len(vs)
                _, _, s = o.matrix_world.decompose()
                l = s.length
                # FIXME: here is error in max with zero length arrays, why are they zero length anyway, putting this single fix for now
                maxd = abs(np.max(vs))
                mind = abs(np.min(vs))
                maxdist = maxd
                if(mind > maxd):
                    maxdist = mind
            shader.uniform_float("maxdist", float(maxdist) * l)
            shader.uniform_float("center", (cx, cy, cz, ))
            shader.uniform_float("brightness", pcv.dev_minimal_shader_variable_size_and_depth_brightness)
            shader.uniform_float("contrast", pcv.dev_minimal_shader_variable_size_and_depth_contrast)
            shader.uniform_float("blend", 1.0 - pcv.dev_minimal_shader_variable_size_and_depth_blend)
            
            batch.draw(shader)
        
        # dev
        if(pcv.dev_billboard_point_cloud_enabled):
            vs = ci['vertices']
            cs = ci['colors']
            l = ci['current_display_length']
            
            use_stored = False
            if('extra' in ci.keys()):
                t = 'BILLBOARD'
                for k, v in ci['extra'].items():
                    if(k == t):
                        if(v['length'] == l):
                            use_stored = True
                            batch = v['batch']
                            shader = v['shader']
                            break
            
            if(not use_stored):
                shader = GPUShader(PCVShaders.billboard_vertex, PCVShaders.billboard_fragment, geocode=PCVShaders.billboard_geometry, )
                # shader = GPUShader(PCVShaders.billboard_vertex, PCVShaders.billboard_fragment, geocode=PCVShaders.billboard_geometry_disc, )
                batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "color": cs[:l], })
                
                if('extra' not in ci.keys()):
                    ci['extra'] = {}
                d = {'shader': shader,
                     'batch': batch,
                     'length': l, }
                ci['extra']['BILLBOARD'] = d
            
            shader.bind()
            shader.uniform_float("view_matrix", bpy.context.region_data.view_matrix)
            shader.uniform_float("window_matrix", bpy.context.region_data.window_matrix)
            shader.uniform_float("object_matrix", o.matrix_world)
            shader.uniform_float("size", pcv.dev_billboard_point_cloud_size)
            shader.uniform_float("alpha", pcv.global_alpha)
            batch.draw(shader)
        
        # dev
        if(pcv.dev_rich_billboard_point_cloud_enabled):
            vs = ci['vertices']
            cs = ci['colors']
            l = ci['current_display_length']
            
            use_stored = False
            if('extra' in ci.keys()):
                t = 'RICH_BILLBOARD'
                for k, v in ci['extra'].items():
                    if(k == t):
                        if(v['length'] == l):
                            use_stored = True
                            batch = v['batch']
                            shader = v['shader']
                            sizesf = v['sizesf']
                            break
            
            if(not use_stored):
                
                if('extra' in ci.keys()):
                    if('RICH_BILLBOARD' in ci['extra'].keys()):
                        sizesf = ci['extra']['RICH_BILLBOARD']['sizesf']
                    else:
                        sizesf = np.random.uniform(low=0.5, high=1.5, size=len(vs), )
                        sizesf = sizesf.astype(np.float32)
                else:
                    sizesf = np.random.uniform(low=0.5, high=1.5, size=len(vs), )
                    sizesf = sizesf.astype(np.float32)
                
                if('extra' in ci.keys()):
                    for k, v in ci['extra'].items():
                        if(k in ('MINIMAL_VARIABLE_SIZE', 'MINIMAL_VARIABLE_SIZE_AND_DEPTH', 'RICH_BILLBOARD', 'RICH_BILLBOARD_NO_DEPTH', )):
                            # FIXME: it was recently switched, try to recover already generated data, both arrays are the same, so thay are in memory just once, no problem here, the problem is with storing reference to it, this should be fixed with something, for example, shader and batch should be generated on one spot, etc.. but have to be done together with all new PCVManager.render and unified shader management. this is just mediocre workaround..
                            sizes = ci['extra'][k]['sizes']
                            sizesf = ci['extra'][k]['sizesf']
                            break
                
                shader = GPUShader(PCVShaders.billboard_vertex_with_depth_and_size, PCVShaders.billboard_fragment_with_depth_and_size, geocode=PCVShaders.billboard_geometry_with_depth_and_size, )
                batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "color": cs[:l], "sizef": sizesf[:l], })
                
                if('extra' not in ci.keys()):
                    ci['extra'] = {}
                d = {'shader': shader,
                     'batch': batch,
                     'sizesf': sizesf,
                     'length': l, }
                ci['extra']['RICH_BILLBOARD'] = d
            
            shader.bind()
            shader.uniform_float("perspective_matrix", bpy.context.region_data.perspective_matrix)
            shader.uniform_float("view_matrix", bpy.context.region_data.view_matrix)
            shader.uniform_float("window_matrix", bpy.context.region_data.window_matrix)
            shader.uniform_float("object_matrix", o.matrix_world)
            shader.uniform_float("size", pcv.dev_rich_billboard_point_cloud_size)
            shader.uniform_float("alpha", pcv.global_alpha)
            
            if(len(vs) == 0):
                maxdist = 1.0
                cx = 0.0
                cy = 0.0
                cz = 0.0
            else:
                # NOTE: precalculating and storing following should speed up things a bit, but then it won't reflect edits..
                # TODO: calculating center of all points is not quite correct, visually it works, but (as i've seen in bounding box shader) it's not working when distribution of points is uneven, so have a check if it might be a bit better..
                cx = np.sum(vs[:, 0]) / len(vs)
                cy = np.sum(vs[:, 1]) / len(vs)
                cz = np.sum(vs[:, 2]) / len(vs)
                _, _, s = o.matrix_world.decompose()
                l = s.length
                # FIXME: here is error in max with zero length arrays, why are they zero length anyway, putting this single fix for now
                maxd = abs(np.max(vs))
                mind = abs(np.min(vs))
                maxdist = maxd
                if(mind > maxd):
                    maxdist = mind
            shader.uniform_float("maxdist", float(maxdist) * l)
            shader.uniform_float("center", (cx, cy, cz, ), )
            
            shader.uniform_float("brightness", pcv.dev_rich_billboard_depth_brightness)
            shader.uniform_float("contrast", pcv.dev_rich_billboard_depth_contrast)
            shader.uniform_float("blend", 1.0 - pcv.dev_rich_billboard_depth_blend)
            
            batch.draw(shader)
        
        # dev
        if(pcv.dev_rich_billboard_point_cloud_no_depth_enabled):
            vs = ci['vertices']
            cs = ci['colors']
            l = ci['current_display_length']
            
            use_stored = False
            if('extra' in ci.keys()):
                t = 'RICH_BILLBOARD_NO_DEPTH'
                for k, v in ci['extra'].items():
                    if(k == t):
                        if(v['length'] == l):
                            use_stored = True
                            batch = v['batch']
                            shader = v['shader']
                            sizesf = v['sizesf']
                            break
            
            if(not use_stored):
                
                if('extra' in ci.keys()):
                    if('RICH_BILLBOARD_NO_DEPTH' in ci['extra'].keys()):
                        sizesf = ci['extra']['RICH_BILLBOARD_NO_DEPTH']['sizesf']
                    else:
                        sizesf = np.random.uniform(low=0.5, high=1.5, size=len(vs), )
                        sizesf = sizesf.astype(np.float32)
                else:
                    sizesf = np.random.uniform(low=0.5, high=1.5, size=len(vs), )
                    sizesf = sizesf.astype(np.float32)
                
                if('extra' in ci.keys()):
                    for k, v in ci['extra'].items():
                        if(k in ('MINIMAL_VARIABLE_SIZE', 'MINIMAL_VARIABLE_SIZE_AND_DEPTH', 'RICH_BILLBOARD', 'RICH_BILLBOARD_NO_DEPTH', )):
                            # FIXME: it was recently switched, try to recover already generated data, both arrays are the same, so thay are in memory just once, no problem here, the problem is with storing reference to it, this should be fixed with something, for example, shader and batch should be generated on one spot, etc.. but have to be done together with all new PCVManager.render and unified shader management. this is just mediocre workaround..
                            sizes = ci['extra'][k]['sizes']
                            sizesf = ci['extra'][k]['sizesf']
                            break
                
                shader = GPUShader(PCVShaders.billboard_vertex_with_no_depth_and_size, PCVShaders.billboard_fragment_with_no_depth_and_size, geocode=PCVShaders.billboard_geometry_with_no_depth_and_size, )
                batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "color": cs[:l], "sizef": sizesf[:l], })
                
                if('extra' not in ci.keys()):
                    ci['extra'] = {}
                d = {'shader': shader,
                     'batch': batch,
                     'sizesf': sizesf,
                     'length': l, }
                ci['extra']['RICH_BILLBOARD_NO_DEPTH'] = d
            
            shader.bind()
            # shader.uniform_float("perspective_matrix", bpy.context.region_data.perspective_matrix)
            shader.uniform_float("view_matrix", bpy.context.region_data.view_matrix)
            shader.uniform_float("window_matrix", bpy.context.region_data.window_matrix)
            shader.uniform_float("object_matrix", o.matrix_world)
            shader.uniform_float("size", pcv.dev_rich_billboard_point_cloud_size)
            shader.uniform_float("alpha", pcv.global_alpha)
            
            batch.draw(shader)
        
        # dev
        if(pcv.dev_phong_shader_enabled):
            vs = ci['vertices']
            ns = ci['normals']
            cs = ci['colors']
            l = ci['current_display_length']
            
            use_stored = False
            if('extra' in ci.keys()):
                t = 'PHONG'
                for k, v in ci['extra'].items():
                    if(k == t):
                        if(v['length'] == l):
                            use_stored = True
                            batch = v['batch']
                            shader = v['shader']
                            break
            
            if(not use_stored):
                shader = GPUShader(PCVShaders.phong_vs, PCVShaders.phong_fs, )
                batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "normal": ns[:l], "color": cs[:l], })
                
                if('extra' not in ci.keys()):
                    ci['extra'] = {}
                d = {'shader': shader,
                     'batch': batch,
                     'length': l, }
                ci['extra']['PHONG'] = d
            
            shader.bind()
            
            # shader.uniform_float("perspective_matrix", bpy.context.region_data.perspective_matrix)
            shader.uniform_float("view", bpy.context.region_data.view_matrix)
            shader.uniform_float("projection", bpy.context.region_data.window_matrix)
            shader.uniform_float("model", o.matrix_world)
            
            shader.uniform_float("light_position", bpy.context.region_data.view_matrix.inverted().translation)
            # shader.uniform_float("light_color", (1.0, 1.0, 1.0))
            shader.uniform_float("light_color", (0.8, 0.8, 0.8, ))
            shader.uniform_float("view_position", bpy.context.region_data.view_matrix.inverted().translation)
            
            shader.uniform_float("ambient_strength", pcv.dev_phong_shader_ambient_strength)
            shader.uniform_float("specular_strength", pcv.dev_phong_shader_specular_strength)
            shader.uniform_float("specular_exponent", pcv.dev_phong_shader_specular_exponent)
            
            shader.uniform_float("alpha", pcv.global_alpha)
            shader.uniform_float("point_size", pcv.point_size)
            shader.uniform_float("alpha_radius", pcv.alpha_radius)
            
            # pm = bpy.context.region_data.perspective_matrix
            # shader.uniform_float("perspective_matrix", pm)
            # shader.uniform_float("object_matrix", o.matrix_world)
            # shader.uniform_float("point_size", pcv.point_size)
            # shader.uniform_float("global_alpha", pcv.global_alpha)
            batch.draw(shader)
        
        # dev
        if(pcv.clip_shader_enabled):
            vs = ci['vertices']
            cs = ci['colors']
            l = ci['current_display_length']
            
            use_stored = False
            if('extra' in ci.keys()):
                t = 'CLIP'
                for k, v in ci['extra'].items():
                    if(k == t):
                        if(v['length'] == l):
                            use_stored = True
                            batch = v['batch']
                            shader = v['shader']
                            break
            
            if(not use_stored):
                shader = GPUShader(PCVShaders.vertex_shader_simple_clip, PCVShaders.fragment_shader_simple_clip, )
                batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "color": cs[:l], })
                
                if('extra' not in ci.keys()):
                    ci['extra'] = {}
                d = {'shader': shader,
                     'batch': batch,
                     'length': l, }
                ci['extra']['CLIP'] = d
            
            if(pcv.clip_plane0_enabled):
                bgl.glEnable(bgl.GL_CLIP_DISTANCE0)
            if(pcv.clip_plane1_enabled):
                bgl.glEnable(bgl.GL_CLIP_DISTANCE1)
            if(pcv.clip_plane2_enabled):
                bgl.glEnable(bgl.GL_CLIP_DISTANCE2)
            if(pcv.clip_plane3_enabled):
                bgl.glEnable(bgl.GL_CLIP_DISTANCE3)
            if(pcv.clip_plane4_enabled):
                bgl.glEnable(bgl.GL_CLIP_DISTANCE4)
            if(pcv.clip_plane5_enabled):
                bgl.glEnable(bgl.GL_CLIP_DISTANCE5)
            
            shader.bind()
            
            shader.uniform_float("perspective_matrix", bpy.context.region_data.perspective_matrix)
            shader.uniform_float("object_matrix", o.matrix_world)
            shader.uniform_float("point_size", pcv.point_size)
            shader.uniform_float("alpha_radius", pcv.alpha_radius)
            shader.uniform_float("global_alpha", pcv.global_alpha)
            
            shader.uniform_float("clip_plane0", pcv.clip_plane0)
            shader.uniform_float("clip_plane1", pcv.clip_plane1)
            shader.uniform_float("clip_plane2", pcv.clip_plane2)
            shader.uniform_float("clip_plane3", pcv.clip_plane3)
            shader.uniform_float("clip_plane4", pcv.clip_plane4)
            shader.uniform_float("clip_plane5", pcv.clip_plane5)
            
            batch.draw(shader)
            
            if(pcv.clip_plane0_enabled):
                bgl.glDisable(bgl.GL_CLIP_DISTANCE0)
            if(pcv.clip_plane1_enabled):
                bgl.glDisable(bgl.GL_CLIP_DISTANCE1)
            if(pcv.clip_plane2_enabled):
                bgl.glDisable(bgl.GL_CLIP_DISTANCE2)
            if(pcv.clip_plane3_enabled):
                bgl.glDisable(bgl.GL_CLIP_DISTANCE3)
            if(pcv.clip_plane4_enabled):
                bgl.glDisable(bgl.GL_CLIP_DISTANCE4)
            if(pcv.clip_plane5_enabled):
                bgl.glDisable(bgl.GL_CLIP_DISTANCE5)
        
        # dev
        if(pcv.billboard_phong_enabled):
            vs = ci['vertices']
            ns = ci['normals']
            cs = ci['colors']
            l = ci['current_display_length']
            
            use_stored = False
            if('extra' in ci.keys()):
                t = 'BILLBOARD_PHONG'
                for k, v in ci['extra'].items():
                    if(k == t):
                        if(v['circles'] != pcv.billboard_phong_circles):
                            use_stored = False
                            break
                        if(v['length'] == l):
                            use_stored = True
                            batch = v['batch']
                            shader = v['shader']
                            break
            
            if(not use_stored):
                use_geocode = PCVShaders.billboard_phong_fast_gs
                if(pcv.billboard_phong_circles):
                    use_geocode = PCVShaders.billboard_phong_circles_gs
                shader = GPUShader(PCVShaders.billboard_phong_vs, PCVShaders.billboard_phong_fs, geocode=use_geocode, )
                batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "normal": ns[:l], "color": cs[:l], })
                
                if('extra' not in ci.keys()):
                    ci['extra'] = {}
                d = {'shader': shader,
                     'batch': batch,
                     'circles': pcv.billboard_phong_circles,
                     'length': l, }
                ci['extra']['BILLBOARD_PHONG'] = d
            
            shader.bind()
            
            shader.uniform_float("model", o.matrix_world)
            # shader.uniform_float("view", bpy.context.region_data.view_matrix)
            # shader.uniform_float("projection", bpy.context.region_data.window_matrix)
            
            shader.uniform_float("view_matrix", bpy.context.region_data.view_matrix)
            shader.uniform_float("window_matrix", bpy.context.region_data.window_matrix)
            
            shader.uniform_float("size", pcv.billboard_phong_size)
            
            shader.uniform_float("alpha", pcv.global_alpha)
            
            shader.uniform_float("light_position", bpy.context.region_data.view_matrix.inverted().translation)
            shader.uniform_float("light_color", (0.8, 0.8, 0.8, ))
            shader.uniform_float("view_position", bpy.context.region_data.view_matrix.inverted().translation)
            
            shader.uniform_float("ambient_strength", pcv.billboard_phong_ambient_strength)
            shader.uniform_float("specular_strength", pcv.billboard_phong_specular_strength)
            shader.uniform_float("specular_exponent", pcv.billboard_phong_specular_exponent)
            
            batch.draw(shader)
        
        # dev
        if(pcv.skip_point_shader_enabled):
            vs = ci['vertices']
            cs = ci['colors']
            
            use_stored = False
            if('extra' in ci.keys()):
                t = 'SKIP'
                for k, v in ci['extra'].items():
                    if(k == t):
                        use_stored = True
                        batch = v['batch']
                        shader = v['shader']
                        break
            
            if(not use_stored):
                indices = np.indices((len(vs), ), dtype=np.int, )
                indices.shape = (-1, )
                
                shader = GPUShader(PCVShaders.vertex_shader_simple_skip_point_vertices, PCVShaders.fragment_shader_simple_skip_point_vertices, )
                batch = batch_for_shader(shader, 'POINTS', {"position": vs[:], "color": cs[:], "index": indices[:], })
                
                if('extra' not in ci.keys()):
                    ci['extra'] = {}
                d = {'shader': shader,
                     'batch': batch, }
                ci['extra']['SKIP'] = d
            
            shader.bind()
            
            shader.uniform_float("perspective_matrix", bpy.context.region_data.perspective_matrix)
            shader.uniform_float("object_matrix", o.matrix_world)
            shader.uniform_float("point_size", pcv.point_size)
            shader.uniform_float("alpha_radius", pcv.alpha_radius)
            shader.uniform_float("global_alpha", pcv.global_alpha)
            
            sp = pcv.skip_point_percentage
            l = int((len(vs) / 100) * sp)
            if(sp >= 99):
                l = len(vs)
            shader.uniform_float("skip_index", l)
            
            batch.draw(shader)
        
        # and now back to some production stuff..
        
        # draw selection as a last step bucause i clear depth buffer for it
        if(pcv.filter_remove_color_selection):
            if('selection_indexes' not in ci):
                return
            vs = ci['vertices']
            indexes = ci['selection_indexes']
            try:
                # if it works, leave it..
                vs = np.take(vs, indexes, axis=0, )
            except IndexError:
                # something has changed.. some other edit hapended, selection is invalid, reset it all..
                pcv.filter_remove_color_selection = False
                del ci['selection_indexes']
            
            shader = GPUShader(PCVShaders.selection_vertex_shader, PCVShaders.selection_fragment_shader, )
            batch = batch_for_shader(shader, 'POINTS', {"position": vs[:], })
            shader.bind()
            pm = bpy.context.region_data.perspective_matrix
            shader.uniform_float("perspective_matrix", pm)
            shader.uniform_float("object_matrix", o.matrix_world)
            sc = bpy.context.preferences.addons[__name__].preferences.selection_color[:]
            shader.uniform_float("color", sc)
            shader.uniform_float("point_size", pcv.point_size)
            shader.uniform_float("alpha_radius", pcv.alpha_radius)
            bgl.glClear(bgl.GL_DEPTH_BUFFER_BIT)
            batch.draw(shader)
        
        bgl.glDisable(bgl.GL_PROGRAM_POINT_SIZE)
        bgl.glDisable(bgl.GL_DEPTH_TEST)
        bgl.glDisable(bgl.GL_BLEND)
    
    @classmethod
    def handler(cls):
        bobjects = bpy.data.objects
        
        run_gc = False
        for k, v in cls.cache.items():
            if(not bobjects.get(v['name'])):
                v['kill'] = True
                run_gc = True
            if(v['ready'] and v['draw'] and not v['kill']):
                cls.render(v['uuid'])
        if(run_gc):
            cls.gc()
    
    @classmethod
    def update(cls, uuid, vs, ns=None, cs=None, ):
        if(uuid not in PCVManager.cache):
            raise KeyError("uuid '{}' not in cache".format(uuid))
        # if(len(vs) == 0):
        #     raise ValueError("zero length")
        
        # get cache item
        c = PCVManager.cache[uuid]
        l = len(vs)
        
        if(ns is None):
            ns = np.column_stack((np.full(l, 0.0, dtype=np.float32, ),
                                  np.full(l, 0.0, dtype=np.float32, ),
                                  np.full(l, 1.0, dtype=np.float32, ), ))
        
        if(cs is None):
            col = bpy.context.preferences.addons[__name__].preferences.default_vertex_color[:]
            col = tuple([c ** (1 / 2.2) for c in col]) + (1.0, )
            cs = np.column_stack((np.full(l, col[0], dtype=np.float32, ),
                                  np.full(l, col[1], dtype=np.float32, ),
                                  np.full(l, col[2], dtype=np.float32, ),
                                  np.ones(l, dtype=np.float32, ), ))
        
        # store data
        c['vertices'] = vs
        c['normals'] = ns
        c['colors'] = cs
        c['length'] = l
        c['stats'] = l
        
        o = c['object']
        pcv = o.point_cloud_visualizer
        dp = pcv.display_percent
        nl = int((l / 100) * dp)
        if(dp >= 99):
            nl = l
        c['display_length'] = nl
        c['current_display_length'] = nl
        
        # setup new shaders
        ienabled = c['illumination']
        if(ienabled):
            shader = GPUShader(PCVShaders.vertex_shader_illumination, PCVShaders.fragment_shader_illumination)
            batch = batch_for_shader(shader, 'POINTS', {"position": vs[:nl], "color": cs[:nl], "normal": ns[:nl], })
        else:
            shader = GPUShader(PCVShaders.vertex_shader_simple, PCVShaders.fragment_shader_simple)
            batch = batch_for_shader(shader, 'POINTS', {"position": vs[:nl], "color": cs[:nl], })
        c['shader'] = shader
        c['batch'] = batch
        
        # redraw all viewports
        for area in bpy.context.screen.areas:
            if(area.type == 'VIEW_3D'):
                area.tag_redraw()
    
    @classmethod
    def gc(cls):
        l = []
        for k, v in cls.cache.items():
            if(v['kill']):
                l.append(k)
        for i in l:
            del cls.cache[i]
    
    @classmethod
    def init(cls):
        if(cls.initialized):
            return
        cls.handle = bpy.types.SpaceView3D.draw_handler_add(cls.handler, (), 'WINDOW', 'POST_VIEW')
        bpy.app.handlers.load_pre.append(watcher)
        cls.initialized = True
    
    @classmethod
    def deinit(cls):
        if(not cls.initialized):
            return
        for k, v in cls.cache.items():
            v['kill'] = True
        cls.gc()
        
        bpy.types.SpaceView3D.draw_handler_remove(cls.handle, 'WINDOW')
        cls.handle = None
        bpy.app.handlers.load_pre.remove(watcher)
        cls.initialized = False
    
    @classmethod
    def add(cls, data, ):
        cls.cache[data['uuid']] = data
    
    @classmethod
    def new(cls):
        # NOTE: this is redundant.. is it?
        return {'uuid': None,
                'filepath': None,
                'vertices': None,
                'normals': None,
                'colors': None,
                'display_length': None,
                'current_display_length': None,
                'illumination': False,
                'shader': False,
                'batch': False,
                'ready': False,
                'draw': False,
                'kill': False,
                'stats': None,
                'length': None,
                'name': None,
                'object': None, }
    
    @classmethod
    def _redraw(cls):
        # force redraw
        
        # for area in bpy.context.screen.areas:
        #     if(area.type == 'VIEW_3D'):
        #         area.tag_redraw()
        
        # seems like sometimes context is different, this should work..
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if(area.type == 'VIEW_3D'):
                    area.tag_redraw()


class PCVControl():
    def __init__(self, o, ):
        self.o = o
        PCVManager.init()
    
    def _prepare(self, vs, ns, cs, ):
        if(vs is not None):
            if(len(vs) == 0):
                vs = None
        if(ns is not None):
            if(len(ns) == 0):
                ns = None
        if(cs is not None):
            if(len(cs) == 0):
                cs = None
        
        if(vs is None):
            vs = np.zeros((0, 3), dtype=np.float32,)
        else:
            # make numpy array if not already
            if(type(vs) != np.ndarray):
                vs = np.array(vs)
            # and ensure data type
            vs = vs.astype(np.float32)
        
        n = len(vs)
        
        # process normals if present, otherwise set to default (0.0, 0.0, 1.0)
        if(ns is None):
            has_normals = False
            ns = np.column_stack((np.full(n, 0.0, dtype=np.float32, ),
                                  np.full(n, 0.0, dtype=np.float32, ),
                                  np.full(n, 1.0, dtype=np.float32, ), ))
        else:
            has_normals = True
            if(type(ns) != np.ndarray):
                ns = np.array(ns)
            ns = ns.astype(np.float32)
        
        # process colors if present, otherwise set to default from preferences, append alpha 1.0
        if(cs is None):
            has_colors = False
            col = bpy.context.preferences.addons[__name__].preferences.default_vertex_color[:]
            col = tuple([c ** (1 / 2.2) for c in col]) + (1.0, )
            cs = np.column_stack((np.full(n, col[0], dtype=np.float32, ),
                                  np.full(n, col[1], dtype=np.float32, ),
                                  np.full(n, col[2], dtype=np.float32, ),
                                  np.ones(n, dtype=np.float32, ), ))
        else:
            has_colors = True
            if(type(cs) != np.ndarray):
                cs = np.array(cs)
            cs = np.column_stack((cs[:, 0], cs[:, 1], cs[:, 2], np.ones(n), ))
            cs = cs.astype(np.float32)
        
        # store points to enable some other functions
        cs8 = cs * 255
        cs8 = cs8.astype(np.uint8)
        dt = [('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('nx', '<f4'), ('ny', '<f4'), ('nz', '<f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        points = np.empty(n, dtype=dt, )
        points['x'] = vs[:, 0]
        points['y'] = vs[:, 1]
        points['z'] = vs[:, 2]
        points['nx'] = ns[:, 0]
        points['ny'] = ns[:, 1]
        points['nz'] = ns[:, 2]
        points['red'] = cs8[:, 0]
        points['green'] = cs8[:, 1]
        points['blue'] = cs8[:, 2]
        
        return vs, ns, cs, points, has_normals, has_colors
    
    def _redraw(self):
        # force redraw
        
        # for area in bpy.context.screen.areas:
        #     if(area.type == 'VIEW_3D'):
        #         area.tag_redraw()
        
        # seems like sometimes context is different, this should work..
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if(area.type == 'VIEW_3D'):
                    area.tag_redraw()
    
    def draw(self, vs=None, ns=None, cs=None, ):
        o = self.o
        pcv = o.point_cloud_visualizer
        
        # check if object has been used before, i.e. has uuid and uuid item is in cache
        if(pcv.uuid != "" and pcv.runtime):
            # was used or blend was saved after it was used and uuid is saved from last time, check cache
            if(pcv.uuid in PCVManager.cache):
                # cache item is found, object has been used before
                self._update(vs, ns, cs, )
                return
        # otherwise setup as new
        
        u = str(uuid.uuid1())
        # use that as path, some checks wants this not empty
        filepath = u
        
        # validate/prepare input data
        vs, ns, cs, points, has_normals, has_colors = self._prepare(vs, ns, cs)
        n = len(vs)
        
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
        d['illumination'] = pcv.illumination
        if(pcv.illumination):
            shader = GPUShader(PCVShaders.vertex_shader_illumination, PCVShaders.fragment_shader_illumination)
            batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "color": cs[:l], "normal": ns[:l], })
        else:
            shader = GPUShader(PCVShaders.vertex_shader_simple, PCVShaders.fragment_shader_simple)
            batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "color": cs[:l], })
        d['shader'] = shader
        d['batch'] = batch
        d['ready'] = True
        d['draw'] = False
        d['kill'] = False
        d['object'] = o
        d['name'] = o.name
        
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
    
    def _update(self, vs, ns, cs, ):
        o = self.o
        pcv = o.point_cloud_visualizer
        
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
        d['illumination'] = pcv.illumination
        if(pcv.illumination):
            shader = GPUShader(PCVShaders.vertex_shader_illumination, PCVShaders.fragment_shader_illumination)
            batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "color": cs[:l], "normal": ns[:l], })
        else:
            shader = GPUShader(PCVShaders.vertex_shader_simple, PCVShaders.fragment_shader_simple)
            batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "color": cs[:l], })
        d['shader'] = shader
        d['batch'] = batch
        
        pcv.has_normals = has_normals
        pcv.has_vcols = has_colors
        
        c = PCVManager.cache[pcv.uuid]
        c['draw'] = True
        
        self._redraw()
    
    def erase(self):
        o = self.o
        pcv = o.point_cloud_visualizer
        
        if(pcv.uuid == ""):
            return
        if(not pcv.runtime):
            return
        if(pcv.uuid not in PCVManager.cache.keys()):
            return
        
        # get cache item and set draw to False
        c = PCVManager.cache[pcv.uuid]
        c['draw'] = False
        
        # # force redraw
        # for area in bpy.context.screen.areas:
        #     if(area.type == 'VIEW_3D'):
        #         area.tag_redraw()
        self._redraw()
    
    def reset(self):
        o = self.o
        pcv = o.point_cloud_visualizer
        
        if(pcv.uuid == ""):
            return
        if(not pcv.runtime):
            return
        if(pcv.uuid not in PCVManager.cache.keys()):
            return
        
        # mark for deletion cache
        c = PCVManager.cache[pcv.uuid]
        c['kill'] = True
        PCVManager.gc()
        
        # reset properties
        pcv.uuid = ""
        pcv.filepath = ""
        pcv.has_normals = False
        pcv.has_vcols = False
        pcv.runtime = False
        
        self._redraw()


class PCVSequence():
    cache = {}
    initialized = False
    
    @classmethod
    def handler(cls, scene, depsgraph, ):
        cf = scene.frame_current
        for k, v in cls.cache.items():
            pcv = v['pcv']
            if(pcv.uuid != k):
                del cls.cache[k]
                if(len(cls.cache.items()) == 0):
                    cls.deinit()
                return
            # if(pcv.sequence_enabled):
            #     PCVManager.init()
            #     ld = len(v['data'])
            #     if(pcv.sequence_use_cyclic):
            #         cf = cf % ld
            #     if(cf > ld):
            #         PCVManager.update(k, [], None, None, )
            #     else:
            #         data = v['data'][cf - 1]
            #         PCVManager.update(k, data['vs'], data['ns'], data['cs'], )
            PCVManager.init()
            ld = len(v['data'])
            if(pcv.sequence_use_cyclic):
                cf = cf % ld
            if(cf > ld):
                PCVManager.update(k, [], None, None, )
            else:
                data = v['data'][cf - 1]
                PCVManager.update(k, data['vs'], data['ns'], data['cs'], )
    
    @classmethod
    def init(cls):
        if(cls.initialized):
            return
        bpy.app.handlers.frame_change_post.append(PCVSequence.handler)
        cls.initialized = True
    
    @classmethod
    def deinit(cls):
        if(not cls.initialized):
            return
        bpy.app.handlers.frame_change_post.remove(PCVSequence.handler)
        cls.initialized = False
        cls.cache = {}


class PCVTriangleSurfaceSampler():
    def __init__(self, context, o, num_samples, rnd, colorize=None, constant_color=None, vcols=None, uvtex=None, vgroup=None, exact_number_of_points=False, ):
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
        
        def distance(a, b, ):
            return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2) ** 0.5
        
        def random_point_in_triangle(a, b, c, ):
            r1 = rnd.random()
            r2 = rnd.random()
            p = (1 - math.sqrt(r1)) * a + (math.sqrt(r1) * (1 - r2)) * b + (math.sqrt(r1) * r2) * c
            return p
        
        depsgraph = context.evaluated_depsgraph_get()
        if(o.modifiers):
            owner = o.evaluated_get(depsgraph)
            me = owner.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph, )
        else:
            owner = o
            me = owner.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph, )
        
        bm = bmesh.new()
        bm.from_mesh(me)
        bmesh.ops.triangulate(bm, faces=bm.faces)
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        
        if(len(bm.faces) == 0):
            raise Exception("Mesh has no faces")
        
        areas = tuple([p.calc_area() for p in bm.faces])
        if(sum(areas) == 0.0):
            raise Exception("Mesh surface area is zero")
        area_min = min(areas)
        area_max = max(areas)
        avg_ppf = num_samples / len(areas)
        area_med = statistics.median(areas)
        nums = []
        for p in bm.faces:
            r = p.calc_area() / area_med
            nums.append(avg_ppf * r)
        
        max_ppf = max(nums)
        min_ppf = min(nums)
        
        vs = []
        ns = []
        cs = []
        
        if(colorize == 'UVTEX'):
            try:
                if(o.active_material is None):
                    raise Exception("Cannot find active material")
                uvtexnode = o.active_material.node_tree.nodes.active
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
                group_layer_index = o.vertex_groups.active.index
            except Exception:
                raise Exception("Cannot find active vertex group")
        
        def generate(poly, vs, ns, cs, override_num=None, ):
            ps = poly.verts
            tri = (ps[0].co, ps[1].co, ps[2].co)
            # if num is 0, it can happen when mesh has large and very small polygons, increase number of samples and eventually all polygons gets covered
            num = int(round(remap(poly.calc_area(), area_min, area_max, min_ppf, max_ppf)))
            if(override_num is not None):
                num = override_num
            for i in range(num):
                v = random_point_in_triangle(*tri)
                vs.append(v.to_tuple())
                if(poly.smooth):
                    a = poly.verts[0].normal
                    b = poly.verts[1].normal
                    c = poly.verts[2].normal
                    nws = poly_3d_calc([a, b, c, ], v)
                    
                    nx = a.x * nws[0] + b.x * nws[1] + c.x * nws[2]
                    ny = a.y * nws[0] + b.y * nws[1] + c.y * nws[2]
                    nz = a.z * nws[0] + b.z * nws[1] + c.z * nws[2]
                    normal = Vector((nx, ny, nz)).normalized()
                    ns.append(normal.to_tuple())
                    
                    # n = Vector((0.0, 0.0, 0.0, ))
                    # n += nws[0] * a
                    # n += nws[1] * b
                    # n += nws[2] * c
                    # n = n / 3
                    # n = n.normalized()
                    # ns.append(n.to_tuple())
                    
                else:
                    ns.append(poly.normal.to_tuple())
                
                if(colorize is None):
                    cs.append((1.0, 0.0, 0.0, ))
                elif(colorize == 'CONSTANT'):
                    cs.append(constant_color)
                elif(colorize == 'VCOLS'):
                    ws = poly_3d_calc([poly.verts[0].co, poly.verts[1].co, poly.verts[2].co, ], v)
                    ac = poly.loops[0][col_layer][:3]
                    bc = poly.loops[1][col_layer][:3]
                    cc = poly.loops[2][col_layer][:3]
                    r = ac[0] * ws[0] + bc[0] * ws[1] + cc[0] * ws[2]
                    g = ac[1] * ws[0] + bc[1] * ws[1] + cc[1] * ws[2]
                    b = ac[2] * ws[0] + bc[2] * ws[1] + cc[2] * ws[2]
                    cs.append((r, g, b, ))
                elif(colorize == 'UVTEX'):
                    uvtriangle = []
                    for l in poly.loops:
                        uvtriangle.append(Vector(l[uvlayer].uv.to_tuple() + (0.0, )))
                    uvpoint = barycentric_transform(v, poly.verts[0].co, poly.verts[1].co, poly.verts[2].co, *uvtriangle, )
                    w, h = uvimage.size
                    # x,y % 1.0 to wrap around if uv coordinate is outside 0.0-1.0 range
                    x = int(round(remap(uvpoint.x % 1.0, 0.0, 1.0, 0, w - 1)))
                    y = int(round(remap(uvpoint.y % 1.0, 0.0, 1.0, 0, h - 1)))
                    cs.append(tuple(uvarray[y][x][:3].tolist()))
                elif(colorize == 'GROUP_MONO'):
                    ws = poly_3d_calc([poly.verts[0].co, poly.verts[1].co, poly.verts[2].co, ], v)
                    aw = poly.verts[0][group_layer].get(group_layer_index, 0.0)
                    bw = poly.verts[1][group_layer].get(group_layer_index, 0.0)
                    cw = poly.verts[2][group_layer].get(group_layer_index, 0.0)
                    m = aw * ws[0] + bw * ws[1] + cw * ws[2]
                    cs.append((m, m, m, ))
                elif(colorize == 'GROUP_COLOR'):
                    ws = poly_3d_calc([poly.verts[0].co, poly.verts[1].co, poly.verts[2].co, ], v)
                    aw = poly.verts[0][group_layer].get(group_layer_index, 0.0)
                    bw = poly.verts[1][group_layer].get(group_layer_index, 0.0)
                    cw = poly.verts[2][group_layer].get(group_layer_index, 0.0)
                    m = aw * ws[0] + bw * ws[1] + cw * ws[2]
                    hue = remap(1.0 - m, 0.0, 1.0, 0.0, 1 / 1.5)
                    c = Color()
                    c.hsv = (hue, 1.0, 1.0, )
                    cs.append((c.r, c.g, c.b, ))
        
        log("generating {} samples:".format(num_samples), 1)
        progress = Progress(len(bm.faces), indent=2, )
        for poly in bm.faces:
            progress.step()
            generate(poly, vs, ns, cs, )
        
        if(exact_number_of_points):
            if(len(vs) < num_samples):
                log("generating extra samples..", 1)
                while(len(vs) < num_samples):
                    # generate one sample in random face until full
                    poly = bm.faces[rnd.randrange(len(bm.faces))]
                    generate(poly, vs, ns, cs, override_num=1, )
            if(len(vs) > num_samples):
                log("throwing out extra samples..", 1)
                a = np.concatenate((vs, ns, cs), axis=1, )
                np.random.shuffle(a)
                a = a[:num_samples]
                # vs = np.column_stack((a[:, 0], a[:, 1], a[:, 2], ))
                # ns = np.column_stack((a[:, 3], a[:, 4], a[:, 5], ))
                # cs = np.column_stack((a[:, 6], a[:, 7], a[:, 8], ))
                vs = a[:, :3]
                ns = a[:, 3:6]
                cs = a[:, 6:]
        
        if(len(vs) == 0):
            raise Exception("No points generated, increase number of points or decrease minimal distance")
        
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


class PCVPoissonDiskSurfaceSampler():
    def __init__(self, context, o, rnd, minimal_distance, sampling_exponent=10, colorize=None, constant_color=None, vcols=None, uvtex=None, vgroup=None, ):
        log("{}:".format(self.__class__.__name__), 0)
        # pregenerate samples, normals and colors will be handled too
        num_presamples = int((sum([p.area for p in o.data.polygons]) / (minimal_distance ** 2)) * sampling_exponent)
        presampler = PCVTriangleSurfaceSampler(context, o, num_presamples, rnd, colorize=colorize, constant_color=constant_color, vcols=vcols, uvtex=uvtex, vgroup=vgroup, exact_number_of_points=False, )
        pre_vs = presampler.vs
        pre_ns = presampler.ns
        pre_cs = presampler.cs
        # join to points
        ppoints = []
        for i in range(len(pre_vs)):
            ppoints.append(tuple(pre_vs[i].tolist()) + tuple(pre_ns[i].tolist()) + tuple(pre_cs[i].tolist()))
        # keep track of used/unused points
        pbools = [True] * len(ppoints)
        
        def random_point():
            while(True):
                i = rnd.randint(0, len(ppoints) - 1)
                if(pbools[i]):
                    pbools[i] = False
                    return ppoints[i]
        
        accepted = []
        candidates = []
        # get starting point
        candidates.append(random_point())
        
        # make kdtree of all points for getting annulus points
        ptree = KDTree(len(ppoints))
        for i in range(len(ppoints)):
            ptree.insert(ppoints[i][:3], i)
        ptree.balance()
        
        def annulus_points(c):
            # generate k point with candidate annulus
            pool = []
            # points within radius minimal_distance
            ball = ptree.find_range(c[:3], minimal_distance)
            ball_indexes = set([i for _, i, _ in ball])
            # points within radius minimal_distance * 2
            ball2 = ptree.find_range(c[:3], minimal_distance * 2)
            ball2_indexes = set([i for _, i, _ in ball2])
            # difference between those two are points in annulus
            s = ball2_indexes.difference(ball_indexes)
            pool = []
            for i in s:
                if(pbools[i]):
                    # pool.append(tuple(ptree.data[i]))
                    pool.append(tuple(pre_vs[i].tolist()) + tuple(pre_ns[i].tolist()) + tuple(pre_cs[i].tolist()))
                    # don't reuse points, if point is ok, will be added to candidate, if not will not be used anymore
                    pbools[i] = False
                    # # isn't the 'hole' problem here?
                    # # this might leave a gap on mesh
                    # # because i am no generating new samples and with this i cut out pre generated neighbours?
                    # if(len(pool) >= k):
                    #     break
            return pool
        
        log("sampling..", 1)
        
        loop = True
        while(loop):
            # choose one random candidate
            c = candidates[rnd.randint(0, len(candidates) - 1)]
            # generate k points in candidate annulus
            ap = annulus_points(c)
            na = False
            for i, a in enumerate(ap):
                # this is not quite right to create kdtree each iteration..
                tps = candidates + accepted
                tpsl = len(tps)
                tree = KDTree(tpsl)
                for j in range(tpsl):
                    tree.insert(tps[j][:3], j)
                tree.balance()
                
                ball = tree.find_range(a[:3], minimal_distance)
                ball_indexes = set([i for _, i, _ in ball])
                if(len(ball) == 0):
                    # add if annulus points is acceptable as candidate - no other candidates or accepted are within minimal_distance
                    candidates.append(a)
                    na = True
            if(not na):
                # if no candidate or accepted is within all annulus points minimal_distance accept candidate and remove from candidates
                accepted.append(c)
                candidates.remove(c)
            # no candidates left, we end here
            if(len(candidates) == 0):
                loop = False
            
            if(debug_mode()):
                sys.stdout.write("\r")
                sys.stdout.write("        accepted: {0} | candidates: {1}  ".format(len(accepted), len(candidates)))
                sys.stdout.write("\b")
                sys.stdout.flush()
        
        if(debug_mode()):
            sys.stdout.write("\n")
        log("done..", 1)
        
        # split point data back
        vs = []
        ns = []
        cs = []
        for i in range(len(accepted)):
            vs.append(accepted[i][:3])
            ns.append(accepted[i][3:6])
            cs.append(accepted[i][6:])
        self.vs = vs
        self.ns = ns
        self.cs = cs


class PCVVertexSampler():
    def __init__(self, context, o, colorize=None, constant_color=None, vcols=None, uvtex=None, vgroup=None, ):
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
        
        depsgraph = context.evaluated_depsgraph_get()
        if(o.modifiers):
            owner = o.evaluated_get(depsgraph)
            me = owner.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph, )
        else:
            owner = o
            me = owner.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph, )
        
        bm = bmesh.new()
        bm.from_mesh(me)
        bmesh.ops.triangulate(bm, faces=bm.faces)
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        
        if(len(bm.verts) == 0):
            raise Exception("Mesh has no vertices")
        if(colorize in ('UVTEX', 'VCOLS', )):
            if(len(bm.faces) == 0):
                raise Exception("Mesh has no faces")
        
        vs = []
        ns = []
        cs = []
        
        if(colorize == 'UVTEX'):
            try:
                if(o.active_material is None):
                    raise Exception("Cannot find active material")
                uvtexnode = o.active_material.node_tree.nodes.active
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
                group_layer_index = o.vertex_groups.active.index
            except Exception:
                raise Exception("Cannot find active vertex group")
        
        vs = []
        ns = []
        cs = []
        for v in bm.verts:
            vs.append(v.co.to_tuple())
            ns.append(v.normal.to_tuple())
            
            if(colorize is None):
                cs.append((1.0, 0.0, 0.0, ))
            elif(colorize == 'CONSTANT'):
                cs.append(constant_color)
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


class PCVParticleSystemSampler():
    def __init__(self, context, o, alive_only=True, colorize=None, constant_color=None, vcols=None, uvtex=None, vgroup=None, ):
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
        
        vs = []
        ns = []
        cs = []
        
        depsgraph = context.evaluated_depsgraph_get()
        if(o.modifiers):
            owner = o.evaluated_get(depsgraph)
            me = owner.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph, )
        else:
            owner = o
            me = owner.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph, )
        
        o = owner
        
        bm = bmesh.new()
        bm.from_mesh(me)
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        
        psys = o.particle_systems.active
        if(psys is None):
            raise Exception("Cannot find active particle system")
        if(len(psys.particles) == 0):
            raise Exception("Active particle system has 0 particles")
        if(alive_only):
            ok = False
            for p in psys.particles:
                if(p.alive_state == "ALIVE"):
                    ok = True
                    break
            if(not ok):
                raise Exception("Active particle system has 0 alive particles")
        
        mod = None
        uv_no = None
        if(colorize in ('VCOLS', 'UVTEX', 'GROUP_MONO', 'GROUP_COLOR', )):
            if(uvtex is None):
                raise Exception("Cannot find active uv layout on emitter")
            for m in o.modifiers:
                if(m.type == 'PARTICLE_SYSTEM'):
                    if(m.particle_system == psys):
                        mod = m
                        break
            uv_no = o.data.uv_layers.active_index
        
        if(colorize == 'UVTEX'):
            try:
                if(o.active_material is None):
                    raise Exception("Cannot find active material")
                uvtexnode = o.active_material.node_tree.nodes.active
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
            # for vertex groups, uv layout is required.. non-overlapping for best results
            uvlayer = bm.loops.layers.uv.active
            if(uvlayer is None):
                raise Exception("Cannot find active UV layout")
        if(colorize in ('GROUP_MONO', 'GROUP_COLOR')):
            try:
                group_layer = bm.verts.layers.deform.active
                if(group_layer is None):
                    raise Exception()
                group_layer_index = o.vertex_groups.active.index
            except Exception:
                raise Exception("Cannot find active vertex group")
            # for vertex groups, uv layout is required.. non-overlapping for best results
            uvlayer = bm.loops.layers.uv.active
            if(uvlayer is None):
                raise Exception("Cannot find active UV layout")
        
        # flatten mesh by uv and add original face indexes as int layer
        def simple_flatten_uv_mesh(bm):
            bm.faces.index_update()
            bm.faces.ensure_lookup_table()
            r = bmesh.new()
            ilayer = r.faces.layers.int.new('face_indexes')
            for f in bm.faces:
                fvs = []
                for i, l in enumerate(f.loops):
                    uv = l[uvlayer].uv
                    rv = r.verts.new((uv.x, uv.y, 0.0))
                    fvs.append(rv)
                rf = r.faces.new(fvs)
                rf[ilayer] = f.index
            r.faces.index_update()
            r.faces.ensure_lookup_table()
            return r
        
        if(colorize in ('VCOLS', 'GROUP_MONO', 'GROUP_COLOR', )):
            # i do not need extra flat mesh each time..
            bmf = simple_flatten_uv_mesh(bm)
            bmf_il = bmf.faces.layers.int['face_indexes']
            # with that i can ray_cast uv location and get original index of face i hit
            bmf_bvh = BVHTree.FromBMesh(bmf)
        
        for i, p in enumerate(psys.particles):
            if(p.alive_state != "ALIVE" and alive_only):
                continue
            
            vs.append(p.location.to_tuple())
            
            n = Vector((1.0, 0.0, 0.0, ))
            n.rotate(p.rotation)
            ns.append(n.normalized().to_tuple())
            
            if(colorize is None):
                cs.append((1.0, 0.0, 0.0, ))
            elif(colorize == 'CONSTANT'):
                cs.append(constant_color)
            elif(colorize == 'VCOLS'):
                uv = p.uv_on_emitter(mod)
                # intersect with flattened mesh and get original index of face from emitter mesh
                fl, fn, fi, di = bmf_bvh.ray_cast(Vector((uv.x, uv.y, -0.1)), Vector((0.0, 0.0, 1.0)), 0.2, )
                fpoly = bmf.faces[fi]
                oi = fpoly[bmf_il]
                poly = bm.faces[oi]
                # get final color from barycentric weights
                ws = poly_3d_calc([fv.co for fv in fpoly.verts], p.location)
                cols = [l[col_layer][:3] for l in poly.loops]
                r = sum([cc[0] * ws[ci] for ci, cc in enumerate(cols)])
                g = sum([cc[1] * ws[ci] for ci, cc in enumerate(cols)])
                b = sum([cc[2] * ws[ci] for ci, cc in enumerate(cols)])
                cs.append((r, g, b, ))
            elif(colorize == 'UVTEX'):
                uv = p.uv_on_emitter(mod)
                w, h = uvimage.size
                # x,y % 1.0 to wrap around if uv coordinate is outside 0.0-1.0 range
                x = int(round(remap(uv.x % 1.0, 0.0, 1.0, 0, w - 1)))
                y = int(round(remap(uv.y % 1.0, 0.0, 1.0, 0, h - 1)))
                cs.append(tuple(uvarray[y][x][:3].tolist()))
            elif(colorize == 'GROUP_MONO'):
                uv = p.uv_on_emitter(mod)
                fl, fn, fi, di = bmf_bvh.ray_cast(Vector((uv.x, uv.y, -0.1)), Vector((0.0, 0.0, 1.0)), 0.2, )
                fpoly = bmf.faces[fi]
                oi = fpoly[bmf_il]
                poly = bm.faces[oi]
                ws = poly_3d_calc([fv.co for fv in fpoly.verts], p.location)
                weights = [pv[group_layer].get(group_layer_index, 0.0) for pv in poly.verts]
                w = sum([ww * ws[wi] for wi, ww in enumerate(weights)])
                cs.append((w, w, w, ))
            elif(colorize == 'GROUP_COLOR'):
                uv = p.uv_on_emitter(mod)
                fl, fn, fi, di = bmf_bvh.ray_cast(Vector((uv.x, uv.y, -0.1)), Vector((0.0, 0.0, 1.0)), 0.2, )
                fpoly = bmf.faces[fi]
                oi = fpoly[bmf_il]
                poly = bm.faces[oi]
                ws = poly_3d_calc([fv.co for fv in fpoly.verts], p.location)
                weights = [pv[group_layer].get(group_layer_index, 0.0) for pv in poly.verts]
                w = sum([ww * ws[wi] for wi, ww in enumerate(weights)])
                hue = remap(1.0 - w, 0.0, 1.0, 0.0, 1 / 1.5)
                c = Color()
                c.hsv = (hue, 1.0, 1.0, )
                cs.append((c.r, c.g, c.b, ))
        
        a = np.concatenate((vs, ns, cs), axis=1, )
        np.random.shuffle(a)
        vs = a[:, :3]
        ns = a[:, 3:6]
        cs = a[:, 6:]
        
        self.vs = vs[:]
        self.ns = ns[:]
        self.cs = cs[:]
        
        if(colorize in ('VCOLS', 'GROUP_MONO', 'GROUP_COLOR', )):
            bmf.free()
        
        bm.free()
        owner.to_mesh_clear()


class PCVRandomVolumeSampler():
    def __init__(self, ob, num_samples, rnd, ):
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
        
        def is_point_inside_mesh_v1(point, ob, mwi, ):
            axes = [Vector((1.0, 0.0, 0.0)), Vector((0.0, 1.0, 0.0)), Vector((0.0, 0.0, 1.0)), ]
            r = False
            for a in axes:
                o = mwi @ point
                c = 0
                while True:
                    _, l, n, i = ob.ray_cast(o, o + a * 10000.0)
                    if(i == -1):
                        break
                    c += 1
                    o = l + a * 0.00001
                if(c % 2 == 0):
                    r = True
                    break
            return not r
        
        def is_point_inside_mesh_v2(p, o, shift=0.000001, search_distance=10000.0, ):
            p = Vector(p)
            path = []
            hits = 0
            direction = Vector((0.0, 0.0, 1.0))
            opposite_direction = direction.copy()
            opposite_direction.negate()
            path.append(p)
            loc = p
            ok = True
            
            def shift_vector(co, no, v):
                return co + (no.normalized() * v)
            
            while(ok):
                end = shift_vector(loc, direction, search_distance)
                loc = shift_vector(loc, direction, shift)
                _, loc, nor, ind = o.ray_cast(loc, end)
                if(ind != -1):
                    a = shift_vector(loc, opposite_direction, shift)
                    path.append(a)
                    hits += 1
                else:
                    ok = False
            if(hits % 2 == 1):
                return True
            return False
        
        def is_point_inside_mesh_v3(p, o):
            _, loc, nor, ind = o.closest_point_on_mesh(p)
            if(ind != -1):
                v = loc - p
                d = v.dot(nor)
                if(d >= 0):
                    return True, loc, nor, ind
            return False, None, None, None
        
        def max_values(o):
            bbox = [Vector(b) for b in o.bound_box]
            nx = sorted(bbox, key=lambda c: c.x, reverse=False)[0].x
            ny = sorted(bbox, key=lambda c: c.y, reverse=False)[0].y
            nz = sorted(bbox, key=lambda c: c.z, reverse=False)[0].z
            x = sorted(bbox, key=lambda c: c.x, reverse=True)[0].x
            y = sorted(bbox, key=lambda c: c.y, reverse=True)[0].y
            z = sorted(bbox, key=lambda c: c.z, reverse=True)[0].z
            return nx, ny, nz, x, y, z
        
        xmin, ymin, zmin, xmax, ymax, zmax = max_values(ob)
        
        def sample():
            x = remap(rnd.random(), 0.0, 1.0, xmin, xmax)
            y = remap(rnd.random(), 0.0, 1.0, ymin, ymax)
            z = remap(rnd.random(), 0.0, 1.0, zmin, zmax)
            return Vector((x, y, z))
        
        vs = []
        ns = []
        cs = []
        progress = Progress(num_samples, indent=1, )
        while(len(vs) < num_samples):
            a = sample()
            ok1 = is_point_inside_mesh_v1(a, ob, Matrix())
            ok2 = is_point_inside_mesh_v2(a, ob)
            ok3, location, normal, index = is_point_inside_mesh_v3(a, ob)
            if(ok1 and ok2 and ok3):
                vs.append(a.to_tuple())
                ns.append(normal.to_tuple())
                cs.append((1.0, 1.0, 1.0, ))
                progress.step()
        
        self.vs = vs[:]
        self.ns = ns[:]
        self.cs = cs[:]


class PCV_OT_init(Operator):
    bl_idname = "point_cloud_visualizer.init"
    bl_label = "init"
    
    def execute(self, context):
        PCVManager.init()
        context.area.tag_redraw()
        return {'FINISHED'}


class PCV_OT_deinit(Operator):
    bl_idname = "point_cloud_visualizer.deinit"
    bl_label = "deinit"
    
    def execute(self, context):
        PCVManager.deinit()
        context.area.tag_redraw()
        return {'FINISHED'}


class PCV_OT_gc(Operator):
    bl_idname = "point_cloud_visualizer.gc"
    bl_label = "gc"
    
    def execute(self, context):
        PCVManager.gc()
        return {'FINISHED'}


class PCV_OT_draw(Operator):
    bl_idname = "point_cloud_visualizer.draw"
    bl_label = "Draw"
    bl_description = "Draw point cloud to viewport"
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        
        pcv = context.object.point_cloud_visualizer
        ok = False
        cached = False
        for k, v in PCVManager.cache.items():
            if(v['uuid'] == pcv.uuid):
                if(v['ready']):
                    cached = True
                    if(not v['draw']):
                        ok = True
        if(not ok and pcv.filepath != "" and pcv.uuid != "" and not cached):
            ok = True
        return ok
    
    def execute(self, context):
        PCVManager.init()
        
        pcv = context.object.point_cloud_visualizer
        
        if(pcv.uuid not in PCVManager.cache):
            pcv.uuid = ""
            ok = PCVManager.load_ply_to_cache(self, context)
            if(not ok):
                return {'CANCELLED'}
        
        c = PCVManager.cache[pcv.uuid]
        c['draw'] = True
        
        context.area.tag_redraw()
        
        return {'FINISHED'}


class PCV_OT_erase(Operator):
    bl_idname = "point_cloud_visualizer.erase"
    bl_label = "Erase"
    bl_description = "Erase point cloud from viewport"
    
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
        c['draw'] = False
        
        context.area.tag_redraw()
        
        return {'FINISHED'}


class PCV_OT_load(Operator):
    bl_idname = "point_cloud_visualizer.load_ply_to_cache"
    bl_label = "Load PLY"
    bl_description = "Load PLY file"
    
    filename_ext = ".ply"
    filter_glob: StringProperty(default="*.ply", options={'HIDDEN'}, )
    filepath: StringProperty(name="File Path", default="", description="", maxlen=1024, subtype='FILE_PATH', )
    order = ["filepath", ]
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        
        pcv = context.object.point_cloud_visualizer
        if(not pcv.runtime):
            return True
        return False
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}
    
    def execute(self, context):
        pcv = context.object.point_cloud_visualizer
        ok = True
        h, t = os.path.split(self.filepath)
        n, e = os.path.splitext(t)
        if(e != '.ply'):
            ok = False
        if(not ok):
            self.report({'ERROR'}, "File at '{}' seems not to be a PLY file.".format(self.filepath))
            return {'CANCELLED'}
        
        pcv.filepath = self.filepath
        
        if(pcv.uuid != ""):
            if(pcv.uuid in PCVManager.cache):
                PCVManager.cache[pcv.uuid]['kill'] = True
                PCVManager.gc()
        
        ok = PCVManager.load_ply_to_cache(self, context)
        
        if(not ok):
            return {'CANCELLED'}
        return {'FINISHED'}


class PCV_OT_render(Operator):
    bl_idname = "point_cloud_visualizer.render"
    bl_label = "Render"
    bl_description = "Render displayed point cloud from active camera view to image"
    
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
        _t = time.time()
        
        bgl.glEnable(bgl.GL_PROGRAM_POINT_SIZE)
        bgl.glEnable(bgl.GL_DEPTH_TEST)
        bgl.glEnable(bgl.GL_BLEND)
        
        scene = context.scene
        render = scene.render
        image_settings = render.image_settings
        
        original_depth = image_settings.color_depth
        image_settings.color_depth = '8'
        
        pcv = context.object.point_cloud_visualizer
        
        if(pcv.render_resolution_linked):
            scale = render.resolution_percentage / 100
            
            if(pcv.render_supersampling > 1):
                scale *= pcv.render_supersampling
            
            width = int(render.resolution_x * scale)
            height = int(render.resolution_y * scale)
        else:
            scale = pcv.render_resolution_percentage / 100
            
            if(pcv.render_supersampling > 1):
                scale *= pcv.render_supersampling
            
            width = int(pcv.render_resolution_x * scale)
            height = int(pcv.render_resolution_y * scale)
        
        cloud = PCVManager.cache[pcv.uuid]
        cam = scene.camera
        if(cam is None):
            self.report({'ERROR'}, "No camera found.")
            return {'CANCELLED'}
        
        def get_output_path():
            p = bpy.path.abspath(pcv.render_path)
            ok = True
            
            # ensure soem directory and filename
            h, t = os.path.split(p)
            if(h == ''):
                # next to blend file
                h = bpy.path.abspath('//')
            
            if(not os.path.exists(h)):
                self.report({'ERROR'}, "Directory does not exist ('{}').".format(h))
                ok = False
            if(not os.path.isdir(h)):
                self.report({'ERROR'}, "Not a directory ('{}').".format(h))
                ok = False
            
            if(t == ''):
                # default name
                t = 'pcv_render_###.png'
            
            # ensure extension
            p = os.path.join(h, t)
            p = bpy.path.ensure_ext(p, '.png', )
            h, t = os.path.split(p)
            
            # ensure frame number
            if('#' not in t):
                n, e = os.path.splitext(t)
                n = '{}_###'.format(n)
                t = '{}{}'.format(n, e)
            
            # return with a bit of luck valid path
            p = os.path.join(h, t)
            return ok, p
        
        def swap_frame_number(p):
            fn = scene.frame_current
            h, t = os.path.split(p)
            
            # swap all occurences of # with zfilled frame number
            pattern = r'#+'
            
            def repl(m):
                l = len(m.group(0))
                return str(fn).zfill(l)
            
            t = re.sub(pattern, repl, t, )
            p = os.path.join(h, t)
            return p
        
        bd = context.blend_data
        if(bd.filepath == "" and not bd.is_saved):
            self.report({'ERROR'}, "Save .blend file first.")
            return {'CANCELLED'}
        
        ok, output_path = get_output_path()
        if(not ok):
            return {'CANCELLED'}
        
        # path is ok, write it back..
        pcv.render_path = output_path
        
        # swap frame number
        output_path = swap_frame_number(output_path)
        
        # TODO: on my machine, maximum size is 16384, is it max in general or just my hardware? check it, and add some warning, or hadle this: RuntimeError: gpu.offscreen.new(...) failed with 'GPUFrameBuffer: framebuffer status GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT somehow..
        offscreen = GPUOffScreen(width, height)
        offscreen.bind()
        try:
            gpu.matrix.load_matrix(Matrix.Identity(4))
            gpu.matrix.load_projection_matrix(Matrix.Identity(4))
            
            bgl.glClear(bgl.GL_COLOR_BUFFER_BIT)
            bgl.glClear(bgl.GL_DEPTH_BUFFER_BIT)
            
            o = cloud['object']
            vs = cloud['vertices']
            cs = cloud['colors']
            ns = cloud['normals']
            
            dp = pcv.render_display_percent
            l = int((len(vs) / 100) * dp)
            if(dp >= 99):
                l = len(vs)
            vs = vs[:l]
            cs = cs[:l]
            ns = ns[:l]
            
            use_smoothstep = False
            if(pcv.render_smoothstep):
                # for anti-aliasing basic shader should be enabled
                # if(not pcv.illumination and not pcv.override_default_shader):
                if(not pcv.override_default_shader):
                    use_smoothstep = True
                    # sort by depth
                    mw = o.matrix_world
                    depth = []
                    for i, v in enumerate(vs):
                        vw = mw @ Vector(v)
                        depth.append(world_to_camera_view(scene, cam, vw)[2])
                    zps = zip(depth, vs, cs, ns)
                    sps = sorted(zps, key=lambda a: a[0])
                    # split and reverse
                    vs = [a for _, a, b, c in sps][::-1]
                    cs = [b for _, a, b, c in sps][::-1]
                    ns = [c for _, a, b, c in sps][::-1]
            
            if(pcv.dev_depth_enabled):
                if(pcv.illumination):
                    shader = GPUShader(PCVShaders.depth_vertex_shader_illumination, PCVShaders.depth_fragment_shader_illumination, )
                    batch = batch_for_shader(shader, 'POINTS', {"position": vs, "normal": ns, })
                elif(pcv.dev_depth_false_colors):
                    shader = GPUShader(PCVShaders.depth_vertex_shader_false_colors, PCVShaders.depth_fragment_shader_false_colors, )
                    batch = batch_for_shader(shader, 'POINTS', {"position": vs, })
                else:
                    shader = GPUShader(PCVShaders.depth_vertex_shader_simple, PCVShaders.depth_fragment_shader_simple, )
                    batch = batch_for_shader(shader, 'POINTS', {"position": vs, })
            elif(pcv.dev_normal_colors_enabled):
                shader = GPUShader(PCVShaders.normal_colors_vertex_shader, PCVShaders.normal_colors_fragment_shader, )
                batch = batch_for_shader(shader, 'POINTS', {"position": vs, "normal": ns, })
            elif(pcv.dev_position_colors_enabled):
                shader = GPUShader(PCVShaders.position_colors_vertex_shader, PCVShaders.position_colors_fragment_shader, )
                batch = batch_for_shader(shader, 'POINTS', {"position": vs, })
            elif(pcv.illumination):
                if(use_smoothstep):
                    shader = GPUShader(PCVShaders.vertex_shader_illumination_render_smooth, PCVShaders.fragment_shader_illumination_render_smooth)
                    batch = batch_for_shader(shader, 'POINTS', {"position": vs, "color": cs, "normal": ns, })
                else:
                    shader = GPUShader(PCVShaders.vertex_shader_illumination, PCVShaders.fragment_shader_illumination)
                    batch = batch_for_shader(shader, 'POINTS', {"position": vs, "color": cs, "normal": ns, })
            else:
                if(use_smoothstep):
                    shader = GPUShader(PCVShaders.vertex_shader_simple_render_smooth, PCVShaders.fragment_shader_simple_render_smooth)
                    batch = batch_for_shader(shader, 'POINTS', {"position": vs, "color": cs, })
                else:
                    shader = GPUShader(PCVShaders.vertex_shader_simple, PCVShaders.fragment_shader_simple)
                    batch = batch_for_shader(shader, 'POINTS', {"position": vs, "color": cs, })
            
            shader.bind()
            
            view_matrix = cam.matrix_world.inverted()
            depsgraph = bpy.context.evaluated_depsgraph_get()
            camera_matrix = cam.calc_matrix_camera(depsgraph, x=render.resolution_x, y=render.resolution_y, scale_x=render.pixel_aspect_x, scale_y=render.pixel_aspect_y, )
            perspective_matrix = camera_matrix @ view_matrix
            
            shader.uniform_float("perspective_matrix", perspective_matrix)
            shader.uniform_float("object_matrix", o.matrix_world)
            if(pcv.render_supersampling > 1):
                shader.uniform_float("point_size", pcv.render_point_size * pcv.render_supersampling)
            else:
                shader.uniform_float("point_size", pcv.render_point_size)
            shader.uniform_float("alpha_radius", pcv.alpha_radius)
            shader.uniform_float("global_alpha", pcv.global_alpha)
            
            if(pcv.dev_depth_enabled):
                # pm = bpy.context.region_data.perspective_matrix
                # shader.uniform_float("perspective_matrix", pm)
                # shader.uniform_float("object_matrix", o.matrix_world)
                
                # NOTE: precalculating and storing following should speed up things a bit, but then it won't reflect edits..
                cx = np.sum(vs[:, 0]) / len(vs)
                cy = np.sum(vs[:, 1]) / len(vs)
                cz = np.sum(vs[:, 2]) / len(vs)
                _, _, s = o.matrix_world.decompose()
                l = s.length
                maxd = abs(np.max(vs))
                mind = abs(np.min(vs))
                maxdist = maxd
                if(mind > maxd):
                    maxdist = mind
                shader.uniform_float("maxdist", float(maxdist) * l)
                shader.uniform_float("center", (cx, cy, cz, ))
                
                shader.uniform_float("brightness", pcv.dev_depth_brightness)
                shader.uniform_float("contrast", pcv.dev_depth_contrast)
                
                # shader.uniform_float("point_size", pcv.point_size)
                # shader.uniform_float("alpha_radius", pcv.alpha_radius)
                # shader.uniform_float("global_alpha", pcv.global_alpha)
                
                if(pcv.illumination):
                    cm = Matrix(((-1.0, 0.0, 0.0, 0.0, ), (0.0, -0.0, 1.0, 0.0, ), (0.0, -1.0, -0.0, 0.0, ), (0.0, 0.0, 0.0, 1.0, ), ))
                    _, obrot, _ = o.matrix_world.decompose()
                    mr = obrot.to_matrix().to_4x4()
                    mr.invert()
                    direction = cm @ pcv.light_direction
                    direction = mr @ direction
                    shader.uniform_float("light_direction", direction)
                    inverted_direction = direction.copy()
                    inverted_direction.negate()
                    c = pcv.light_intensity
                    shader.uniform_float("light_intensity", (c, c, c, ))
                    shader.uniform_float("shadow_direction", inverted_direction)
                    c = pcv.shadow_intensity
                    shader.uniform_float("shadow_intensity", (c, c, c, ))
                    if(pcv.dev_depth_false_colors):
                        shader.uniform_float("color_a", pcv.dev_depth_color_a)
                        shader.uniform_float("color_b", pcv.dev_depth_color_b)
                    else:
                        shader.uniform_float("color_a", (1.0, 1.0, 1.0))
                        shader.uniform_float("color_b", (0.0, 0.0, 0.0))
                else:
                    if(pcv.dev_depth_false_colors):
                        shader.uniform_float("color_a", pcv.dev_depth_color_a)
                        shader.uniform_float("color_b", pcv.dev_depth_color_b)
            elif(pcv.dev_normal_colors_enabled):
                pass
            elif(pcv.dev_position_colors_enabled):
                pass
            elif(pcv.illumination and pcv.has_normals and cloud['illumination']):
                cm = Matrix(((-1.0, 0.0, 0.0, 0.0, ), (0.0, -0.0, 1.0, 0.0, ), (0.0, -1.0, -0.0, 0.0, ), (0.0, 0.0, 0.0, 1.0, ), ))
                _, obrot, _ = o.matrix_world.decompose()
                mr = obrot.to_matrix().to_4x4()
                mr.invert()
                direction = cm @ pcv.light_direction
                direction = mr @ direction
                shader.uniform_float("light_direction", direction)
                
                inverted_direction = direction.copy()
                inverted_direction.negate()
                
                c = pcv.light_intensity
                shader.uniform_float("light_intensity", (c, c, c, ))
                shader.uniform_float("shadow_direction", inverted_direction)
                c = pcv.shadow_intensity
                shader.uniform_float("shadow_intensity", (c, c, c, ))
                # shader.uniform_float("show_normals", float(pcv.show_normals))
                # shader.uniform_float("show_illumination", float(pcv.illumination))
            else:
                pass
            
            batch.draw(shader)
            
            buffer = bgl.Buffer(bgl.GL_BYTE, width * height * 4)
            bgl.glReadBuffer(bgl.GL_BACK)
            bgl.glReadPixels(0, 0, width, height, bgl.GL_RGBA, bgl.GL_UNSIGNED_BYTE, buffer)
            
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
            
        finally:
            bgl.glDisable(bgl.GL_PROGRAM_POINT_SIZE)
            bgl.glDisable(bgl.GL_DEPTH_TEST)
            bgl.glDisable(bgl.GL_BLEND)
            offscreen.unbind()
            offscreen.free()
        
        # image from buffer
        image_name = "pcv_output"
        if(image_name not in bpy.data.images):
            bpy.data.images.new(image_name, width, height)
        image = bpy.data.images[image_name]
        image.pixels = [v / 255 for v in buffer]
        # image.scale(width, height)
        
        if(pcv.render_supersampling > 1):
            width = int(width / pcv.render_supersampling)
            height = int(height / pcv.render_supersampling)
            image.scale(width, height)
        
        # save as image file
        def save_render(operator, scene, image, output_path, ):
            rs = scene.render
            s = rs.image_settings
            ff = s.file_format
            cm = s.color_mode
            cd = s.color_depth
            
            vs = scene.view_settings
            vsvt = vs.view_transform
            vsl = vs.look
            vs.view_transform = 'Standard'
            vs.look = 'None'
            
            s.file_format = 'PNG'
            s.color_mode = 'RGBA'
            s.color_depth = '8'
            
            try:
                image.save_render(output_path)
                log("image '{}' saved".format(output_path))
            except Exception as e:
                s.file_format = ff
                s.color_mode = cm
                s.color_depth = cd
                vs.view_transform = vsvt
                vs.look = vsl
                
                log("error: {}".format(e))
                operator.report({'ERROR'}, "Unable to save render image, see console for details.")
                return
            
            s.file_format = ff
            s.color_mode = cm
            s.color_depth = cd
            vs.view_transform = vsvt
            vs.look = vsl
        
        save_render(self, scene, image, output_path, )
        
        # restore
        image_settings.color_depth = original_depth
        # cleanup
        bpy.data.images.remove(image)
        
        _d = datetime.timedelta(seconds=time.time() - _t)
        print("PCV: Frame completed in {}.".format(_d))
        
        return {'FINISHED'}


class PCV_OT_render_animation(Operator):
    bl_idname = "point_cloud_visualizer.render_animation"
    bl_label = "Animation"
    bl_description = "Render displayed point cloud from active camera view to animation frames"
    
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
        scene = context.scene
        
        if(scene.camera is None):
            self.report({'ERROR'}, "No camera found.")
            return {'CANCELLED'}
        
        def rm_ms(d):
            return d - datetime.timedelta(microseconds=d.microseconds)
        
        user_frame = scene.frame_current
        
        _t = time.time()
        log_format = 'PCV: Frame: {} ({}/{}) | Time: {} | Remaining: {}'
        frames = [i for i in range(scene.frame_start, scene.frame_end + 1, 1)]
        num_frames = len(frames)
        times = []
        
        for i, n in enumerate(frames):
            t = time.time()
            
            scene.frame_set(n)
            bpy.ops.point_cloud_visualizer.render()
            
            d = time.time() - t
            times.append(d)
            print(log_format.format(scene.frame_current, i + 1, num_frames,
                                    rm_ms(datetime.timedelta(seconds=d)),
                                    rm_ms(datetime.timedelta(seconds=(sum(times) / len(times)) * (num_frames - i - 1)), ), ))
            
        scene.frame_set(user_frame)
        
        _d = datetime.timedelta(seconds=time.time() - _t)
        print("PCV: Animation completed in {}.".format(_d))
        
        return {'FINISHED'}


class PCV_OT_convert(Operator):
    bl_idname = "point_cloud_visualizer.convert"
    bl_label = "Convert"
    bl_description = "Convert point cloud to mesh"
    
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
        _t = time.time()
        
        scene = context.scene
        pcv = context.object.point_cloud_visualizer
        o = context.object
        
        cache = PCVManager.cache[pcv.uuid]
        
        l = cache['stats']
        if(not pcv.mesh_all):
            nump = l
            mps = pcv.mesh_percentage
            l = int((nump / 100) * mps)
            if(mps >= 99):
                l = nump
        
        vs = cache['vertices'][:l]
        ns = cache['normals'][:l]
        cs = cache['colors'][:l]
        
        points = []
        for i in range(l):
            c = tuple([int(255 * cs[i][j]) for j in range(3)])
            points.append(tuple(vs[i]) + tuple(ns[i]) + c)
        
        def apply_matrix(points, matrix):
            matrot = matrix.decompose()[1].to_matrix().to_4x4()
            r = [None] * len(points)
            for i, p in enumerate(points):
                co = matrix @ Vector((p[0], p[1], p[2]))
                no = matrot @ Vector((p[3], p[4], p[5]))
                r[i] = (co.x, co.y, co.z, no.x, no.y, no.z, p[6], p[7], p[8])
            return r
        
        _, t = os.path.split(pcv.filepath)
        n, _ = os.path.splitext(t)
        m = o.matrix_world.copy()
        
        points = apply_matrix(points, m)
        
        g = None
        if(pcv.mesh_type in ('INSTANCER', 'PARTICLES', )):
            # TODO: if normals are missing or align to normal is not required, make just vertices instead of triangles and use that as source for particles and instances, will be a bit faster
            g = PCMeshInstancerMeshGenerator(mesh_type='TRIANGLE', )
        else:
            g = PCMeshInstancerMeshGenerator(mesh_type=pcv.mesh_type, )
        
        names = {'VERTEX': "{}-vertices",
                 'TRIANGLE': "{}-triangles",
                 'TETRAHEDRON': "{}-tetrahedrons",
                 'CUBE': "{}-cubes",
                 'ICOSPHERE': "{}-icospheres",
                 'INSTANCER': "{}-instancer",
                 'PARTICLES': "{}-particles", }
        n = names[pcv.mesh_type].format(n)
        
        # hide the point cloud, 99% of time i go back, hide cloud and then go to conversion product again..
        bpy.ops.point_cloud_visualizer.erase()
        
        s = pcv.mesh_size
        a = pcv.mesh_normal_align
        c = pcv.mesh_vcols
        if(not pcv.has_normals):
            a = False
        if(not pcv.has_vcols):
            c = False
        d = {'name': n, 'points': points, 'generator': g, 'matrix': Matrix(),
             'size': s, 'normal_align': a, 'vcols': c, }
        if(pcv.mesh_type == 'VERTEX'):
            # faster than instancer.. single vertices can't have normals and colors, so no need for instancer
            bm = bmesh.new()
            for p in points:
                bm.verts.new(p[:3])
            me = bpy.data.meshes.new(n)
            bm.to_mesh(me)
            bm.free()
            o = bpy.data.objects.new(n, me)
            view_layer = context.view_layer
            collection = view_layer.active_layer_collection.collection
            collection.objects.link(o)
            bpy.ops.object.select_all(action='DESELECT')
            o.select_set(True)
            view_layer.objects.active = o
        else:
            
            if(pcv.mesh_use_instancer2):
                # import cProfile
                # import pstats
                # import io
                # pr = cProfile.Profile()
                # pr.enable()
                
                d = {
                    'name': n,
                    'vs': vs,
                    'ns': ns,
                    'cs': cs,
                    'generator': g,
                    'matrix': Matrix(),
                    'size': s,
                    'with_normal_align': a,
                    'with_vertex_colors': c,
                }
                instancer = PCMeshInstancer2(**d)
                
                # pr.disable()
                # s = io.StringIO()
                # sortby = 'cumulative'
                # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
                # ps.print_stats()
                # print(s.getvalue())
            else:
                instancer = PCMeshInstancer(**d)
            
            o = instancer.object
        
        me = o.data
        me.transform(m.inverted())
        o.matrix_world = m
        
        if(pcv.mesh_type == 'INSTANCER'):
            pci = PCInstancer(o, pcv.mesh_size, pcv.mesh_base_sphere_subdivisions, )
        if(pcv.mesh_type == 'PARTICLES'):
            pcp = PCParticles(o, pcv.mesh_size, pcv.mesh_base_sphere_subdivisions, )
        
        _d = datetime.timedelta(seconds=time.time() - _t)
        log("completed in {}.".format(_d))
        
        return {'FINISHED'}


class PCV_OT_export(Operator, ExportHelper):
    bl_idname = "point_cloud_visualizer.export"
    bl_label = "Export PLY"
    bl_description = "Export point cloud to ply file"
    
    filename_ext = ".ply"
    filter_glob: StringProperty(default="*.ply", options={'HIDDEN'}, )
    check_extension = True
    
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
    
    def draw(self, context):
        l = self.layout
        c = l.column()
        pcv = context.object.point_cloud_visualizer
        c.prop(pcv, 'export_apply_transformation')
        c.prop(pcv, 'export_convert_axes')
        c.prop(pcv, 'export_visible_only')
    
    def execute(self, context):
        log("Export:", 0)
        _t = time.time()
        
        pcv = context.object.point_cloud_visualizer
        c = PCVManager.cache[pcv.uuid]
        
        o = c['object']
        
        if(pcv.export_use_viewport):
            log("using viewport points..", 1)
            vs = c['vertices']
            ns = c['normals']
            cs = c['colors']
            
            if(pcv.export_visible_only):
                log("visible only..", 1)
                # points in cache are stored already shuffled (or not), so this should work the same as in viewport..
                l = c['display_length']
                vs = vs[:l]
                ns = ns[:l]
                cs = cs[:l]
            
            # TODO: viewport points have always some normals and colors, should i keep it how it was loaded or should i include also generic data created for viewing?
            normals = True
            colors = True
            
        else:
            log("using original loaded points..", 1)
            # get original loaded points
            points = c['points']
            # check for normals
            normals = True
            if(not set(('nx', 'ny', 'nz')).issubset(points.dtype.names)):
                normals = False
            # check for colors
            colors = True
            if(not set(('red', 'green', 'blue')).issubset(points.dtype.names)):
                colors = False
            # make vertices, normals, colors arrays, use None if data is not available, colors leave as they are
            vs = np.column_stack((points['x'], points['y'], points['z'], ))
            ns = None
            if(normals):
                ns = np.column_stack((points['nx'], points['ny'], points['nz'], ))
            cs = None
            if(colors):
                cs = np.column_stack((points['red'], points['green'], points['blue'], ))
        
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
        
        # fabricate matrix
        m = Matrix.Identity(4)
        if(pcv.export_apply_transformation):
            if(o.matrix_world != Matrix.Identity(4)):
                log("apply transformation..", 1)
                vs, ns = apply_matrix(o.matrix_world.copy(), vs, ns, )
        if(pcv.export_convert_axes):
            log("convert axes..", 1)
            axis_forward = '-Z'
            axis_up = 'Y'
            cm = axis_conversion(to_forward=axis_forward, to_up=axis_up).to_4x4()
            vs, ns = apply_matrix(cm, vs, ns, )
        
        # TODO: make whole PCV data type agnostic, load anything, keep original, convert to what is needed for display (float32), use original for export if not set to use viewport/edited data. now i am forcing float32 for x, y, z, nx, ny, nz and uint8 for red, green, blue. 99% of ply files i've seen is like that, but specification is not that strict (read again the best resource: http://paulbourke.net/dataformats/ply/ )
        
        # somehow along the way i am getting double dtype, so correct that
        vs = vs.astype(np.float32)
        if(normals):
            ns = ns.astype(np.float32)
        if(colors):
            # cs = cs.astype(np.float32)
            
            if(pcv.export_use_viewport):
                # viewport colors are in float32 now, back to uint8 colors, loaded data should be in uint8, so no need for conversion
                cs = cs.astype(np.float32)
                cs = cs * 255
                cs = cs.astype(np.uint8)
        
        log("write..", 1)
        
        # combine back to points, using original dtype
        dt = (('x', vs[0].dtype.str, ),
              ('y', vs[1].dtype.str, ),
              ('z', vs[2].dtype.str, ), )
        if(normals):
            dt += (('nx', ns[0].dtype.str, ),
                   ('ny', ns[1].dtype.str, ),
                   ('nz', ns[2].dtype.str, ), )
        if(colors):
            dt += (('red', cs[0].dtype.str, ),
                   ('green', cs[1].dtype.str, ),
                   ('blue', cs[2].dtype.str, ), )
        log("dtype: {}".format(dt), 1)
        
        l = len(vs)
        dt = list(dt)
        a = np.empty(l, dtype=dt, )
        a['x'] = vs[:, 0]
        a['y'] = vs[:, 1]
        a['z'] = vs[:, 2]
        if(normals):
            a['nx'] = ns[:, 0]
            a['ny'] = ns[:, 1]
            a['nz'] = ns[:, 2]
        if(colors):
            a['red'] = cs[:, 0]
            a['green'] = cs[:, 1]
            a['blue'] = cs[:, 2]
        
        w = BinPlyPointCloudWriter(self.filepath, a, )
        
        _d = datetime.timedelta(seconds=time.time() - _t)
        log("completed in {}.".format(_d), 1)
        
        return {'FINISHED'}


class PCV_OT_filter_simplify(Operator):
    bl_idname = "point_cloud_visualizer.filter_simplify"
    bl_label = "Simplify"
    bl_description = "Simplify point cloud to exact number of evenly distributed samples, all loaded points are processed"
    
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
    
    def resample(self, context):
        scene = context.scene
        pcv = context.object.point_cloud_visualizer
        
        c = PCVManager.cache[pcv.uuid]
        vs = c['vertices']
        ns = c['normals']
        cs = c['colors']
        
        num_samples = pcv.filter_simplify_num_samples
        if(num_samples >= len(vs)):
            self.report({'ERROR'}, "Number of samples must be < number of points.")
            return False, []
        candidates = pcv.filter_simplify_num_candidates
        log("num_samples: {}, candidates: {}".format(num_samples, candidates), 1)
        
        l = len(vs)
        dt = [('x', '<f8'), ('y', '<f8'), ('z', '<f8'), ('nx', '<f8'), ('ny', '<f8'), ('nz', '<f8'), ('red', '<f8'), ('green', '<f8'), ('blue', '<f8'), ('alpha', '<f8'), ('index', '<i8')]
        pool = np.empty(l, dtype=dt, )
        pool['x'] = vs[:, 0]
        pool['y'] = vs[:, 1]
        pool['z'] = vs[:, 2]
        pool['nx'] = ns[:, 0]
        pool['ny'] = ns[:, 1]
        pool['nz'] = ns[:, 2]
        pool['red'] = cs[:, 0]
        pool['green'] = cs[:, 1]
        pool['blue'] = cs[:, 2]
        pool['alpha'] = cs[:, 3]
        pool['index'] = np.indices((l, ), dtype='<i8', )
        
        # to get random points, shuffle pool if not shuffled upon loading
        preferences = bpy.context.preferences
        addon_prefs = preferences.addons[__name__].preferences
        if(not addon_prefs.shuffle_points):
            np.random.shuffle(pool)
            # and set indexes again
            pool['index'] = np.indices((l, ), dtype='<i8', )
        
        tree = KDTree(len(pool))
        samples = np.array(pool[0])
        last_used = 0
        # unused_pool = np.empty(0, dtype=dt, )
        unused_pool = []
        
        tree.insert([pool[0]['x'], pool[0]['y'], pool[0]['z']], 0)
        tree.balance()
        
        log("sampling:", 1)
        use_unused = False
        prgs = Progress(num_samples - 1, indent=2, prefix="> ")
        for i in range(num_samples - 1):
            prgs.step()
            # choose candidates
            cands = pool[last_used + 1:last_used + 1 + candidates]
            
            if(len(cands) <= 0):
                # lets pretend that last set of candidates was filled full..
                use_unused = True
                cands = unused_pool[:candidates]
                unused_pool = unused_pool[candidates:]
            
            if(len(cands) == 0):
                # out of candidates, nothing to do here now.. number of desired samples was too close to total of points
                return True, samples
            
            # get the most distant candidate
            dists = []
            inds = []
            for cl in cands:
                vec, ci, cd = tree.find((cl['x'], cl['y'], cl['z']))
                inds.append(ci)
                dists.append(cd)
            maxd = 0.0
            maxi = 0
            for j, d in enumerate(dists):
                if(d > maxd):
                    maxd = d
                    maxi = j
            # chosen candidate
            ncp = cands[maxi]
            # get unused candidates to recycle
            uncands = np.delete(cands, maxi)
            # unused_pool = np.append(unused_pool, uncands)
            unused_pool.extend(uncands)
            # store accepted sample
            samples = np.append(samples, ncp)
            # update kdtree
            tree.insert([ncp['x'], ncp['y'], ncp['z']], ncp['index'])
            tree.balance()
            # use next points
            last_used += 1 + candidates
        
        return True, samples
    
    def execute(self, context):
        log("Simplify:", 0)
        _t = time.time()
        
        # if(debug_mode()):
        #     import cProfile
        #     import pstats
        #     import io
        #     pr = cProfile.Profile()
        #     pr.enable()
        
        ok, a = self.resample(context)
        if(not ok):
            return {'CANCELLED'}
        
        # if(debug_mode()):
        #     pr.disable()
        #     s = io.StringIO()
        #     sortby = 'cumulative'
        #     ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        #     ps.print_stats()
        #     print(s.getvalue())
        
        vs = np.column_stack((a['x'], a['y'], a['z'], ))
        ns = np.column_stack((a['nx'], a['ny'], a['nz'], ))
        cs = np.column_stack((a['red'], a['green'], a['blue'], a['alpha'], ))
        vs = vs.astype(np.float32)
        ns = ns.astype(np.float32)
        cs = cs.astype(np.float32)
        
        # put to cache
        pcv = context.object.point_cloud_visualizer
        PCVManager.update(pcv.uuid, vs, ns, cs, )
        
        _d = datetime.timedelta(seconds=time.time() - _t)
        log("completed in {}.".format(_d), 1)
        
        return {'FINISHED'}


class PCV_OT_filter_project(Operator):
    bl_idname = "point_cloud_visualizer.filter_project"
    bl_label = "Project"
    bl_description = "Project points on mesh surface"
    
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
                        o = pcv.filter_project_object
                        if(o is not None):
                            if(o.type in ('MESH', 'CURVE', 'SURFACE', 'FONT', )):
                                ok = True
        return ok
    
    def execute(self, context):
        log("Project:", 0)
        _t = time.time()
        
        # if(debug_mode()):
        #     import cProfile
        #     import pstats
        #     import io
        #     pr = cProfile.Profile()
        #     pr.enable()
        
        log("preprocessing..", 1)
        
        pcv = context.object.point_cloud_visualizer
        o = pcv.filter_project_object
        
        if(o is None):
            raise Exception()
        
        c = PCVManager.cache[pcv.uuid]
        vs = c['vertices']
        ns = c['normals']
        cs = c['colors']
        
        # apply parent matrix to points
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
        
        m = c['object'].matrix_world.copy()
        vs, ns = apply_matrix(m, vs, ns, )
        vs = vs.astype(np.float32)
        ns = ns.astype(np.float32)
        
        # combine
        l = len(vs)
        dt = [('x', '<f8'), ('y', '<f8'), ('z', '<f8'), ('nx', '<f8'), ('ny', '<f8'), ('nz', '<f8'), ('red', '<f8'), ('green', '<f8'), ('blue', '<f8'), ('alpha', '<f8'), ('index', '<i8'), ('delete', '?')]
        points = np.empty(l, dtype=dt, )
        points['x'] = vs[:, 0]
        points['y'] = vs[:, 1]
        points['z'] = vs[:, 2]
        points['nx'] = ns[:, 0]
        points['ny'] = ns[:, 1]
        points['nz'] = ns[:, 2]
        points['red'] = cs[:, 0]
        points['green'] = cs[:, 1]
        points['blue'] = cs[:, 2]
        points['alpha'] = cs[:, 3]
        points['index'] = np.indices((l, ), dtype='<i8', )
        points['delete'] = np.zeros((l, ), dtype='?', )
        
        search_distance = pcv.filter_project_search_distance
        negative = pcv.filter_project_negative
        positive = pcv.filter_project_positive
        discard = pcv.filter_project_discard
        shift = pcv.filter_project_shift
        
        sc = context.scene
        depsgraph = bpy.context.evaluated_depsgraph_get()
        
        # make target
        tmp_mesh = o.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph, )
        target_mesh = tmp_mesh.copy()
        target_mesh.name = 'target_mesh_{}'.format(pcv.uuid)
        target_mesh.transform(o.matrix_world.copy())
        view_layer = context.view_layer
        collection = view_layer.active_layer_collection.collection
        target = bpy.data.objects.new('target_mesh_{}'.format(pcv.uuid), target_mesh)
        collection.objects.link(target)
        # TODO use BVHTree for ray_cast without need to add object to scene
        # still no idea, have to read about it..
        depsgraph.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        
        # this should be None if not available
        vcols = o.data.vertex_colors.active
        uvtex = o.data.uv_layers.active
        vgroup = o.vertex_groups.active
        # now check if color source is available, if not, cancel
        if(pcv.filter_project_colorize):
            
            # depsgraph = context.evaluated_depsgraph_get()
            # if(o.modifiers):
            #     owner = o.evaluated_get(depsgraph)
            #     me = owner.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph, )
            # else:
            #     owner = o
            #     me = owner.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph, )
            
            bm = bmesh.new()
            bm.from_mesh(target_mesh)
            bmesh.ops.triangulate(bm, faces=bm.faces)
            bm.verts.ensure_lookup_table()
            bm.faces.ensure_lookup_table()
            
            if(pcv.filter_project_colorize_from == 'VCOLS'):
                try:
                    col_layer = bm.loops.layers.color.active
                    if(col_layer is None):
                        raise Exception()
                except Exception:
                    self.report({'ERROR'}, "Cannot find active vertex colors", )
                    return {'CANCELLED'}
            elif(pcv.filter_project_colorize_from == 'UVTEX'):
                try:
                    if(o.active_material is None):
                        raise Exception("Cannot find active material")
                    uvtexnode = o.active_material.node_tree.nodes.active
                    if(uvtexnode is None):
                        raise Exception("Cannot find active image texture in active material")
                    if(uvtexnode.type != 'TEX_IMAGE'):
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
                    self.report({'ERROR'}, str(e), )
                    return {'CANCELLED'}
            elif(pcv.filter_project_colorize_from in ['GROUP_MONO', 'GROUP_COLOR', ]):
                try:
                    group_layer = bm.verts.layers.deform.active
                    if(group_layer is None):
                        raise Exception()
                    group_layer_index = o.vertex_groups.active.index
                except Exception:
                    self.report({'ERROR'}, "Cannot find active vertex group", )
                    return {'CANCELLED'}
            else:
                self.report({'ERROR'}, "Unsupported color source", )
                return {'CANCELLED'}
        
        def distance(a, b, ):
            return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5
        
        def shift_vert_along_normal(co, no, v):
            c = Vector(co)
            n = Vector(no)
            return c + (n.normalized() * v)
        
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
        
        def gen_color(bm, result, location, normal, index, distance, ):
            col = (0.0, 0.0, 0.0, )
            
            poly = bm.faces[index]
            v = location
            
            if(pcv.filter_project_colorize_from == 'VCOLS'):
                ws = poly_3d_calc([poly.verts[0].co, poly.verts[1].co, poly.verts[2].co, ], v)
                ac = poly.loops[0][col_layer][:3]
                bc = poly.loops[1][col_layer][:3]
                cc = poly.loops[2][col_layer][:3]
                r = ac[0] * ws[0] + bc[0] * ws[1] + cc[0] * ws[2]
                g = ac[1] * ws[0] + bc[1] * ws[1] + cc[1] * ws[2]
                b = ac[2] * ws[0] + bc[2] * ws[1] + cc[2] * ws[2]
                col = (r, g, b, )
            elif(pcv.filter_project_colorize_from == 'UVTEX'):
                uvtriangle = []
                for l in poly.loops:
                    uvtriangle.append(Vector(l[uvlayer].uv.to_tuple() + (0.0, )))
                uvpoint = barycentric_transform(v, poly.verts[0].co, poly.verts[1].co, poly.verts[2].co, *uvtriangle, )
                w, h = uvimage.size
                # x,y % 1.0 to wrap around if uv coordinate is outside 0.0-1.0 range
                x = int(round(remap(uvpoint.x % 1.0, 0.0, 1.0, 0, w - 1)))
                y = int(round(remap(uvpoint.y % 1.0, 0.0, 1.0, 0, h - 1)))
                col = tuple(uvarray[y][x][:3].tolist())
            elif(pcv.filter_project_colorize_from == 'GROUP_MONO'):
                ws = poly_3d_calc([poly.verts[0].co, poly.verts[1].co, poly.verts[2].co, ], v)
                aw = poly.verts[0][group_layer].get(group_layer_index, 0.0)
                bw = poly.verts[1][group_layer].get(group_layer_index, 0.0)
                cw = poly.verts[2][group_layer].get(group_layer_index, 0.0)
                m = aw * ws[0] + bw * ws[1] + cw * ws[2]
                col = (m, m, m, )
            elif(pcv.filter_project_colorize_from == 'GROUP_COLOR'):
                ws = poly_3d_calc([poly.verts[0].co, poly.verts[1].co, poly.verts[2].co, ], v)
                aw = poly.verts[0][group_layer].get(group_layer_index, 0.0)
                bw = poly.verts[1][group_layer].get(group_layer_index, 0.0)
                cw = poly.verts[2][group_layer].get(group_layer_index, 0.0)
                m = aw * ws[0] + bw * ws[1] + cw * ws[2]
                hue = remap(1.0 - m, 0.0, 1.0, 0.0, 1 / 1.5)
                c = Color()
                c.hsv = (hue, 1.0, 1.0, )
                col = (c.r, c.g, c.b, )
            
            return col
        
        a = np.empty(l, dtype=dt, )
        
        log("projecting:", 1)
        prgr = Progress(len(points), 2)
        for i, p in enumerate(points):
            prgr.step()
            
            v = Vector((p['x'], p['y'], p['z'], ))
            n = Vector((p['nx'], p['ny'], p['nz'], ))
            
            if(positive):
                p_result, p_location, p_normal, p_index = target.ray_cast(v, n, distance=search_distance, depsgraph=depsgraph, )
                p_distance = None
                if(p_result):
                    p_distance = distance(p_location, v[:])
            else:
                p_result = False
            
            nn = shift_vert_along_normal(v, n, -search_distance)
            ndir = Vector(nn - v).normalized()
            
            if(negative):
                n_result, n_location, n_normal, n_index = target.ray_cast(v, ndir, distance=search_distance, depsgraph=depsgraph, )
                n_distance = None
                if(n_result):
                    n_distance = distance(n_location, v[:])
            else:
                n_result = False
            
            rp = np.copy(p)
            
            # store ray_cast results which was used for current point and pass them to gen_color
            used = None
            
            if(p_result and n_result):
                if(p_distance < n_distance):
                    rp['x'] = p_location[0]
                    rp['y'] = p_location[1]
                    rp['z'] = p_location[2]
                    used = (p_result, p_location, p_normal, p_index, p_distance)
                else:
                    rp['x'] = n_location[0]
                    rp['y'] = n_location[1]
                    rp['z'] = n_location[2]
                    used = (n_result, n_location, n_normal, n_index, n_distance)
            elif(p_result):
                rp['x'] = p_location[0]
                rp['y'] = p_location[1]
                rp['z'] = p_location[2]
                used = (p_result, p_location, p_normal, p_index, p_distance)
            elif(n_result):
                rp['x'] = n_location[0]
                rp['y'] = n_location[1]
                rp['z'] = n_location[2]
                used = (n_result, n_location, n_normal, n_index, n_distance)
            else:
                rp['delete'] = 1
            
            if(pcv.filter_project_colorize):
                if(used is not None):
                    col = gen_color(bm, *used)
                    rp['red'] = col[0]
                    rp['green'] = col[1]
                    rp['blue'] = col[2]
            
            a[i] = rp
        
        if(discard):
            log("discarding:", 1)
            prgr = Progress(len(a), 2)
            indexes = []
            for i in a:
                prgr.step()
                if(i['delete']):
                    indexes.append(i['index'])
            a = np.delete(a, indexes)
        
        if(shift != 0.0):
            log("shifting:", 1)
            prgr = Progress(len(a), 2)
            for i in a:
                prgr.step()
                l = shift_vert_along_normal((i['x'], i['y'], i['z'], ), (i['nx'], i['ny'], i['nz'], ), shift, )
                i['x'] = l[0]
                i['y'] = l[1]
                i['z'] = l[2]
        
        log("postprocessing..", 1)
        # split back
        vs = np.column_stack((a['x'], a['y'], a['z'], ))
        ns = np.column_stack((a['nx'], a['ny'], a['nz'], ))
        cs = np.column_stack((a['red'], a['green'], a['blue'], a['alpha'], ))
        vs = vs.astype(np.float32)
        ns = ns.astype(np.float32)
        cs = cs.astype(np.float32)
        
        # unapply parent matrix to points
        m = c['object'].matrix_world.copy()
        m = m.inverted()
        vs, ns = apply_matrix(m, vs, ns, )
        vs = vs.astype(np.float32)
        ns = ns.astype(np.float32)
        
        # cleanup
        if(pcv.filter_project_colorize):
            bm.free()
            # owner.to_mesh_clear()
        
        collection.objects.unlink(target)
        bpy.data.objects.remove(target)
        bpy.data.meshes.remove(target_mesh)
        o.to_mesh_clear()
        
        # put to cache..
        pcv = context.object.point_cloud_visualizer
        PCVManager.update(pcv.uuid, vs, ns, cs, )
        
        # if(debug_mode()):
        #     pr.disable()
        #     s = io.StringIO()
        #     sortby = 'cumulative'
        #     ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        #     ps.print_stats()
        #     print(s.getvalue())
        
        _d = datetime.timedelta(seconds=time.time() - _t)
        log("completed in {}.".format(_d), 1)
        
        return {'FINISHED'}


class PCV_OT_filter_remove_color(Operator):
    bl_idname = "point_cloud_visualizer.filter_remove_color"
    bl_label = "Select Color"
    bl_description = "Select points with exact/similar color"
    
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
        log("Remove Color:", 0)
        _t = time.time()
        
        pcv = context.object.point_cloud_visualizer
        # cache item
        c = PCVManager.cache[pcv.uuid]
        vs = c['vertices']
        ns = c['normals']
        cs = c['colors']
        # join to indexed points
        l = len(vs)
        dt = [('x', '<f8'), ('y', '<f8'), ('z', '<f8'), ('nx', '<f8'), ('ny', '<f8'), ('nz', '<f8'), ('red', '<f8'), ('green', '<f8'), ('blue', '<f8'), ('alpha', '<f8'), ('index', '<i8')]
        points = np.empty(l, dtype=dt, )
        points['x'] = vs[:, 0]
        points['y'] = vs[:, 1]
        points['z'] = vs[:, 2]
        points['nx'] = ns[:, 0]
        points['ny'] = ns[:, 1]
        points['nz'] = ns[:, 2]
        points['red'] = cs[:, 0]
        points['green'] = cs[:, 1]
        points['blue'] = cs[:, 2]
        points['alpha'] = cs[:, 3]
        points['index'] = np.indices((l, ), dtype='<i8', )
        
        # evaluate
        indexes = []
        
        # black magic..
        c = [c ** (1 / 2.2) for c in pcv.filter_remove_color]
        c = [int(i * 256) for i in c]
        c = [i / 256 for i in c]
        rmcolor = Color(c)
        
        # take half of the value because 1/2 <- v -> 1/2, plus and minus => full range
        dh = pcv.filter_remove_color_delta_hue / 2
        # only for hue, because i take in consideration its radial nature
        ds = pcv.filter_remove_color_delta_saturation
        dv = pcv.filter_remove_color_delta_value
        uh = pcv.filter_remove_color_delta_hue_use
        us = pcv.filter_remove_color_delta_saturation_use
        uv = pcv.filter_remove_color_delta_value_use
        
        prgr = Progress(len(points), 1)
        for p in points:
            prgr.step()
            # get point color
            c = Color((p['red'], p['green'], p['blue']))
            # check for more or less same color, a few decimals should be more than enough, ply should have 8bit colors
            fpr = 5
            same = (round(c.r, fpr) == round(rmcolor.r, fpr),
                    round(c.g, fpr) == round(rmcolor.g, fpr),
                    round(c.b, fpr) == round(rmcolor.b, fpr))
            if(all(same)):
                indexes.append(p['index'])
                continue
            
            # check
            h = False
            s = False
            v = False
            if(uh):
                rm_hue = rmcolor.h
                hd = min(abs(rm_hue - c.h), 1.0 - abs(rm_hue - c.h))
                if(hd <= dh):
                    h = True
            if(us):
                if(rmcolor.s - ds < c.s < rmcolor.s + ds):
                    s = True
            if(uv):
                if(rmcolor.v - dv < c.v < rmcolor.v + dv):
                    v = True
            
            a = False
            if(uh and not us and not uv):
                if(h):
                    a = True
            elif(not uh and us and not uv):
                if(s):
                    a = True
            elif(not uh and not us and uv):
                if(v):
                    a = True
            elif(uh and us and not uv):
                if(h and s):
                    a = True
            elif(not uh and us and uv):
                if(s and v):
                    a = True
            elif(uh and not us and uv):
                if(h and v):
                    a = True
            elif(uh and us and uv):
                if(h and s and v):
                    a = True
            else:
                pass
            if(a):
                indexes.append(p['index'])
        
        log("selected: {} points".format(len(indexes)), 1)
        
        if(len(indexes) == 0):
            # self.report({'ERROR'}, "Nothing selected.")
            self.report({'INFO'}, "Nothing selected.")
        else:
            pcv.filter_remove_color_selection = True
            c = PCVManager.cache[pcv.uuid]
            c['selection_indexes'] = indexes
        
        context.area.tag_redraw()
        
        _d = datetime.timedelta(seconds=time.time() - _t)
        log("completed in {}.".format(_d), 1)
        
        return {'FINISHED'}


class PCV_OT_filter_remove_color_deselect(Operator):
    bl_idname = "point_cloud_visualizer.filter_remove_color_deselect"
    bl_label = "Deselect"
    bl_description = "Deselect points"
    
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
                        if(pcv.filter_remove_color_selection):
                            if('selection_indexes' in v.keys()):
                                ok = True
        return ok
    
    def execute(self, context):
        pcv = context.object.point_cloud_visualizer
        c = PCVManager.cache[pcv.uuid]
        
        pcv.filter_remove_color_selection = False
        del c['selection_indexes']
        
        context.area.tag_redraw()
        
        return {'FINISHED'}


class PCV_OT_filter_remove_color_delete_selected(Operator):
    bl_idname = "point_cloud_visualizer.filter_remove_color_delete_selected"
    bl_label = "Delete Selected"
    bl_description = "Remove selected points"
    
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
                        if(pcv.filter_remove_color_selection):
                            if('selection_indexes' in v.keys()):
                                ok = True
        return ok
    
    def execute(self, context):
        pcv = context.object.point_cloud_visualizer
        c = PCVManager.cache[pcv.uuid]
        vs = c['vertices']
        ns = c['normals']
        cs = c['colors']
        indexes = c['selection_indexes']
        vs = np.delete(vs, indexes, axis=0, )
        ns = np.delete(ns, indexes, axis=0, )
        cs = np.delete(cs, indexes, axis=0, )
        
        PCVManager.update(pcv.uuid, vs, ns, cs, )
        
        pcv.filter_remove_color_selection = False
        del c['selection_indexes']
        
        return {'FINISHED'}


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


class PCV_OT_filter_merge(Operator):
    bl_idname = "point_cloud_visualizer.filter_merge"
    bl_label = "Merge With Other PLY"
    bl_description = "Merge with other ply file"
    
    filename_ext = ".ply"
    filter_glob: StringProperty(default="*.ply", options={'HIDDEN'}, )
    filepath: StringProperty(name="File Path", default="", description="", maxlen=1024, subtype='FILE_PATH', )
    order = ["filepath", ]
    
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
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}
    
    def execute(self, context):
        pcv = context.object.point_cloud_visualizer
        
        filepath = self.filepath
        h, t = os.path.split(filepath)
        n, e = os.path.splitext(t)
        if(e != '.ply'):
            self.report({'ERROR'}, "File at '{}' seems not to be a PLY file.".format(filepath))
            return {'CANCELLED'}
        
        points = []
        try:
            points = PlyPointCloudReader(filepath).points
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        if(len(points) == 0):
            self.report({'ERROR'}, "No vertices loaded from file at {}".format(filepath))
            return {'CANCELLED'}
        
        if(not set(('x', 'y', 'z')).issubset(points.dtype.names)):
            # this is very unlikely..
            self.report({'ERROR'}, "Loaded data seems to miss vertex locations.")
            return False
        normals = True
        if(not set(('nx', 'ny', 'nz')).issubset(points.dtype.names)):
            normals = False
        pcv.has_normals = normals
        if(not pcv.has_normals):
            pcv.illumination = False
        vcols = True
        if(not set(('red', 'green', 'blue')).issubset(points.dtype.names)):
            vcols = False
        pcv.has_vcols = vcols
        
        vs = np.column_stack((points['x'], points['y'], points['z'], ))
        
        if(normals):
            ns = np.column_stack((points['nx'], points['ny'], points['nz'], ))
        else:
            n = len(points)
            ns = np.column_stack((np.full(n, 0.0, dtype=np.float32, ),
                                  np.full(n, 0.0, dtype=np.float32, ),
                                  np.full(n, 1.0, dtype=np.float32, ), ))
        
        if(vcols):
            preferences = bpy.context.preferences
            addon_prefs = preferences.addons[__name__].preferences
            if(addon_prefs.convert_16bit_colors and points['red'].dtype == 'uint16'):
                r8 = (points['red'] / 256).astype('uint8')
                g8 = (points['green'] / 256).astype('uint8')
                b8 = (points['blue'] / 256).astype('uint8')
                if(addon_prefs.gamma_correct_16bit_colors):
                    cs = np.column_stack(((r8 / 255) ** (1 / 2.2),
                                          (g8 / 255) ** (1 / 2.2),
                                          (b8 / 255) ** (1 / 2.2),
                                          np.ones(len(points), dtype=float, ), ))
                else:
                    cs = np.column_stack((r8 / 255, g8 / 255, b8 / 255, np.ones(len(points), dtype=float, ), ))
                cs = cs.astype(np.float32)
            else:
                # 'uint8'
                cs = np.column_stack((points['red'] / 255, points['green'] / 255, points['blue'] / 255, np.ones(len(points), dtype=float, ), ))
                cs = cs.astype(np.float32)
        else:
            n = len(points)
            preferences = bpy.context.preferences
            addon_prefs = preferences.addons[__name__].preferences
            col = addon_prefs.default_vertex_color[:]
            col = tuple([c ** (1 / 2.2) for c in col]) + (1.0, )
            cs = np.column_stack((np.full(n, col[0], dtype=np.float32, ),
                                  np.full(n, col[1], dtype=np.float32, ),
                                  np.full(n, col[2], dtype=np.float32, ),
                                  np.ones(n, dtype=np.float32, ), ))
        
        # append
        c = PCVManager.cache[pcv.uuid]
        ovs = c['vertices']
        ons = c['normals']
        ocs = c['colors']
        vs = np.concatenate((ovs, vs, ))
        ns = np.concatenate((ons, ns, ))
        cs = np.concatenate((ocs, cs, ))
        
        preferences = bpy.context.preferences
        addon_prefs = preferences.addons[__name__].preferences
        if(addon_prefs.shuffle_points):
            l = len(vs)
            dt = [('x', '<f8'), ('y', '<f8'), ('z', '<f8'), ('nx', '<f8'), ('ny', '<f8'), ('nz', '<f8'), ('red', '<f8'), ('green', '<f8'), ('blue', '<f8'), ('alpha', '<f8'), ]
            a = np.empty(l, dtype=dt, )
            a['x'] = vs[:, 0]
            a['y'] = vs[:, 1]
            a['z'] = vs[:, 2]
            a['nx'] = ns[:, 0]
            a['ny'] = ns[:, 1]
            a['nz'] = ns[:, 2]
            a['red'] = cs[:, 0]
            a['green'] = cs[:, 1]
            a['blue'] = cs[:, 2]
            a['alpha'] = cs[:, 3]
            np.random.shuffle(a)
            vs = np.column_stack((a['x'], a['y'], a['z'], ))
            ns = np.column_stack((a['nx'], a['ny'], a['nz'], ))
            cs = np.column_stack((a['red'], a['green'], a['blue'], a['alpha'], ))
            vs = vs.astype(np.float32)
            ns = ns.astype(np.float32)
            cs = cs.astype(np.float32)
        
        PCVManager.update(pcv.uuid, vs, ns, cs, )
        
        return {'FINISHED'}


class PCV_OT_filter_join(Operator):
    bl_idname = "point_cloud_visualizer.filter_join"
    bl_label = "Join"
    bl_description = "Join with another PCV instance cloud"
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        
        pcv = context.object.point_cloud_visualizer
        pcv2 = None
        ok = False
        for k, v in PCVManager.cache.items():
            if(v['uuid'] == pcv.uuid):
                if(v['ready']):
                    if(v['draw']):
                        o = pcv.filter_join_object
                        if(o is not None):
                            pcv2 = o.point_cloud_visualizer
                break
        if(pcv2 is not None):
            for k, v in PCVManager.cache.items():
                if(v['uuid'] == pcv2.uuid):
                    if(v['ready']):
                        ok = True
                    break
        return ok
    
    def execute(self, context):
        pcv = context.object.point_cloud_visualizer
        pcv2 = pcv.filter_join_object.point_cloud_visualizer
        
        c = PCVManager.cache[pcv.uuid]
        c2 = PCVManager.cache[pcv2.uuid]
        
        ovs = c['vertices']
        ons = c['normals']
        ocs = c['colors']
        nvs = c2['vertices']
        nns = c2['normals']
        ncs = c2['colors']
        
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
        
        nvs, nns = apply_matrix(pcv.filter_join_object.matrix_world, nvs, nns, )
        nvs, nns = apply_matrix(context.object.matrix_world.inverted(), nvs, nns, )
        
        vs = np.concatenate((ovs, nvs, ))
        ns = np.concatenate((ons, nns, ))
        cs = np.concatenate((ocs, ncs, ))
        
        preferences = bpy.context.preferences
        addon_prefs = preferences.addons[__name__].preferences
        if(addon_prefs.shuffle_points):
            l = len(vs)
            dt = [('x', '<f8'), ('y', '<f8'), ('z', '<f8'), ('nx', '<f8'), ('ny', '<f8'), ('nz', '<f8'), ('red', '<f8'), ('green', '<f8'), ('blue', '<f8'), ('alpha', '<f8'), ]
            a = np.empty(l, dtype=dt, )
            a['x'] = vs[:, 0]
            a['y'] = vs[:, 1]
            a['z'] = vs[:, 2]
            a['nx'] = ns[:, 0]
            a['ny'] = ns[:, 1]
            a['nz'] = ns[:, 2]
            a['red'] = cs[:, 0]
            a['green'] = cs[:, 1]
            a['blue'] = cs[:, 2]
            a['alpha'] = cs[:, 3]
            np.random.shuffle(a)
            vs = np.column_stack((a['x'], a['y'], a['z'], ))
            ns = np.column_stack((a['nx'], a['ny'], a['nz'], ))
            cs = np.column_stack((a['red'], a['green'], a['blue'], a['alpha'], ))
            vs = vs.astype(np.float32)
            ns = ns.astype(np.float32)
            cs = cs.astype(np.float32)
        
        PCVManager.update(pcv.uuid, vs, ns, cs, )
        
        c2['draw'] = False
        context.area.tag_redraw()
        
        return {'FINISHED'}


class PCV_OT_filter_boolean_intersect(Operator):
    bl_idname = "point_cloud_visualizer.filter_boolean_intersect"
    bl_label = "Intersect"
    bl_description = ""
    
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
                        o = pcv.filter_boolean_object
                        if(o is not None):
                            if(o.type in ('MESH', 'CURVE', 'SURFACE', 'FONT', )):
                                ok = True
        return ok
    
    def execute(self, context):
        log("Intersect:", 0)
        _t = time.time()
        
        pcv = context.object.point_cloud_visualizer
        o = pcv.filter_boolean_object
        if(o is None):
            raise Exception()
        
        def apply_matrix(vs, ns, matrix, ):
            matrot = matrix.decompose()[1].to_matrix().to_4x4()
            dtv = vs.dtype
            dtn = ns.dtype
            rvs = np.zeros(vs.shape, dtv)
            rns = np.zeros(ns.shape, dtn)
            for i in range(len(vs)):
                co = matrix @ Vector(vs[i])
                no = matrot @ Vector(ns[i])
                rvs[i] = np.array(co.to_tuple(), dtv)
                rns[i] = np.array(no.to_tuple(), dtn)
            return rvs, rns
        
        c = PCVManager.cache[pcv.uuid]
        vs = c['vertices']
        ns = c['normals']
        cs = c['colors']
        
        # apply parent matrix to points
        m = c['object'].matrix_world.copy()
        # vs, ns = apply_matrix(vs, ns, m)
        
        sc = context.scene
        depsgraph = bpy.context.evaluated_depsgraph_get()
        # make target
        tmp_mesh = o.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph, )
        target_mesh = tmp_mesh.copy()
        target_mesh.name = 'target_mesh_{}'.format(pcv.uuid)
        target_mesh.transform(o.matrix_world.copy())
        target_mesh.transform(m.inverted())
        view_layer = context.view_layer
        collection = view_layer.active_layer_collection.collection
        target = bpy.data.objects.new('target_mesh_{}'.format(pcv.uuid), target_mesh)
        collection.objects.link(target)
        # still no idea, have to read about it..
        depsgraph.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        
        # use object bounding box for fast check if point can even be inside/outside of mesh and then use ray casting etc..
        bounds = [bv[:] for bv in target.bound_box]
        xmin = min([v[0] for v in bounds])
        xmax = max([v[0] for v in bounds])
        ymin = min([v[1] for v in bounds])
        ymax = max([v[1] for v in bounds])
        zmin = min([v[2] for v in bounds])
        zmax = max([v[2] for v in bounds])
        
        def is_in_bound_box(v):
            x = False
            if(xmin < v[0] < xmax):
                x = True
            y = False
            if(ymin < v[1] < ymax):
                y = True
            z = False
            if(zmin < v[2] < zmax):
                z = True
            if(x and y and z):
                return True
            return False
        
        # v1 raycasting in three axes and counting hits
        def is_point_inside_mesh_v1(p, o, ):
            axes = [Vector((1.0, 0.0, 0.0)), Vector((0.0, 1.0, 0.0)), Vector((0.0, 0.0, 1.0)), ]
            r = False
            for a in axes:
                v = p
                c = 0
                while True:
                    _, l, n, i = o.ray_cast(v, v + a * 10000.0)
                    if(i == -1):
                        break
                    c += 1
                    v = l + a * 0.00001
                if(c % 2 == 0):
                    r = True
                    break
            return not r
        
        # v2 raycasting and counting hits
        def is_point_inside_mesh_v2(p, o, shift=0.000001, search_distance=10000.0, ):
            p = Vector(p)
            path = []
            hits = 0
            direction = Vector((0.0, 0.0, 1.0))
            opposite_direction = direction.copy()
            opposite_direction.negate()
            path.append(p)
            loc = p
            ok = True
            
            def shift_vector(co, no, v):
                return co + (no.normalized() * v)
            
            while(ok):
                end = shift_vector(loc, direction, search_distance)
                loc = shift_vector(loc, direction, shift)
                _, loc, nor, ind = o.ray_cast(loc, end)
                if(ind != -1):
                    a = shift_vector(loc, opposite_direction, shift)
                    path.append(a)
                    hits += 1
                else:
                    ok = False
            if(hits % 2 == 1):
                return True
            return False
        
        # v3 closest point on mesh normal checking
        def is_point_inside_mesh_v3(p, o, ):
            _, loc, nor, ind = o.closest_point_on_mesh(p)
            if(ind != -1):
                v = loc - p
                d = v.dot(nor)
                if(d >= 0):
                    return True
            return False
        
        # if(debug_mode()):
        #     import cProfile
        #     import pstats
        #     import io
        #     pr = cProfile.Profile()
        #     pr.enable()
        
        indexes = []
        prgs = Progress(len(vs), indent=1, prefix="> ")
        for i, v in enumerate(vs):
            prgs.step()
            vv = Vector(v)
            '''
            inside1 = is_point_inside_mesh_v1(vv, target, )
            inside2 = is_point_inside_mesh_v2(vv, target, )
            inside3 = is_point_inside_mesh_v3(vv, target, )
            # intersect
            if(not inside1 and not inside2 and not inside3):
                indexes.append(i)
            # # exclude
            # if(inside1 and inside2 and inside3):
            #     indexes.append(i)
            '''
            
            in_bb = is_in_bound_box(v)
            if(not in_bb):
                # is not in bounds i can skip completely
                indexes.append(i)
                continue
            
            inside3 = is_point_inside_mesh_v3(vv, target, )
            if(not inside3):
                indexes.append(i)
        
        # if(debug_mode()):
        #     pr.disable()
        #     s = io.StringIO()
        #     sortby = 'cumulative'
        #     ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        #     ps.print_stats()
        #     print(s.getvalue())
        
        c = PCVManager.cache[pcv.uuid]
        vs = c['vertices']
        ns = c['normals']
        cs = c['colors']
        vs = np.delete(vs, indexes, axis=0, )
        ns = np.delete(ns, indexes, axis=0, )
        cs = np.delete(cs, indexes, axis=0, )
        
        log("removed: {} points".format(len(indexes)), 1)
        
        # cleanup
        collection.objects.unlink(target)
        bpy.data.objects.remove(target)
        bpy.data.meshes.remove(target_mesh)
        o.to_mesh_clear()
        
        # put to cache..
        pcv = context.object.point_cloud_visualizer
        PCVManager.update(pcv.uuid, vs, ns, cs, )
        
        _d = datetime.timedelta(seconds=time.time() - _t)
        log("completed in {}.".format(_d), 1)
        
        return {'FINISHED'}


class PCV_OT_filter_boolean_exclude(Operator):
    bl_idname = "point_cloud_visualizer.filter_boolean_exclude"
    bl_label = "Exclude"
    bl_description = ""
    
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
                        o = pcv.filter_boolean_object
                        if(o is not None):
                            if(o.type in ('MESH', 'CURVE', 'SURFACE', 'FONT', )):
                                ok = True
        return ok
    
    def execute(self, context):
        log("Exclude:", 0)
        _t = time.time()
        
        pcv = context.object.point_cloud_visualizer
        o = pcv.filter_boolean_object
        if(o is None):
            raise Exception()
        
        def apply_matrix(vs, ns, matrix, ):
            matrot = matrix.decompose()[1].to_matrix().to_4x4()
            dtv = vs.dtype
            dtn = ns.dtype
            rvs = np.zeros(vs.shape, dtv)
            rns = np.zeros(ns.shape, dtn)
            for i in range(len(vs)):
                co = matrix @ Vector(vs[i])
                no = matrot @ Vector(ns[i])
                rvs[i] = np.array(co.to_tuple(), dtv)
                rns[i] = np.array(no.to_tuple(), dtn)
            return rvs, rns
        
        c = PCVManager.cache[pcv.uuid]
        vs = c['vertices']
        ns = c['normals']
        cs = c['colors']
        
        # apply parent matrix to points
        m = c['object'].matrix_world.copy()
        # vs, ns = apply_matrix(vs, ns, m)
        
        sc = context.scene
        depsgraph = bpy.context.evaluated_depsgraph_get()
        # make target
        tmp_mesh = o.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph, )
        target_mesh = tmp_mesh.copy()
        target_mesh.name = 'target_mesh_{}'.format(pcv.uuid)
        target_mesh.transform(o.matrix_world.copy())
        target_mesh.transform(m.inverted())
        view_layer = context.view_layer
        collection = view_layer.active_layer_collection.collection
        target = bpy.data.objects.new('target_mesh_{}'.format(pcv.uuid), target_mesh)
        collection.objects.link(target)
        # still no idea, have to read about it..
        depsgraph.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        
        # use object bounding box for fast check if point can even be inside/outside of mesh and then use ray casting etc..
        bounds = [bv[:] for bv in target.bound_box]
        xmin = min([v[0] for v in bounds])
        xmax = max([v[0] for v in bounds])
        ymin = min([v[1] for v in bounds])
        ymax = max([v[1] for v in bounds])
        zmin = min([v[2] for v in bounds])
        zmax = max([v[2] for v in bounds])
        
        def is_in_bound_box(v):
            x = False
            if(xmin < v[0] < xmax):
                x = True
            y = False
            if(ymin < v[1] < ymax):
                y = True
            z = False
            if(zmin < v[2] < zmax):
                z = True
            if(x and y and z):
                return True
            return False
        
        # v1 raycasting in three axes and counting hits
        def is_point_inside_mesh_v1(p, o, ):
            axes = [Vector((1.0, 0.0, 0.0)), Vector((0.0, 1.0, 0.0)), Vector((0.0, 0.0, 1.0)), ]
            r = False
            for a in axes:
                v = p
                c = 0
                while True:
                    _, l, n, i = o.ray_cast(v, v + a * 10000.0)
                    if(i == -1):
                        break
                    c += 1
                    v = l + a * 0.00001
                if(c % 2 == 0):
                    r = True
                    break
            return not r
        
        # v2 raycasting and counting hits
        def is_point_inside_mesh_v2(p, o, shift=0.000001, search_distance=10000.0, ):
            p = Vector(p)
            path = []
            hits = 0
            direction = Vector((0.0, 0.0, 1.0))
            opposite_direction = direction.copy()
            opposite_direction.negate()
            path.append(p)
            loc = p
            ok = True
            
            def shift_vector(co, no, v):
                return co + (no.normalized() * v)
            
            while(ok):
                end = shift_vector(loc, direction, search_distance)
                loc = shift_vector(loc, direction, shift)
                _, loc, nor, ind = o.ray_cast(loc, end)
                if(ind != -1):
                    a = shift_vector(loc, opposite_direction, shift)
                    path.append(a)
                    hits += 1
                else:
                    ok = False
            if(hits % 2 == 1):
                return True
            return False
        
        # v3 closest point on mesh normal checking
        def is_point_inside_mesh_v3(p, o, ):
            _, loc, nor, ind = o.closest_point_on_mesh(p)
            if(ind != -1):
                v = loc - p
                d = v.dot(nor)
                if(d >= 0):
                    return True
            return False
        
        indexes = []
        prgs = Progress(len(vs), indent=1, prefix="> ")
        for i, v in enumerate(vs):
            prgs.step()
            vv = Vector(v)
            '''
            inside1 = is_point_inside_mesh_v1(vv, target, )
            inside2 = is_point_inside_mesh_v2(vv, target, )
            inside3 = is_point_inside_mesh_v3(vv, target, )
            # # intersect
            # if(not inside1 and not inside2 and not inside3):
            #     indexes.append(i)
            # exclude
            if(inside1 and inside2 and inside3):
                indexes.append(i)
            '''
            
            in_bb = is_in_bound_box(v)
            if(in_bb):
                # indexes.append(i)
                # continue
                
                # is in bound so i can check further
                inside3 = is_point_inside_mesh_v3(vv, target, )
                if(inside3):
                    indexes.append(i)
            
            # inside3 = is_point_inside_mesh_v3(vv, target, )
            # if(inside3):
            #     indexes.append(i)
        
        c = PCVManager.cache[pcv.uuid]
        vs = c['vertices']
        ns = c['normals']
        cs = c['colors']
        vs = np.delete(vs, indexes, axis=0, )
        ns = np.delete(ns, indexes, axis=0, )
        cs = np.delete(cs, indexes, axis=0, )
        
        log("removed: {} points".format(len(indexes)), 1)
        
        # cleanup
        collection.objects.unlink(target)
        bpy.data.objects.remove(target)
        bpy.data.meshes.remove(target_mesh)
        o.to_mesh_clear()
        
        # put to cache..
        pcv = context.object.point_cloud_visualizer
        PCVManager.update(pcv.uuid, vs, ns, cs, )
        
        _d = datetime.timedelta(seconds=time.time() - _t)
        log("completed in {}.".format(_d), 1)
        
        return {'FINISHED'}


class PCV_OT_reload(Operator):
    bl_idname = "point_cloud_visualizer.reload"
    bl_label = "Reload"
    bl_description = "Reload points from file"
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        
        pcv = context.object.point_cloud_visualizer
        # if(pcv.filepath != '' and pcv.uuid != '' and not pcv.runtime):
        if(pcv.filepath != '' and pcv.uuid != ''):
            return True
        return False
    
    def execute(self, context):
        pcv = context.object.point_cloud_visualizer
        
        draw = False
        for k, v in PCVManager.cache.items():
            if(v['uuid'] == pcv.uuid):
                if(v['ready']):
                    if(v['draw']):
                        draw = True
                        bpy.ops.point_cloud_visualizer.erase()
        
        if(pcv.runtime):
            c = PCVManager.cache[pcv.uuid]
            points = c['points']
            vs = np.column_stack((points['x'], points['y'], points['z'], ))
            ns = np.column_stack((points['nx'], points['ny'], points['nz'], ))
            cs = c['colors_original']
            PCVManager.update(pcv.uuid, vs, ns, cs, )
        else:
            bpy.ops.point_cloud_visualizer.load_ply_to_cache(filepath=pcv.filepath)
        
        if(draw):
            bpy.ops.point_cloud_visualizer.draw()
        
        return {'FINISHED'}


class PCV_OT_sequence_preload(Operator):
    bl_idname = "point_cloud_visualizer.sequence_preload"
    bl_label = "Preload Sequence"
    bl_description = "Preload sequence of PLY files. Files should be numbered starting at 1. Missing files in sequence will be skipped."
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        
        pcv = context.object.point_cloud_visualizer
        if(pcv.uuid in PCVSequence.cache.keys()):
            return False
        ok = False
        for k, v in PCVManager.cache.items():
            if(v['uuid'] == pcv.uuid):
                if(v['ready']):
                    if(v['draw']):
                        ok = True
        return ok
    
    def execute(self, context):
        log('Preload Sequence..')
        pcv = context.object.point_cloud_visualizer
        
        # pcv.sequence_enabled = True
        
        dirpath = os.path.dirname(pcv.filepath)
        files = []
        for (p, ds, fs) in os.walk(dirpath):
            files.extend(fs)
            break
        
        fs = [f for f in files if f.lower().endswith('.ply')]
        f = os.path.split(pcv.filepath)[1]
        
        pattern = re.compile(r'(\d+)(?!.*(\d+))')
        
        m = re.search(pattern, f, )
        if(m is not None):
            prefix = f[:m.start()]
            suffix = f[m.end():]
        else:
            self.report({'ERROR'}, 'Filename does not contain any sequence number')
            return {'CANCELLED'}
        
        sel = []
        
        for n in fs:
            m = re.search(pattern, n, )
            if(m is not None):
                # some numbers present, lets compare with selected file prefix/suffix
                pre = n[:m.start()]
                if(pre == prefix):
                    # prefixes match
                    suf = n[m.end():]
                    if(suf == suffix):
                        # suffixes match, extract number
                        si = n[m.start():m.end()]
                        try:
                            # try convert it to integer
                            i = int(si)
                            # and store as selected file
                            sel.append((i, n))
                        except ValueError:
                            pass
        
        # sort by sequence number
        sel.sort()
        # fabricate list with missing sequence numbers as None
        sequence = [[None] for i in range(sel[-1][0])]
        for i, n in sel:
            sequence[i - 1] = (i, n)
        for i in range(len(sequence)):
            if(sequence[i][0] is None):
                sequence[i] = [i, None]
        
        log('found files:', 1)
        for i, n in sequence:
            log('{}: {}'.format(i, n), 2)
        
        log('preloading..', 1)
        # this is our sequence with matching filenames, sorted by numbers with missing as None, now load it all..
        cache = []
        for i, n in sequence:
            if(n is not None):
                p = os.path.join(dirpath, n)
                points = []
                try:
                    points = PlyPointCloudReader(p).points
                except Exception as e:
                    self.report({'ERROR'}, str(e))
                if(len(points) == 0):
                    self.report({'ERROR'}, "No vertices loaded from file at {}".format(p))
                else:
                    if(not set(('x', 'y', 'z')).issubset(points.dtype.names)):
                        self.report({'ERROR'}, "Loaded data seems to miss vertex locations.")
                        return {'CANCELLED'}
                    
                    vs = np.column_stack((points['x'], points['y'], points['z'], ))
                    vs = vs.astype(np.float32)
                    
                    if(not set(('nx', 'ny', 'nz')).issubset(points.dtype.names)):
                        ns = None
                    else:
                        ns = np.column_stack((points['nx'], points['ny'], points['nz'], ))
                        ns = ns.astype(np.float32)
                    if(not set(('red', 'green', 'blue')).issubset(points.dtype.names)):
                        cs = None
                    else:
                        cs = np.column_stack((points['red'] / 255, points['green'] / 255, points['blue'] / 255, np.ones(len(points), dtype=float, ), ))
                        cs = cs.astype(np.float32)
                    
                    cache.append({'index': i,
                                  'name': n,
                                  'path': p,
                                  'vs': vs,
                                  'ns': ns,
                                  'cs': cs,
                                  'points': points, })
        
        log('...', 1)
        log('loaded {} item(s)'.format(len(cache)), 1)
        log('initializing..', 1)
        
        PCVSequence.init()
        
        ci = {'data': cache,
              'uuid': pcv.uuid,
              'pcv': pcv, }
        PCVSequence.cache[pcv.uuid] = ci
        
        log('force frame update..', 1)
        sc = bpy.context.scene
        cf = sc.frame_current
        sc.frame_current = cf
        log('done.', 1)
        
        return {'FINISHED'}


class PCV_OT_sequence_clear(Operator):
    bl_idname = "point_cloud_visualizer.sequence_clear"
    bl_label = "Clear Sequence"
    bl_description = "Clear preloaded sequence cache and reset all"
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        
        pcv = context.object.point_cloud_visualizer
        if(pcv.uuid in PCVSequence.cache.keys()):
            return True
        return False
        # ok = False
        # for k, v in PCVManager.cache.items():
        #     if(v['uuid'] == pcv.uuid):
        #         if(v['ready']):
        #             if(v['draw']):
        #                 ok = True
        # return ok
    
    def execute(self, context):
        pcv = context.object.point_cloud_visualizer
        
        del PCVSequence.cache[pcv.uuid]
        if(len(PCVSequence.cache.items()) == 0):
            PCVSequence.deinit()
        
        # c = PCVManager.cache[pcv.uuid]
        # vs = c['vertices']
        # ns = c['normals']
        # cs = c['colors']
        # PCVManager.update(pcv.uuid, vs, ns, cs, )
        
        bpy.ops.point_cloud_visualizer.reload()
        
        return {'FINISHED'}


class PCV_OT_seq_init(Operator):
    bl_idname = "point_cloud_visualizer.seq_init"
    bl_label = "seq_init"
    
    def execute(self, context):
        PCVSequence.init()
        context.area.tag_redraw()
        return {'FINISHED'}


class PCV_OT_seq_deinit(Operator):
    bl_idname = "point_cloud_visualizer.seq_deinit"
    bl_label = "seq_deinit"
    
    def execute(self, context):
        PCVSequence.deinit()
        context.area.tag_redraw()
        return {'FINISHED'}


class PCV_OT_generate_point_cloud(Operator):
    bl_idname = "point_cloud_visualizer.generate_from_mesh"
    bl_label = "Generate"
    bl_description = "Generate colored point cloud from mesh (or object convertible to mesh)"
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        
        # pcv = context.object.point_cloud_visualizer
        # if(pcv.uuid in PCVSequence.cache.keys()):
        #     return False
        # ok = False
        # for k, v in PCVManager.cache.items():
        #     if(v['uuid'] == pcv.uuid):
        #         if(v['ready']):
        #             if(v['draw']):
        #                 ok = True
        # return ok
        
        return True
    
    def execute(self, context):
        log("Generate From Mesh:", 0)
        _t = time.time()
        
        o = context.object
        
        if(o.type not in ('MESH', 'CURVE', 'SURFACE', 'FONT', )):
            self.report({'ERROR'}, "Object does not have geometry data.")
            return {'CANCELLED'}
        
        pcv = o.point_cloud_visualizer
        
        if(pcv.generate_source not in ('SURFACE', 'VERTICES', 'PARTICLES', )):
            self.report({'ERROR'}, "Source not implemented.")
            return {'CANCELLED'}
        
        n = pcv.generate_number_of_points
        r = random.Random(pcv.generate_seed)
        
        if(pcv.generate_colors in ('CONSTANT', 'UVTEX', 'VCOLS', 'GROUP_MONO', 'GROUP_COLOR', )):
            if(o.type in ('CURVE', 'SURFACE', 'FONT', ) and pcv.generate_colors != 'CONSTANT'):
                self.report({'ERROR'}, "Object type does not support UV textures, vertex colors or vertex groups.")
                return {'CANCELLED'}
        else:
            self.report({'ERROR'}, "Color generation not implemented.")
            return {'CANCELLED'}
        
        if(o.type != 'MESH'):
            vcols = None
            uvtex = None
            vgroup = None
        else:
            # all of following should return None is not available, at least for mesh object
            vcols = o.data.vertex_colors.active
            uvtex = o.data.uv_layers.active
            vgroup = o.vertex_groups.active
        
        if(pcv.generate_source == 'VERTICES'):
            try:
                sampler = PCVVertexSampler(context, o,
                                           colorize=pcv.generate_colors,
                                           constant_color=pcv.generate_constant_color,
                                           vcols=vcols, uvtex=uvtex, vgroup=vgroup, )
            except Exception as e:
                self.report({'ERROR'}, str(e), )
                return {'CANCELLED'}
        elif(pcv.generate_source == 'SURFACE'):
            if(pcv.generate_algorithm == 'WEIGHTED_RANDOM_IN_TRIANGLE'):
                try:
                    sampler = PCVTriangleSurfaceSampler(context, o, n, r,
                                                        colorize=pcv.generate_colors,
                                                        constant_color=pcv.generate_constant_color,
                                                        vcols=vcols, uvtex=uvtex, vgroup=vgroup,
                                                        exact_number_of_points=pcv.generate_exact_number_of_points, )
                except Exception as e:
                    self.report({'ERROR'}, str(e), )
                    return {'CANCELLED'}
            elif(pcv.generate_algorithm == 'POISSON_DISK_SAMPLING'):
                try:
                    sampler = PCVPoissonDiskSurfaceSampler(context, o, r, minimal_distance=pcv.generate_minimal_distance,
                                                           sampling_exponent=pcv.generate_sampling_exponent,
                                                           colorize=pcv.generate_colors,
                                                           constant_color=pcv.generate_constant_color,
                                                           vcols=vcols, uvtex=uvtex, vgroup=vgroup, )
                except Exception as e:
                    self.report({'ERROR'}, str(e), )
                    return {'CANCELLED'}
            else:
                self.report({'ERROR'}, "Algorithm not implemented.")
                return {'CANCELLED'}
            
        elif(pcv.generate_source == 'PARTICLES'):
            try:
                alive_only = True
                if(pcv.generate_source_psys == 'ALL'):
                    alive_only = False
                sampler = PCVParticleSystemSampler(context, o, alive_only=alive_only,
                                                   colorize=pcv.generate_colors,
                                                   constant_color=pcv.generate_constant_color,
                                                   vcols=vcols, uvtex=uvtex, vgroup=vgroup, )
            except Exception as e:
                self.report({'ERROR'}, str(e), )
                return {'CANCELLED'}
        else:
            self.report({'ERROR'}, "Source type not implemented.")
            return {'CANCELLED'}
        
        vs = sampler.vs
        ns = sampler.ns
        cs = sampler.cs
        
        log("generated {} points.".format(len(vs)), 1)
        
        ok = False
        for k, v in PCVManager.cache.items():
            if(v['uuid'] == pcv.uuid):
                if(v['ready']):
                    if(v['draw']):
                        ok = True
        if(ok):
            bpy.ops.point_cloud_visualizer.erase()
        
        c = PCVControl(o)
        c.draw(vs, ns, cs)
        
        if(debug_mode()):
            o.display_type = 'BOUNDS'
        
        _d = datetime.timedelta(seconds=time.time() - _t)
        log("completed in {}.".format(_d), 1)
        
        return {'FINISHED'}


class PCV_OT_reset_runtime(Operator):
    bl_idname = "point_cloud_visualizer.reset_runtime"
    bl_label = "Reset Runtime"
    bl_description = "Reset PCV to its default state if in runtime mode (displayed data is set with python and not with ui)"
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        pcv = context.object.point_cloud_visualizer
        if(pcv.runtime):
            return True
        return False
    
    def execute(self, context):
        o = context.object
        c = PCVControl(o)
        c.erase()
        c.reset()
        return {'FINISHED'}


class PCV_OT_generate_volume_point_cloud(Operator):
    bl_idname = "point_cloud_visualizer.generate_volume_from_mesh"
    bl_label = "Generate Volume"
    bl_description = "Generate colored point cloud in mesh (or object convertible to mesh) volume"
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        
        # pcv = context.object.point_cloud_visualizer
        # if(pcv.uuid in PCVSequence.cache.keys()):
        #     return False
        # ok = False
        # for k, v in PCVManager.cache.items():
        #     if(v['uuid'] == pcv.uuid):
        #         if(v['ready']):
        #             if(v['draw']):
        #                 ok = True
        # return ok
        
        return True
    
    def execute(self, context):
        log("Generate From Mesh:", 0)
        _t = time.time()
        
        o = context.object
        pcv = o.point_cloud_visualizer
        n = pcv.generate_number_of_points
        r = random.Random(pcv.generate_seed)
        g = PCVRandomVolumeSampler(o, n, r, )
        vs = g.vs
        ns = g.ns
        cs = g.cs
        
        log("generated {} points.".format(len(vs)), 1)
        
        ok = False
        for k, v in PCVManager.cache.items():
            if(v['uuid'] == pcv.uuid):
                if(v['ready']):
                    if(v['draw']):
                        ok = True
        if(ok):
            bpy.ops.point_cloud_visualizer.erase()
        
        c = PCVControl(o)
        c.draw(vs, ns, cs)
        
        if(debug_mode()):
            o.display_type = 'BOUNDS'
        
        _d = datetime.timedelta(seconds=time.time() - _t)
        log("completed in {}.".format(_d), 1)
        
        return {'FINISHED'}


class PCV_OT_color_adjustment_shader_reset(Operator):
    bl_idname = "point_cloud_visualizer.color_adjustment_shader_reset"
    bl_label = "Reset"
    bl_description = "Reset color adjustment values"
    
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
                        if(pcv.color_adjustment_shader_enabled):
                            ok = True
        return ok
    
    def execute(self, context):
        pcv = context.object.point_cloud_visualizer
        
        pcv.color_adjustment_shader_exposure = 0.0
        pcv.color_adjustment_shader_gamma = 1.0
        pcv.color_adjustment_shader_brightness = 0.0
        pcv.color_adjustment_shader_contrast = 1.0
        pcv.color_adjustment_shader_hue = 0.0
        pcv.color_adjustment_shader_saturation = 0.0
        pcv.color_adjustment_shader_value = 0.0
        pcv.color_adjustment_shader_invert = False
        
        for area in bpy.context.screen.areas:
            if(area.type == 'VIEW_3D'):
                area.tag_redraw()
        
        return {'FINISHED'}


class PCV_OT_color_adjustment_shader_apply(Operator):
    bl_idname = "point_cloud_visualizer.color_adjustment_shader_apply"
    bl_label = "Apply"
    bl_description = "Apply color adjustments to points, reset and exit"
    
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
                        if(pcv.color_adjustment_shader_enabled):
                            ok = True
        return ok
    
    def execute(self, context):
        pcv = context.object.point_cloud_visualizer
        
        c = PCVManager.cache[pcv.uuid]
        vs = c['vertices']
        ns = c['normals']
        cs = c['colors']
        
        cs = cs * (2 ** pcv.color_adjustment_shader_exposure)
        cs = np.clip(cs, 0.0, 1.0, )
        cs = cs ** (1 / pcv.color_adjustment_shader_gamma)
        cs = np.clip(cs, 0.0, 1.0, )
        cs = (cs - 0.5) * pcv.color_adjustment_shader_contrast + 0.5 + pcv.color_adjustment_shader_brightness
        cs = np.clip(cs, 0.0, 1.0, )
        
        h = pcv.color_adjustment_shader_hue
        s = pcv.color_adjustment_shader_saturation
        v = pcv.color_adjustment_shader_value
        if(h > 1.0):
            h = h % 1.0
        for _i, ca in enumerate(cs):
            col = Color(ca[:3])
            _h, _s, _v = col.hsv
            _h = (_h + h) % 1.0
            _s += s
            _v += v
            col.hsv = (_h, _s, _v)
            cs[_i][0] = col.r
            cs[_i][1] = col.g
            cs[_i][2] = col.b
        cs = np.clip(cs, 0.0, 1.0, )
        
        if(pcv.color_adjustment_shader_invert):
            cs = 1.0 - cs
        cs = np.clip(cs, 0.0, 1.0, )
        
        bpy.ops.point_cloud_visualizer.color_adjustment_shader_reset()
        pcv.color_adjustment_shader_enabled = False
        
        if('extra' in c.keys()):
            del c['extra']
        
        PCVManager.update(pcv.uuid, vs, ns, cs, )
        
        return {'FINISHED'}


class PCV_OT_clip_planes_from_bbox(Operator):
    bl_idname = "point_cloud_visualizer.clip_planes_from_bbox"
    bl_label = "Set Clip Planes From Object Bounding Box"
    bl_description = "Set clip planes from object bounding box"
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        
        pcv = context.object.point_cloud_visualizer
        ok = False
        
        bbo = pcv.clip_planes_from_bbox_object
        if(bbo is not None):
            ok = True
        
        return ok
    
    def execute(self, context):
        pcv = context.object.point_cloud_visualizer
        bbo = pcv.clip_planes_from_bbox_object
        
        mw = bbo.matrix_world
        vs = [mw @ Vector(v) for v in bbo.bound_box]
        
        # 0 front left down
        # 1 front left up
        # 2 back left up
        # 3 back left down
        # 4 front right down
        # 5 front right up
        # 6 back right up
        # 7 back right down
        #          2-----------6
        #         /           /|
        #       1-----------5  |
        #       |           |  |
        #       |           |  |
        #       |  3        |  7
        #       |           | /
        #       0-----------4
        
        fs = (
            (0, 4, 5, 1),  # front
            (3, 2, 6, 7),  # back
            (1, 5, 6, 2),  # top
            (0, 3, 7, 4),  # bottom
            (0, 1, 2, 3),  # left
            (4, 7, 6, 5),  # right
        )
        
        quads = [[vs[fs[i][j]] for j in range(4)] for i in range(6)]
        ns = [mathutils.geometry.normal(quads[i]) for i in range(6)]
        for i in range(6):
            # FIXME: if i need to do this, it is highly probable i have something wrong.. somewhere..
            ns[i].negate()
        
        ds = []
        for i in range(6):
            v = quads[i][0]
            n = ns[i]
            d = mathutils.geometry.distance_point_to_plane(Vector(), v, n)
            ds.append(d)
        
        a = [ns[i].to_tuple() + (ds[i], ) for i in range(6)]
        pcv.clip_plane0 = a[0]
        pcv.clip_plane1 = a[1]
        pcv.clip_plane2 = a[2]
        pcv.clip_plane3 = a[3]
        pcv.clip_plane4 = a[4]
        pcv.clip_plane5 = a[5]
        
        pcv.clip_shader_enabled = True
        pcv.clip_plane0_enabled = True
        pcv.clip_plane1_enabled = True
        pcv.clip_plane2_enabled = True
        pcv.clip_plane3_enabled = True
        pcv.clip_plane4_enabled = True
        pcv.clip_plane5_enabled = True
        
        return {'FINISHED'}


class PCV_OT_clip_planes_reset(Operator):
    bl_idname = "point_cloud_visualizer.clip_planes_reset"
    bl_label = "Reset Clip Planes"
    bl_description = "Reset all clip planes"
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        
        pcv = context.object.point_cloud_visualizer
        ok = True
        
        return ok
    
    def execute(self, context):
        pcv = context.object.point_cloud_visualizer
        
        pcv.clip_planes_from_bbox_object = None
        
        z = (0.0, 0.0, 0.0, 0.0, )
        pcv.clip_plane0 = z
        pcv.clip_plane1 = z
        pcv.clip_plane2 = z
        pcv.clip_plane3 = z
        pcv.clip_plane4 = z
        pcv.clip_plane5 = z
        
        pcv.clip_shader_enabled = False
        pcv.clip_plane0_enabled = False
        pcv.clip_plane1_enabled = False
        pcv.clip_plane2_enabled = False
        pcv.clip_plane3_enabled = False
        pcv.clip_plane4_enabled = False
        pcv.clip_plane5_enabled = False
        
        return {'FINISHED'}


class PCV_OT_clip_planes_from_camera_view(Operator):
    bl_idname = "point_cloud_visualizer.clip_planes_from_camera_view"
    bl_label = "Set Clip Planes From Camera View"
    bl_description = "Set clip planes from active camera view"
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        
        pcv = context.object.point_cloud_visualizer
        ok = False
        
        s = bpy.context.scene
        o = s.camera
        if(o):
            ok = True
        
        return ok
    
    def execute(self, context):
        pcv = context.object.point_cloud_visualizer
        
        o = bpy.context.object
        
        s = bpy.context.scene
        c = s.camera
        cd = c.data
        rx = s.render.resolution_x
        ry = s.render.resolution_y
        w = 0.5 * cd.sensor_width / cd.lens
        if(rx > ry):
            x = w
            y = w * ry / rx
        else:
            x = w * rx / ry
            y = w
        
        lr = Vector((x, -y, -1.0, ))
        ur = Vector((x, y, -1.0, ))
        ll = Vector((-x, -y, -1.0, ))
        ul = Vector((-x, y, -1.0, ))
        
        z = Vector()
        n0 = mathutils.geometry.normal((z, lr, ur))
        n1 = mathutils.geometry.normal((z, ur, ul))
        n2 = mathutils.geometry.normal((z, ul, ll))
        n3 = mathutils.geometry.normal((z, ll, lr))
        
        n0.negate()
        n1.negate()
        n2.negate()
        n3.negate()
        
        m = c.matrix_world
        l, r, _ = m.decompose()
        rm = r.to_matrix().to_4x4()
        lrm = Matrix.Translation(l).to_4x4() @ rm
        
        omi = o.matrix_world.inverted()
        _, r, _ = omi.decompose()
        orm = r.to_matrix().to_4x4()
        
        n0 = orm @ rm @ n0
        n1 = orm @ rm @ n1
        n2 = orm @ rm @ n2
        n3 = orm @ rm @ n3
        
        v0 = omi @ lrm @ lr
        v1 = omi @ lrm @ ur
        v2 = omi @ lrm @ ul
        v3 = omi @ lrm @ ll
        
        d0 = mathutils.geometry.distance_point_to_plane(Vector(), v0, n0)
        d1 = mathutils.geometry.distance_point_to_plane(Vector(), v1, n1)
        d2 = mathutils.geometry.distance_point_to_plane(Vector(), v2, n2)
        d3 = mathutils.geometry.distance_point_to_plane(Vector(), v3, n3)
        
        # TODO: add plane behind camera (not much needed anyway, but for consistency), but more important, add plane in clipping distance set on camera
        # TODO: ORTHO camera does not work
        
        pcv.clip_plane0 = n0.to_tuple() + (d0, )
        pcv.clip_plane1 = n1.to_tuple() + (d1, )
        pcv.clip_plane2 = n2.to_tuple() + (d2, )
        pcv.clip_plane3 = n3.to_tuple() + (d3, )
        pcv.clip_plane4 = (0.0, 0.0, 0.0, 0.0, )
        pcv.clip_plane5 = (0.0, 0.0, 0.0, 0.0, )
        
        pcv.clip_shader_enabled = True
        pcv.clip_plane0_enabled = True
        pcv.clip_plane1_enabled = True
        pcv.clip_plane2_enabled = True
        pcv.clip_plane3_enabled = True
        pcv.clip_plane4_enabled = False
        pcv.clip_plane5_enabled = False
        
        return {'FINISHED'}


class PCV_PT_panel(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    # bl_category = "View"
    bl_category = "Point Cloud Visualizer"
    bl_label = "Point Cloud Visualizer"
    # bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        # o = context.active_object
        # if(o):
        #     return True
        # return False
        return True
    
    def draw(self, context):
        o = context.active_object
        if(not o):
            self.layout.label(text='Select an object..', icon='ERROR', )
            return
        
        pcv = context.object.point_cloud_visualizer
        l = self.layout
        sub = l.column()
        
        if(pcv.instance_visualizer_active_hidden_value):
            r = sub.row()
            r.alert = True
            r.prop(pcv, 'instance_visualizer_active', toggle=True, icon='ERROR', )
            sub.separator()
        
        # edit mode, main pcv object panel
        if(pcv.edit_initialized):
            sub.label(text='PCV Edit in progress..', icon='ERROR', )
            sub.separator()
            sub.operator('point_cloud_visualizer.edit_cancel')
            return
        
        # edit mode, helper object panel
        if(pcv.edit_is_edit_mesh):
            sub.label(text='PCV Edit helper mesh', icon='INFO', )
            sub.separator()
            c = sub.column()
            c.label(text='• Transform, delete and duplicate vertices.')
            c.label(text='• Update button will refresh point cloud.')
            c.label(text='• End button will refresh point cloud and delete helper mesh.')
            c.label(text='• All other functions are disabled until finished.')
            c.scale_y = 0.66
            
            sub.separator()
            
            sub.prop(pcv, 'edit_overlay_alpha')
            sub.prop(pcv, 'edit_overlay_size')
            
            sub.separator()
            
            r = sub.row(align=True)
            r.operator('point_cloud_visualizer.edit_update')
            r.operator('point_cloud_visualizer.edit_end')
            
            if(context.mode != 'EDIT_MESH'):
                sub.label(text="Must be in Edit Mode", icon='ERROR', )
            
            sub.enabled = PCV_OT_edit_update.poll(context)
            
            return
        
        # ----------->>> file selector
        def prop_name(cls, prop, colon=False, ):
            for p in cls.bl_rna.properties:
                if(p.identifier == prop):
                    if(colon):
                        return "{}:".format(p.name)
                    return p.name
            return ''
        
        # f = 0.275
        f = 0.33
        
        r = sub.row(align=True, )
        s = r.split(factor=f)
        s.label(text=prop_name(pcv, 'filepath', True, ))
        s = s.split(factor=1.0)
        r = s.row(align=True, )
        c = r.column(align=True)
        c.prop(pcv, 'filepath', text='', )
        c.enabled = False
        r.operator('point_cloud_visualizer.load_ply_to_cache', icon='FILEBROWSER', text='', )
        
        r.operator('point_cloud_visualizer.reload', icon='FILE_REFRESH', text='', )
        
        # <<<----------- file selector
        
        # ----------->>> info block
        def human_readable_number(num, suffix='', ):
            # https://stackoverflow.com/questions/1094841/reusable-library-to-get-human-readable-version-of-file-size
            f = 1000.0
            for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z', ]:
                if(abs(num) < f):
                    return "{:3.1f}{}{}".format(num, unit, suffix)
                num /= f
            return "{:.1f}{}{}".format(num, 'Y', suffix)
        
        l0c0 = "Selected: "
        l0c1 = "{}".format("n/a")
        l1c0 = "Displayed: "
        # l1c1 = "{} of {}".format("0.0", "n/a")
        l1c1 = "{}".format("n/a")
        
        if(pcv.filepath != ""):
            _, t = os.path.split(pcv.filepath)
            l0c1 = "{}".format(t)
            if(pcv.uuid in PCVManager.cache):
                l0c0 = "Loaded: "
                l0c1 = "{}".format(t)
                cache = PCVManager.cache[pcv.uuid]
                
                n = human_readable_number(cache['display_length'])
                # don't use it when less or equal to 999
                if(cache['display_length'] < 1000):
                    n = str(cache['display_length'])
                
                if(not cache['draw']):
                    # n = "0.0"
                    n = "0"
                nn = human_readable_number(cache['stats'])
                if(nn.endswith('.0')):
                    nn = nn[:-2]
                l1c1 = "{} of {}".format(n, nn)
        
        f = 0.33
        c = sub.column()
        c.scale_y = 0.66
        r = c.row()
        s = r.split(factor=f)
        s.label(text=l0c0)
        s = s.split(factor=1.0)
        s.label(text=l0c1)
        r = c.row()
        s = r.split(factor=f)
        s.label(text=l1c0)
        s = s.split(factor=1.0)
        s.label(text=l1c1)
        
        sub.separator()
        # <<<----------- info block
        
        e = not (pcv.filepath == "")
        r = sub.row(align=True)
        r.operator('point_cloud_visualizer.draw')
        r.operator('point_cloud_visualizer.erase')
        r.scale_y = 1.5
        r.enabled = e
        r = sub.row()
        r.prop(pcv, 'display_percent')
        r.enabled = e
        r = sub.row()
        r.prop(pcv, 'point_size')
        r.enabled = e
        
        r = sub.row()
        r.prop(pcv, 'global_alpha')
        r.enabled = e
        
        r = sub.row(align=True)
        r.prop(pcv, 'vertex_normals', toggle=True, icon_only=True, icon='SNAP_NORMAL', )
        r.prop(pcv, 'vertex_normals_size')
        r.enabled = e
        if(not pcv.has_normals):
            r.enabled = False
        
        pcv = context.object.point_cloud_visualizer
        ok = False
        zero_length = False
        for k, v in PCVManager.cache.items():
            if(v['uuid'] == pcv.uuid):
                if(v['ready']):
                    if(v['draw']):
                        ok = True
                    if(len(v['vertices']) == 0):
                        zero_length = True
        
        if(ok):
            if(not pcv.has_normals):
                if(not zero_length):
                    sub.label(text="Missing vertex normals.", icon='ERROR', )
        
        c = sub.column()
        r = c.row(align=True)
        r.prop(pcv, 'illumination', toggle=True, )
        r.prop(pcv, 'illumination_edit', toggle=True, icon_only=True, icon='TOOL_SETTINGS', )
        # r.prop(pcv, 'illumination_edit', toggle=True, icon_only=True, icon='SETTINGS', )
        if(ok):
            if(not pcv.has_normals):
                # c.label(text="Missing vertex normals.", icon='ERROR', )
                c.enabled = False
        else:
            c.enabled = False
        if(pcv.illumination_edit):
            cc = c.column()
            cc.prop(pcv, 'light_direction', text="", )
            ccc = cc.column(align=True)
            ccc.prop(pcv, 'light_intensity')
            ccc.prop(pcv, 'shadow_intensity')
            if(not pcv.has_normals):
                cc.enabled = e
            
            sub.separator()
        
        # # other shaders
        # e = ok
        # c = sub.column()
        # r = c.row(align=True)
        # r.prop(pcv, 'dev_depth_enabled', toggle=True, )
        # r.prop(pcv, 'dev_depth_edit', toggle=True, icon_only=True, icon='TOOL_SETTINGS', )
        # if(pcv.dev_depth_edit):
        #     cc = c.column(align=True)
        #     cc.prop(pcv, 'dev_depth_brightness')
        #     cc.prop(pcv, 'dev_depth_contrast')
        #     c.prop(pcv, 'dev_depth_false_colors')
        #     r = c.row(align=True)
        #     r.prop(pcv, 'dev_depth_color_a', text="", )
        #     r.prop(pcv, 'dev_depth_color_b', text="", )
        #     r.enabled = pcv.dev_depth_false_colors
        #
        #     sub.separator()
        # c.enabled = e
        #
        # c = sub.column()
        # c.prop(pcv, 'dev_normal_colors_enabled', toggle=True, )
        # c.enabled = e
        #
        # c = sub.column()
        # c.prop(pcv, 'dev_position_colors_enabled', toggle=True, )
        # c.enabled = e
        
        # other shaders
        c = sub.column()
        c.enabled = ok
        r = c.row(align=True)
        r.prop(pcv, 'dev_depth_enabled', toggle=True, )
        # r.prop(pcv, 'dev_normal_colors_enabled', toggle=True, )
        cc = r.column(align=True)
        cc.prop(pcv, 'dev_normal_colors_enabled', toggle=True, )
        if(ok):
            if(not pcv.has_normals):
                cc.enabled = False
        r.prop(pcv, 'dev_position_colors_enabled', toggle=True, )
        
        # r = c.row(align=True)
        # r.prop(pcv, 'debug_shader', expand=True, )
        
        if(pcv.dev_depth_enabled):
            cc = c.column(align=True)
            cc.prop(pcv, 'dev_depth_brightness')
            cc.prop(pcv, 'dev_depth_contrast')
            c.prop(pcv, 'dev_depth_false_colors')
            r = c.row(align=True)
            r.prop(pcv, 'dev_depth_color_a', text="", )
            r.prop(pcv, 'dev_depth_color_b', text="", )
            r.enabled = pcv.dev_depth_false_colors
            # sub.separator()
        if(pcv.dev_normal_colors_enabled):
            pass
        if(pcv.dev_position_colors_enabled):
            pass
        
        # r = c.row(align=True)
        # r.prop(pcv, 'dev_bbox_enabled', toggle=True, icon='SHADING_BBOX', text="", icon_only=True, )
        # if(pcv.dev_bbox_enabled):
        #     r.prop(pcv, 'dev_bbox_color', text="", )
        #     r = c.row(align=True)
        #     r.prop(pcv, 'dev_bbox_size')
        #     r.prop(pcv, 'dev_bbox_alpha')
        
        # c.prop(pcv, 'dev_bbox_enabled', toggle=True, )
        # if(pcv.dev_bbox_enabled):
        #     r = c.row()
        #     r.prop(pcv, 'dev_bbox_color', text="", )
        #     c.prop(pcv, 'dev_bbox_size')
        #     c.prop(pcv, 'dev_bbox_alpha')


class PCV_PT_render(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "View"
    bl_label = "Render"
    bl_parent_id = "PCV_PT_panel"
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
        return True
    
    def draw(self, context):
        pcv = context.object.point_cloud_visualizer
        l = self.layout
        sub = l.column()
        
        c = sub.column()
        c.prop(pcv, 'render_display_percent')
        c.prop(pcv, 'render_point_size')
        c.prop(pcv, 'render_supersampling')
        r = c.row()
        r.prop(pcv, 'render_smoothstep')
        ok = False
        # if(not pcv.illumination and not pcv.override_default_shader):
        if(not pcv.override_default_shader):
            ok = True
        r.enabled = ok
        
        c = sub.column()
        
        f = 0.33
        r = sub.row(align=True, )
        s = r.split(factor=f)
        s.label(text='Output:')
        s = s.split(factor=1.0)
        r = s.row(align=True, )
        c = r.column(align=True)
        c.prop(pcv, 'render_path', text='', )
        
        r = sub.row(align=True)
        c0 = r.column(align=True)
        c0.prop(pcv, 'render_resolution_linked', toggle=True, text='', icon='LINKED' if pcv.render_resolution_linked else 'UNLINKED', icon_only=True, )
        c0.prop(pcv, 'render_resolution_linked', toggle=True, text='', icon='LINKED' if pcv.render_resolution_linked else 'UNLINKED', icon_only=True, )
        c0.prop(pcv, 'render_resolution_linked', toggle=True, text='', icon='LINKED' if pcv.render_resolution_linked else 'UNLINKED', icon_only=True, )
        c1 = r.column(align=True)
        if(pcv.render_resolution_linked):
            render = context.scene.render
            c1.prop(render, 'resolution_x')
            c1.prop(render, 'resolution_y')
            c1.prop(render, 'resolution_percentage')
            c1.active = False
        else:
            c1.prop(pcv, 'render_resolution_x')
            c1.prop(pcv, 'render_resolution_y')
            c1.prop(pcv, 'render_resolution_percentage')
        
        r = sub.row(align=True)
        r.operator('point_cloud_visualizer.render')
        r.operator('point_cloud_visualizer.render_animation')
        
        sub.enabled = PCV_OT_render.poll(context)


class PCV_PT_convert(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "View"
    bl_label = "Convert"
    bl_parent_id = "PCV_PT_panel"
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
        return True
    
    def draw(self, context):
        pcv = context.object.point_cloud_visualizer
        l = self.layout
        sub = l.column()
        c = sub.column()
        c.prop(pcv, 'mesh_type')
        
        f = 0.245
        r = c.row(align=True)
        s = r.split(factor=f, align=True, )
        s.prop(pcv, 'mesh_all', toggle=True, )
        s = s.split(factor=1.0, align=True, )
        s.prop(pcv, 'mesh_percentage')
        if(pcv.mesh_all):
            s.enabled = False
        
        cc = c.column()
        cc.prop(pcv, 'mesh_size')
        
        if(pcv.mesh_type in ('INSTANCER', 'PARTICLES', )):
            cc.prop(pcv, 'mesh_base_sphere_subdivisions')
        
        cc_n = cc.row()
        cc_n.prop(pcv, 'mesh_normal_align')
        if(not pcv.has_normals):
            cc_n.enabled = False
        
        cc_c = cc.row()
        cc_c.prop(pcv, 'mesh_vcols')
        if(not pcv.has_vcols):
            cc_c.enabled = False
        
        if(pcv.mesh_type == 'VERTEX'):
            cc.enabled = False
        
        # c.operator('point_cloud_visualizer.convert')
        
        r = c.row(align=True)
        r.operator('point_cloud_visualizer.convert')
        r.prop(pcv, 'mesh_use_instancer2', toggle=True, text='', icon='AUTO', )
        
        c.enabled = PCV_OT_convert.poll(context)


class PCV_PT_filter(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "View"
    bl_label = "Filter"
    bl_parent_id = "PCV_PT_panel"
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
        return True
    
    def draw(self, context):
        pcv = context.object.point_cloud_visualizer
        l = self.layout
        c = l.column()


class PCV_PT_filter_simplify(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "View"
    bl_label = "Simplify"
    bl_parent_id = "PCV_PT_filter"
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
        return True
    
    def draw(self, context):
        pcv = context.object.point_cloud_visualizer
        l = self.layout
        c = l.column()
        
        a = c.column(align=True)
        a.prop(pcv, 'filter_simplify_num_samples')
        a.prop(pcv, 'filter_simplify_num_candidates')
        
        c.operator('point_cloud_visualizer.filter_simplify')
        
        c.enabled = PCV_OT_filter_simplify.poll(context)


class PCV_PT_filter_project(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "View"
    bl_label = "Project"
    bl_parent_id = "PCV_PT_filter"
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
        return True
    
    def draw(self, context):
        pcv = context.object.point_cloud_visualizer
        l = self.layout
        c = l.column()
        c.prop(pcv, 'filter_project_object')
        
        a = c.column(align=True)
        a.prop(pcv, 'filter_project_search_distance')
        r = a.row(align=True)
        r.prop(pcv, 'filter_project_negative', toggle=True, )
        r.prop(pcv, 'filter_project_positive', toggle=True, )
        
        c.prop(pcv, 'filter_project_discard')
        
        cc = c.column(align=True)
        f = 0.5
        r = cc.row(align=True, )
        s = r.split(factor=f, align=True, )
        s.prop(pcv, 'filter_project_colorize', toggle=True, )
        s = s.split(factor=1.0, align=True, )
        r = s.row(align=True, )
        ccc = r.column(align=True)
        ccc.prop(pcv, 'filter_project_colorize_from', text="", )
        ccc.enabled = pcv.filter_project_colorize
        
        c.prop(pcv, 'filter_project_shift')
        c.operator('point_cloud_visualizer.filter_project')
        
        # conditions are the same, also `filter_project_object` has to be set
        c.enabled = PCV_OT_filter_simplify.poll(context)
        
        if(pcv.filepath != '' and pcv.uuid != ''):
            if(not pcv.has_normals):
                c.label(text="Missing vertex normals.", icon='ERROR', )
                c.enabled = False


class PCV_PT_filter_remove_color(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "View"
    bl_label = "Remove Color"
    bl_parent_id = "PCV_PT_filter"
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
        return True
    
    def draw(self, context):
        pcv = context.object.point_cloud_visualizer
        l = self.layout
        c = l.column()
        r = c.row()
        r.prop(pcv, 'filter_remove_color', text='', )
        
        a = c.column(align=True)
        r = a.row(align=True)
        r.prop(pcv, 'filter_remove_color_delta_hue_use', text='', toggle=True, icon_only=True, icon='CHECKBOX_HLT' if pcv.filter_remove_color_delta_hue_use else 'CHECKBOX_DEHLT', )
        cc = r.column(align=True)
        cc.prop(pcv, 'filter_remove_color_delta_hue')
        cc.active = pcv.filter_remove_color_delta_hue_use
        
        r = a.row(align=True)
        r.prop(pcv, 'filter_remove_color_delta_saturation_use', text='', toggle=True, icon_only=True, icon='CHECKBOX_HLT' if pcv.filter_remove_color_delta_saturation_use else 'CHECKBOX_DEHLT', )
        cc = r.column(align=True)
        cc.prop(pcv, 'filter_remove_color_delta_saturation')
        cc.active = pcv.filter_remove_color_delta_saturation_use
        
        r = a.row(align=True)
        r.prop(pcv, 'filter_remove_color_delta_value_use', text='', toggle=True, icon_only=True, icon='CHECKBOX_HLT' if pcv.filter_remove_color_delta_value_use else 'CHECKBOX_DEHLT', )
        cc = r.column(align=True)
        cc.prop(pcv, 'filter_remove_color_delta_value')
        cc.active = pcv.filter_remove_color_delta_value_use
        
        cc = c.column(align=True)
        r = cc.row(align=True)
        r.operator('point_cloud_visualizer.filter_remove_color')
        r.operator('point_cloud_visualizer.filter_remove_color_deselect', text="", icon='X', )
        cc.operator('point_cloud_visualizer.filter_remove_color_delete_selected')
        
        c.enabled = PCV_OT_filter_remove_color.poll(context)


class PCV_PT_filter_merge(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "View"
    bl_label = "Merge"
    bl_parent_id = "PCV_PT_filter"
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
        return True
    
    def draw(self, context):
        pcv = context.object.point_cloud_visualizer
        l = self.layout
        c = l.column()
        
        c.operator('point_cloud_visualizer.filter_merge')
        
        c.enabled = PCV_OT_filter_merge.poll(context)


class PCV_PT_filter_join(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "View"
    bl_label = "Join"
    bl_parent_id = "PCV_PT_filter"
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
        return True
    
    def draw(self, context):
        pcv = context.object.point_cloud_visualizer
        l = self.layout
        c = l.column()
        
        c.prop(pcv, 'filter_join_object')
        c.operator('point_cloud_visualizer.filter_join')
        
        ok = False
        for k, v in PCVManager.cache.items():
            if(v['uuid'] == pcv.uuid):
                if(v['ready']):
                    if(v['draw']):
                        ok = True
                break
        c.enabled = ok


class PCV_PT_filter_boolean(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "View"
    bl_label = "Boolean"
    bl_parent_id = "PCV_PT_filter"
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
        return True
    
    def draw(self, context):
        pcv = context.object.point_cloud_visualizer
        l = self.layout
        c = l.column()
        
        c.prop(pcv, 'filter_boolean_object')
        c.operator('point_cloud_visualizer.filter_boolean_intersect')
        c.operator('point_cloud_visualizer.filter_boolean_exclude')
        
        c.enabled = PCV_OT_filter_merge.poll(context)


class PCV_PT_filter_color_adjustment(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "View"
    bl_label = "Color Adjustment"
    bl_parent_id = "PCV_PT_filter"
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
        return True
    
    def draw(self, context):
        pcv = context.object.point_cloud_visualizer
        l = self.layout
        c = l.column()
        
        c.prop(pcv, 'color_adjustment_shader_enabled')
        cc = c.column(align=True)
        cc.prop(pcv, 'color_adjustment_shader_exposure')
        cc.prop(pcv, 'color_adjustment_shader_gamma')
        cc.prop(pcv, 'color_adjustment_shader_brightness')
        cc.prop(pcv, 'color_adjustment_shader_contrast')
        cc.prop(pcv, 'color_adjustment_shader_hue')
        cc.prop(pcv, 'color_adjustment_shader_saturation')
        cc.prop(pcv, 'color_adjustment_shader_value')
        cc.prop(pcv, 'color_adjustment_shader_invert')
        r = cc.row(align=True)
        r.operator('point_cloud_visualizer.color_adjustment_shader_reset')
        r.operator('point_cloud_visualizer.color_adjustment_shader_apply')
        cc.enabled = pcv.color_adjustment_shader_enabled
        
        c.enabled = PCV_OT_filter_merge.poll(context)


class PCV_PT_clip(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "View"
    bl_label = "Clip"
    bl_parent_id = "PCV_PT_panel"
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
        return True
    
    def draw(self, context):
        pcv = context.object.point_cloud_visualizer
        l = self.layout
        c = l.column()
        c.prop(pcv, 'clip_shader_enabled', toggle=True, text='Enable Clipping Planes Shader', )
        
        a = l.column()
        c = a.column(align=True)
        r = c.row(align=True)
        r.prop(pcv, 'clip_plane0_enabled', text='', toggle=True, icon_only=True, icon='HIDE_OFF' if pcv.clip_plane0_enabled else 'HIDE_ON', )
        r.prop(pcv, 'clip_plane0', )
        r = c.row(align=True)
        r.prop(pcv, 'clip_plane1_enabled', text='', toggle=True, icon_only=True, icon='HIDE_OFF' if pcv.clip_plane1_enabled else 'HIDE_ON', )
        r.prop(pcv, 'clip_plane1', )
        r = c.row(align=True)
        r.prop(pcv, 'clip_plane2_enabled', text='', toggle=True, icon_only=True, icon='HIDE_OFF' if pcv.clip_plane2_enabled else 'HIDE_ON', )
        r.prop(pcv, 'clip_plane2', )
        r = c.row(align=True)
        r.prop(pcv, 'clip_plane3_enabled', text='', toggle=True, icon_only=True, icon='HIDE_OFF' if pcv.clip_plane3_enabled else 'HIDE_ON', )
        r.prop(pcv, 'clip_plane3', )
        r = c.row(align=True)
        r.prop(pcv, 'clip_plane4_enabled', text='', toggle=True, icon_only=True, icon='HIDE_OFF' if pcv.clip_plane4_enabled else 'HIDE_ON', )
        r.prop(pcv, 'clip_plane4', )
        r = c.row(align=True)
        r.prop(pcv, 'clip_plane5_enabled', text='', toggle=True, icon_only=True, icon='HIDE_OFF' if pcv.clip_plane5_enabled else 'HIDE_ON', )
        r.prop(pcv, 'clip_plane5', )
        
        c = a.column(align=True)
        c.prop(pcv, 'clip_planes_from_bbox_object')
        r = c.row(align=True)
        r.operator('point_cloud_visualizer.clip_planes_from_bbox')
        r.operator('point_cloud_visualizer.clip_planes_reset', text='', icon='X', )
        
        a.enabled = pcv.clip_shader_enabled
        
        pcv = context.object.point_cloud_visualizer
        ok = False
        for k, v in PCVManager.cache.items():
            if(v['uuid'] == pcv.uuid):
                if(v['ready']):
                    if(v['draw']):
                        ok = True
        l.enabled = ok


class PCV_PT_edit(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "View"
    bl_label = "Edit"
    bl_parent_id = "PCV_PT_panel"
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
        return True
    
    def draw(self, context):
        pcv = context.object.point_cloud_visualizer
        l = self.layout
        c = l.column()
        c.operator('point_cloud_visualizer.edit_start', text='Enable Edit Mode', )


class PCV_PT_export(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "View"
    bl_label = "Export"
    bl_parent_id = "PCV_PT_panel"
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
        return True
    
    def draw(self, context):
        pcv = context.object.point_cloud_visualizer
        l = self.layout
        c = l.column()
        c.prop(pcv, 'export_use_viewport')
        cc = c.column()
        cc.prop(pcv, 'export_visible_only')
        if(not pcv.export_use_viewport):
            cc.enabled = False
        c.prop(pcv, 'export_apply_transformation')
        c.prop(pcv, 'export_convert_axes')
        c.operator('point_cloud_visualizer.export')
        
        c.enabled = PCV_OT_export.poll(context)


class PCV_PT_sequence(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "View"
    bl_label = "Sequence"
    bl_parent_id = "PCV_PT_panel"
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
        return True
    
    # def draw_header(self, context):
    #     pcv = context.object.point_cloud_visualizer
    #     l = self.layout
    #     l.label(text='', icon='EXPERIMENTAL', )
    
    def draw(self, context):
        pcv = context.object.point_cloud_visualizer
        l = self.layout
        # c = l.column()
        # c.prop(pcv, 'sequence_enabled')
        # c.enabled = PCV_OT_sequence_preload.poll(context)
        
        c = l.column()
        
        # c.label(text='Experimental', icon='ERROR', )
        
        c.operator('point_cloud_visualizer.sequence_preload')
        if(pcv.uuid in PCVSequence.cache.keys()):
            c.label(text="Loaded {} item(s)".format(len(PCVSequence.cache[pcv.uuid]['data'])))
            # c.enabled = pcv.sequence_enabled
        else:
            c.label(text="Loaded {} item(s)".format(0))
            c.enabled = PCV_OT_sequence_preload.poll(context)
        # c.enabled = pcv.sequence_enabled
        
        # c = l.column()
        # c.prop(pcv, 'sequence_frame_duration')
        # c.prop(pcv, 'sequence_frame_start')
        # c.prop(pcv, 'sequence_frame_offset')
        c.prop(pcv, 'sequence_use_cyclic')
        # c.enabled = False
        # if(pcv.sequence_enabled):
        #     c.enabled = True
        # c.enabled = (PCV_OT_sequence_preload.poll(context) and pcv.sequence_enabled)
        # c.enabled = PCV_OT_sequence_preload.poll(context)
        # c.enabled = pcv.sequence_enabled
        c.operator('point_cloud_visualizer.sequence_clear')
        
        l.enabled = not pcv.runtime


class PCV_PT_generate(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "View"
    bl_label = "Generate"
    bl_parent_id = "PCV_PT_panel"
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
        return True
    
    def draw(self, context):
        pcv = context.object.point_cloud_visualizer
        l = self.layout
        c = l.column()
        
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
        
        third_label_two_thirds_prop(pcv, 'generate_source', c, )
        
        if(pcv.generate_source == 'PARTICLES'):
            third_label_two_thirds_prop(pcv, 'generate_source_psys', c, )
        
        if(pcv.generate_source in ('SURFACE', )):
            third_label_two_thirds_prop(pcv, 'generate_algorithm', c, )
        
        if(pcv.generate_source in ('SURFACE', )):
            if(pcv.generate_algorithm in ('WEIGHTED_RANDOM_IN_TRIANGLE', )):
                c.prop(pcv, 'generate_number_of_points')
                c.prop(pcv, 'generate_seed')
                c.prop(pcv, 'generate_exact_number_of_points')
            if(pcv.generate_algorithm in ('POISSON_DISK_SAMPLING', )):
                c.prop(pcv, 'generate_minimal_distance')
                c.prop(pcv, 'generate_sampling_exponent')
                # c.prop(pcv, 'generate_seed')
        
        third_label_two_thirds_prop(pcv, 'generate_colors', c, )
        if(pcv.generate_colors == 'CONSTANT'):
            r = c.row()
            third_label_two_thirds_prop(pcv, 'generate_constant_color', c, )
        
        c.operator('point_cloud_visualizer.generate_from_mesh')
        c.operator('point_cloud_visualizer.reset_runtime', text="Remove Generated", )
        
        c.enabled = PCV_OT_generate_point_cloud.poll(context)


class PCV_PT_development(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "View"
    bl_label = "Development"
    bl_parent_id = "PCV_PT_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        if(not debug_mode()):
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
        # l.label(text='', icon='SETTINGS', )
        l.label(text='', icon='EXPERIMENTAL', )
    
    def draw(self, context):
        pcv = context.object.point_cloud_visualizer
        l = self.layout
        sub = l.column()
        
        # sub.label(text="Color Adjustment Shader:")
        # c = sub.column(align=True)
        # c.prop(pcv, 'color_adjustment_shader_enabled')
        # cc = c.column(align=True)
        # cc.prop(pcv, 'color_adjustment_shader_exposure')
        # cc.prop(pcv, 'color_adjustment_shader_gamma')
        # cc.prop(pcv, 'color_adjustment_shader_brightness')
        # cc.prop(pcv, 'color_adjustment_shader_contrast')
        # cc.prop(pcv, 'color_adjustment_shader_hue')
        # cc.prop(pcv, 'color_adjustment_shader_saturation')
        # cc.prop(pcv, 'color_adjustment_shader_value')
        # cc.prop(pcv, 'color_adjustment_shader_invert')
        # r = cc.row(align=True)
        # r.operator('point_cloud_visualizer.color_adjustment_shader_reset')
        # r.operator('point_cloud_visualizer.color_adjustment_shader_apply')
        # cc.enabled = pcv.color_adjustment_shader_enabled
        # sub.separator()
        
        sub.label(text="Shaders:")
        e = False
        for k, v in PCVManager.cache.items():
            if(v['uuid'] == pcv.uuid):
                if(v['ready']):
                    if(v['draw']):
                        e = True
        
        c = sub.column()
        
        cc = c.column(align=True)
        cc.prop(pcv, 'dev_selection_shader_display', toggle=True, )
        if(pcv.dev_selection_shader_display):
            r = cc.row(align=True)
            r.prop(pcv, 'dev_selection_shader_color', text="", )
        
        c.prop(pcv, 'dev_minimal_shader_enabled', toggle=True, text="Minimal Shader", )
        
        c.prop(pcv, 'dev_minimal_shader_variable_size_enabled', toggle=True, text="Minimal Shader With Variable Size", )
        
        cc = c.column(align=True)
        cc.prop(pcv, 'dev_minimal_shader_variable_size_and_depth_enabled', toggle=True, text="Minimal Shader With Variable Size And Depth", )
        if(pcv.dev_minimal_shader_variable_size_and_depth_enabled):
            cc.prop(pcv, 'dev_minimal_shader_variable_size_and_depth_brightness')
            cc.prop(pcv, 'dev_minimal_shader_variable_size_and_depth_contrast')
            cc.prop(pcv, 'dev_minimal_shader_variable_size_and_depth_blend')
        
        cc = c.column(align=True)
        cc.prop(pcv, 'dev_billboard_point_cloud_enabled', toggle=True, text='BIllboard Shader', )
        if(pcv.dev_billboard_point_cloud_enabled):
            cc.prop(pcv, 'dev_billboard_point_cloud_size')
        
        cc = c.column(align=True)
        cc.prop(pcv, 'dev_rich_billboard_point_cloud_enabled', toggle=True, text='Rich BIllboard Shader', )
        if(pcv.dev_rich_billboard_point_cloud_enabled):
            cc.prop(pcv, 'dev_rich_billboard_point_cloud_size')
            cc.prop(pcv, 'dev_rich_billboard_depth_brightness')
            cc.prop(pcv, 'dev_rich_billboard_depth_contrast')
            cc.prop(pcv, 'dev_rich_billboard_depth_blend')
        
        cc = c.column(align=True)
        cc.prop(pcv, 'dev_phong_shader_enabled', toggle=True, text='Phong Shader', )
        if(pcv.dev_phong_shader_enabled):
            cc.prop(pcv, 'dev_phong_shader_ambient_strength')
            cc.prop(pcv, 'dev_phong_shader_specular_strength')
            cc.prop(pcv, 'dev_phong_shader_specular_exponent')
        
        cc = c.column(align=True)
        cc.prop(pcv, 'clip_shader_enabled', toggle=True, text='Clip', )
        if(pcv.clip_shader_enabled):
            r = cc.row(align=True)
            r.prop(pcv, 'clip_plane0_enabled', text='', toggle=True, icon_only=True, icon='HIDE_OFF' if pcv.clip_plane0_enabled else 'HIDE_ON', )
            r.prop(pcv, 'clip_plane0', )
            r = cc.row(align=True)
            r.prop(pcv, 'clip_plane1_enabled', text='', toggle=True, icon_only=True, icon='HIDE_OFF' if pcv.clip_plane1_enabled else 'HIDE_ON', )
            r.prop(pcv, 'clip_plane1', )
            r = cc.row(align=True)
            r.prop(pcv, 'clip_plane2_enabled', text='', toggle=True, icon_only=True, icon='HIDE_OFF' if pcv.clip_plane2_enabled else 'HIDE_ON', )
            r.prop(pcv, 'clip_plane2', )
            r = cc.row(align=True)
            r.prop(pcv, 'clip_plane3_enabled', text='', toggle=True, icon_only=True, icon='HIDE_OFF' if pcv.clip_plane3_enabled else 'HIDE_ON', )
            r.prop(pcv, 'clip_plane3', )
            r = cc.row(align=True)
            r.prop(pcv, 'clip_plane4_enabled', text='', toggle=True, icon_only=True, icon='HIDE_OFF' if pcv.clip_plane4_enabled else 'HIDE_ON', )
            r.prop(pcv, 'clip_plane4', )
            r = cc.row(align=True)
            r.prop(pcv, 'clip_plane5_enabled', text='', toggle=True, icon_only=True, icon='HIDE_OFF' if pcv.clip_plane5_enabled else 'HIDE_ON', )
            r.prop(pcv, 'clip_plane5', )
            cc.prop(pcv, 'clip_planes_from_bbox_object')
            cc.operator('point_cloud_visualizer.clip_planes_from_bbox')
        
        cc = c.column(align=True)
        cc.prop(pcv, 'billboard_phong_enabled', toggle=True, text='Billboard Phong', )
        if(pcv.billboard_phong_enabled):
            cc.prop(pcv, 'billboard_phong_circles', toggle=True, )
            cc.prop(pcv, 'billboard_phong_size')
            cc.prop(pcv, 'billboard_phong_ambient_strength')
            cc.prop(pcv, 'billboard_phong_specular_strength')
            cc.prop(pcv, 'billboard_phong_specular_exponent')
        
        cc = c.column(align=True)
        cc.prop(pcv, 'skip_point_shader_enabled', toggle=True, text='Skip Point Shader', )
        if(pcv.skip_point_shader_enabled):
            cc.prop(pcv, 'skip_point_percentage')
        
        cc = c.column(align=True)
        cc.prop(pcv, 'dev_bbox_enabled', toggle=True, text='Bounding Box', )
        if(pcv.dev_bbox_enabled):
            cc.prop(pcv, 'dev_bbox_color')
            cc.prop(pcv, 'dev_bbox_size')
            cc.prop(pcv, 'dev_bbox_alpha')
        
        sub.separator()
        
        sub.label(text="Generate Volume:")
        c = sub.column(align=True)
        c.prop(pcv, 'generate_number_of_points')
        c.prop(pcv, 'generate_seed')
        c.operator('point_cloud_visualizer.generate_volume_from_mesh')
        
        # sub.separator()
        
        """
        c.separator()
        
        c.label(text="new ui for shaders")
        c.separator()
        
        r = c.row(align=True)
        s = r.split(factor=0.25, align=True, )
        s.label(text='Shader:')
        s = s.split(factor=0.75, align=True, )
        r = s.row(align=True)
        r.prop(pcv, 'shader', text='', )
        s = s.split(factor=0.25, align=True, )
        
        cc = s.column(align=True)
        cc.prop(pcv, 'shader_illumination', text='', icon='LIGHT', toggle=True, icon_only=True, )
        if(pcv.shader not in ('DEFAULT', 'DEPTH', )):
            cc.enabled = False

        cc = s.column(align=True)
        cc.prop(pcv, 'shader_options_show', text='', icon='TOOL_SETTINGS', toggle=True, icon_only=True, )
        if(pcv.shader not in ('DEPTH', )):
            cc.enabled = False

        cc = s.column(align=True)
        cc.prop(pcv, 'shader_normal_lines', text='', icon='SNAP_NORMAL', toggle=True, icon_only=True, )
        
        c.separator()
        c.separator()
        
        r = c.row(align=True)
        r.prop(pcv, 'shader', expand=True, )
        
        cc = r.column(align=True)
        cc.prop(pcv, 'shader_illumination', text='', icon='LIGHT', toggle=True, icon_only=True, )
        if(pcv.shader not in ('DEFAULT', 'DEPTH', )):
            cc.enabled = False
        
        cc = r.column(align=True)
        cc.prop(pcv, 'shader_options_show', text='', icon='TOOL_SETTINGS', toggle=True, icon_only=True, )
        if(pcv.shader not in ('DEPTH', )):
            cc.enabled = False
        
        cc = r.column(align=True)
        cc.prop(pcv, 'shader_normal_lines', text='', icon='SNAP_NORMAL', toggle=True, icon_only=True, )
        
        if(pcv.shader_illumination):
            if(pcv.shader in ('DEFAULT', 'DEPTH', )):
                c.label(text='shader illumination options..')
        
        if(pcv.shader_options_show):
            if(pcv.shader in ('DEPTH', )):
                c.label(text='shader options..')
        
        if(pcv.shader_normal_lines):
            c.label(text='shader normal lines options..')
        
        c.separator()
        """
        
        # sub.label(text="Numpy Vertices And Normals Transform")
        # c = sub.column()
        # c.prop(pcv, 'dev_transform_normals_target_object')
        # c.operator('point_cloud_visualizer.pcviv_dev_transform_normals')
        
        # sub.label(text="Clip To Active Camera Cone")
        # c = sub.column()
        # c.operator('point_cloud_visualizer.clip_planes_from_camera_view')


class PCV_PT_debug(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "View"
    bl_label = "Debug"
    bl_parent_id = "PCV_PT_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        o = context.active_object
        if(o is None):
            return False
        
        if(o):
            pcv = o.point_cloud_visualizer
            if(debug_mode()):
                return True
        return False
    
    def draw_header(self, context):
        pcv = context.object.point_cloud_visualizer
        l = self.layout
        l.label(text='', icon='SETTINGS', )
    
    def draw(self, context):
        pcv = context.object.point_cloud_visualizer
        l = self.layout
        sub = l.column()
        sub.operator('script.reload')
        sub.separator()
        
        b = sub.box()
        r = b.row()
        r.prop(pcv, 'debug_panel_show_properties', icon='TRIA_DOWN' if pcv.debug_panel_show_properties else 'TRIA_RIGHT', icon_only=True, emboss=False, )
        r.label(text="properties")
        if(pcv.debug_panel_show_properties):
            c = b.column()
            for k, p in pcv.bl_rna.properties.items():
                v = 'n/a'
                if(p.type == 'POINTER'):
                    v = 'POINTER'
                else:
                    v = p.default
                    if(k in pcv.keys()):
                        v = pcv[k]
                if(p.type == 'BOOLEAN'):
                    v = bool(v)
                c.label(text="{}: {}".format(k, v))
            c.scale_y = 0.5
        
        b = sub.box()
        r = b.row()
        r.prop(pcv, 'debug_panel_show_manager', icon='TRIA_DOWN' if pcv.debug_panel_show_manager else 'TRIA_RIGHT', icon_only=True, emboss=False, )
        r.label(text="manager")
        if(pcv.debug_panel_show_manager):
            c = b.column(align=True)
            rr = c.row(align=True)
            rr.operator('point_cloud_visualizer.init')
            rr.operator('point_cloud_visualizer.deinit')
            rr.operator('point_cloud_visualizer.gc')
            bb = b.box()
            c = bb.column()
            c.label(text="cache: {} item(s)".format(len(PCVManager.cache.items())))
            c.label(text="handle: {}".format(PCVManager.handle))
            c.label(text="initialized: {}".format(PCVManager.initialized))
            c.scale_y = 0.5
            
            if(len(PCVManager.cache)):
                b.label(text="cache details:")
                for k, v in PCVManager.cache.items():
                    bb = b.box()
                    r = bb.row()
                    r.prop(pcv, 'debug_panel_show_cache_items', icon='TRIA_DOWN' if pcv.debug_panel_show_cache_items else 'TRIA_RIGHT', icon_only=True, emboss=False, )
                    r.label(text=k)
                    if(pcv.debug_panel_show_cache_items):
                        c = bb.column()
                        c.scale_y = 0.5
                        for ki, vi in sorted(v.items()):
                            if(type(vi) == np.ndarray):
                                c.label(text="{}: numpy.ndarray ({} items)".format(ki, len(vi)))
                            elif(type(vi) == dict):
                                c.label(text="{}: dict ({} items)".format(ki, len(vi.keys())))
                                t = '    '
                                for dk, dv in vi.items():
                                    c.label(text="{}{}: {}".format(t, dk, dv))
                            else:
                                c.label(text="{}: {}".format(ki, vi))
        
        b = sub.box()
        r = b.row()
        r.prop(pcv, 'debug_panel_show_sequence', icon='TRIA_DOWN' if pcv.debug_panel_show_sequence else 'TRIA_RIGHT', icon_only=True, emboss=False, )
        r.label(text="sequence")
        if(pcv.debug_panel_show_sequence):
            c = b.column(align=True)
            rr = c.row(align=True)
            rr.operator('point_cloud_visualizer.seq_init')
            rr.operator('point_cloud_visualizer.seq_deinit')
            bb = b.box()
            c = bb.column()
            c.label(text="cache: {} item(s)".format(len(PCVSequence.cache.items())))
            c.label(text="initialized: {}".format(PCVSequence.initialized))
            c.scale_y = 0.5
            
            if(len(PCVSequence.cache)):
                b.label(text="cache details:")
                for k, v in PCVSequence.cache.items():
                    bb = b.box()
                    c = bb.column()
                    c.scale_y = 0.5
                    c.label(text="{}: {}".format('uuid', v['uuid']))
                    c.label(text="{}: {}".format('pcv', v['pcv']))
                    c.label(text="{}: {}".format('data', '{} item(s)'.format(len(v['data']))))


class PCV_properties(PropertyGroup):
    filepath: StringProperty(name="PLY File", default="", description="", )
    uuid: StringProperty(default="", options={'HIDDEN', }, )
    
    def _instance_visualizer_active_get(self, ):
        return self.instance_visualizer_active_hidden_value
    
    def _instance_visualizer_active_set(self, value, ):
        pass
    
    # for setting value, there are handlers for save, pre which sets to False, and post which sets back to True if it was True before, instance visualizer have to be activated at runtime and this value should not be saved, this way it works ok.. if only there was a way to specify which properties should not save, and/or save only as default value..
    instance_visualizer_active_hidden_value: BoolProperty(default=False, options={'HIDDEN', }, )
    # for display, read-only
    instance_visualizer_active: BoolProperty(name="Instance Visualizer Active", default=False, get=_instance_visualizer_active_get, set=_instance_visualizer_active_set, )
    
    """
    def _shader_items():
        # closure - keep a reference to the list
        items = None
        
        def func(self, context, ):
            items = [
                # # user selectable shaders
                
                # basic shader, round shape, unaltered colors
                ('DEFAULT', 'Default', "", '', 2 ** 0, ),
                # # default with illumination on top
                # ('ILLUMINATION', 'Illumination', "", '', 2 ** 1, ),
                # color by depth
                ('DEPTH', 'Depth', "", '', 2 ** 2, ),
                # # depth with illumination on top, maybe join illuminated and non-illuminated variants somehow together
                # ('DEPTH_ILLUMINATION', 'Depth With Illumination', "", '', 2 ** 3, ),
                # color by normal
                ('NORMAL', 'Normal', "", '', 2 ** 4, ),
                # color by position
                ('POSITION', 'Position', "", '', 2 ** 5, ),
                
                # # internal shaders, used only under certain conditions or development shaders
                
                # # this is internal shader active only when color adjustment filter is active
                # ('COLOR_ADJUSTMENT', 'Color Adjustment', "", '', 2 ** 6, ),
                # # basically this is the same as Default, only without point shape rounding, there is not much use for it apart from that it is slightly faster to draw
                # ('MINIMAL', 'Minimal', "", '', 2 ** 7, ),
                # # this is meant to be used only with instance visualizer, there is not much to do in regular ply files
                # ('MINIMAL_VARIABLE', 'Minimal With Variable Size', "", '', 2 ** 8, ),
                # # not ready for production yet
                # ('BOUNDING_BOX', 'Bounding Box', "", '', 2 ** 9, ),
                # # this is extra drawn on top while using remove color filter
                # ('SELECTION', 'Selection', "", '', 2 ** 10, ),
            ]
            return items
        
        return func
    
    def _shader_update(self, context, ):
        pass
    """
    """
    shader_items = [
        ('DEFAULT', 'Default', "", ),
        ('DEPTH', 'Depth', "", ),
        ('NORMAL', 'Normal', "", ),
        ('POSITION', 'Position', "", ),
    ]
    # FIXMENOT: would be nice to have this initialized at least to 'DEFAULT' if '' and PCVManager is initialized, so i don't need to handle situations if(shader == ''): do like it's 'DEFAULT'
    # NOTTODO: split illumination to its own BoolProperty, and enable only when possible
    # FIXMENOT: it looks like i don't need dynamic enum, items that might be dynamic are not user selectable anyway.
    # shader: EnumProperty(name="Shader", items=_shader_items(), update=_shader_update, description="Shader to draw points with", )
    shader: EnumProperty(name="Shader", items=shader_items, default='DEFAULT', description="Shader to draw points with", )
    shader_illumination: BoolProperty(name="Illumination", default=False, description="Enable extra illumination on point cloud", )
    shader_options_show: BoolProperty(name="Shader Options", default=False, description="Show shader options", )
    shader_normal_lines: BoolProperty(name="Normals", default=False, description="Show normals as lines", )
    """
    
    runtime: BoolProperty(default=False, options={'HIDDEN', }, )
    
    # TODO: add some prefix to global props, like global_size, global_display_percent, .. leave unprefixed only essentials, like uuid, runtime, ..
    point_size: IntProperty(name="Size", default=3, min=1, max=10, subtype='PIXEL', description="Point size", )
    alpha_radius: FloatProperty(name="Radius", default=1.0, min=0.001, max=1.0, precision=3, subtype='FACTOR', description="Adjust point circular discard radius", )
    
    def _display_percent_update(self, context, ):
        if(self.uuid not in PCVManager.cache):
            return
        d = PCVManager.cache[self.uuid]
        dp = self.display_percent
        vl = d['length']
        l = int((vl / 100) * dp)
        if(dp >= 99):
            l = vl
        d['display_length'] = l
    
    display_percent: FloatProperty(name="Display", default=100.0, min=0.0, max=100.0, precision=0, subtype='PERCENTAGE', update=_display_percent_update, description="Adjust percentage of points displayed", )
    global_alpha: FloatProperty(name="Alpha", default=1.0, min=0.0, max=1.0, precision=2, subtype='FACTOR', description="Adjust alpha of points displayed", )
    
    vertex_normals: BoolProperty(name="Normals", description="Draw normals of points", default=False, )
    vertex_normals_size: FloatProperty(name="Length", description="Length of point normal line", default=0.01, min=0.00001, max=1.0, soft_min=0.001, soft_max=0.2, step=1, precision=3, )
    vertex_normals_alpha: FloatProperty(name="Alpha", description="Alpha of point normal line", default=0.5, min=0.0, max=1.0, soft_min=0.0, soft_max=1.0, step=1, precision=3, )
    
    render_point_size: IntProperty(name="Size", default=3, min=1, max=100, subtype='PIXEL', description="Point size", )
    render_display_percent: FloatProperty(name="Count", default=100.0, min=0.0, max=100.0, precision=0, subtype='PERCENTAGE', description="Adjust percentage of points rendered", )
    render_path: StringProperty(name="Output Path", default="//pcv_render_###.png", description="Directory/name to save rendered images, # characters defines the position and length of frame numbers, filetype is always png", subtype='FILE_PATH', )
    render_resolution_x: IntProperty(name="Resolution X", default=1920, min=4, max=65536, description="Number of horizontal pixels in rendered image", subtype='PIXEL', )
    render_resolution_y: IntProperty(name="Resolution Y", default=1080, min=4, max=65536, description="Number of vertical pixels in rendered image", subtype='PIXEL', )
    render_resolution_percentage: IntProperty(name="Resolution %", default=100, min=1, max=100, description="Percentage scale for render resolution", subtype='PERCENTAGE', )
    render_smoothstep: BoolProperty(name="Smooth Circles", default=False, description="Currently works only for basic shader with/without illumination and generally is much slower than Supersampling, use only when Supersampling fails", )
    render_supersampling: IntProperty(name="Supersampling", default=1, soft_min=1, soft_max=4, min=1, max=10, description="Render larger image and then resize back, 1 - disabled, 2 - render 200%, 3 - render 300%, ...", )
    
    def _render_resolution_linked_update(self, context, ):
        if(not self.render_resolution_linked):
            # now it is False, so it must have been True, so for convenience, copy values
            r = context.scene.render
            self.render_resolution_x = r.resolution_x
            self.render_resolution_y = r.resolution_y
            self.render_resolution_percentage = r.resolution_percentage
    
    render_resolution_linked: BoolProperty(name="Resolution Linked", description="Link resolution settings to scene", default=True, update=_render_resolution_linked_update, )
    
    has_normals: BoolProperty(default=False, options={'HIDDEN', }, )
    # TODO: rename to 'has_colors'
    has_vcols: BoolProperty(default=False, options={'HIDDEN', }, )
    illumination: BoolProperty(name="Illumination", description="Enable extra illumination on point cloud", default=False, )
    illumination_edit: BoolProperty(name="Edit", description="Edit illumination properties", default=False, )
    light_direction: FloatVectorProperty(name="Light Direction", description="Light direction", default=(0.0, 1.0, 0.0), subtype='DIRECTION', size=3, )
    # light_color: FloatVectorProperty(name="Light Color", description="", default=(0.2, 0.2, 0.2), min=0, max=1, subtype='COLOR', size=3, )
    light_intensity: FloatProperty(name="Light Intensity", description="Light intensity", default=0.3, min=0, max=1, subtype='FACTOR', )
    shadow_intensity: FloatProperty(name="Shadow Intensity", description="Shadow intensity", default=0.2, min=0, max=1, subtype='FACTOR', )
    # show_normals: BoolProperty(name="Colorize By Vertex Normals", description="", default=False, )
    
    mesh_type: EnumProperty(name="Type", items=[('VERTEX', "Vertex", ""),
                                                ('TRIANGLE', "Equilateral Triangle", ""),
                                                ('TETRAHEDRON', "Tetrahedron", ""),
                                                ('CUBE', "Cube", ""),
                                                ('ICOSPHERE', "Ico Sphere", ""),
                                                ('INSTANCER', "Instancer", ""),
                                                ('PARTICLES', "Particle System", ""), ], default='CUBE', description="Instance mesh type", )
    mesh_size: FloatProperty(name="Size", description="Mesh instance size, instanced mesh has size 1.0", default=0.01, min=0.000001, precision=4, max=100.0, )
    mesh_normal_align: BoolProperty(name="Align To Normal", description="Align instance to point normal", default=True, )
    mesh_vcols: BoolProperty(name="Colors", description="Assign point color to instance vertex colors", default=True, )
    mesh_all: BoolProperty(name="All", description="Convert all points", default=True, )
    mesh_percentage: FloatProperty(name="Subset", default=100.0, min=0.0, max=100.0, precision=0, subtype='PERCENTAGE', description="Convert random subset of points by given percentage", )
    mesh_base_sphere_subdivisions: IntProperty(name="Sphere Subdivisions", default=2, min=1, max=6, description="Particle instance (Ico Sphere) subdivisions, instance mesh can be change later", )
    mesh_use_instancer2: BoolProperty(name="Use Faster Conversion", description="Faster (especially with icosphere) Numpy implementation, use if you don't mind all triangles in result", default=False, )
    
    export_use_viewport: BoolProperty(name="Use Viewport Points", default=True, description="When checked, export points currently displayed in viewport or when unchecked, export data loaded from original ply file", )
    export_apply_transformation: BoolProperty(name="Apply Transformation", default=False, description="Apply parent object transformation to points", )
    export_convert_axes: BoolProperty(name="Convert Axes", default=False, description="Convert from blender (y forward, z up) to forward -z, up y axes", )
    export_visible_only: BoolProperty(name="Visible Points Only", default=False, description="Export currently visible points only (controlled by 'Display' on main panel)", )
    
    filter_simplify_num_samples: IntProperty(name="Samples", default=10000, min=1, subtype='NONE', description="Number of points in simplified point cloud, best result when set to less than 20% of points, when samples has value close to total expect less points in result", )
    filter_simplify_num_candidates: IntProperty(name="Candidates", default=10, min=3, max=100, subtype='NONE', description="Number of candidates used during resampling, the higher value, the slower calculation, but more even", )
    
    filter_remove_color: FloatVectorProperty(name="Color", default=(1.0, 1.0, 1.0, ), min=0, max=1, subtype='COLOR', size=3, description="Color to remove from point cloud", )
    filter_remove_color_delta_hue: FloatProperty(name="Δ Hue", default=0.1, min=0.0, max=1.0, precision=3, subtype='FACTOR', description="", )
    filter_remove_color_delta_hue_use: BoolProperty(name="Use Δ Hue", description="", default=True, )
    filter_remove_color_delta_saturation: FloatProperty(name="Δ Saturation", default=0.1, min=0.0, max=1.0, precision=3, subtype='FACTOR', description="", )
    filter_remove_color_delta_saturation_use: BoolProperty(name="Use Δ Saturation", description="", default=True, )
    filter_remove_color_delta_value: FloatProperty(name="Δ Value", default=0.1, min=0.0, max=1.0, precision=3, subtype='FACTOR', description="", )
    filter_remove_color_delta_value_use: BoolProperty(name="Use Δ Value", description="", default=True, )
    filter_remove_color_selection: BoolProperty(default=False, options={'HIDDEN', }, )
    
    def _project_positive_radio_update(self, context):
        if(not self.filter_project_negative and not self.filter_project_positive):
            self.filter_project_negative = True
    
    def _project_negative_radio_update(self, context):
        if(not self.filter_project_negative and not self.filter_project_positive):
            self.filter_project_positive = True
    
    def _filter_project_object_poll(self, o, ):
        if(o and o.type in ('MESH', 'CURVE', 'SURFACE', 'FONT', )):
            return True
        return False
    
    filter_project_object: PointerProperty(type=bpy.types.Object, name="Object", description="", poll=_filter_project_object_poll, )
    filter_project_search_distance: FloatProperty(name="Search Distance", default=0.1, min=0.0, max=10000.0, precision=3, subtype='DISTANCE', description="Maximum search distance in which to search for surface", )
    filter_project_positive: BoolProperty(name="Positive", description="Search along point normal forwards", default=True, update=_project_positive_radio_update, )
    filter_project_negative: BoolProperty(name="Negative", description="Search along point normal backwards", default=True, update=_project_negative_radio_update, )
    filter_project_discard: BoolProperty(name="Discard Unprojectable", description="Discard points which didn't hit anything", default=False, )
    filter_project_colorize: BoolProperty(name="Colorize", description="Colorize projected points", default=False, )
    filter_project_colorize_from: EnumProperty(name="Source", items=[('VCOLS', "Vertex Colors", "Use active vertex colors from target"),
                                                                     ('UVTEX', "UV Texture", "Use colors from active image texture node in active material using active UV layout from target"),
                                                                     ('GROUP_MONO', "Vertex Group Monochromatic", "Use active vertex group from target, result will be shades of grey"),
                                                                     ('GROUP_COLOR', "Vertex Group Colorized", "Use active vertex group from target, result will be colored from red (1.0) to blue (0.0) like in weight paint viewport"),
                                                                     ], default='UVTEX', description="Color source for projected point cloud", )
    filter_project_shift: FloatProperty(name="Shift", default=0.0, precision=3, subtype='DISTANCE', description="Shift points after projection above (positive) or below (negative) surface", )
    
    def _filter_boolean_object_poll(self, o, ):
        if(o and o.type in ('MESH', 'CURVE', 'SURFACE', 'FONT', )):
            return True
        return False
    
    filter_boolean_object: PointerProperty(type=bpy.types.Object, name="Object", description="", poll=_filter_boolean_object_poll, )
    
    def _filter_join_object_poll(self, o, ):
        ok = False
        if(o):
            pcv = o.point_cloud_visualizer
            if(pcv.uuid != ''):
                for k, v in PCVManager.cache.items():
                    if(v['uuid'] == pcv.uuid):
                        if(v['ready']):
                            # if(v['draw']):
                            #     ok = True
                            ok = True
                        break
        return ok
    
    filter_join_object: PointerProperty(type=bpy.types.Object, name="Object", description="", poll=_filter_join_object_poll, )
    
    edit_initialized: BoolProperty(default=False, options={'HIDDEN', }, )
    edit_is_edit_mesh: BoolProperty(default=False, options={'HIDDEN', }, )
    edit_is_edit_uuid: StringProperty(default="", options={'HIDDEN', }, )
    edit_pre_edit_alpha: FloatProperty(default=0.5, options={'HIDDEN', }, )
    edit_pre_edit_display: FloatProperty(default=100.0, options={'HIDDEN', }, )
    edit_pre_edit_size: IntProperty(default=3, options={'HIDDEN', }, )
    
    def _edit_overlay_alpha_update(self, context, ):
        o = context.object
        p = o.parent
        pcv = p.point_cloud_visualizer
        pcv.global_alpha = self.edit_overlay_alpha
    
    def _edit_overlay_size_update(self, context, ):
        o = context.object
        p = o.parent
        pcv = p.point_cloud_visualizer
        pcv.point_size = self.edit_overlay_size
    
    edit_overlay_alpha: FloatProperty(name="Overlay Alpha", default=0.5, min=0.0, max=1.0, precision=2, subtype='FACTOR', description="Overlay point alpha", update=_edit_overlay_alpha_update, )
    edit_overlay_size: IntProperty(name="Overlay Size", default=3, min=1, max=10, subtype='PIXEL', description="Overlay point size", update=_edit_overlay_size_update, )
    
    # sequence_enabled: BoolProperty(default=False, options={'HIDDEN', }, )
    # sequence_frame_duration: IntProperty(name="Frames", default=1, min=1, description="", )
    # sequence_frame_start: IntProperty(name="Start Frame", default=1, description="", )
    # sequence_frame_offset: IntProperty(name="Offset", default=0, description="", )
    sequence_use_cyclic: BoolProperty(name="Cycle Forever", default=True, description="Cycle preloaded point clouds (ply_index = (current_frame % len(ply_files)) - 1)", )
    
    generate_source: EnumProperty(name="Source", items=[('VERTICES', "Vertices", "Use mesh vertices"),
                                                        ('SURFACE', "Surface", "Use triangulated mesh surface"),
                                                        ('PARTICLES', "Particle System", "Use active particle system"),
                                                        ], default='SURFACE', description="Points generation source", )
    generate_source_psys: EnumProperty(name="Particles", items=[('ALL', "All", "Use all particles"),
                                                                ('ALIVE', "Alive", "Use alive particles"),
                                                                ], default='ALIVE', description="Particles source", )
    generate_algorithm: EnumProperty(name="Algorithm", items=[('WEIGHTED_RANDOM_IN_TRIANGLE', "Weighted Random In Triangle", "Average triangle areas to approximate number of random points in each to get even distribution of points. If some very small polygons are left without points, increase number of samples. Mesh is triangulated before processing, on non-planar polygons, points will not be exactly on original polygon surface."),
                                                              ('POISSON_DISK_SAMPLING', "Poisson Disk Sampling", "Warning: slow, very slow indeed.. Uses Weighted Random In Triangle algorithm to pregenerate samples with all its inconveniences."),
                                                              ], default='WEIGHTED_RANDOM_IN_TRIANGLE', description="Point generating algorithm", )
    generate_number_of_points: IntProperty(name="Approximate Number Of Points", default=100000, min=1, description="Number of points to generate, some algorithms may not generate exact number of points.", )
    generate_seed: IntProperty(name="Seed", default=0, min=0, description="Random number generator seed", )
    generate_colors: EnumProperty(name="Colors", items=[('CONSTANT', "Constant Color", "Use constant color value"),
                                                        ('VCOLS', "Vertex Colors", "Use active vertex colors"),
                                                        ('UVTEX', "UV Texture", "Generate colors from active image texture node in active material using active UV layout"),
                                                        ('GROUP_MONO', "Vertex Group Monochromatic", "Use active vertex group, result will be shades of grey"),
                                                        ('GROUP_COLOR', "Vertex Group Colorized", "Use active vertex group, result will be colored from red (1.0) to blue (0.0) like in weight paint viewport"),
                                                        ], default='CONSTANT', description="Color source for generated point cloud", )
    generate_constant_color: FloatVectorProperty(name="Color", description="Constant color", default=(0.7, 0.7, 0.7, ), min=0, max=1, subtype='COLOR', size=3, )
    generate_exact_number_of_points: BoolProperty(name="Exact Number of Samples", default=False, description="Generate exact number of points, if selected algorithm result is less points, more points will be calculated on random polygons at the end, if result is more points, points will be shuffled and sliced to match exact value", )
    generate_minimal_distance: FloatProperty(name="Minimal Distance", default=0.1, precision=3, subtype='DISTANCE', description="Poisson Disk minimal distance between points, the smaller value, the slower calculation", )
    generate_sampling_exponent: IntProperty(name="Sampling Exponent", default=5, min=1, description="Poisson Disk presampling exponent, lower values are faster but less even, higher values are slower exponentially", )
    
    # debug_shader: EnumProperty(name="Debug Shader", items=[('NONE', "None", ""),
    #                                                        ('DEPTH', "Depth", ""),
    #                                                        ('NORMAL', "Normal", ""),
    #                                                        ('POSITION', "Position", ""),
    #                                                        ], default='NONE', description="", )
    override_default_shader: BoolProperty(default=False, options={'HIDDEN', }, )
    
    # def _update_override_default_shader(self, context, ):
    #     if(self.dev_depth_enabled or self.dev_normal_colors_enabled or self.dev_position_colors_enabled):
    #         self.override_default_shader = True
    #     else:
    #         self.override_default_shader = False
    
    def _update_dev_depth(self, context, ):
        if(self.dev_depth_enabled):
            # FIXME: this is really getting ridiculous
            self.dev_normal_colors_enabled = False
            self.dev_position_colors_enabled = False
            
            self.dev_minimal_shader_enabled = False
            self.dev_minimal_shader_variable_size_enabled = False
            self.dev_minimal_shader_variable_size_and_depth_enabled = False
            self.dev_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_no_depth_enabled = False
            self.dev_phong_shader_enabled = False
            self.clip_shader_enabled = False
            self.billboard_phong_enabled = False
            self.skip_point_shader_enabled = False
            
            self.override_default_shader = True
        else:
            self.override_default_shader = False
    
    def _update_dev_normal(self, context, ):
        if(self.dev_normal_colors_enabled):
            # FIXME: this is really getting ridiculous
            self.dev_depth_enabled = False
            self.dev_position_colors_enabled = False
            
            self.dev_minimal_shader_enabled = False
            self.dev_minimal_shader_variable_size_enabled = False
            self.dev_minimal_shader_variable_size_and_depth_enabled = False
            self.dev_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_no_depth_enabled = False
            self.dev_phong_shader_enabled = False
            self.clip_shader_enabled = False
            self.billboard_phong_enabled = False
            self.skip_point_shader_enabled = False
            
            self.override_default_shader = True
        else:
            self.override_default_shader = False
    
    def _update_dev_position(self, context, ):
        if(self.dev_position_colors_enabled):
            # FIXME: this is really getting ridiculous
            self.dev_depth_enabled = False
            self.dev_normal_colors_enabled = False
            
            self.dev_minimal_shader_enabled = False
            self.dev_minimal_shader_variable_size_enabled = False
            self.dev_minimal_shader_variable_size_and_depth_enabled = False
            self.dev_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_no_depth_enabled = False
            self.dev_phong_shader_enabled = False
            self.clip_shader_enabled = False
            self.billboard_phong_enabled = False
            self.skip_point_shader_enabled = False
            
            self.override_default_shader = True
        else:
            self.override_default_shader = False
    
    # dev_depth_enabled: BoolProperty(name="Depth", default=False, description="", update=_update_override_default_shader, )
    dev_depth_enabled: BoolProperty(name="Depth", default=False, description="Enable depth debug shader", update=_update_dev_depth, )
    # dev_depth_edit: BoolProperty(name="Edit", description="Edit depth shader properties", default=False, )
    dev_depth_brightness: FloatProperty(name="Brightness", description="Depth shader color brightness", default=0.0, min=-10.0, max=10.0, )
    dev_depth_contrast: FloatProperty(name="Contrast", description="Depth shader color contrast", default=1.0, min=-10.0, max=10.0, )
    dev_depth_false_colors: BoolProperty(name="False Colors", default=False, description="Display depth shader in false colors", )
    dev_depth_color_a: FloatVectorProperty(name="Color A", description="Depth shader false colors front color", default=(0.0, 1.0, 0.0, ), min=0, max=1, subtype='COLOR', size=3, )
    dev_depth_color_b: FloatVectorProperty(name="Color B", description="Depth shader false colors back color", default=(0.0, 0.0, 1.0, ), min=0, max=1, subtype='COLOR', size=3, )
    # dev_normal_colors_enabled: BoolProperty(name="Normal", default=False, description="", update=_update_override_default_shader, )
    dev_normal_colors_enabled: BoolProperty(name="Normal", default=False, description="Enable normal debug shader", update=_update_dev_normal, )
    # dev_position_colors_enabled: BoolProperty(name="Position", default=False, description="", update=_update_override_default_shader, )
    dev_position_colors_enabled: BoolProperty(name="Position", default=False, description="Enable position debug shader", update=_update_dev_position, )
    
    # NOTE: icon for bounding box 'SHADING_BBOX' ?
    dev_bbox_enabled: BoolProperty(name="Bounding Box", default=False, description="", )
    dev_bbox_color: FloatVectorProperty(name="Color", description="", default=(0.7, 0.7, 0.7), min=0, max=1, subtype='COLOR', size=3, )
    dev_bbox_size: FloatProperty(name="Size", description="", default=0.3, min=0.1, max=0.9, subtype='FACTOR', )
    dev_bbox_alpha: FloatProperty(name="Alpha", description="", default=0.7, min=0.0, max=1.0, subtype='FACTOR', )
    
    def _dev_sel_color_update(self, context, ):
        bpy.context.preferences.addons[__name__].preferences.selection_color = self.dev_selection_shader_color
    
    dev_selection_shader_display: BoolProperty(name="Selection", default=False, description="", )
    dev_selection_shader_color: FloatVectorProperty(name="Color", description="", default=(1.0, 0.0, 0.0, 0.5), min=0, max=1, subtype='COLOR', size=4, update=_dev_sel_color_update, )
    
    def _update_color_adjustment(self, context, ):
        if(self.color_adjustment_shader_enabled):
            # FIXME: this is really getting ridiculous
            self.dev_depth_enabled = False
            self.dev_normal_colors_enabled = False
            self.dev_position_colors_enabled = False
            
            self.dev_minimal_shader_enabled = False
            self.dev_minimal_shader_variable_size_enabled = False
            self.dev_minimal_shader_variable_size_and_depth_enabled = False
            self.dev_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_no_depth_enabled = False
            self.dev_phong_shader_enabled = False
            self.clip_shader_enabled = False
            self.billboard_phong_enabled = False
            self.skip_point_shader_enabled = False
            
            self.illumination = False
            self.override_default_shader = True
        else:
            self.override_default_shader = False
    
    color_adjustment_shader_enabled: BoolProperty(name="Enabled", default=False, description="Enable color adjustment shader, other shaders will be overrided until disabled", update=_update_color_adjustment, )
    color_adjustment_shader_exposure: FloatProperty(name="Exposure", description="formula: color = color * (2 ** value)", default=0.0, min=-5.0, max=5.0, )
    color_adjustment_shader_gamma: FloatProperty(name="Gamma", description="formula: color = color ** (1 / value)", default=1.0, min=0.01, max=9.99, )
    color_adjustment_shader_brightness: FloatProperty(name="Brightness", description="formula: color = (color - 0.5) * contrast + 0.5 + brightness", default=0.0, min=-5.0, max=5.0, )
    color_adjustment_shader_contrast: FloatProperty(name="Contrast", description="formula: color = (color - 0.5) * contrast + 0.5 + brightness", default=1.0, min=0.0, max=10.0, )
    color_adjustment_shader_hue: FloatProperty(name="Hue", description="formula: color.h = (color.h + (value % 1.0)) % 1.0", default=0.0, min=0.0, max=1.0, )
    color_adjustment_shader_saturation: FloatProperty(name="Saturation", description="formula: color.s += value", default=0.0, min=-1.0, max=1.0, )
    color_adjustment_shader_value: FloatProperty(name="Value", description="formula: color.v += value", default=0.0, min=-1.0, max=1.0, )
    color_adjustment_shader_invert: BoolProperty(name="Invert", description="formula: color = 1.0 - color", default=False, )
    
    def _update_minimal_shader(self, context, ):
        if(self.dev_minimal_shader_enabled):
            # FIXME: this is really getting ridiculous
            self.illumination = False
            self.dev_depth_enabled = False
            self.dev_normal_colors_enabled = False
            self.dev_position_colors_enabled = False
            self.color_adjustment_shader_enabled = False
            self.dev_minimal_shader_variable_size_enabled = False
            self.dev_minimal_shader_variable_size_and_depth_enabled = False
            self.dev_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_no_depth_enabled = False
            self.dev_phong_shader_enabled = False
            self.clip_shader_enabled = False
            self.billboard_phong_enabled = False
            self.skip_point_shader_enabled = False
            
            self.override_default_shader = True
        else:
            self.override_default_shader = False
    
    def _update_minimal_shader_variable_size(self, context, ):
        if(self.dev_minimal_shader_variable_size_enabled):
            # FIXME: this is really getting ridiculous
            self.illumination = False
            self.dev_depth_enabled = False
            self.dev_normal_colors_enabled = False
            self.dev_position_colors_enabled = False
            self.color_adjustment_shader_enabled = False
            self.dev_minimal_shader_enabled = False
            self.dev_minimal_shader_variable_size_and_depth_enabled = False
            self.dev_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_no_depth_enabled = False
            self.dev_phong_shader_enabled = False
            self.clip_shader_enabled = False
            self.billboard_phong_enabled = False
            self.skip_point_shader_enabled = False
            
            self.override_default_shader = True
        else:
            self.override_default_shader = False
    
    dev_minimal_shader_enabled: BoolProperty(name="Enabled", default=False, description="Enable minimal shader", update=_update_minimal_shader, )
    dev_minimal_shader_variable_size_enabled: BoolProperty(name="Enabled", default=False, description="Enable minimal shader with variable size", update=_update_minimal_shader_variable_size, )
    
    def _update_minimal_shader_variable_size_with_depth(self, context, ):
        if(self.dev_minimal_shader_variable_size_and_depth_enabled):
            # FIXME: this is really getting ridiculous
            self.illumination = False
            self.dev_depth_enabled = False
            self.dev_normal_colors_enabled = False
            self.dev_position_colors_enabled = False
            self.color_adjustment_shader_enabled = False
            self.dev_minimal_shader_enabled = False
            self.dev_minimal_shader_variable_size_enabled = False
            self.dev_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_no_depth_enabled = False
            self.dev_phong_shader_enabled = False
            self.clip_shader_enabled = False
            self.billboard_phong_enabled = False
            self.skip_point_shader_enabled = False
            
            self.override_default_shader = True
        else:
            self.override_default_shader = False
    
    dev_minimal_shader_variable_size_and_depth_enabled: BoolProperty(name="Enabled", default=False, description="Enable minimal shader with variable size with depth", update=_update_minimal_shader_variable_size_with_depth, )
    dev_minimal_shader_variable_size_and_depth_brightness: FloatProperty(name="Brightness", default=0.25, min=-10.0, max=10.0, description="Depth shader color brightness", )
    dev_minimal_shader_variable_size_and_depth_contrast: FloatProperty(name="Contrast", default=0.5, min=-10.0, max=10.0, description="Depth shader color contrast", )
    dev_minimal_shader_variable_size_and_depth_blend: FloatProperty(name="Blend", default=0.75, min=0.0, max=1.0, subtype='FACTOR', description="Depth shader blending with original colors", )
    
    def _update_dev_billboard_point_cloud_enabled(self, context, ):
        if(self.dev_billboard_point_cloud_enabled):
            # FIXME: this is really getting ridiculous
            self.illumination = False
            self.dev_depth_enabled = False
            self.dev_normal_colors_enabled = False
            self.dev_position_colors_enabled = False
            self.color_adjustment_shader_enabled = False
            self.dev_minimal_shader_enabled = False
            self.dev_minimal_shader_variable_size_enabled = False
            self.dev_minimal_shader_variable_size_and_depth_enabled = False
            self.dev_rich_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_no_depth_enabled = False
            self.dev_phong_shader_enabled = False
            self.clip_shader_enabled = False
            self.billboard_phong_enabled = False
            self.skip_point_shader_enabled = False
            
            self.override_default_shader = True
        else:
            self.override_default_shader = False
    
    dev_billboard_point_cloud_enabled: BoolProperty(name="Enabled", default=False, description="Enable Billboard Shader", update=_update_dev_billboard_point_cloud_enabled, )
    dev_billboard_point_cloud_size: FloatProperty(name="Size", default=0.002, min=0.0001, max=0.2, description="", precision=6, )
    
    def _update_dev_rich_billboard_point_cloud_enabled(self, context):
        if(self.dev_rich_billboard_point_cloud_enabled):
            # FIXME: this is really getting ridiculous
            self.illumination = False
            self.dev_depth_enabled = False
            self.dev_normal_colors_enabled = False
            self.dev_position_colors_enabled = False
            self.color_adjustment_shader_enabled = False
            self.dev_minimal_shader_enabled = False
            self.dev_minimal_shader_variable_size_enabled = False
            self.dev_minimal_shader_variable_size_and_depth_enabled = False
            self.dev_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_no_depth_enabled = False
            self.dev_phong_shader_enabled = False
            self.clip_shader_enabled = False
            self.billboard_phong_enabled = False
            self.skip_point_shader_enabled = False
            
            self.override_default_shader = True
        else:
            self.override_default_shader = False
    
    dev_rich_billboard_point_cloud_enabled: BoolProperty(name="Enabled", default=False, description="Enable Rich Billboard Shader", update=_update_dev_rich_billboard_point_cloud_enabled, )
    dev_rich_billboard_point_cloud_size: FloatProperty(name="Size", default=0.01, min=0.0001, max=1.0, description="", precision=6, )
    dev_rich_billboard_depth_brightness: FloatProperty(name="Brightness", default=0.25, min=-10.0, max=10.0, description="Depth shader color brightness", )
    dev_rich_billboard_depth_contrast: FloatProperty(name="Contrast", default=0.5, min=-10.0, max=10.0, description="Depth shader color contrast", )
    dev_rich_billboard_depth_blend: FloatProperty(name="Blend", default=0.75, min=0.0, max=1.0, subtype='FACTOR', description="Depth shader blending with original colors", )
    
    def _update_dev_rich_billboard_point_cloud_no_depth_enabled(self, context):
        if(self.dev_rich_billboard_point_cloud_no_depth_enabled):
            # FIXME: this is really getting ridiculous
            self.illumination = False
            self.dev_depth_enabled = False
            self.dev_normal_colors_enabled = False
            self.dev_position_colors_enabled = False
            self.color_adjustment_shader_enabled = False
            self.dev_minimal_shader_enabled = False
            self.dev_minimal_shader_variable_size_enabled = False
            self.dev_minimal_shader_variable_size_and_depth_enabled = False
            self.dev_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_enabled = False
            self.dev_phong_shader_enabled = False
            self.clip_shader_enabled = False
            self.billboard_phong_enabled = False
            self.skip_point_shader_enabled = False
            
            self.override_default_shader = True
        else:
            self.override_default_shader = False
    
    dev_rich_billboard_point_cloud_no_depth_enabled: BoolProperty(name="Enabled", default=False, description="Enable Rich Billboard Shader Without Depth", update=_update_dev_rich_billboard_point_cloud_no_depth_enabled, )
    
    def _update_dev_phong_shader_enabled(self, context):
        if(self.dev_phong_shader_enabled):
            # FIXME: this is really getting ridiculous
            self.illumination = False
            self.dev_depth_enabled = False
            self.dev_normal_colors_enabled = False
            self.dev_position_colors_enabled = False
            self.color_adjustment_shader_enabled = False
            self.dev_minimal_shader_enabled = False
            self.dev_minimal_shader_variable_size_enabled = False
            self.dev_minimal_shader_variable_size_and_depth_enabled = False
            self.dev_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_no_depth_enabled = False
            self.clip_shader_enabled = False
            self.billboard_phong_enabled = False
            self.skip_point_shader_enabled = False
            
            self.override_default_shader = True
        else:
            self.override_default_shader = False
    
    dev_phong_shader_enabled: BoolProperty(name="Enabled", default=False, description="", update=_update_dev_phong_shader_enabled, )
    dev_phong_shader_ambient_strength: FloatProperty(name="ambient_strength", default=0.5, min=0.0, max=1.0, description="", )
    dev_phong_shader_specular_strength: FloatProperty(name="specular_strength", default=0.5, min=0.0, max=1.0, description="", )
    dev_phong_shader_specular_exponent: FloatProperty(name="specular_exponent", default=8.0, min=1.0, max=512.0, description="", )
    
    debug_panel_show_properties: BoolProperty(default=False, options={'HIDDEN', }, )
    debug_panel_show_manager: BoolProperty(default=False, options={'HIDDEN', }, )
    debug_panel_show_sequence: BoolProperty(default=False, options={'HIDDEN', }, )
    debug_panel_show_cache_items: BoolProperty(default=False, options={'HIDDEN', }, )
    
    # store info how long was last draw call, ie get points from cache, join, draw
    pcviv_debug_draw: StringProperty(default="", )
    pcviv_debug_panel_show_info: BoolProperty(default=False, options={'HIDDEN', }, )
    # have to provide prop for indexing, not needed for anything in this case
    pcviv_material_list_active_index: IntProperty(name="Index", default=0, description="", options={'HIDDEN', }, )
    
    # testing / development stuff
    def _dev_transform_normals_target_object_poll(self, o, ):
        if(o and o.type in ('MESH', )):
            return True
        return False
    
    dev_transform_normals_target_object: PointerProperty(type=bpy.types.Object, name="Object", description="", poll=_dev_transform_normals_target_object_poll, )
    
    # dev
    def _clip_shader_enabled(self, context):
        if(self.clip_shader_enabled):
            # FIXME: this is really getting ridiculous
            self.illumination = False
            self.dev_depth_enabled = False
            self.dev_normal_colors_enabled = False
            self.dev_position_colors_enabled = False
            self.color_adjustment_shader_enabled = False
            self.dev_minimal_shader_enabled = False
            self.dev_minimal_shader_variable_size_enabled = False
            self.dev_minimal_shader_variable_size_and_depth_enabled = False
            self.dev_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_no_depth_enabled = False
            self.dev_phong_shader_enabled = False
            self.billboard_phong_enabled = False
            self.skip_point_shader_enabled = False
            
            self.override_default_shader = True
        else:
            self.override_default_shader = False
    
    clip_shader_enabled: BoolProperty(name="Enabled", default=False, description="", update=_clip_shader_enabled, )
    clip_plane0_enabled: BoolProperty(name="Enabled", default=False, description="", )
    clip_plane1_enabled: BoolProperty(name="Enabled", default=False, description="", )
    clip_plane2_enabled: BoolProperty(name="Enabled", default=False, description="", )
    clip_plane3_enabled: BoolProperty(name="Enabled", default=False, description="", )
    clip_plane4_enabled: BoolProperty(name="Enabled", default=False, description="", )
    clip_plane5_enabled: BoolProperty(name="Enabled", default=False, description="", )
    clip_plane0: FloatVectorProperty(name="Plane 0", default=(0.0, 0.0, 0.0, 0.0), subtype='NONE', size=4, description="", )
    clip_plane1: FloatVectorProperty(name="Plane 1", default=(0.0, 0.0, 0.0, 0.0), subtype='NONE', size=4, description="", )
    clip_plane2: FloatVectorProperty(name="Plane 2", default=(0.0, 0.0, 0.0, 0.0), subtype='NONE', size=4, description="", )
    clip_plane3: FloatVectorProperty(name="Plane 3", default=(0.0, 0.0, 0.0, 0.0), subtype='NONE', size=4, description="", )
    clip_plane4: FloatVectorProperty(name="Plane 4", default=(0.0, 0.0, 0.0, 0.0), subtype='NONE', size=4, description="", )
    clip_plane5: FloatVectorProperty(name="Plane 5", default=(0.0, 0.0, 0.0, 0.0), subtype='NONE', size=4, description="", )
    
    def _clip_planes_from_bbox_object_poll(self, o, ):
        if(o and o.type in ('MESH', )):
            return True
        return False
    
    clip_planes_from_bbox_object: PointerProperty(type=bpy.types.Object, name="Object", description="", poll=_clip_planes_from_bbox_object_poll, )
    
    def _billboard_phong_enabled(self, context):
        if(self.billboard_phong_enabled):
            # FIXME: this is really getting ridiculous
            self.illumination = False
            self.dev_depth_enabled = False
            self.dev_normal_colors_enabled = False
            self.dev_position_colors_enabled = False
            self.color_adjustment_shader_enabled = False
            self.dev_minimal_shader_enabled = False
            self.dev_minimal_shader_variable_size_enabled = False
            self.dev_minimal_shader_variable_size_and_depth_enabled = False
            self.dev_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_no_depth_enabled = False
            self.dev_phong_shader_enabled = False
            self.clip_shader_enabled = False
            self.skip_point_shader_enabled = False
            
            self.override_default_shader = True
        else:
            self.override_default_shader = False
    
    billboard_phong_enabled: BoolProperty(name="Enabled", default=False, description="", update=_billboard_phong_enabled, )
    billboard_phong_circles: BoolProperty(name="Circles (slower)", default=False, description="", )
    billboard_phong_size: FloatProperty(name="Size", default=0.002, min=0.0001, max=0.2, description="", precision=6, )
    billboard_phong_ambient_strength: FloatProperty(name="Ambient", default=0.5, min=0.0, max=1.0, description="", )
    billboard_phong_specular_strength: FloatProperty(name="Specular", default=0.5, min=0.0, max=1.0, description="", )
    billboard_phong_specular_exponent: FloatProperty(name="Hardness", default=8.0, min=1.0, max=512.0, description="", )
    
    def _skip_point_shader_enabled(self, context):
        if(self.skip_point_shader_enabled):
            # FIXME: this is really getting ridiculous
            self.illumination = False
            self.dev_depth_enabled = False
            self.dev_normal_colors_enabled = False
            self.dev_position_colors_enabled = False
            self.color_adjustment_shader_enabled = False
            self.dev_minimal_shader_enabled = False
            self.dev_minimal_shader_variable_size_enabled = False
            self.dev_minimal_shader_variable_size_and_depth_enabled = False
            self.dev_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_no_depth_enabled = False
            self.dev_phong_shader_enabled = False
            self.clip_shader_enabled = False
            self.billboard_phong_enabled = False
            
            self.override_default_shader = True
        else:
            self.override_default_shader = False
    
    skip_point_shader_enabled: BoolProperty(name="Enabled", default=False, description="", update=_skip_point_shader_enabled, )
    # skip_point_percentage: FloatProperty(name="Skip Percentage", default=100.0, min=0.0, max=100.0, precision=0, subtype='PERCENTAGE', description="", )
    skip_point_percentage: FloatProperty(name="Skip Percentage", default=100.0, min=0.0, max=100.0, precision=3, description="", )
    
    @classmethod
    def register(cls):
        bpy.types.Object.point_cloud_visualizer = PointerProperty(type=cls)
    
    @classmethod
    def unregister(cls):
        del bpy.types.Object.point_cloud_visualizer


def _update_panel_bl_category(self, context, ):
    _main_panel = PCV_PT_panel
    # NOTE: maybe generate those from 'classes' tuple, or, just don't forget to append new panel also here..
    _sub_panels = (
        PCV_PT_clip,
        PCV_PT_edit, PCV_PT_filter, PCV_PT_filter_simplify, PCV_PT_filter_project, PCV_PT_filter_boolean, PCV_PT_filter_remove_color,
        PCV_PT_filter_merge, PCV_PT_filter_join, PCV_PT_filter_color_adjustment, PCV_PT_render, PCV_PT_convert, PCV_PT_generate, PCV_PT_export, PCV_PT_sequence,
        PCV_PT_development,
        PCV_PT_debug,
    )
    try:
        p = _main_panel
        bpy.utils.unregister_class(p)
        for sp in _sub_panels:
            bpy.utils.unregister_class(sp)
        prefs = context.preferences.addons[__name__].preferences
        c = prefs.category_custom
        n = ''
        if(c):
            n = prefs.category_custom_name
        else:
            v = prefs.category
            ei = prefs.bl_rna.properties['category'].enum_items
            for e in ei:
                if(e.identifier == v):
                    n = e.name
        if(n == ''):
            raise Exception('Name is empty string')
        p.bl_category = n
        bpy.utils.register_class(p)
        for sp in _sub_panels:
            bpy.utils.register_class(sp)
    except Exception as e:
        log('PCV: setting tab name failed ({})'.format(str(e)))


class PCV_preferences(AddonPreferences):
    bl_idname = __name__
    
    default_vertex_color: FloatVectorProperty(name="Default", default=(0.65, 0.65, 0.65, ), min=0, max=1, subtype='COLOR', size=3, description="Default color to be used upon loading PLY to cache when vertex colors are missing", )
    normal_color: FloatVectorProperty(name="Normal", default=((35 / 255) ** 2.2, (97 / 255) ** 2.2, (221 / 255) ** 2.2, ), min=0, max=1, subtype='COLOR', size=3, description="Display color for vertex normals lines", )
    selection_color: FloatVectorProperty(name="Selection", description="Display color for selection", default=(1.0, 0.0, 0.0, 0.5), min=0, max=1, subtype='COLOR', size=4, )
    convert_16bit_colors: BoolProperty(name="Convert 16bit Colors", description="Convert 16bit colors to 8bit, applied when Red channel has 'uint16' dtype", default=True, )
    gamma_correct_16bit_colors: BoolProperty(name="Gamma Correct 16bit Colors", description="When 16bit colors are encountered apply gamma as 'c ** (1 / 2.2)'", default=False, )
    shuffle_points: BoolProperty(name="Shuffle Points", description="Shuffle points upon loading, display percentage is more useable if points are shuffled", default=True, )
    category: EnumProperty(name="Tab Name", items=[('POINT_CLOUD_VISUALIZER', "Point Cloud Visualizer", ""),
                                                   ('PCV', "PCV", ""), ], default='POINT_CLOUD_VISUALIZER', description="To have PCV in its own separate tab, choose one", update=_update_panel_bl_category, )
    category_custom: BoolProperty(name="Custom Tab Name", default=False, description="Check if you want to have PCV in custom named tab or in existing tab", update=_update_panel_bl_category, )
    category_custom_name: StringProperty(name="Name", default="View", description="Custom PCV tab name, if you choose one from already existing tabs it will append to that tab", update=_update_panel_bl_category, )
    
    def draw(self, context):
        l = self.layout
        r = l.row()
        r.prop(self, "default_vertex_color")
        r.prop(self, "normal_color")
        r.prop(self, "selection_color")
        r = l.row()
        r.prop(self, "shuffle_points")
        r.prop(self, "convert_16bit_colors")
        c = r.column()
        c.prop(self, "gamma_correct_16bit_colors")
        if(not self.convert_16bit_colors):
            c.active = False
        
        f = 0.5
        r = l.row()
        s = r.split(factor=f)
        c = s.column()
        c.prop(self, "category")
        if(self.category_custom):
            c.enabled = False
        s = s.split(factor=1.0)
        r = s.row()
        r.prop(self, "category_custom")
        c = r.column()
        c.prop(self, "category_custom_name")
        if(not self.category_custom):
            c.enabled = False
    
    # @classmethod
    # def prefs(cls, context=None, ):
    #     if(context is None):
    #         context = bpy.context
    #     return context.preferences.addons[__name__].preferences


@persistent
def watcher(scene):
    PCVSequence.deinit()
    PCVManager.deinit()


classes = (
    PCV_properties, PCV_preferences,
    
    PCV_PT_panel, PCV_PT_clip, PCV_PT_edit,
    PCV_PT_filter, PCV_PT_filter_simplify, PCV_PT_filter_project, PCV_PT_filter_boolean, PCV_PT_filter_remove_color, PCV_PT_filter_merge,
    PCV_PT_filter_join, PCV_PT_filter_color_adjustment,
    PCV_PT_render, PCV_PT_convert, PCV_PT_generate, PCV_PT_export, PCV_PT_sequence,
    
    PCV_OT_load, PCV_OT_draw, PCV_OT_erase, PCV_OT_render, PCV_OT_render_animation, PCV_OT_convert, PCV_OT_reload, PCV_OT_export,
    PCV_OT_filter_simplify, PCV_OT_filter_remove_color, PCV_OT_filter_remove_color_delete_selected, PCV_OT_filter_remove_color_deselect,
    PCV_OT_filter_project, PCV_OT_filter_merge, PCV_OT_filter_boolean_intersect, PCV_OT_filter_boolean_exclude,
    PCV_OT_edit_start, PCV_OT_edit_update, PCV_OT_edit_end, PCV_OT_edit_cancel,
    PCV_OT_sequence_preload, PCV_OT_sequence_clear, PCV_OT_generate_point_cloud, PCV_OT_reset_runtime,
    PCV_OT_color_adjustment_shader_reset, PCV_OT_color_adjustment_shader_apply, PCV_OT_filter_join,
    
    PCV_PT_development,
    PCV_OT_generate_volume_point_cloud,
    
    PCV_OT_clip_planes_from_bbox, PCV_OT_clip_planes_reset, PCV_OT_clip_planes_from_camera_view,
    
    PCV_PT_debug,
    PCV_OT_init, PCV_OT_deinit, PCV_OT_gc, PCV_OT_seq_init, PCV_OT_seq_deinit,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    
    _update_panel_bl_category(None, bpy.context)


def unregister():
    PCVSequence.deinit()
    PCVManager.deinit()
    
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    """
    > Well, that doesn't explain... why you've come all the way out here, all the way out here to hell.
    > I, uh, have a job out in the town of Machine.
    > Machine? That's the end of the line.
    Jim Jarmusch, Dead Man (1995)
    """
    register()
