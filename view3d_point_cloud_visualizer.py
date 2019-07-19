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
           "description": "Display, render and convert to mesh colored point cloud PLY files.",
           "author": "Jakub Uhlik",
           "version": (0, 9, 4),
           "blender": (2, 80, 0),
           "location": "3D Viewport > Sidebar > View > Point Cloud Visualizer",
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
import textwrap
import sys

import bpy
import bmesh
from bpy.props import PointerProperty, BoolProperty, StringProperty, FloatProperty, IntProperty, FloatVectorProperty, EnumProperty
from bpy.types import PropertyGroup, Panel, Operator, AddonPreferences
import gpu
from gpu.types import GPUOffScreen, GPUShader, GPUBatch, GPUVertBuf, GPUVertFormat
from gpu_extras.batch import batch_for_shader
from bpy.app.handlers import persistent
import bgl
from mathutils import Matrix, Vector, Quaternion, Color
from bpy_extras.object_utils import world_to_camera_view
from bpy_extras.io_utils import axis_conversion, ExportHelper
from mathutils.kdtree import KDTree


# FIXME undo still doesn't work in some cases, from what i've seen, only when i am undoing operations on parent object, especially when you undo/redo e.g. transforms around load/draw operators, filepath property gets reset and the whole thing is drawn, but ui looks like loding never happened, i've added a quick fix storing path in cache, but it all depends on object name and this is bad.
# FIXME ply loading might not work with all ply files, for example, file spec seems does not forbid having two or more blocks of vertices with different props, currently i load only first block of vertices. maybe construct some messed up ply and test how for example meshlab behaves
# FIXME checking for normals/colors in points is kinda scattered all over
# TODO better docs, some gifs would be the best, i personally hate watching video tutorials when i need just sigle bit of information buried in 10+ minutes video, what a waste of time
# TODO try to remove manual depth test during offscreen rendering
# NOTE parent object reference check should be before drawing, not in the middle, it's not that bad, it's pretty early, but it's still messy, this will require rewrite of handler and render functions in manager.. so don't touch until broken
# NOTE ~2k lines, maybe time to break into modules, but having sigle file is not a bad thing.. update: >3k now and having a sigle file is still better..
# NOTE $ pycodestyle --ignore=W293,E501,E741,E402 --exclude='io_mesh_fast_obj/blender' .


DEBUG = False
EXPERIMENTAL = False


def log(msg, indent=0, ):
    m = "{0}> {1}".format("    " * indent, msg)
    if(DEBUG):
        print(m)


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
        if(not DEBUG):
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


class InstanceMeshGenerator():
    def __init__(self):
        self.def_verts, self.def_edges, self.def_faces = self.generate()
    
    def generate(self):
        return [(0, 0, 0, ), ], [], []


class VertexMeshGenerator(InstanceMeshGenerator):
    def __init__(self):
        log("{}:".format(self.__class__.__name__), 0, )
        super(VertexMeshGenerator, self).__init__()


class TetrahedronMeshGenerator(InstanceMeshGenerator):
    def __init__(self, length=1.0, ):
        log("{}:".format(self.__class__.__name__), 0, )
        if(length <= 0):
            log("length is (or less than) 0, which is ridiculous. setting to 0.001..", 1)
            length = 0.001
        self.length = length
        super(TetrahedronMeshGenerator, self).__init__()
    
    def generate(self):
        def circle2d_coords(radius, steps, offset, ox, oy):
            r = []
            angstep = 2 * math.pi / steps
            for i in range(steps):
                x = math.sin(i * angstep + offset) * radius + ox
                y = math.cos(i * angstep + offset) * radius + oy
                r.append((x, y))
            return r
        
        l = self.length
        excircle_radius = math.sqrt(3) / 3 * l
        c = circle2d_coords(excircle_radius, 3, 0, 0, 0)
        h = l / 3 * math.sqrt(6)
        dv = [(c[0][0], c[0][1], 0, ),
              (c[1][0], c[1][1], 0, ),
              (c[2][0], c[2][1], 0, ),
              (0, 0, h, ), ]
        df = ([(0, 1, 2),
               (3, 2, 1),
               (3, 1, 0),
               (3, 0, 2), ])
        return dv, [], df


class EquilateralTriangleMeshGenerator(InstanceMeshGenerator):
    def __init__(self, length=1.0, offset=0.0, ):
        log("{}:".format(self.__class__.__name__), 0, )
        if(length <= 0):
            log("got ridiculous length value (smaller or equal to 0).. setting to 0.001", 1)
            length = 0.001
        self.length = length
        self.offset = offset
        super(EquilateralTriangleMeshGenerator, self).__init__()
    
    def generate(self):
        def circle2d_coords(radius, steps, offset, ox, oy):
            r = []
            angstep = 2 * math.pi / steps
            for i in range(steps):
                x = math.sin(i * angstep + offset) * radius + ox
                y = math.cos(i * angstep + offset) * radius + oy
                r.append((x, y))
            return r
        
        r = math.sqrt(3) / 3 * self.length
        c = circle2d_coords(r, 3, self.offset, 0, 0)
        dv = []
        for i in c:
            dv.append((i[0], i[1], 0, ))
        df = [(0, 2, 1, ), ]
        return dv, [], df


class IcoSphereMeshGenerator(InstanceMeshGenerator):
    def __init__(self, radius=1, subdivision=2, ):
        log("{}:".format(self.__class__.__name__), 0, )
        if(radius <= 0):
            log("radius is (or less than) 0, which is ridiculous. setting to 0.001..", 1)
            radius = 0.001
        self.radius = radius
        subdivision = int(subdivision)
        if(not (0 < subdivision <= 2)):
            log("subdivision 1 or 2 allowed, not {}, setting to 1".format(subdivision), 1)
            subdivision = 1
        self.subdivision = subdivision
        super(IcoSphereMeshGenerator, self).__init__()
    
    def generate(self):
        if(self.subdivision == 1):
            dv = [(0.0, 0.0, -0.5), (0.3617999851703644, -0.2628600001335144, -0.22360749542713165), (-0.13819250464439392, -0.42531999945640564, -0.22360749542713165), (-0.44721248745918274, 0.0, -0.22360749542713165), (-0.13819250464439392, 0.42531999945640564, -0.22360749542713165), (0.3617999851703644, 0.2628600001335144, -0.22360749542713165), (0.13819250464439392, -0.42531999945640564, 0.22360749542713165), (-0.3617999851703644, -0.2628600001335144, 0.22360749542713165), (-0.3617999851703644, 0.2628600001335144, 0.22360749542713165), (0.13819250464439392, 0.42531999945640564, 0.22360749542713165), (0.44721248745918274, 0.0, 0.22360749542713165), (0.0, 0.0, 0.5)]
            df = [(0, 1, 2), (1, 0, 5), (0, 2, 3), (0, 3, 4), (0, 4, 5), (1, 5, 10), (2, 1, 6), (3, 2, 7), (4, 3, 8), (5, 4, 9), (1, 10, 6), (2, 6, 7), (3, 7, 8), (4, 8, 9), (5, 9, 10), (6, 10, 11), (7, 6, 11), (8, 7, 11), (9, 8, 11), (10, 9, 11)]
        elif(self.subdivision == 2):
            dv = [(0.0, 0.0, -0.5), (0.36180365085601807, -0.2628626525402069, -0.22360976040363312), (-0.1381940096616745, -0.42532461881637573, -0.22360992431640625), (-0.4472131133079529, 0.0, -0.22360780835151672), (-0.1381940096616745, 0.42532461881637573, -0.22360992431640625), (0.36180365085601807, 0.2628626525402069, -0.22360976040363312), (0.1381940096616745, -0.42532461881637573, 0.22360992431640625), (-0.36180365085601807, -0.2628626525402069, 0.22360976040363312), (-0.36180365085601807, 0.2628626525402069, 0.22360976040363312), (0.1381940096616745, 0.42532461881637573, 0.22360992431640625), (0.4472131133079529, 0.0, 0.22360780835151672), (0.0, 0.0, 0.5), (-0.08122777938842773, -0.24999763071537018, -0.42532721161842346), (0.21266134083271027, -0.15450569987297058, -0.4253270924091339), (0.13143441081047058, -0.40450581908226013, -0.26286882162094116), (0.4253239333629608, 0.0, -0.2628679573535919), (0.21266134083271027, 0.15450569987297058, -0.4253270924091339), (-0.262864887714386, 0.0, -0.42532584071159363), (-0.3440946936607361, -0.24999846518039703, -0.26286810636520386), (-0.08122777938842773, 0.24999763071537018, -0.42532721161842346), (-0.3440946936607361, 0.24999846518039703, -0.26286810636520386), (0.13143441081047058, 0.40450581908226013, -0.26286882162094116), (0.47552892565727234, -0.15450631082057953, 0.0), (0.47552892565727234, 0.15450631082057953, 0.0), (0.0, -0.4999999701976776, 0.0), (0.2938928008079529, -0.4045083522796631, 0.0), (-0.47552892565727234, -0.15450631082057953, 0.0), (-0.2938928008079529, -0.4045083522796631, 0.0), (-0.2938928008079529, 0.4045083522796631, 0.0), (-0.47552892565727234, 0.15450631082057953, 0.0), (0.2938928008079529, 0.4045083522796631, 0.0), (0.0, 0.4999999701976776, 0.0), (0.3440946936607361, -0.24999846518039703, 0.26286810636520386), (-0.13143441081047058, -0.40450581908226013, 0.26286882162094116), (-0.4253239333629608, 0.0, 0.2628679573535919), (-0.13143441081047058, 0.40450581908226013, 0.26286882162094116), (0.3440946936607361, 0.24999846518039703, 0.26286810636520386), (0.08122777938842773, -0.24999763071537018, 0.4253271818161011), (0.262864887714386, 0.0, 0.42532584071159363), (-0.21266134083271027, -0.15450569987297058, 0.4253270924091339), (-0.21266134083271027, 0.15450569987297058, 0.4253270924091339), (0.08122777938842773, 0.24999763071537018, 0.4253271818161011)]
            df = [(0, 13, 12), (1, 13, 15), (0, 12, 17), (0, 17, 19), (0, 19, 16), (1, 15, 22), (2, 14, 24), (3, 18, 26), (4, 20, 28), (5, 21, 30), (1, 22, 25), (2, 24, 27), (3, 26, 29), (4, 28, 31), (5, 30, 23), (6, 32, 37), (7, 33, 39), (8, 34, 40), (9, 35, 41), (10, 36, 38), (38, 41, 11), (38, 36, 41), (36, 9, 41), (41, 40, 11), (41, 35, 40), (35, 8, 40), (40, 39, 11), (40, 34, 39), (34, 7, 39), (39, 37, 11), (39, 33, 37), (33, 6, 37), (37, 38, 11), (37, 32, 38), (32, 10, 38), (23, 36, 10), (23, 30, 36), (30, 9, 36), (31, 35, 9), (31, 28, 35), (28, 8, 35), (29, 34, 8), (29, 26, 34), (26, 7, 34), (27, 33, 7), (27, 24, 33), (24, 6, 33), (25, 32, 6), (25, 22, 32), (22, 10, 32), (30, 31, 9), (30, 21, 31), (21, 4, 31), (28, 29, 8), (28, 20, 29), (20, 3, 29), (26, 27, 7), (26, 18, 27), (18, 2, 27), (24, 25, 6), (24, 14, 25), (14, 1, 25), (22, 23, 10), (22, 15, 23), (15, 5, 23), (16, 21, 5), (16, 19, 21), (19, 4, 21), (19, 20, 4), (19, 17, 20), (17, 3, 20), (17, 18, 3), (17, 12, 18), (12, 2, 18), (15, 16, 5), (15, 13, 16), (13, 0, 16), (12, 14, 2), (12, 13, 14), (13, 1, 14)]
        else:
            raise ValueError("IcoSphereMeshGenerator: unsupported subdivision: {}".format(self.subdivision))
        return dv, [], df


class CubeMeshGenerator(InstanceMeshGenerator):
    def __init__(self, length=1.0, ):
        log("{}:".format(self.__class__.__name__), 0, )
        if(length <= 0):
            log("less is (or less than) 0, which is ridiculous. setting to 0.001..", 1)
            radius = 0.001
        self.length = length
        super(CubeMeshGenerator, self).__init__()
    
    def generate(self):
        l = self.length / 2
        dv = [(+l, +l, -l),
              (+l, -l, -l),
              (-l, -l, -l),
              (-l, +l, -l),
              (+l, +l, +l),
              (+l, -l, +l),
              (-l, -l, +l),
              (-l, +l, +l), ]
        df = [(0, 1, 2, 3),
              (4, 7, 6, 5),
              (0, 4, 5, 1),
              (1, 5, 6, 2),
              (2, 6, 7, 3),
              (4, 0, 3, 7), ]
        return dv, [], df


class PCMeshInstancer():
    def __init__(self, name, points, generator=None, matrix=None, size=0.01, normal_align=False, vcols=False, ):
        log("{}:".format(self.__class__.__name__), 0, )
        
        self.name = name
        self.points = points
        if(generator is None):
            generator = InstanceMeshGenerator()
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
                # colors.append(Color(col))
                colors.append(col)
            
            num = len(self.def_verts)
            vc = self.mesh.vertex_colors.new()
            for l in self.mesh.loops:
                vi = l.vertex_index
                li = l.index
                c = colors[int(vi / num)]
                # vc.data[li].color = (c.r, c.g, c.b, 1.0, )
                vc.data[li].color = c + (1.0, )
        else:
            log("no mesh loops in mesh", 2, )


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
        # node_emit = nodes.new(type='ShaderNodeEmission')
        # link = links.new(node_tex.outputs[0], node_emit.inputs[0])
        # node_output = nodes.new(type='ShaderNodeOutputMaterial')
        # link = links.new(node_emit.outputs[0], node_output.inputs[0])
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
    _types = {'char': 'c', 'uchar': 'B', 'short': 'h', 'ushort': 'H', 'int': 'i', 'uint': 'I', 'float': 'f', 'double': 'd', }
    
    def __init__(self, path, ):
        log("{}:".format(self.__class__.__name__), 0)
        if(os.path.exists(path) is False or os.path.isdir(path) is True):
            raise OSError("did you point me to an imaginary file? ('{}')".format(path))
        
        self.path = path
        log("will read file at: '{}'".format(self.path), 1)
        log("reading header..", 1)
        self._header()
        log("reading data..", 1)
        # log("data format: {}".format(self._ply_format), 1)
        # log("vertex element properties:", 1)
        # for n, p in self._props:
        #     log("{}: {}".format(n, p), 2)
        if(self._ply_format == 'ascii'):
            self._data_ascii()
        else:
            self._data_binary()
        log("loaded {} vertices".format(len(self.points)), 1)
        # remove alpha if present
        self.points = self.points[[b for b in list(self.points.dtype.names) if b != 'alpha']]
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
        # stream = open(self.path, mode='rb')
        # raw = []
        # h = []
        # for l in stream:
        #     raw.append(l)
        #     a = l.decode('ascii').rstrip()
        #     h.append(a)
        #     if(a == "end_header"):
        #         break
        # # stream.close()
        
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
        
        # if(self._ply_format == 'ascii'):
        #     stream.close()
        # else:
        #     self._stream = stream
        
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
            
            # self._stream.seek(read_from)
            # a = np.fromfile(self._stream, dtype=dt, count=element['count'], )
            
            self.points = a
            read_from += element['count']
        
        # self._stream.close()
    
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
    """Save binary ply file from data PCV is using for drawing on screen
    
    Note:
        colors are stored as float64, writer converts them to uint8
    
    Args:
        path: path to ply file
        vs: vertex array, (x, y, z), (number_of_points, 3) shape, float64 dtype, PCVManager.cache[uuid].vertices
        ns: normal array, (nx, ny, nz), (number_of_points, 3) shape, float64 dtype, PCVManager.cache[uuid].normals
        cs: color array, (r, g, b), (number_of_points, 3) shape, float64 dtype, PCVManager.cache[uuid].colors
    
    Attributes:
        path (str): real path to ply file
    
    """
    
    def __init__(self, path, vs, ns, cs, ):
        log("{}:".format(self.__class__.__name__), 0)
        self.path = os.path.realpath(path)
        
        # join back to structured array and ensure data type
        vs = vs.astype(np.float32)
        ns = ns.astype(np.float32)
        cs = cs.astype(np.float32)
        # back to uint8 colors
        # TODO: maybe store original color values (or all original loaded data) for saving, just for accuracy, it won't use as much extra memory..
        cs = cs * 255
        cs = cs.astype(np.uint8)
        
        l = len(vs)
        dt = [('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('nx', '<f4'), ('ny', '<f4'), ('nz', '<f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
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
        
        # write
        log("will write to: {}".format(self.path), 1)
        # write to temp file first
        n = os.path.splitext(os.path.split(self.path)[1])[0]
        t = "{}.temp.ply".format(n)
        p = os.path.join(os.path.dirname(self.path), t)
        
        with open(p, 'wb') as f:
            # write header
            log("writing header..", 2)
            h = textwrap.dedent("""\
                                ply
                                format binary_little_endian 1.0
                                element vertex {}
                                property float x
                                property float y
                                property float z
                                property float nx
                                property float ny
                                property float nz
                                property uchar red
                                property uchar green
                                property uchar blue
                                comment {}
                                end_header
                                """.format(l, "created with Point Cloud Visualizer", ), )
            f.write(h.encode('ascii'))
            # write data
            log("writing data.. ({} points)".format(l), 2)
            f.write(a.tobytes())
        
        # remove original file (if needed) and rename temp
        if(os.path.exists(self.path)):
            os.remove(self.path)
        shutil.move(p, self.path)
        
        log("done.", 1)


class PCVShaders():
    vertex_shader = '''
        in vec3 position;
        in vec3 normal;
        in vec4 color;
        
        uniform float show_illumination;
        uniform vec3 light_direction;
        uniform vec3 light_intensity;
        uniform vec3 shadow_direction;
        uniform vec3 shadow_intensity;
        uniform float show_normals;
        
        uniform mat4 perspective_matrix;
        uniform mat4 object_matrix;
        uniform float point_size;
        uniform float alpha_radius;
        
        out vec4 f_color;
        out float f_alpha_radius;
        out vec3 f_normal;
        
        out vec3 f_light_direction;
        out vec3 f_light_intensity;
        out vec3 f_shadow_direction;
        out vec3 f_shadow_intensity;
        out float f_show_normals;
        out float f_show_illumination;
        
        void main()
        {
            gl_Position = perspective_matrix * object_matrix * vec4(position, 1.0f);
            gl_PointSize = point_size;
            f_normal = normal;
            f_color = color;
            f_alpha_radius = alpha_radius;
            
            // f_light_direction = normalize(vec3(inverse(object_matrix) * vec4(light_direction, 1.0)));
            f_light_direction = light_direction;
            f_light_intensity = light_intensity;
            // f_shadow_direction = normalize(vec3(inverse(object_matrix) * vec4(shadow_direction, 1.0)));
            f_shadow_direction = shadow_direction;
            f_shadow_intensity = shadow_intensity;
            f_show_normals = show_normals;
            f_show_illumination = show_illumination;
        }
    '''
    fragment_shader = '''
        in vec4 f_color;
        in vec3 f_normal;
        in float f_alpha_radius;
        
        in vec3 f_light_direction;
        in vec3 f_light_intensity;
        in vec3 f_shadow_direction;
        in vec3 f_shadow_intensity;
        in float f_show_normals;
        in float f_show_illumination;
        
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
            if(f_show_normals > 0.5){
                col = vec4(f_normal, 1.0) * a;
            }else if(f_show_illumination > 0.5){
                vec4 light = vec4(max(dot(f_light_direction, -f_normal), 0) * f_light_intensity, 1);
                vec4 shadow = vec4(max(dot(f_shadow_direction, -f_normal), 0) * f_shadow_intensity, 1);
                col = (f_color + light - shadow) * a;
            }else{
                col = f_color * a;
            }
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
        out vec4 f_color;
        out float f_alpha_radius;
        void main()
        {
            gl_Position = perspective_matrix * object_matrix * vec4(position, 1.0f);
            gl_PointSize = point_size;
            f_color = color;
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
    vertex_shader_normals = '''
        uniform mat4 perspective_matrix;
        uniform mat4 object_matrix;
        in vec3 position;
        void main()
        {
            gl_Position = perspective_matrix * object_matrix * vec4(position, 1.0f);
        }
    '''
    fragment_shader_normals = '''
        uniform vec4 color;
        out vec4 fragColor;
        void main()
        {
            fragColor = color;
        }
    '''


class PCVManager():
    cache = {}
    handle = None
    initialized = False
    
    @classmethod
    def load_ply_to_cache(cls, operator, context, ):
        pcv = context.object.point_cloud_visualizer
        filepath = pcv.filepath
        
        __t = time.time()
        
        log('load data..')
        _t = time.time()
        
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
        
        np.random.shuffle(points)
        
        _d = datetime.timedelta(seconds=time.time() - _t)
        log("completed in {}.".format(_d))
        
        log('process data..')
        _t = time.time()
        
        if(not set(('x', 'y', 'z')).issubset(points.dtype.names)):
            # this is very unlikely..
            operator.report({'ERROR'}, "Loaded data seems to miss vertex locations.")
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
            # default_color = 0.65
            # cs = np.column_stack((np.full(n, default_color, dtype=np.float32, ),
            #                       np.full(n, default_color, dtype=np.float32, ),
            #                       np.full(n, default_color, dtype=np.float32, ),
            #                       np.ones(n, dtype=np.float32, ), ))
            
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
        
        # FIXME: names 'display_percent' and 'current_display_percent' are really badly chosen, value is number of points displayed, not presentage, this is not the first time it confused me, rename it not so distant future, right?
        
        d['display_percent'] = l
        d['current_display_percent'] = l
        
        ienabled = pcv.illumination
        d['illumination'] = ienabled
        if(ienabled):
            shader = GPUShader(PCVShaders.vertex_shader, PCVShaders.fragment_shader)
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
        
        return True
    
    @classmethod
    def render(cls, uuid, ):
        bgl.glEnable(bgl.GL_PROGRAM_POINT_SIZE)
        bgl.glEnable(bgl.GL_DEPTH_TEST)
        
        ci = PCVManager.cache[uuid]
        
        shader = ci['shader']
        batch = ci['batch']
        
        if(ci['current_display_percent'] != ci['display_percent']):
            l = ci['display_percent']
            ci['current_display_percent'] = l
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
            bgl.glDisable(bgl.GL_DEPTH_TEST)
            return
        
        if(ci['illumination'] != pcv.illumination):
            vs = ci['vertices']
            cs = ci['colors']
            ns = ci['normals']
            l = ci['current_display_percent']
            if(pcv.illumination):
                shader = GPUShader(PCVShaders.vertex_shader, PCVShaders.fragment_shader)
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
            shader.uniform_float("show_normals", float(pcv.show_normals))
            shader.uniform_float("show_illumination", float(pcv.illumination))
        else:
            # z = (0, 0, 0)
            # shader.uniform_float("light_direction", z)
            # shader.uniform_float("light_intensity", z)
            # shader.uniform_float("shadow_direction", z)
            # shader.uniform_float("shadow_intensity", z)
            # shader.uniform_float("show_normals", float(False))
            # shader.uniform_float("show_illumination", float(False))
            pass
        
        batch.draw(shader)
        
        if(pcv.vertex_normals and pcv.has_normals):
            
            def make_arrays(vs, ns, s, ):
                l = len(vs)
                coords = [None] * (l * 2)
                indices = [None] * l
                for i, v in enumerate(vs):
                    n = Vector(ns[i])
                    v = Vector(v)
                    coords[i * 2 + 0] = v
                    coords[i * 2 + 1] = v + (n.normalized() * s)
                    indices[i] = (i * 2 + 0, i * 2 + 1, )
                return coords, indices
            
            def make(ci):
                s = pcv.vertex_normals_size
                l = ci['current_display_percent']
                vs = ci['vertices'][:l]
                ns = ci['normals'][:l]
                coords, indices = make_arrays(vs, ns, s, )
                # shader = gpu.shader.from_builtin('3D_UNIFORM_COLOR')
                shader = GPUShader(PCVShaders.vertex_shader_normals, PCVShaders.fragment_shader_normals)
                batch = batch_for_shader(shader, 'LINES', {'position': coords}, indices=indices, )
                d = {'shader': shader,
                     'batch': batch,
                     'coords': coords,
                     'indices': indices,
                     'current_display_percent': l,
                     'size': s,
                     'current_size': s, }
                ci['vertex_normals'] = d
                return shader, batch
            
            if("vertex_normals" not in ci.keys()):
                shader, batch = make(ci)
            else:
                d = ci['vertex_normals']
                shader = d['shader']
                batch = d['batch']
                ok = True
                if(ci['current_display_percent'] != d['current_display_percent']):
                    ok = False
                if(d['current_size'] != pcv.vertex_normals_size):
                    ok = False
                if(not ok):
                    shader, batch = make(ci)
            
            shader.bind()
            pm = bpy.context.region_data.perspective_matrix
            shader.uniform_float("perspective_matrix", pm)
            shader.uniform_float("object_matrix", o.matrix_world)
            
            preferences = bpy.context.preferences
            addon_prefs = preferences.addons[__name__].preferences
            col = addon_prefs.normal_color[:]
            col = tuple([c ** (1 / 2.2) for c in col]) + (1.0, )
            # shader.uniform_float("color", (35 / 255, 97 / 255, 221 / 255, 1, ), )
            shader.uniform_float("color", col, )
            batch.draw(shader)
        
        bgl.glDisable(bgl.GL_DEPTH_TEST)
    
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
        # NOTE: this is redundant
        return {'uuid': None,
                'filepath': None,
                'vertices': None,
                'normals': None,
                'colors': None,
                'display_percent': None,
                'current_display_percent': None,
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
    bl_description = "Load PLY"
    
    filename_ext = ".ply"
    filter_glob: StringProperty(default="*.ply", options={'HIDDEN'}, )
    filepath: StringProperty(name="File Path", default="", description="", maxlen=1024, subtype='FILE_PATH', )
    order = ["filepath", ]
    
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
        pcv = context.object.point_cloud_visualizer
        ok = False
        for k, v in PCVManager.cache.items():
            if(v['uuid'] == pcv.uuid):
                if(v['ready']):
                    if(v['draw']):
                        ok = True
        return ok
    
    def execute(self, context):
        bgl.glEnable(bgl.GL_PROGRAM_POINT_SIZE)
        
        scene = context.scene
        render = scene.render
        image_settings = render.image_settings
        
        original_depth = image_settings.color_depth
        image_settings.color_depth = '8'
        
        pcv = context.object.point_cloud_visualizer
        
        if(pcv.render_resolution_linked):
            scale = render.resolution_percentage / 100
            width = int(render.resolution_x * scale)
            height = int(render.resolution_y * scale)
        else:
            scale = pcv.render_resolution_percentage / 100
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
        
        # render_suffix = pcv.render_suffix
        # render_zeros = pcv.render_zeros
        
        offscreen = GPUOffScreen(width, height)
        offscreen.bind()
        try:
            gpu.matrix.load_matrix(Matrix.Identity(4))
            gpu.matrix.load_projection_matrix(Matrix.Identity(4))
            
            bgl.glClear(bgl.GL_COLOR_BUFFER_BIT)
            
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
            
            if(pcv.illumination):
                shader = GPUShader(PCVShaders.vertex_shader, PCVShaders.fragment_shader)
                batch = batch_for_shader(shader, 'POINTS', {"position": vs, "color": cs, "normal": ns, })
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
            shader.uniform_float("point_size", pcv.render_point_size)
            shader.uniform_float("alpha_radius", pcv.alpha_radius)
            
            if(pcv.illumination and pcv.has_normals and cloud['illumination']):
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
                shader.uniform_float("show_normals", float(pcv.show_normals))
                shader.uniform_float("show_illumination", float(pcv.illumination))
            else:
                # z = (0, 0, 0)
                # shader.uniform_float("light_direction", z)
                # shader.uniform_float("light_intensity", z)
                # shader.uniform_float("shadow_direction", z)
                # shader.uniform_float("shadow_intensity", z)
                # shader.uniform_float("show_normals", float(False))
                # shader.uniform_float("show_illumination", float(False))
                pass
            
            batch.draw(shader)
            
            buffer = bgl.Buffer(bgl.GL_BYTE, width * height * 4)
            bgl.glReadBuffer(bgl.GL_BACK)
            bgl.glReadPixels(0, 0, width, height, bgl.GL_RGBA, bgl.GL_UNSIGNED_BYTE, buffer)
            
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
            
        finally:
            offscreen.unbind()
            offscreen.free()
        
        # image from buffer
        image_name = "pcv_output"
        if(image_name not in bpy.data.images):
            bpy.data.images.new(image_name, width, height)
        image = bpy.data.images[image_name]
        image.scale(width, height)
        image.pixels = [v / 255 for v in buffer]
        
        # save as image file
        def save_render(operator, scene, image, output_path, ):
            # f = False
            # n = render_suffix
            # rs = bpy.context.scene.render
            # op = rs.filepath
            # if(len(op) > 0):
            #     if(not op.endswith(os.path.sep)):
            #         f = True
            #         op, n = os.path.split(op)
            # else:
            #     log("error: output path is not set".format(e))
            #     operator.report({'ERROR'}, "Output path is not set.")
            #     return
            #
            # if(f):
            #     n = "{}_{}".format(n, render_suffix)
            #
            # fnm = "{}_{:0{z}d}.png".format(n, scene.frame_current, z=render_zeros)
            # p = os.path.join(os.path.realpath(bpy.path.abspath(op)), fnm)
            
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
        
        # save_render(self, scene, image, render_suffix, render_zeros, )
        save_render(self, scene, image, output_path, )
        
        # restore
        image_settings.color_depth = original_depth
        
        # cleanup
        bpy.data.images.remove(image)
        
        return {'FINISHED'}


class PCV_OT_animation(Operator):
    bl_idname = "point_cloud_visualizer.animation"
    bl_label = "Animation"
    bl_description = "Render displayed point cloud from active camera view to animation frames"
    
    @classmethod
    def poll(cls, context):
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
        
        fc = scene.frame_current
        for i in range(scene.frame_start, scene.frame_end + 1, 1):
            scene.frame_set(i)
            bpy.ops.point_cloud_visualizer.render()
        scene.frame_set(fc)
        return {'FINISHED'}


class PCV_OT_convert(Operator):
    bl_idname = "point_cloud_visualizer.convert"
    bl_label = "Convert"
    bl_description = "Convert point cloud to mesh"
    
    @classmethod
    def poll(cls, context):
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
        if(pcv.mesh_type == 'VERTEX'):
            g = VertexMeshGenerator()
            n = "{}-vertices".format(n)
        elif(pcv.mesh_type == 'TRIANGLE'):
            g = EquilateralTriangleMeshGenerator()
            n = "{}-triangles".format(n)
        elif(pcv.mesh_type == 'TETRAHEDRON'):
            g = TetrahedronMeshGenerator()
            n = "{}-tetrahedrons".format(n)
        elif(pcv.mesh_type == 'CUBE'):
            g = CubeMeshGenerator()
            n = "{}-cubes".format(n)
        elif(pcv.mesh_type == 'ICOSPHERE'):
            g = IcoSphereMeshGenerator()
            n = "{}-icospheres".format(n)
        elif(pcv.mesh_type == 'INSTANCER'):
            g = EquilateralTriangleMeshGenerator()
            n = "{}-instancer".format(n)
        elif(pcv.mesh_type == 'PARTICLES'):
            g = EquilateralTriangleMeshGenerator()
            n = "{}-particles".format(n)
        else:
            pass
        
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
        instancer = PCMeshInstancer(**d)
        
        o = instancer.object
        me = o.data
        me.transform(m.inverted())
        o.matrix_world = m
        
        if(pcv.mesh_type == 'INSTANCER'):
            pci = PCInstancer(o, pcv.mesh_size, pcv.mesh_base_sphere_subdivisions, )
        
        if(pcv.mesh_type == 'PARTICLES'):
            pcp = PCParticles(o, pcv.mesh_size, pcv.mesh_base_sphere_subdivisions, )
        
        return {'FINISHED'}


class PCV_OT_export(Operator, ExportHelper):
    bl_idname = "point_cloud_visualizer.export"
    bl_label = "Export PLY"
    bl_description = "Export point cloud to ply file"
    # bl_options = {'PRESET'}
    
    filename_ext = ".ply"
    filter_glob: StringProperty(default="*.ply", options={'HIDDEN'}, )
    check_extension = True
    
    # apply_transformation: BoolProperty(name="Apply Transformation", default=True, description="Apply parent object transformation to points", )
    # convert_axes: BoolProperty(name="Convert Axes", default=False, description="Convert from blender (y forward, z up) to forward -z, up y axes", )
    # visible_only: BoolProperty(name="Visible Points Only", default=False, description="Export currently visible points only", )
    
    @classmethod
    def poll(cls, context):
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
        # c.prop(self, 'apply_transformation')
        # c.prop(self, 'convert_axes')
        # c.prop(self, 'visible_only')
        pcv = context.object.point_cloud_visualizer
        c.prop(pcv, 'export_apply_transformation')
        c.prop(pcv, 'export_convert_axes')
        c.prop(pcv, 'export_visible_only')
    
    def execute(self, context):
        pcv = context.object.point_cloud_visualizer
        c = PCVManager.cache[pcv.uuid]
        
        o = c['object']
        vs = c['vertices']
        ns = c['normals']
        cs = c['colors']
        
        # if(self.visible_only):
        if(pcv.export_visible_only):
            l = c['display_percent']
            vs = vs[:l]
            ns = ns[:l]
            cs = cs[:l]
        
        def apply_matrix(vs, ns, m):
            # https://blender.stackexchange.com/questions/139511/replace-matrix-vector-list-comprehensions-with-something-more-efficient/
            l = len(vs)
            mw = np.array(m.inverted(), dtype=np.float, )
            mwrot = np.array(m.inverted().decompose()[1].to_matrix().to_4x4(), dtype=np.float, )
            
            a = np.ones((l, 4), vs.dtype)
            a[:, :-1] = vs
            # a = np.einsum('ij,aj->ai', mw, a)
            a = np.dot(a, mw)
            a = np.float32(a)
            vs = a[:, :-1]
            
            a = np.ones((l, 4), ns.dtype)
            a[:, :-1] = ns
            # a = np.einsum('ij,aj->ai', mwrot, a)
            a = np.dot(a, mwrot)
            a = np.float32(a)
            ns = a[:, :-1]
            
            return vs, ns
        
        # if(self.apply_transformation):
        if(pcv.export_apply_transformation):
            vs, ns = apply_matrix(vs, ns, o.matrix_world.copy())
        
        # if(self.convert_axes):
        if(pcv.export_convert_axes):
            axis_forward = '-Z'
            axis_up = 'Y'
            cm = axis_conversion(to_forward=axis_forward, to_up=axis_up).to_4x4()
            vs, ns = apply_matrix(vs, ns, cm)
        
        w = BinPlyPointCloudWriter(self.filepath, vs, ns, cs, )
        return {'FINISHED'}


class PCV_OT_simplify(Operator):
    bl_idname = "point_cloud_visualizer.simplify"
    bl_label = "Simplify"
    bl_description = "Simplify point cloud to exact number of evenly distributed samples, all loaded points are processed"
    
    @classmethod
    def poll(cls, context):
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
        
        num_samples = pcv.modify_simplify_num_samples
        if(num_samples >= len(vs)):
            self.report({'ERROR'}, "Number of samples must be < number of points.")
            return False, []
        candidates = pcv.modify_simplify_num_candidates
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
        
        # if(DEBUG):
        #     import cProfile
        #     import pstats
        #     import io
        #     pr = cProfile.Profile()
        #     pr.enable()
        
        ok, a = self.resample(context)
        if(not ok):
            return {'CANCELLED'}
        
        # if(DEBUG):
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
        c = PCVManager.cache[pcv.uuid]
        c['vertices'] = vs
        c['normals'] = ns
        c['colors'] = cs
        l = len(vs)
        c['length'] = l
        c['stats'] = l
        
        pcv.display_percent = 100.0
        
        c['display_percent'] = l
        c['current_display_percent'] = l
        
        # force PCVManager to redraw cloud
        ienabled = pcv.illumination
        c['illumination'] = ienabled
        if(ienabled):
            shader = GPUShader(PCVShaders.vertex_shader, PCVShaders.fragment_shader)
            batch = batch_for_shader(shader, 'POINTS', {"position": vs[:], "color": cs[:], "normal": ns[:], })
        else:
            shader = GPUShader(PCVShaders.vertex_shader_simple, PCVShaders.fragment_shader_simple)
            batch = batch_for_shader(shader, 'POINTS', {"position": vs[:], "color": cs[:], })
        c['shader'] = shader
        c['batch'] = batch
        
        context.area.tag_redraw()
        
        _d = datetime.timedelta(seconds=time.time() - _t)
        log("completed in {}.".format(_d), 1)
        
        return {'FINISHED'}


class PCV_OT_remove_color(Operator):
    bl_idname = "point_cloud_visualizer.remove_color"
    bl_label = "Remove Color"
    bl_description = "Remove points with exact/similar color"
    
    @classmethod
    def poll(cls, context):
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
        
        # rmcolor = Color(pcv.modify_remove_color)
        # rmcolor = Color([c ** (1 / 2.2) for c in pcv.modify_remove_color])
        
        # black magic..
        c = [c ** (1 / 2.2) for c in pcv.modify_remove_color]
        c = [int(i * 256) for i in c]
        c = [i / 256 for i in c]
        rmcolor = Color(c)
        
        dh = pcv.modify_remove_color_delta_hue
        ds = pcv.modify_remove_color_delta_saturation
        dv = pcv.modify_remove_color_delta_value
        # if(not pcv.modify_remove_color_delta_hue_use):
        #     dhue = None
        # if(not pcv.modify_remove_color_delta_saturation_use):
        #     dsaturation = None
        # if(not pcv.modify_remove_color_delta_value_use):
        #     dvalue = None
        uh = pcv.modify_remove_color_delta_hue_use
        us = pcv.modify_remove_color_delta_saturation_use
        uv = pcv.modify_remove_color_delta_value_use
        
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
                if(rmcolor.h - dh < c.h < rmcolor.h + dh):
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
        
        log("removed: {} points".format(len(indexes)), 1)
        
        # delete marked points
        points = np.delete(points, indexes)
        
        # split back
        vs = np.column_stack((points['x'], points['y'], points['z'], ))
        ns = np.column_stack((points['nx'], points['ny'], points['nz'], ))
        cs = np.column_stack((points['red'], points['green'], points['blue'], points['alpha'], ))
        vs = vs.astype(np.float32)
        ns = ns.astype(np.float32)
        cs = cs.astype(np.float32)
        # put to cache
        pcv = context.object.point_cloud_visualizer
        c = PCVManager.cache[pcv.uuid]
        c['vertices'] = vs
        c['normals'] = ns
        c['colors'] = cs
        l = len(vs)
        c['length'] = l
        c['stats'] = l
        
        pcv.display_percent = 100.0
        
        c['display_percent'] = l
        c['current_display_percent'] = l
        
        # force PCVManager to redraw cloud
        ienabled = pcv.illumination
        c['illumination'] = ienabled
        if(ienabled):
            shader = GPUShader(PCVShaders.vertex_shader, PCVShaders.fragment_shader)
            batch = batch_for_shader(shader, 'POINTS', {"position": vs[:], "color": cs[:], "normal": ns[:], })
        else:
            shader = GPUShader(PCVShaders.vertex_shader_simple, PCVShaders.fragment_shader_simple)
            batch = batch_for_shader(shader, 'POINTS', {"position": vs[:], "color": cs[:], })
        c['shader'] = shader
        c['batch'] = batch
        
        context.area.tag_redraw()
        
        _d = datetime.timedelta(seconds=time.time() - _t)
        log("completed in {}.".format(_d), 1)
        
        return {'FINISHED'}


class PCV_OT_reload(Operator):
    bl_idname = "point_cloud_visualizer.reload"
    bl_label = "Reload"
    bl_description = "Reload points from original file"
    
    @classmethod
    def poll(cls, context):
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
        
        bpy.ops.point_cloud_visualizer.erase()
        bpy.ops.point_cloud_visualizer.load_ply_to_cache(filepath=c['filepath'])
        bpy.ops.point_cloud_visualizer.draw()
        
        return {'FINISHED'}


class PCV_PT_panel(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "View"
    bl_label = "Point Cloud Visualizer"
    bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        o = context.active_object
        if(o):
            return True
        return False
    
    def draw(self, context):
        pcv = context.object.point_cloud_visualizer
        l = self.layout
        sub = l.column()
        
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
                
                n = human_readable_number(cache['display_percent'])
                # don't use it when less or equal to 999
                if(cache['display_percent'] < 1000):
                    n = str(cache['display_percent'])
                
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
        
        # sub.prop(pcv, 'ply_info', text="", emboss=False, )
        # sub.prop(pcv, 'ply_display_info', text="", emboss=False, )
        
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
        # r = sub.row()
        # r.prop(pcv, 'alpha_radius')
        # r.enabled = e
        
        r = sub.row(align=True)
        r.prop(pcv, 'vertex_normals', toggle=True, icon_only=True, icon='SNAP_NORMAL', )
        r.prop(pcv, 'vertex_normals_size')
        r.enabled = e
        if(not pcv.has_normals):
            r.enabled = False
        
        sub.separator()
        
        pcv = context.object.point_cloud_visualizer
        ok = False
        for k, v in PCVManager.cache.items():
            if(v['uuid'] == pcv.uuid):
                if(v['ready']):
                    if(v['draw']):
                        ok = True
        
        c = sub.column()
        r = c.row(align=True)
        r.prop(pcv, 'illumination', toggle=True, )
        r.prop(pcv, 'illumination_edit', toggle=True, icon_only=True, icon='TOOL_SETTINGS', )
        # r.prop(pcv, 'illumination_edit', toggle=True, icon_only=True, icon='SETTINGS', )
        if(ok):
            if(not pcv.has_normals):
                c.label(text="Missing vertex normals.", icon='ERROR', )
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
        
        # if(pcv.uuid in PCVManager.cache):
        #     sub.separator()
        #     # r = sub.row()
        #     # h, t = os.path.split(pcv.filepath)
        #     # n = human_readable_number(PCVManager.cache[pcv.uuid]['stats'])
        #     # r.label(text='{}: {} points'.format(t, n))
        #     sub.prop(pcv, 'ply_info', text="", emboss=False, )
        #     sub.prop(pcv, 'ply_display_info', text="", emboss=False, )
        
        sub.separator()
        b = sub.box()
        b.alert = True
        b.prop(pcv, 'experimental')


class PCV_PT_render(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "View"
    bl_label = "Render"
    bl_parent_id = "PCV_PT_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        pcv = context.object.point_cloud_visualizer
        l = self.layout
        sub = l.column()
        
        c = sub.column()
        c.prop(pcv, 'render_display_percent')
        c.prop(pcv, 'render_point_size')
        
        sub.separator()
        
        c = sub.column()
        # c.prop(pcv, 'render_path', text='Output', )
        
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
        
        sub.separator()
        
        r = sub.row(align=True)
        r.operator('point_cloud_visualizer.render')
        r.operator('point_cloud_visualizer.animation')
        
        sub.enabled = PCV_OT_render.poll(context)


class PCV_PT_convert(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "View"
    bl_label = "Convert"
    bl_parent_id = "PCV_PT_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
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
        
        c.separator()
        c.operator('point_cloud_visualizer.convert')
        c.enabled = PCV_OT_convert.poll(context)


class PCV_PT_modify(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "View"
    bl_label = "Modify"
    bl_parent_id = "PCV_PT_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        pcv = context.object.point_cloud_visualizer
        l = self.layout
        c = l.column()
        
        # b = c.box()
        # cc = b.column(align=True)
        cc = c.column(align=True)
        cc.label(text="Simplify Point Cloud:")
        
        cc.prop(pcv, 'modify_simplify_num_samples')
        cc.prop(pcv, 'modify_simplify_num_candidates')
        cc.operator('point_cloud_visualizer.simplify')
        c.separator()
        
        # b = c.box()
        # cc = b.column(align=True)
        cc = c.column(align=True)
        cc.label(text="Remove Color:")
        
        r = cc.row(align=True)
        r.prop(pcv, 'modify_remove_color', text='', )
        
        r = cc.row(align=True)
        r.prop(pcv, 'modify_remove_color_delta_hue_use', text='', toggle=True, icon_only=True, icon='CHECKBOX_HLT' if pcv.modify_remove_color_delta_hue_use else 'CHECKBOX_DEHLT', )
        ccc = r.column(align=True)
        ccc.prop(pcv, 'modify_remove_color_delta_hue')
        ccc.active = pcv.modify_remove_color_delta_hue_use
        
        r = cc.row(align=True)
        r.prop(pcv, 'modify_remove_color_delta_saturation_use', text='', toggle=True, icon_only=True, icon='CHECKBOX_HLT' if pcv.modify_remove_color_delta_saturation_use else 'CHECKBOX_DEHLT', )
        ccc = r.column(align=True)
        ccc.prop(pcv, 'modify_remove_color_delta_saturation')
        ccc.active = pcv.modify_remove_color_delta_saturation_use
        
        r = cc.row(align=True)
        r.prop(pcv, 'modify_remove_color_delta_value_use', text='', toggle=True, icon_only=True, icon='CHECKBOX_HLT' if pcv.modify_remove_color_delta_value_use else 'CHECKBOX_DEHLT', )
        ccc = r.column(align=True)
        ccc.prop(pcv, 'modify_remove_color_delta_value')
        ccc.active = pcv.modify_remove_color_delta_value_use
        
        cc.operator('point_cloud_visualizer.remove_color')
        c.separator()
        
        c.label(text="Something went wrong?")
        # c.operator('point_cloud_visualizer.reload', icon='RECOVER_LAST', )
        c.operator('point_cloud_visualizer.reload')
        
        c.enabled = PCV_OT_simplify.poll(context)


class PCV_PT_export(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "View"
    bl_label = "Export"
    bl_parent_id = "PCV_PT_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        pcv = context.object.point_cloud_visualizer
        l = self.layout
        c = l.column(align=True)
        c.prop(pcv, 'export_apply_transformation')
        c.prop(pcv, 'export_convert_axes')
        c.prop(pcv, 'export_visible_only')
        c.operator('point_cloud_visualizer.export')
        
        c.enabled = PCV_OT_export.poll(context)


class PCV_PT_debug(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "View"
    bl_label = "Debug"
    bl_parent_id = "PCV_PT_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        pcv = context.object.point_cloud_visualizer
        l = self.layout
        sub = l.column()
        
        sub.label(text="properties:")
        b = sub.box()
        c = b.column()
        
        # c.label(text="uuid: {}".format(pcv.uuid))
        # c.label(text="filepath: {}".format(pcv.filepath))
        # c.label(text="point_size: {}".format(pcv.point_size))
        # c.label(text="alpha_radius: {}".format(pcv.alpha_radius))
        # c.label(text="display_percent: {}".format(pcv.display_percent))
        # c.label(text="render_expanded: {}".format(pcv.render_expanded))
        # c.label(text="render_point_size: {}".format(pcv.render_point_size))
        # c.label(text="render_display_percent: {}".format(pcv.render_display_percent))
        # # c.label(text="render_suffix: {}".format(pcv.render_suffix))
        # # c.label(text="render_zeros: {}".format(pcv.render_zeros))
        # c.label(text="has_normals: {}".format(pcv.has_normals))
        # c.label(text="has_vcols: {}".format(pcv.has_vcols))
        # c.label(text="illumination: {}".format(pcv.illumination))
        # c.label(text="light_direction: {}".format(pcv.light_direction))
        # c.label(text="light_intensity: {}".format(pcv.light_intensity))
        # c.label(text="shadow_intensity: {}".format(pcv.shadow_intensity))
        
        c.label(text="uuid: {}".format(pcv.uuid))
        c.label(text="filepath: {}".format(pcv.filepath))
        c.label(text="has_normals: {}".format(pcv.has_normals))
        c.label(text="has_vcols: {}".format(pcv.has_vcols))
        c.label(text="point_size: {}".format(pcv.point_size))
        c.label(text="alpha_radius: {}".format(pcv.alpha_radius))
        c.label(text="display_percent: {}".format(pcv.display_percent))
        c.label(text="illumination: {}".format(pcv.illumination))
        c.label(text="illumination_edit: {}".format(pcv.illumination_edit))
        c.label(text="light_direction: {}".format(pcv.light_direction))
        c.label(text="light_intensity: {}".format(pcv.light_intensity))
        c.label(text="shadow_intensity: {}".format(pcv.shadow_intensity))
        c.label(text="show_normals: {}".format(pcv.show_normals))
        c.label(text="vertex_normals: {}".format(pcv.vertex_normals))
        c.label(text="vertex_normals_size: {}".format(pcv.vertex_normals_size))
        c.label(text="render_expanded: {}".format(pcv.render_expanded))
        c.label(text="render_point_size: {}".format(pcv.render_point_size))
        c.label(text="render_display_percent: {}".format(pcv.render_display_percent))
        c.label(text="render_path: {}".format(pcv.render_path))
        c.label(text="render_resolution_x: {}".format(pcv.render_resolution_x))
        c.label(text="render_resolution_y: {}".format(pcv.render_resolution_y))
        c.label(text="render_resolution_percentage: {}".format(pcv.render_resolution_percentage))
        c.label(text="render_resolution_linked: {}".format(pcv.render_resolution_linked))
        c.label(text="mesh_type: {}".format(pcv.mesh_type))
        c.label(text="mesh_size: {}".format(pcv.mesh_size))
        c.label(text="mesh_normal_align: {}".format(pcv.mesh_normal_align))
        c.label(text="mesh_vcols: {}".format(pcv.mesh_vcols))
        c.label(text="mesh_all: {}".format(pcv.mesh_all))
        c.label(text="mesh_percentage: {}".format(pcv.mesh_percentage))
        c.label(text="mesh_base_sphere_subdivisions: {}".format(pcv.mesh_base_sphere_subdivisions))
        c.label(text="modify_simplify_num_samples: {}".format(pcv.modify_simplify_num_samples))
        c.label(text="modify_simplify_num_candidates: {}".format(pcv.modify_simplify_num_candidates))
        
        c.label(text="debug: {}".format(pcv.debug))
        c.scale_y = 0.5
        
        sub.label(text="manager:")
        c = sub.column(align=True)
        c.operator('point_cloud_visualizer.init')
        c.operator('point_cloud_visualizer.deinit')
        c.operator('point_cloud_visualizer.gc')
        b = sub.box()
        c = b.column()
        c.label(text="cache: {} item(s)".format(len(PCVManager.cache.items())))
        c.label(text="handle: {}".format(PCVManager.handle))
        c.label(text="initialized: {}".format(PCVManager.initialized))
        c.scale_y = 0.5
        
        if(len(PCVManager.cache)):
            sub.label(text="cache details:")
            for k, v in PCVManager.cache.items():
                b = sub.box()
                c = b.column()
                c.scale_y = 0.5
                for ki, vi in sorted(v.items()):
                    if(type(vi) == np.ndarray):
                        c.label(text="{}: numpy.ndarray ({} items)".format(ki, len(vi)))
                    else:
                        c.label(text="{}: {}".format(ki, vi))


class PCV_properties(PropertyGroup):
    filepath: StringProperty(name="PLY File", default="", description="", )
    uuid: StringProperty(default="", options={'HIDDEN', }, )
    # point_size: FloatProperty(name="Size", default=3.0, min=0.001, max=100.0, precision=3, subtype='FACTOR', description="Point size", )
    # point_size: IntProperty(name="Size", default=3, min=1, max=100, subtype='PIXEL', description="Point size", )
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
        d['display_percent'] = l
    
    display_percent: FloatProperty(name="Display", default=100.0, min=0.0, max=100.0, precision=0, subtype='PERCENTAGE', update=_display_percent_update, description="Adjust percentage of points displayed", )
    
    # ply_info: StringProperty(name="PLY Info", default="", description="", )
    # ply_display_info: StringProperty(name="PLY Display Info", default="Display:", description="", )
    
    vertex_normals: BoolProperty(name="Normals", description="Draw normals of points", default=False, )
    vertex_normals_size: FloatProperty(name="Length", description="Length of point normal line", default=0.01, min=0.00001, max=1.0, soft_min=0.001, soft_max=0.2, step=1, precision=3, )
    
    render_expanded: BoolProperty(default=False, options={'HIDDEN', }, )
    # render_point_size: FloatProperty(name="Size", default=3.0, min=0.001, max=100.0, precision=3, subtype='FACTOR', description="Render point size", )
    render_point_size: IntProperty(name="Size", default=3, min=1, max=100, subtype='PIXEL', description="Point size", )
    render_display_percent: FloatProperty(name="Count", default=100.0, min=0.0, max=100.0, precision=0, subtype='PERCENTAGE', description="Adjust percentage of points rendered", )
    # render_suffix: StringProperty(name="Suffix", default="pcv_frame", description="Render filename or suffix, depends on render output path. Frame number will be appended automatically", )
    # # render_zeros: IntProperty(name="Leading Zeros", default=6, min=3, max=10, subtype='FACTOR', description="Number of leading zeros in render filename", )
    # render_zeros: IntProperty(name="Leading Zeros", default=6, min=3, max=10, description="Number of leading zeros in render filename", )
    
    render_path: StringProperty(name="Output Path", default="//pcv_render_###.png", description="Directory/name to save rendered images, # characters defines the position and length of frame numbers, filetype is always png", subtype='FILE_PATH', )
    render_resolution_x: IntProperty(name="Resolution X", default=1920, min=4, max=65536, description="Number of horizontal pixels in rendered image", subtype='PIXEL', )
    render_resolution_y: IntProperty(name="Resolution Y", default=1080, min=4, max=65536, description="Number of vertical pixels in rendered image", subtype='PIXEL', )
    render_resolution_percentage: IntProperty(name="Resolution %", default=100, min=1, max=100, description="Percentage scale for render resolution", subtype='PERCENTAGE', )
    
    def _render_resolution_linked_update(self, context, ):
        if(not self.render_resolution_linked):
            # now it is False, so it must have been True, so for convenience, copy values
            r = context.scene.render
            self.render_resolution_x = r.resolution_x
            self.render_resolution_y = r.resolution_y
            self.render_resolution_percentage = r.resolution_percentage
    
    render_resolution_linked: BoolProperty(name="Resolution Linked", description="Link resolution settings to scene", default=True, update=_render_resolution_linked_update, )
    
    has_normals: BoolProperty(default=False, options={'HIDDEN', }, )
    has_vcols: BoolProperty(default=False, options={'HIDDEN', }, )
    illumination: BoolProperty(name="Illumination", description="Enable extra illumination on point cloud", default=False, )
    illumination_edit: BoolProperty(name="Edit", description="Edit illumination properties", default=False, )
    light_direction: FloatVectorProperty(name="Light Direction", description="Light direction", default=(0.0, 1.0, 0.0), subtype='DIRECTION', size=3, )
    # light_color: FloatVectorProperty(name="Light Color", description="", default=(0.2, 0.2, 0.2), min=0, max=1, subtype='COLOR', size=3, )
    light_intensity: FloatProperty(name="Light Intensity", description="Light intensity", default=0.3, min=0, max=1, subtype='FACTOR', )
    shadow_intensity: FloatProperty(name="Shadow Intensity", description="Shadow intensity", default=0.2, min=0, max=1, subtype='FACTOR', )
    show_normals: BoolProperty(name="Colorize By Vertex Normals", description="", default=False, )
    
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
    
    export_apply_transformation: BoolProperty(name="Apply Transformation", default=True, description="Apply parent object transformation to points", )
    export_convert_axes: BoolProperty(name="Convert Axes", default=False, description="Convert from blender (y forward, z up) to forward -z, up y axes", )
    export_visible_only: BoolProperty(name="Visible Points Only", default=False, description="Export currently visible points only (controlled by 'Display' on main panel)", )
    
    modify_simplify_num_samples: IntProperty(name="Samples", default=10000, min=1, subtype='NONE', description="Number of points in simplified point cloud, best result when set to less than 20% of points, when samples has value close to total expect less points in result", )
    modify_simplify_num_candidates: IntProperty(name="Candidates", default=10, min=3, max=100, subtype='NONE', description="Number of candidates used during resampling, the higher value, the slower calculation, but more even", )
    
    # def _rmcol_radio_update(self, context):
    #     if(not self.modify_remove_color_delta_hue_use and not self.modify_remove_color_delta_saturation_use and not self.modify_remove_color_delta_value_use):
    #         self.modify_remove_color_delta_hue_use = True
    
    modify_remove_color: FloatVectorProperty(name="Color", default=(1.0, 1.0, 1.0, ), min=0, max=1, subtype='COLOR', size=3, description="", )
    modify_remove_color_delta_hue: FloatProperty(name="Δ Hue", default=0.1, min=0.0, max=1.0, precision=3, subtype='FACTOR', description="", )
    modify_remove_color_delta_hue_use: BoolProperty(name="Use Δ Hue", description="", default=True, )
    modify_remove_color_delta_saturation: FloatProperty(name="Δ Saturation", default=0.1, min=0.0, max=1.0, precision=3, subtype='FACTOR', description="", )
    modify_remove_color_delta_saturation_use: BoolProperty(name="Use Δ Saturation", description="", default=True, )
    modify_remove_color_delta_value: FloatProperty(name="Δ Value", default=0.1, min=0.0, max=1.0, precision=3, subtype='FACTOR', description="", )
    modify_remove_color_delta_value_use: BoolProperty(name="Use Δ Value", description="", default=True, )
    
    def _debug_update(self, context, ):
        global DEBUG, debug_classes
        DEBUG = self.debug
        if(DEBUG):
            for cls in debug_classes:
                bpy.utils.register_class(cls)
        else:
            for cls in reversed(debug_classes):
                bpy.utils.unregister_class(cls)
    
    debug: BoolProperty(default=DEBUG, options={'HIDDEN', }, update=_debug_update, )
    
    def _experimental_update(self, context, ):
        global EXPERIMENTAL, experimental_classes
        EXPERIMENTAL = self.experimental
        if(EXPERIMENTAL):
            for cls in experimental_classes:
                bpy.utils.register_class(cls)
            self.debug = True
        else:
            for cls in reversed(experimental_classes):
                bpy.utils.unregister_class(cls)
            self.debug = False
    
    experimental: BoolProperty(name="Experimental Features", description="Enable experimental, unfinished, unoptimized or otherwise useless features", default=EXPERIMENTAL, update=_experimental_update, )
    
    @classmethod
    def register(cls):
        bpy.types.Object.point_cloud_visualizer = PointerProperty(type=cls)
    
    @classmethod
    def unregister(cls):
        del bpy.types.Object.point_cloud_visualizer


class PCV_preferences(AddonPreferences):
    bl_idname = __name__
    
    default_vertex_color: FloatVectorProperty(name="Default Color", default=(0.65, 0.65, 0.65, ), min=0, max=1, subtype='COLOR', size=3, description="Default color to be used upon loading PLY to cache when vertex colors are missing", )
    normal_color: FloatVectorProperty(name="Normal Color", default=((35 / 255) ** 2.2, (97 / 255) ** 2.2, (221 / 255) ** 2.2, ), min=0, max=1, subtype='COLOR', size=3, description="Display color for vertex normals", )
    convert_16bit_colors: BoolProperty(name="Convert 16bit Colors", description="Convert 16bit colors to 8bit, applied when Red channel has 'uint16' dtype", default=True, )
    gamma_correct_16bit_colors: BoolProperty(name="Gamma Correct 16bit Colors", description="When 16bit colors are encountered apply gamma as 'c ** (1 / 2.2)'", default=False, )
    
    def draw(self, context):
        l = self.layout
        r = l.row()
        r.prop(self, "default_vertex_color")
        r.prop(self, "normal_color")
        r = l.row()
        r.prop(self, "convert_16bit_colors")
        c = r.column()
        c.prop(self, "gamma_correct_16bit_colors")
        if(not self.convert_16bit_colors):
            c.active = False


@persistent
def watcher(scene):
    PCVManager.deinit()


classes = (
    PCV_properties,
    PCV_preferences,
    PCV_PT_panel,
    PCV_PT_render,
    PCV_PT_convert,
    PCV_OT_load,
    PCV_OT_draw,
    PCV_OT_erase,
    PCV_OT_render,
    PCV_OT_animation,
    PCV_OT_convert,
)

experimental_classes = (
    PCV_PT_modify,
    PCV_PT_export,
    PCV_OT_export,
    PCV_OT_simplify,
    PCV_OT_reload,
    PCV_OT_remove_color,
)
if(EXPERIMENTAL):
    classes = classes + experimental_classes

debug_classes = (
    PCV_PT_debug,
    PCV_OT_init,
    PCV_OT_deinit,
    PCV_OT_gc,
)
if(DEBUG):
    classes = classes + debug_classes


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
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
