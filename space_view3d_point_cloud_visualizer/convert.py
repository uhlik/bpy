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
import uuid
import time
import datetime
import math
import numpy as np

import bpy
import bmesh
from bpy.types import Operator
from mathutils import Matrix, Vector, Quaternion, Color

from .debug import log, debug_mode
from .machine import PCVManager


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
