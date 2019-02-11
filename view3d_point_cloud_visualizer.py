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
           "description": "Display colored point cloud PLY files in 3D viewport.",
           "author": "Jakub Uhlik",
           "version": (0, 8, 0),
           "blender": (2, 80, 0),
           "location": "3D Viewport > Sidebar > Point Cloud Visualizer",
           "warning": "",
           "wiki_url": "",
           "tracker_url": "",
           "category": "3D View", }


import os
import struct
import uuid
import time
import datetime
import math
import numpy as np

import bpy
from bpy.props import PointerProperty, BoolProperty, StringProperty, FloatProperty, IntProperty, FloatVectorProperty, EnumProperty
from bpy.types import PropertyGroup, Panel, Operator
import gpu
from gpu.types import GPUOffScreen, GPUShader, GPUBatch, GPUVertBuf, GPUVertFormat
from gpu_extras.batch import batch_for_shader
from bpy.app.handlers import persistent
import bgl
from mathutils import Matrix, Vector, Quaternion
from bpy_extras.object_utils import world_to_camera_view
from bpy_extras.io_utils import axis_conversion


# FIXME undo still doesn't work in some cases, from what i've seen, only when i am undoing operations on parent object, especially when you undo/redo e.g. transforms around load/draw operators, filepath property gets reset and the whole thing is drawn, but ui looks like loding never happened, i've added a quick fix storing path in cache, but it all depends on object name and this is bad.
# TODO better docs, some gifs would be the best, i personally hate watching video tutorials when i need just sigle bit of information buried in 10+ minutes video, what a waste of time
# TODO ~2k lines, maybe time to break into modules
# TODO any new functionality? any ideas?

# NOTE $ pycodestyle --ignore=W293,E501,E741,E402 --exclude='io_mesh_fast_obj/blender' .
#      W293 blank line contains whitespace              --> and i will do it, i hate caret jumping all around
#      E501 line too long (81 > 79 characters)          --> yea sure, i have 24" screen and i will write in small stripe in the middle, fantastic
#      E741 ambiguous variable name 'l'                 --> this one is new? good old pep8, never had an issue to confuse letters, sorry, i like one letter variables and i will keep using them
#      E402 module level import not at top of file      --> hmm, why do we have to have bl_info on top? 99% of other addons have it too


DEBUG = False


def log(msg, indent=0, ):
    m = "{0}> {1}".format("    " * indent, msg)
    if(DEBUG):
        print(m)


def human_readable_number(num, suffix='', ):
    # https://stackoverflow.com/questions/1094841/reusable-library-to-get-human-readable-version-of-file-size
    f = 1000.0
    for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z', ]:
        if(abs(num) < f):
            return "{:3.1f}{}{}".format(num, unit, suffix)
        num /= f
    return "{:.1f}{}{}".format(num, 'Y', suffix)


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
    def __init__(self, radius=1, recursion=1, ):
        log("{}:".format(self.__class__.__name__), 0, )
        if(radius <= 0):
            log("radius is (or less than) 0, which is ridiculous. setting to 0.001..", 1)
            radius = 0.001
        self.radius = radius
        recursion = int(recursion)
        if(recursion < 0):
            log("recursion is (or less than) 0, which is ridiculous. setting to 0..", 1)
            recursion = 0
        self.recursion = recursion
        super(IcoSphereMeshGenerator, self).__init__()
    
    def generate(self):
        def generate_icosphere(recursion, radius, ):
            # http://blog.andreaskahler.com/2009/06/creating-icosphere-mesh-in-code.html
            # http://www.kynd.info/library/mathandphysics/Icosphere/
            # http://www.csee.umbc.edu/~squire/reference/polyhedra.shtml
            
            verts = []
            faces = []
            cache = {}
            
            phiaa = 26.56505
            r = 1.0
            phia = math.pi * phiaa / 180.0
            theb = math.pi * 36.0 / 180.0
            the72 = math.pi * 72.0 / 180
            
            verts.append((0.0, 0.0, r))
            the = 0.0
            for i in range(5):
                verts.append((r * math.cos(the) * math.cos(phia),
                              r * math.sin(the) * math.cos(phia),
                              r * math.sin(phia), ))
                the = the + the72
            
            the = theb
            for i in range(5):
                verts.append((r * math.cos(the) * math.cos(-phia),
                              r * math.sin(the) * math.cos(-phia),
                              r * math.sin(-phia), ))
                the = the + the72
            verts.append((0.0, 0.0, -r))
            
            faces = [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 5), (0, 5, 1), (11, 7, 6), (11, 8, 7), (11, 9, 8), (11, 10, 9), (11, 6, 10),
                     (1, 6, 2), (2, 7, 3), (3, 8, 4), (4, 9, 5), (5, 10, 1), (6, 7, 2), (7, 8, 3), (8, 9, 4), (9, 10, 5), (10, 6, 1), ]
            
            def add_vert(x, y, z):
                l = math.sqrt(x * x + y * y + z * z)
                verts.append((x / l, y / l, z / l))
                return len(verts) - 1
            
            def get_middle(i1, i2):
                p1 = i1
                p2 = i2
                firstIsSmaller = p1 < p2
                if(firstIsSmaller):
                    smallerIndex = p1
                    greaterIndex = p2
                else:
                    smallerIndex = p2
                    greaterIndex = p1
                
                key = (smallerIndex << 32) + greaterIndex
                if(str(key) in cache):
                    return cache[str(key)]
            
                middle = ((verts[p1][0] + verts[p2][0]) / 2,
                          (verts[p1][1] + verts[p2][1]) / 2,
                          (verts[p1][2] + verts[p2][2]) / 2, )
                i = add_vert(middle[0], middle[1], middle[2])
                cache[str(key)] = i
                return i
            
            for i in range(recursion):
                faces2 = []
                for j in range(len(faces)):
                    a = get_middle(faces[j][0], faces[j][1])
                    b = get_middle(faces[j][1], faces[j][2])
                    c = get_middle(faces[j][2], faces[j][0])
                    faces2.append((faces[j][0], a, c))
                    faces2.append((faces[j][1], b, a))
                    faces2.append((faces[j][2], c, b))
                    faces2.append((a, b, c))
                faces = faces2
            
            return verts, [], faces
        
        dv, de, df = generate_icosphere(self.recursion, self.radius)
        return dv, de, df


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
        self.mesh.show_double_sided = False
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
            """Calculates shortest Quaternion from Vector a to Vector b"""
            # a - up vector
            # b - direction to point to
            
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
                    if(n == 'alpha'):
                        # skip alpha, maybe use it in future versions, but now it is useless
                        continue
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
        in vec3 normal;
        in vec4 color;
        uniform mat4 perspective_matrix;
        uniform mat4 object_matrix;
        uniform float point_size;
        uniform float alpha_radius;
        out vec4 f_color;
        out float f_alpha_radius;
        void main()
        {
            vec3 n = normal;
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


def load_ply_to_cache(operator, context, ):
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
        cs = np.column_stack((points['red'] / 255, points['green'] / 255, points['blue'] / 255, np.ones(len(points), dtype=float, ), ))
        cs = cs.astype(np.float32)
    else:
        n = len(points)
        default_color = 0.65
        cs = np.column_stack((np.full(n, default_color, dtype=np.float32, ),
                              np.full(n, default_color, dtype=np.float32, ),
                              np.full(n, default_color, dtype=np.float32, ),
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
    d['display_percent'] = l
    d['current_display_percent'] = l
    
    ienabled = pcv.illumination
    d['illumination'] = ienabled
    if(ienabled):
        shader = GPUShader(PCVShaders.vertex_shader, PCVShaders.fragment_shader)
    else:
        shader = GPUShader(PCVShaders.vertex_shader_simple, PCVShaders.fragment_shader_simple)
    batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "color": cs[:l], "normal": ns[:l], })
    
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


def save_render(operator, scene, image, render_suffix, render_zeros, ):
    f = False
    n = render_suffix
    rs = bpy.context.scene.render
    op = rs.filepath
    if(len(op) > 0):
        if(not op.endswith(os.path.sep)):
            f = True
            op, n = os.path.split(op)
    else:
        log("error: output path is not set".format(e))
        operator.report({'ERROR'}, "Output path is not set.")
        return
    
    if(f):
        n = "{}_{}".format(n, render_suffix)
    
    fnm = "{}_{:0{z}d}.png".format(n, scene.frame_current, z=render_zeros)
    p = os.path.join(os.path.realpath(bpy.path.abspath(op)), fnm)
    
    s = rs.image_settings
    ff = s.file_format
    cm = s.color_mode
    cd = s.color_depth
    
    vs = scene.view_settings
    vsvt = vs.view_transform
    vsl = vs.look
    vs.view_transform = 'Default'
    vs.look = 'None'
    
    s.file_format = 'PNG'
    s.color_mode = 'RGBA'
    s.color_depth = '8'
    
    try:
        image.save_render(p)
        log("image '{}' saved".format(p))
    except Exception as e:
        s.file_format = ff
        s.color_mode = cm
        s.color_depth = cd
        
        log("error: {}".format(e))
        operator.report({'ERROR'}, "Unable to save render image, see console for details.")
        return
    
    s.file_format = ff
    s.color_mode = cm
    s.color_depth = cd
    vs.view_transform = vsvt
    vs.look = vsl


class PCVManager():
    cache = {}
    handle = None
    initialized = False
    
    @classmethod
    def render(cls, uuid, ):
        bgl.glEnable(bgl.GL_PROGRAM_POINT_SIZE)
        
        ci = PCVManager.cache[uuid]
        
        shader = ci['shader']
        batch = ci['batch']
        
        if(ci['current_display_percent'] != ci['display_percent']):
            l = ci['display_percent']
            ci['current_display_percent'] = l
            vs = ci['vertices']
            cs = ci['colors']
            ns = ci['normals']
            batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "color": cs[:l], "normal": ns[:l], })
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
        
        if(ci['illumination'] != pcv.illumination):
            if(pcv.illumination):
                shader = GPUShader(PCVShaders.vertex_shader, PCVShaders.fragment_shader)
                ci['illumination'] = True
            else:
                shader = GPUShader(PCVShaders.vertex_shader_simple, PCVShaders.fragment_shader_simple)
                ci['illumination'] = False
            vs = ci['vertices']
            cs = ci['colors']
            ns = ci['normals']
            l = ci['current_display_percent']
            batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "color": cs[:l], "normal": ns[:l], })
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
        return {'uuid': None,
                'vertices': None,
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
            ok = load_ply_to_cache(self, context)
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
        
        ok = load_ply_to_cache(self, context)
        
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
        
        scale = render.resolution_percentage / 100
        width = int(render.resolution_x * scale)
        height = int(render.resolution_y * scale)
        
        pcv = context.object.point_cloud_visualizer
        cloud = PCVManager.cache[pcv.uuid]
        cam = scene.camera
        if(cam is None):
            self.report({'ERROR'}, "No camera found.")
            return {'CANCELLED'}
        
        render_suffix = pcv.render_suffix
        render_zeros = pcv.render_zeros
        
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
            else:
                shader = GPUShader(PCVShaders.vertex_shader_simple, PCVShaders.fragment_shader_simple)
            batch = batch_for_shader(shader, 'POINTS', {"position": vs, "color": cs, "normal": ns, })
            shader.bind()
            
            view_matrix = cam.matrix_world.inverted()
            camera_matrix = cam.calc_matrix_camera(bpy.context.depsgraph, x=render.resolution_x, y=render.resolution_y, scale_x=render.pixel_aspect_x, scale_y=render.pixel_aspect_y, )
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
        save_render(self, scene, image, render_suffix, render_zeros, )
        
        # restore
        image_settings.color_depth = original_depth
        
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
        for i in range(scene.frame_start, scene.frame_end, 1):
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
        
        # def real_length_to_relative(matrix, length, ):
        #     """From matrix_world and desired real size length in meters, calculate relative length
        #     without matrix applied. Apply matrix, and you will get desired length."""
        #     l, r, s = matrix.decompose()
        #     ms = Matrix.Scale(s.x, 4)
        #     l = Vector((length, 0, 0))
        #     v = ms.inverted() @ l
        #     # v = ms @ l
        #     return v.x
        
        def apply_matrix(points, matrix):
            log("apply_matrix:", 0, )
            ms = str(matrix).replace('            ', ', ').replace('\n', '')
            log("points: {0}, matrix: {1}".format(len(points), ms), 1)
            matrot = matrix.decompose()[1].to_matrix().to_4x4()
            r = [None] * len(points)
            for i, p in enumerate(points):
                co = matrix @ Vector((p[0], p[1], p[2]))
                no = matrot @ Vector((p[3], p[4], p[5]))
                r[i] = (co.x, co.y, co.z, no.x, no.y, no.z, p[6], p[7], p[8])
            return r
        
        g = None
        if(pcv.mesh_type == 'VERTEX'):
            g = VertexMeshGenerator()
        elif(pcv.mesh_type == 'TRIANGLE'):
            g = EquilateralTriangleMeshGenerator()
        elif(pcv.mesh_type == 'TETRAHEDRON'):
            g = TetrahedronMeshGenerator()
        elif(pcv.mesh_type == 'CUBE'):
            g = CubeMeshGenerator()
        elif(pcv.mesh_type == 'ICOSPHERE'):
            g = IcoSphereMeshGenerator()
        else:
            pass
        
        _, t = os.path.split(pcv.filepath)
        n, _ = os.path.splitext(t)
        m = o.matrix_world.copy()
        points = PlyPointCloudReader(pcv.filepath).points
        points = apply_matrix(points, m)
        # s = real_length_to_relative(m, pcv.mesh_size)
        s = pcv.mesh_size
        a = pcv.mesh_normal_align
        c = pcv.mesh_vcols
        
        d = {'name': n, 'points': points, 'generator': g, 'matrix': Matrix(),
             'size': s, 'normal_align': a, 'vcols': c, }
        instancer = PCMeshInstancer(**d)
        
        o = instancer.object
        me = o.data
        me.transform(m.inverted())
        o.matrix_world = m
        
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
        
        # -------------- file selector
        def prop_name(cls, prop, colon=False, ):
            for p in cls.bl_rna.properties:
                if(p.identifier == prop):
                    if(colon):
                        return "{}:".format(p.name)
                    return p.name
            return ''
        
        r = sub.row(align=True, )
        s = r.split(factor=0.33)
        s.label(text=prop_name(pcv, 'filepath', True, ))
        s = s.split(factor=1.0)
        r = s.row(align=True, )
        c = r.column(align=True)
        c.prop(pcv, 'filepath', text='', )
        c.enabled = False
        r.operator('point_cloud_visualizer.load_ply_to_cache', icon='FILEBROWSER', text='', )
        # -------------- file selector
        
        e = not (pcv.filepath == "")
        r = sub.row(align=True)
        r.operator('point_cloud_visualizer.draw')
        r.operator('point_cloud_visualizer.erase')
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
        
        sub.separator()
        
        pcv = context.object.point_cloud_visualizer
        ok = False
        for k, v in PCVManager.cache.items():
            if(v['uuid'] == pcv.uuid):
                if(v['ready']):
                    if(v['draw']):
                        ok = True
        
        c = sub.column()
        c.prop(pcv, 'illumination', toggle=True, )
        if(ok):
            if(not pcv.has_normals):
                c.label(text="Missing vertex normals.", icon='ERROR', )
                c.enabled = False
        else:
            c.enabled = False
        if(pcv.illumination):
            cc = c.column()
            cc.prop(pcv, 'light_direction', text="", )
            ccc = cc.column(align=True)
            ccc.prop(pcv, 'light_intensity')
            ccc.prop(pcv, 'shadow_intensity')
            if(not pcv.has_normals):
                cc.enabled = e
        
        if(pcv.uuid in PCVManager.cache):
            sub.separator()
            r = sub.row()
            h, t = os.path.split(pcv.filepath)
            n = human_readable_number(PCVManager.cache[pcv.uuid]['stats'])
            r.label(text='{}: {} points'.format(t, n))


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
        r = c.row(align=True)
        r.operator('point_cloud_visualizer.render')
        r.operator('point_cloud_visualizer.animation')
        c = sub.column()
        c.prop(pcv, 'render_display_percent')
        c.prop(pcv, 'render_point_size')
        c.separator()
        c.prop(pcv, 'render_suffix')
        c.prop(pcv, 'render_zeros')
        c.enabled = PCV_OT_render.poll(context)


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
        cc = c.column()
        cc.prop(pcv, 'mesh_size')
        cc.prop(pcv, 'mesh_normal_align')
        cc.prop(pcv, 'mesh_vcols')
        if(pcv.mesh_type == 'VERTEX'):
            cc.enabled = False
        c.operator('point_cloud_visualizer.convert')
        c.enabled = PCV_OT_convert.poll(context)


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
        c.label(text="uuid: {}".format(pcv.uuid))
        c.label(text="filepath: {}".format(pcv.filepath))
        c.label(text="point_size: {}".format(pcv.point_size))
        c.label(text="alpha_radius: {}".format(pcv.alpha_radius))
        c.label(text="display_percent: {}".format(pcv.display_percent))
        c.label(text="render_expanded: {}".format(pcv.render_expanded))
        c.label(text="render_point_size: {}".format(pcv.render_point_size))
        c.label(text="render_display_percent: {}".format(pcv.render_display_percent))
        c.label(text="render_suffix: {}".format(pcv.render_suffix))
        c.label(text="render_zeros: {}".format(pcv.render_zeros))
        
        c.label(text="has_normals: {}".format(pcv.has_normals))
        c.label(text="has_vcols: {}".format(pcv.has_vcols))
        c.label(text="illumination: {}".format(pcv.illumination))
        c.label(text="light_direction: {}".format(pcv.light_direction))
        c.label(text="light_intensity: {}".format(pcv.light_intensity))
        c.label(text="shadow_intensity: {}".format(pcv.shadow_intensity))
        
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
    filepath: StringProperty(name="PLY file", default="", description="", )
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
    
    render_expanded: BoolProperty(default=False, options={'HIDDEN', }, )
    # render_point_size: FloatProperty(name="Size", default=3.0, min=0.001, max=100.0, precision=3, subtype='FACTOR', description="Render point size", )
    render_point_size: IntProperty(name="Size", default=3, min=1, max=100, subtype='PIXEL', description="Point size", )
    render_display_percent: FloatProperty(name="Count", default=100.0, min=0.0, max=100.0, precision=0, subtype='PERCENTAGE', description="Adjust percentage of points rendered", )
    render_suffix: StringProperty(name="Suffix", default="pcv_frame", description="Render filename or suffix, depends on render output path. Frame number will be appended automatically", )
    render_zeros: IntProperty(name="Leading Zeros", default=6, min=3, max=10, subtype='FACTOR', description="Number of leading zeros in render filename", )
    
    has_normals: BoolProperty(default=False)
    has_vcols: BoolProperty(default=False)
    illumination: BoolProperty(name="Illumination", description="Enable extra illumination on point cloud", default=False, )
    light_direction: FloatVectorProperty(name="Light Direction", description="Light direction", default=(0.0, 1.0, 0.0), subtype='DIRECTION', size=3, )
    # light_color: FloatVectorProperty(name="Light Color", description="", default=(0.2, 0.2, 0.2), min=0, max=1, subtype='COLOR', size=3, )
    light_intensity: FloatProperty(name="Light Intensity", description="Light intensity", default=0.3, min=0, max=1, subtype='FACTOR', )
    shadow_intensity: FloatProperty(name="Shadow Intensity", description="Shadow intensity", default=0.2, min=0, max=1, subtype='FACTOR', )
    show_normals: BoolProperty(name="Colorize By Vertex Normals", description="", default=False, )
    
    mesh_type: EnumProperty(name="Type", items=[('VERTEX', "Vertex", ""),
                                                ('TRIANGLE', "Equilateral Triangle", ""),
                                                ('TETRAHEDRON', "Tetrahedron", ""),
                                                ('CUBE', "Cube", ""),
                                                ('ICOSPHERE', "Ico Sphere", ""), ], default='CUBE', description="Instance mesh type", )
    mesh_size: FloatProperty(name="Size", description="Mesh instance size, instanced mesh has size 1.0", default=0.01, min=0.000001, precision=4, max=100.0, )
    mesh_normal_align: BoolProperty(name="Align To Normal", description="Align instance to point normal", default=True, )
    mesh_vcols: BoolProperty(name="Colors", description="Assign point color to instance vertex colors", default=True, )
    
    debug: BoolProperty(default=DEBUG, options={'HIDDEN', }, )
    
    @classmethod
    def register(cls):
        bpy.types.Object.point_cloud_visualizer = PointerProperty(type=cls)
    
    @classmethod
    def unregister(cls):
        del bpy.types.Object.point_cloud_visualizer


@persistent
def watcher(scene):
    PCVManager.deinit()


classes = (
    PCV_properties,
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
if(DEBUG):
    classes = classes + (
        PCV_PT_debug,
        PCV_OT_init,
        PCV_OT_deinit,
        PCV_OT_gc,
    )


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
