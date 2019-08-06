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
           "description": "Display colored point cloud PLY in Blender's 3d viewport. Works with binary point cloud PLY files with 'x, y, z, red, green, blue' vertex values. All other values are ignored.",
           "author": "Jakub Uhlik",
           "version": (0, 3, 0),
           "blender": (2, 78, 0),
           "location": "View3d > Properties > Point Cloud Visualizer (with an Empty object active)",
           "warning": "",
           "wiki_url": "",
           "tracker_url": "",
           "category": "3D View", }


import os
import math
import struct
import uuid
import numpy
import random

import bpy
import bgl
from mathutils import Matrix, Vector
from bpy.props import PointerProperty, BoolProperty, StringProperty, FloatProperty, IntProperty
from bpy.types import PropertyGroup, Panel, Operator


def log(msg, indent=0):
    m = "{0}> {1}".format("    " * indent, msg)
    print(m)


def int_to_short_notation(n, precision=1, ):
    if(round(n / 10 ** 12, precision) >= 1):
        r = int(round(n / 10 ** 12, precision))
        return '{}t'.format(r)
    elif(round(n / 10 ** 9, precision) >= 1):
        r = int(round(n / 10 ** 9, precision))
        return '{}g'.format(r)
    elif(round(n / 10 ** 6, precision) >= 1):
        r = int(round(n / 10 ** 6, precision))
        return '{}m'.format(r)
    elif(round(n / 10 ** 3, precision) >= 1):
        r = int(round(n / 10 ** 3, precision))
        return '{}k'.format(r)
    else:
        r = round(n / 10 ** 3, precision)
        if(r >= 0.1):
            return '{}k'.format(r)
        else:
            return '{}'.format(n)


def clamp(f, t, l):
    if(f != 0 or t != 0):
        # only when something is set
        if(f != 0 and t == 0):
            # from is set
            f, _ = tuple(numpy.clip([f, t], 0, l))
        elif(f == 0 and t != 0):
            # to is set
            _, t = tuple(numpy.clip([f, t], 0, l))
        elif(f != 0 and t != 0):
            # both are set
            f, t = tuple(numpy.clip([f, t], 0, l))
        else:
            print("wtf?")
            pass
    
    if(f > t):
        # swap
        a = f
        f = t
        t = a
    
    if(f == t and f != 0 and t != 0):
        f = f - 1
    
    return f, t


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
        self._data()
        
        self._stream.close()
        
        props = ['x', 'y', 'z', 'red', 'green', 'blue', ]
        es = self._elements
        for e in es:
            if(e['name'] == 'vertex'):
                ps = e['properties']
        q = []
        for i, n in enumerate(props):
            for j, p in enumerate(ps):
                if(n == p[0]):
                    q.append(j)
        vd = self.data['vertex']
        self.points = [(i[q[0]], i[q[1]], i[q[2]], i[q[3]], i[q[4]], i[q[5]], ) for i in vd]
        
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
    
    def _data(self):
        self.data = {}
        for i, d in enumerate(self._elements):
            nm = d['name']
            if(nm != 'vertex'):
                # read only vertices
                continue
            a = []
            f = self._endianness
            f += ''.join([i[1] for i in d['properties']])
            c = d['count']
            sz = struct.calcsize(f)
            log("reading {} {} elements..".format(c, nm), 2)
            self._stream.seek(self._header_length)
            for i in range(c):
                r = self._stream.read(sz)
                v = struct.unpack(f, r)
                a.append(v)
            self.data[nm] = a


class PCVCache():
    cache = {}
    
    @classmethod
    def add(cls, data, ):
        cls.cache[data['uuid']] = data
    
    @classmethod
    def new(cls):
        return {'uuid': None,
                'path': None,
                'ready': False,
                'length': None,
                'vertex_buffer': None,
                'color_buffer': None,
                'smooth': False,
                'drawing': False,
                'matrix': None,
                'matrix_buffer': None,
                'display_percent': None,
                'object': None, }


def PCV_draw_callback(self, context, ):
    def draw_one(u):
        c = PCVCache.cache[u]
        # update matrix, every frame for now, it should be done better.. but it works well..
        m = c['object'].matrix_world
        matrix = []
        for v in m.transposed():
            matrix.extend(list(v.to_tuple()))
        matrix_buffer = bgl.Buffer(bgl.GL_FLOAT, len(matrix), matrix)
        c['matrix'] = m
        c['matrix_buffer'] = matrix_buffer
        
        bgl.glPushMatrix()
        bgl.glMultMatrixf(c['matrix_buffer'])
        
        bgl.glEnable(bgl.GL_DEPTH_TEST)
        bgl.glEnableClientState(bgl.GL_VERTEX_ARRAY)
        bgl.glVertexPointer(3, bgl.GL_FLOAT, 0, c['vertex_buffer'])
        bgl.glEnableClientState(bgl.GL_COLOR_ARRAY)
        bgl.glColorPointer(3, bgl.GL_FLOAT, 0, c['color_buffer'])
        
        if(PCVCache.cache[u]['smooth']):
            bgl.glEnable(bgl.GL_POINT_SMOOTH)
        
        l = int((c['length'] / 100) * c['display_percent'])
        bgl.glDrawArrays(bgl.GL_POINTS, 0, l)
        
        bgl.glDisableClientState(bgl.GL_VERTEX_ARRAY)
        bgl.glDisableClientState(bgl.GL_COLOR_ARRAY)
        
        if(c['smooth']):
            bgl.glDisable(bgl.GL_POINT_SMOOTH)
        bgl.glDisable(bgl.GL_DEPTH_TEST)
        
        bgl.glPopMatrix()
    
    # draw each 'ready' and 'drawing' cloud from cache
    for k, v in PCVCache.cache.items():
        if(v['ready'] and v['drawing']):
            draw_one(k)


class PCVDraw(Operator):
    bl_idname = "point_cloud_visualizer.draw"
    bl_label = "Draw"
    bl_description = ""
    
    @classmethod
    def poll(cls, context):
        pcv = context.object.point_cloud_visualizer
        if(pcv.uuid != "" and pcv.uuid in PCVCache.cache):
            return PCVCache.cache[pcv.uuid]['ready']
        return False
    
    def modal(self, context, event):
        context.area.tag_redraw()
        
        c = 0
        for k, v in PCVCache.cache.items():
            if(v['drawing']):
                c += 1
        if(not c):
            # remove only when number of drawing clouds is zero
            bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
            return {'CANCELLED'}
        
        return {'PASS_THROUGH'}
    
    def invoke(self, context, event):
        if(context.area.type == 'VIEW_3D'):
            args = (self, context)
            self._handle = bpy.types.SpaceView3D.draw_handler_add(PCV_draw_callback, args, 'WINDOW', 'POST_VIEW')
            context.window_manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        else:
            self.report({'WARNING'}, "View3D not found, cannot run operator")
            return {'CANCELLED'}


class PCVReset(Operator):
    bl_idname = "point_cloud_visualizer.reset"
    bl_label = "Reset"
    bl_description = "Reset"
    
    def execute(self, context):
        pcv = context.object.point_cloud_visualizer
        u = pcv.uuid
        
        pcv.filepath = ""
        pcv.smooth = False
        pcv.draw = False
        pcv.uuid = ""
        pcv.display_percent = 100.0
        
        if(u in PCVCache.cache):
            # if reseting duplicated object, do not remove cache, can be used by another object still in scene
            if(context.object == PCVCache.cache[u]['object']):
                del PCVCache.cache[u]
        
        return {'FINISHED'}


class PCVLoader(Operator):
    bl_idname = "point_cloud_visualizer.load"
    bl_label = "Load Points"
    bl_description = "Load Points"
    
    @classmethod
    def poll(cls, context):
        pcv = context.object.point_cloud_visualizer
        if(pcv.filepath != ""):
            return True
        return False
    
    def load(self, context):
        pcv = context.object.point_cloud_visualizer
        p = os.path.abspath(bpy.path.abspath(pcv.filepath))
        if(not os.path.exists(p)):
            self.report({'WARNING'}, "File does not exist")
            return {'CANCELLED'}
        
        points = BinPlyPointCloudReader(p).points
        
        rnd = random.Random()
        random.shuffle(points, rnd.random)
        
        # process points
        vertices = []
        colors = []
        for i, p in enumerate(points):
            v = Vector(p[:3])
            vertices.extend(v.to_tuple())
            c = [v / 255 for v in p[3:]]
            colors.extend(c)
        
        # make buffers
        length = len(points)
        vertex_buffer = bgl.Buffer(bgl.GL_FLOAT, len(vertices), vertices)
        color_buffer = bgl.Buffer(bgl.GL_FLOAT, len(colors), colors)
        
        o = context.object
        m = o.matrix_world
        matrix = []
        for v in m.transposed():
            matrix.extend(list(v.to_tuple()))
        matrix_buffer = bgl.Buffer(bgl.GL_FLOAT, len(matrix), matrix)
        
        d = PCVCache.new()
        u = str(uuid.uuid1())
        d['uuid'] = u
        d['path'] = pcv.filepath
        d['ready'] = True
        d['length'] = length
        d['vertex_buffer'] = vertex_buffer
        d['color_buffer'] = color_buffer
        d['matrix'] = m
        d['matrix_buffer'] = matrix_buffer
        d['object'] = o
        d['display_percent'] = pcv.display_percent
        PCVCache.add(d)
        
        pcv.uuid = u
    
    def execute(self, context):
        try:
            self.load(context)
            
            # auto draw cloud
            pcv = context.object.point_cloud_visualizer
            pcv.draw = True
            bpy.ops.point_cloud_visualizer.draw('INVOKE_DEFAULT')
            
        except Exception as e:
            self.report({'ERROR'}, 'Unable to load .ply file.')
            # self.report({'ERROR'}, e)
            return {'CANCELLED'}
        return {'FINISHED'}


class PCVPanel(Panel):
    bl_label = "Point Cloud Visualizer"
    bl_idname = "PointCloudVisualizer"
    bl_space_type = 'VIEW_3D'
    bl_context = "scene"
    bl_region_type = 'UI'
    bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        o = context.active_object
        if(o and o.type == 'EMPTY'):
            return True
        return False
    
    def draw(self, context):
        l = self.layout
        sub = l.column()
        pcv = context.object.point_cloud_visualizer
        
        # StringProperty subtype FILE_PATH file selector remake
        def prop_name(cls, prop, colon=False, ):
            for p in cls.bl_rna.properties:
                if(p.identifier == prop):
                    if(colon):
                        return "{}:".format(p.name)
                    return p.name
            return ''
        
        r = sub.row(align=True, )
        s = r.split(percentage=0.33)
        s.label(prop_name(pcv, 'filepath', True, ))
        s = s.split(percentage=1.0)
        r = s.row(align=True, )
        r.prop(pcv, 'filepath', text='', )
        r.operator('point_cloud_visualizer.auto_load', icon='FILESEL', text='', )
        
        sub.separator()
        # r = sub.row(align=True, )
        # r.prop(pcv, 'load_from')
        # r.prop(pcv, 'load_to')
        r = sub.row(align=True, )
        r.operator('point_cloud_visualizer.load')
        r.prop(pcv, 'auto', toggle=True, icon='AUTO', icon_only=True, )
        if(pcv.uuid != ""):
            if(pcv.draw or pcv.uuid in PCVCache.cache):
                sub.enabled = False
        
        # sub.separator()
        c = l.column(align=True, )
        r = c.row(align=True, )
        r.prop(pcv, 'draw', toggle=True, )
        r.prop(pcv, 'smooth', toggle=True, icon='ANTIALIASED', icon_only=True, )
        r = c.row(align=True, )
        r.prop(pcv, 'display_percent')
        # r.prop(pcv, 'display_max')
        c.enabled = False
        
        if(pcv.uuid != "" and pcv.uuid in PCVCache.cache):
            if(PCVCache.cache[pcv.uuid]['ready']):
                c.enabled = True
        
        c = l.column()
        r = c.row()
        if(pcv.uuid in PCVCache.cache):
            h, t = os.path.split(pcv.filepath)
            n = int_to_short_notation(PCVCache.cache[pcv.uuid]['length'], precision=1, )
            r.label("{}: {} points".format(t, n))
        else:
            r.label("n/a")
        r.operator('point_cloud_visualizer.reset', icon='X', text='', )


class PCVAutoLoadHelper():
    filepath = StringProperty(name="File Path", default="", description="", maxlen=1024, subtype='FILE_PATH', )
    order = ["filepath", ]
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}
    
    def check(self, context):
        return True


class PCVAutoLoad(Operator, PCVAutoLoadHelper, ):
    bl_idname = "point_cloud_visualizer.auto_load"
    bl_label = "Choose file"
    bl_description = "Choose file"
    
    filename_ext = ".ply"
    filter_glob = bpy.props.StringProperty(default="*.ply", options={'HIDDEN'}, )
    
    def execute(self, context):
        pcv = context.object.point_cloud_visualizer
        pcv.filepath = self.filepath
        if(pcv.auto):
            bpy.ops.point_cloud_visualizer.load('INVOKE_DEFAULT')
        return {'FINISHED'}


class PCVProperties(PropertyGroup):
    def _smooth_update(self, context, ):
        if(self.uuid != "" and self.uuid in PCVCache.cache):
            PCVCache.cache[self.uuid]['smooth'] = self.smooth
    
    def _draw_update(self, context, ):
        if(self.uuid != "" and self.uuid in PCVCache.cache):
            if(self.draw):
                PCVCache.cache[self.uuid]['drawing'] = True
                bpy.ops.point_cloud_visualizer.draw('INVOKE_DEFAULT')
            else:
                PCVCache.cache[self.uuid]['drawing'] = False
    
    def _percentage_update(self, context, ):
        if(self.uuid != "" and self.uuid in PCVCache.cache):
            PCVCache.cache[self.uuid]['display_percent'] = self.display_percent
    
    filepath = StringProperty(name="PLY file", default="", description="", )
    auto = BoolProperty(name="Autoload", default=False, description="Load chosen file automatically", )
    uuid = StringProperty(default="", options={'HIDDEN', 'SKIP_SAVE', }, )
    smooth = BoolProperty(name="GL_POINT_SMOOTH", default=False, description="Use GL_POINT_SMOOTH", update=_smooth_update, )
    draw = BoolProperty(name="Draw", default=False, description="Enable/disable drawing", update=_draw_update, )
    # load_from = IntProperty(name="From", default=0, min=0, )
    # load_to = IntProperty(name="To", default=0, min=0, )
    display_percent = FloatProperty(name="Display", default=100.0, min=0.0, max=100.0, precision=0, subtype='PERCENTAGE', update=_percentage_update, )
    # display_max = IntProperty(name="Max", default=0, min=0, )
    
    @classmethod
    def register(cls):
        bpy.types.Object.point_cloud_visualizer = PointerProperty(type=cls)
    
    @classmethod
    def unregister(cls):
        del bpy.types.Object.point_cloud_visualizer


def register():
    bpy.utils.register_module(__name__)


def unregister():
    bpy.utils.unregister_module(__name__)


if __name__ == "__main__":
    register()
