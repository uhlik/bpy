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
           "description": "Display colored point cloud PLY in Blender's 3d viewport. Works with binary point cloud PLY files with 'x, y, z, red, green, blue' vertex values.",
           "author": "Jakub Uhlik",
           "version": (0, 4, 3),
           "blender": (2, 80, 0),
           "location": "3D Viewport > Sidebar > Point Cloud Visualizer",
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
import time
import datetime

import bpy
from bpy.props import PointerProperty, BoolProperty, StringProperty, FloatProperty
from bpy.types import PropertyGroup, Panel, Operator
import gpu
from gpu_extras.batch import batch_for_shader
from bpy.app.handlers import persistent


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


class BinPlyPointCloudReader():
    def __init__(self, path, ):
        # t = time.time()
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
        # d = datetime.timedelta(seconds=time.time() - t)
        # log("completed in {}.".format(d), 1)
    
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


vertex_shader = '''
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

fragment_shader = '''
    in vec4 f_color;
    in float f_alpha_radius;
    out vec4 fragColor;
    
    void main()
    {
        float r = 0.0;
        float delta = 0.0;
        float alpha = 1.0;
        vec2 cxy = 2.0 * gl_PointCoord - 1.0;
        r = dot(cxy, cxy);
        if(r > f_alpha_radius){
            discard;
        }
        fragColor = f_color * alpha;
    }
'''


def load_ply_to_cache(context, operator=None, ):
    pcv = context.object.point_cloud_visualizer
    filepath = pcv.filepath
    
    log('load data..')
    t = time.time()
    
    points = []
    try:
        points = BinPlyPointCloudReader(filepath).points
    except Exception as e:
        if(operator is not None):
            operator.report({'ERROR'}, str(e))
        else:
            raise e
    if(len(points) == 0):
        operator.report({'ERROR'}, "No vertices loaded from file at {}".format(filepath))
        return False
    
    log('done.')
    d = datetime.timedelta(seconds=time.time() - t)
    log("completed in {}.".format(d))
    
    # # no need to shuffle because display percent is not yet implemented
    # rnd = random.Random()
    # random.shuffle(points, rnd.random)
    
    log('process data..')
    t = time.time()
    
    # vs = []
    # cs = []
    # for i, p in enumerate(points):
    #     vs.append(tuple(p[:3]))
    #     c = [v / 255 for v in p[3:]]
    #     cs.append(tuple(c) + (1.0, ))
    
    l = len(points)
    vs = [None] * l
    cs = [None] * l
    for i, p in enumerate(points):
        vs[i] = p[:3]
        cs[i] = tuple(v / 255 for v in p[3:]) + (1.0, )
    
    u = str(uuid.uuid1())
    o = context.object
    
    pcv.uuid = u
    
    d = PCVManager.new()
    d['uuid'] = u
    d['stats'] = len(vs)
    
    shader = gpu.types.GPUShader(vertex_shader, fragment_shader)
    batch = batch_for_shader(shader, 'POINTS', {"position": vs, "color": cs, })
    
    d['shader'] = shader
    d['batch'] = batch
    d['ready'] = True
    d['object'] = o
    d['name'] = o.name
    
    PCVManager.add(d)
    
    log('done.')
    d = datetime.timedelta(seconds=time.time() - t)
    log("completed in {}.".format(d))
    
    return True


class PCVManager():
    cache = {}
    handle = None
    initialized = False
    
    @classmethod
    def handler(cls):
        bobjects = bpy.data.objects
        
        def draw(uuid):
            pm = bpy.context.region_data.perspective_matrix
            
            ci = PCVManager.cache[uuid]
            if(not bobjects.get(ci['name'])):
                ci['kill'] = True
                cls.gc()
                return
            if(not ci['draw']):
                cls.gc()
                return
            
            shader = ci['shader']
            batch = ci['batch']
            
            o = ci['object']
            pcv = o.point_cloud_visualizer
            
            shader.bind()
            shader.uniform_float("perspective_matrix", pm)
            shader.uniform_float("object_matrix", o.matrix_world)
            shader.uniform_float("point_size", pcv.point_size)
            shader.uniform_float("alpha_radius", pcv.alpha_radius)
            batch.draw(shader)
            
            ci['drawing'] = True
        
        for k, v in cls.cache.items():
            if(v['ready'] and v['draw'] and not v['drawing']):
                v['handle'] = bpy.types.SpaceView3D.draw_handler_add(draw, (v['uuid'], ), 'WINDOW', 'POST_VIEW')
    
    @classmethod
    def gc(cls):
        l = []
        for k, v in cls.cache.items():
            if(v['kill']):
                l.append(k)
                if(v['drawing']):
                    bpy.types.SpaceView3D.draw_handler_remove(v['handle'], 'WINDOW')
                    v['handle'] = None
                    v['drawing'] = False
            if(v['drawing'] and not v['draw']):
                bpy.types.SpaceView3D.draw_handler_remove(v['handle'], 'WINDOW')
                v['handle'] = None
                v['drawing'] = False
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
        
        cls.initialized = False
        
        bpy.app.handlers.load_pre.remove(watcher)
    
    @classmethod
    def add(cls, data, ):
        cls.cache[data['uuid']] = data
    
    @classmethod
    def new(cls):
        return {'uuid': None,
                'shader': False,
                'batch': False,
                'ready': False,
                'draw': False,
                'drawing': False,
                'handle': None,
                'kill': False,
                'stats': None,
                'object': None, }


class PCV_OT_deinit(Operator):
    bl_idname = "point_cloud_visualizer.deinit"
    bl_label = "deinit"
    bl_description = ""
    
    @classmethod
    def poll(cls, context):
        return True
    
    def execute(self, context):
        PCVManager.deinit()
        return {'FINISHED'}


class PCV_OT_draw(Operator):
    bl_idname = "point_cloud_visualizer.draw"
    bl_label = "Draw"
    bl_description = "Draw point cloud to viewport"
    
    @classmethod
    def poll(cls, context):
        return True
    
    def execute(self, context):
        PCVManager.init()
        
        pcv = context.object.point_cloud_visualizer
        cached = False
        for k, v in PCVManager.cache.items():
            if(v['uuid'] == pcv.uuid):
                cached = True
                if(v['ready']):
                    if(not v['draw']):
                        v['draw'] = True
                else:
                    # why is this here? if it is cached, it means it has been loaded already, so ready attribute is not even needed..
                    # bpy.ops.point_cloud_visualizer.load_ply_to_cache('INVOKE_DEFAULT')
                    v['draw'] = True
        if(not cached):
            if(pcv.filepath != ""):
                pcv.uuid = ""
                ok = load_ply_to_cache(context, self)
                if(not ok):
                    return {'CANCELLED'}
                v = PCVManager.cache[pcv.uuid]
                v['draw'] = True
        
        context.area.tag_redraw()
        
        return {'FINISHED'}


class PCV_OT_erase(Operator):
    bl_idname = "point_cloud_visualizer.erase"
    bl_label = "Erase"
    bl_description = "Erase point cloud from viewport"
    
    @classmethod
    def poll(cls, context):
        return True
    
    def execute(self, context):
        pcv = context.object.point_cloud_visualizer
        ok = False
        for k, v in PCVManager.cache.items():
            if(v['uuid'] == pcv.uuid):
                if(v['ready']):
                    if(v['draw']):
                        v['draw'] = False
                        ok = True
        if(not ok):
            return {'CANCELLED'}
        
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
    
    def check(self, context):
        return True
    
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
        
        ok = load_ply_to_cache(context, self)
        
        if(not ok):
            return {'CANCELLED'}
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
        r.operator('point_cloud_visualizer.draw', icon='HIDE_OFF', )
        r.operator('point_cloud_visualizer.erase', icon='HIDE_ON', )
        r.enabled = e
        r = sub.row()
        r.prop(pcv, 'alpha_radius')
        r.enabled = e
        
        if(pcv.uuid in PCVManager.cache):
            r = sub.row()
            h, t = os.path.split(pcv.filepath)
            n = int_to_short_notation(PCVManager.cache[pcv.uuid]['stats'], precision=1, )
            r.label(text='{}: {} points'.format(t, n))
        
        if(pcv.debug):
            sub.separator()
            # sub.operator('point_cloud_visualizer.deinit')
            sub.separator()
            sub.label(text="PCV uuid: {}".format(pcv.uuid))
            sub.label(text="PCVManager:")
            sub.separator()
            for k, v in PCVManager.cache.items():
                sub.label(text="key: {}".format(k))
                sub.label(text="uuid: {}".format(v['uuid']))
                sub.label(text="ready: {}".format(v['ready']))
                sub.label(text="draw: {}".format(v['draw']))
                sub.label(text="drawing: {}".format(v['drawing']))
                sub.label(text="handle: {}".format(v['handle']))
                sub.label(text="----------------------")


class PCV_properties(PropertyGroup):
    filepath: StringProperty(name="PLY file", default="", description="", )
    uuid: StringProperty(default="", options={'HIDDEN', }, )
    point_size: FloatProperty(name="Size", default=1.0, min=0.001, max=100.0, precision=3, description="", )
    alpha_radius: FloatProperty(name="Radius", default=0.5, min=0.001, max=1.0, precision=3, subtype='FACTOR', description="Adjust point radius", )
    
    debug: BoolProperty(default=False, options={'HIDDEN', }, )
    
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
    PCV_OT_load,
    PCV_OT_draw,
    PCV_OT_erase,
    # PCV_OT_deinit,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    PCVManager.deinit()
    
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
