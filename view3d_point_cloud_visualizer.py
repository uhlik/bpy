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
           "version": (0, 7, 0),
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
from bpy.props import PointerProperty, BoolProperty, StringProperty, FloatProperty, IntProperty, FloatVectorProperty
from bpy.types import PropertyGroup, Panel, Operator
import gpu
from gpu.types import GPUOffScreen, GPUShader, GPUBatch, GPUVertBuf, GPUVertFormat
from gpu_extras.batch import batch_for_shader
from bpy.app.handlers import persistent
import bgl
from mathutils import Matrix, Vector
from bpy_extras.object_utils import world_to_camera_view
from bpy_extras.io_utils import axis_conversion


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
        pcv.light_enabled = False
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
    shader = GPUShader(PCVShaders.vertex_shader, PCVShaders.fragment_shader)
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
        
        shader.bind()
        pm = bpy.context.region_data.perspective_matrix
        shader.uniform_float("perspective_matrix", pm)
        shader.uniform_float("object_matrix", o.matrix_world)
        shader.uniform_float("point_size", pcv.point_size)
        shader.uniform_float("alpha_radius", pcv.alpha_radius)
        
        if(pcv.light_enabled and pcv.has_normals):
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
            shader.uniform_float("show_illumination", float(pcv.light_enabled))
        else:
            z = (0, 0, 0)
            shader.uniform_float("light_direction", z)
            shader.uniform_float("light_intensity", z)
            shader.uniform_float("shadow_direction", z)
            shader.uniform_float("shadow_intensity", z)
            shader.uniform_float("show_normals", float(False))
            shader.uniform_float("show_illumination", float(False))
        
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
            
            shader = GPUShader(PCVShaders.vertex_shader, PCVShaders.fragment_shader)
            batch = batch_for_shader(shader, 'POINTS', {"position": vs, "color": cs, "normal": ns, })
            shader.bind()
            
            view_matrix = cam.matrix_world.inverted()
            camera_matrix = cam.calc_matrix_camera(bpy.context.depsgraph, x=render.resolution_x, y=render.resolution_y, scale_x=render.pixel_aspect_x, scale_y=render.pixel_aspect_y, )
            perspective_matrix = camera_matrix @ view_matrix
            
            shader.uniform_float("perspective_matrix", perspective_matrix)
            shader.uniform_float("object_matrix", o.matrix_world)
            shader.uniform_float("point_size", pcv.render_point_size)
            shader.uniform_float("alpha_radius", pcv.alpha_radius)
            
            if(pcv.light_enabled and pcv.has_normals):
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
                shader.uniform_float("show_illumination", float(pcv.light_enabled))
            else:
                z = (0, 0, 0)
                shader.uniform_float("light_direction", z)
                shader.uniform_float("light_intensity", z)
                shader.uniform_float("shadow_direction", z)
                shader.uniform_float("shadow_intensity", z)
                shader.uniform_float("show_normals", float(False))
                shader.uniform_float("show_illumination", float(False))
            
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
        c.prop(pcv, 'light_enabled', toggle=True, )
        if(ok):
            if(not pcv.has_normals):
                c.label(text="Missing vertex normals.", icon='ERROR', )
                c.enabled = False
        else:
            c.enabled = False
        if(pcv.light_enabled):
            cc = c.column()
            cc.prop(pcv, 'light_direction', text="", )
            ccc = cc.column(align=True)
            ccc.prop(pcv, 'light_intensity')
            ccc.prop(pcv, 'shadow_intensity')
            if(not pcv.has_normals):
                cc.enabled = e
        
        sub.separator()
        
        b = sub.box()
        r = b.row()
        r.prop(pcv, 'render_expanded', icon='TRIA_DOWN' if pcv.render_expanded else 'TRIA_RIGHT', icon_only=True, emboss=False, )
        r.label(text="Render")
        if(pcv.render_expanded):
            c = b.column()
            r = c.row(align=True)
            r.operator('point_cloud_visualizer.render')
            r.operator('point_cloud_visualizer.animation')
            c = b.column()
            c.prop(pcv, 'render_display_percent')
            c.prop(pcv, 'render_point_size')
            c.separator()
            c.prop(pcv, 'render_suffix')
            c.prop(pcv, 'render_zeros')
            c.enabled = PCV_OT_render.poll(context)
        
        if(pcv.uuid in PCVManager.cache):
            r = sub.row()
            h, t = os.path.split(pcv.filepath)
            n = human_readable_number(PCVManager.cache[pcv.uuid]['stats'])
            r.label(text='{}: {} points'.format(t, n))
        
        if(pcv.debug):
            sub.separator()
            
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
            c.label(text="light_enabled: {}".format(pcv.light_enabled))
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
    light_enabled: BoolProperty(name="Illumination", description="Enable extra illumination on point cloud", default=False, )
    light_direction: FloatVectorProperty(name="Light Direction", description="Light direction", default=(0.0, 1.0, 0.0), subtype='DIRECTION', size=3, )
    # light_color: FloatVectorProperty(name="Light Color", description="", default=(0.2, 0.2, 0.2), min=0, max=1, subtype='COLOR', size=3, )
    light_intensity: FloatProperty(name="Light Intensity", description="Light intensity", default=0.3, min=0, max=1, subtype='FACTOR', )
    shadow_intensity: FloatProperty(name="Shadow Intensity", description="Shadow intensity", default=0.2, min=0, max=1, subtype='FACTOR', )
    show_normals: BoolProperty(name="Colorize By Vertex Normals", description="", default=False, )
    
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
    PCV_OT_load,
    PCV_OT_draw,
    PCV_OT_erase,
    PCV_OT_render,
    PCV_OT_animation,
)
if(DEBUG):
    classes = classes + (
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
    register()
