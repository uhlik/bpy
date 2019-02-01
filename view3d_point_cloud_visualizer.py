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
           "version": (0, 6, 0),
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
import numpy as np

import bpy
from bpy.props import PointerProperty, BoolProperty, StringProperty, FloatProperty, IntProperty
from bpy.types import PropertyGroup, Panel, Operator
import gpu
from gpu_extras.batch import batch_for_shader
from bpy.app.handlers import persistent
import bgl
from mathutils import Matrix, Vector
from bpy_extras.object_utils import world_to_camera_view


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


vertex_shader = '''
    in vec3 position;
    in vec4 color;
    uniform mat4 perspective_matrix;
    uniform mat4 object_matrix;
    // uniform float point_size;
    uniform float alpha_radius;
    out vec4 f_color;
    out float f_alpha_radius;
    
    void main()
    {
        gl_Position = perspective_matrix * object_matrix * vec4(position, 1.0f);
        // gl_PointSize = point_size;
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
        points = BinPlyPointCloudReader(filepath).points
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
    # # normals are not needed yet
    # if(not set(('nx', 'ny', 'nz')).issubset(points.dtype.names)):
    #     operator.report({'ERROR'}, "Loaded data seems to miss vertex normals.")
    #     return False
    vcols = True
    if(not set(('red', 'green', 'blue')).issubset(points.dtype.names)):
        vcols = False
    
    vs = np.column_stack((points['x'], points['y'], points['z'], ))
    if(vcols):
        cs = np.column_stack((points['red'] / 255, points['green'] / 255, points['blue'] / 255, np.ones(len(points), dtype=float, ), ))
        cs = cs.astype(np.float32)
    else:
        n = len(points)
        cs = np.column_stack((np.full(n, 0.75, dtype=np.float32, ),
                              np.full(n, 0.75, dtype=np.float32, ),
                              np.full(n, 0.75, dtype=np.float32, ),
                              np.ones(n, dtype=np.float32, ), ))
    
    u = str(uuid.uuid1())
    o = context.object
    
    pcv.uuid = u
    
    d = PCVManager.new()
    d['uuid'] = u
    d['stats'] = len(vs)
    d['vertices'] = vs
    d['colors'] = cs
    
    d['length'] = len(vs)
    dp = pcv.display_percent
    l = int((len(vs) / 100) * dp)
    if(dp >= 99):
        l = len(vs)
    d['display_percent'] = l
    d['current_display_percent'] = l
    shader = gpu.types.GPUShader(vertex_shader, fragment_shader)
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


def draw_circle_2d(position, color, radius, segments=32):
    # modified to draw filled circles from blender.app/Contents/Resources/2.80/scripts/modules/gpu_extras/presets.py
    
    from math import sin, cos, pi
    import gpu
    from gpu.types import (
        GPUBatch,
        GPUVertBuf,
        GPUVertFormat,
    )
    
    if segments <= 0:
        raise ValueError("Amount of segments must be greater than 0.")
    
    with gpu.matrix.push_pop():
        gpu.matrix.translate(position)
        gpu.matrix.scale_uniform(radius)
        mul = (1.0 / (segments - 1)) * (pi * 2)
        verts = [(sin(i * mul), cos(i * mul)) for i in range(segments)]
        fmt = GPUVertFormat()
        pos_id = fmt.attr_add(id="pos", comp_type='F32', len=2, fetch_mode='FLOAT')
        vbo = GPUVertBuf(len=len(verts), format=fmt)
        vbo.attr_fill(id=pos_id, data=verts)
        # batch = GPUBatch(type='LINE_STRIP', buf=vbo)
        batch = GPUBatch(type='TRI_FAN', buf=vbo)
        shader = gpu.shader.from_builtin('2D_UNIFORM_COLOR')
        batch.program_set(shader)
        shader.uniform_float("color", color)
        batch.draw()


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
            
            if(ci['current_display_percent'] != ci['display_percent']):
                l = ci['display_percent']
                ci['current_display_percent'] = l
                vs = ci['vertices']
                cs = ci['colors']
                batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "color": cs[:l], })
                ci['batch'] = batch
            
            o = ci['object']
            pcv = o.point_cloud_visualizer
            
            shader.bind()
            shader.uniform_float("perspective_matrix", pm)
            shader.uniform_float("object_matrix", o.matrix_world)
            # shader.uniform_float("point_size", pcv.point_size)
            shader.uniform_float("alpha_radius", pcv.alpha_radius)
            batch.draw(shader)
            
            ci['drawing'] = True
        
        run_gc = False
        for k, v in cls.cache.items():
            if(v['ready'] and v['draw'] and not v['drawing']):
                v['handle'] = bpy.types.SpaceView3D.draw_handler_add(draw, (v['uuid'], ), 'WINDOW', 'POST_VIEW')
            
            if(not bobjects.get(v['name'])):
                v['kill'] = True
                run_gc = True
        if(run_gc):
            cls.gc()
    
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
                'vertices': None,
                'colors': None,
                'display_percent': None,
                'current_display_percent': None,
                'shader': False,
                'batch': False,
                'ready': False,
                'draw': False,
                'drawing': False,
                'handle': None,
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
        scene = context.scene
        render = scene.render
        image_settings = render.image_settings
        
        original_depth = image_settings.color_depth
        image_settings.color_depth = '8'
        
        scale = render.resolution_percentage / 100
        width = int(render.resolution_x * scale)
        height = int(render.resolution_y * scale)
        
        offscreen = gpu.types.GPUOffScreen(width, height)
        view_matrix = Matrix(((2 / width, 0, 0, -1), (0, 2 / height, 0, -1), (0, 0, 1, 0), (0, 0, 0, 1), ))
        
        pcv = context.object.point_cloud_visualizer
        cloud = PCVManager.cache[pcv.uuid]
        model_matrix = cloud['object'].matrix_world
        cam = scene.camera
        render_segments = pcv.render_segments
        render_suffix = pcv.render_suffix
        render_zeros = pcv.render_zeros
        
        radius = pcv.render_size
        
        with offscreen.bind():
            bgl.glClear(bgl.GL_COLOR_BUFFER_BIT)
            
            gpu.matrix.reset()
            gpu.matrix.load_matrix(view_matrix)
            gpu.matrix.load_projection_matrix(Matrix.Identity(4))
            
            locs_2d = []
            for i, v in enumerate(cloud['vertices']):
                # covert point location to camera view coordinates
                vw = model_matrix @ Vector(v)
                loc = world_to_camera_view(scene, cam, vw)
                # join with color to be able to z sort points in next step
                col = cloud['colors'][i]
                locs_2d.append(loc.to_tuple() + tuple(col))
            # sort point by z (which is depth returned from world_to_camera_view)
            points_2d = sorted(locs_2d, key=lambda v: v[2])
            # draw circles in reversed order to have closest on top
            for p in reversed(points_2d):
                draw_circle_2d((width * p[0], height * p[1]), p[3:], radius, segments=render_segments, )
            
            buff = bgl.Buffer(bgl.GL_BYTE, width * height * 4)
            bgl.glReadBuffer(bgl.GL_COLOR_ATTACHMENT0)
            bgl.glReadPixels(0, 0, width, height, bgl.GL_RGBA, bgl.GL_UNSIGNED_BYTE, buff)
        
        offscreen.free()
        
        # image from buffer
        image_name = "pcv_output"
        if(not image_name in bpy.data.images):
            bpy.data.images.new(image_name, width, height)
        image = bpy.data.images[image_name]
        image.scale(width, height)
        image.pixels = [v / 255 for v in buff]
        
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
        r.prop(pcv, 'alpha_radius')
        r.enabled = e
        
        if(pcv.uuid in PCVManager.cache):
            r = sub.row()
            h, t = os.path.split(pcv.filepath)
            n = human_readable_number(PCVManager.cache[pcv.uuid]['stats'])
            r.label(text='{}: {} points'.format(t, n))
        
        sub.separator()
        
        c = sub.column()
        r = c.row(align=True)
        r.operator('point_cloud_visualizer.render')
        r.operator('point_cloud_visualizer.animation')
        c.enabled = PCV_OT_render.poll(context)
        
        b = sub.box()
        r = b.row()
        r.prop(pcv, 'render_expanded', icon='TRIA_DOWN' if pcv.render_expanded else 'TRIA_RIGHT', icon_only=True, emboss=False, )
        r.label(text="Render Options")
        if(pcv.render_expanded):
            c = b.column(align=True)
            c.prop(pcv, 'render_size')
            c.prop(pcv, 'render_segments')
            c.enabled = PCV_OT_render.poll(context)
            c = b.column()
            c.prop(pcv, 'render_suffix')
            c.prop(pcv, 'render_zeros')
            c.enabled = PCV_OT_render.poll(context)
        
        if(pcv.debug):
            sub.separator()
            sub.label(text="PCV uuid: {}".format(pcv.uuid))
            c = sub.column(align=True)
            c.operator('point_cloud_visualizer.init')
            c.operator('point_cloud_visualizer.deinit')
            c.operator('point_cloud_visualizer.gc')
            if(len(PCVManager.cache)):
                sub.separator()
                sub.label(text="PCVManager:")
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
    # point_size: FloatProperty(name="Size", default=1.0, min=0.001, max=100.0, precision=3, description="", )
    alpha_radius: FloatProperty(name="Radius", default=0.5, min=0.001, max=1.0, precision=3, subtype='FACTOR', description="Adjust point radius", )
    
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
    render_size: FloatProperty(name="Point Radius", default=2.0, min=0.1, max=10.0, precision=1, subtype='PIXEL', description="Adjust point render radius in pixels", )
    render_segments: IntProperty(name="Point Segments", default=32, min=3, max=256, subtype='NONE', description="Number of segments in each circle / points", )
    render_suffix: StringProperty(name="Suffix", default="pcv_frame", description="Render filename or suffix, depends on render output path. Frame number will be appended automatically", )
    render_zeros: IntProperty(name="Leading Zeros", default=6, min=3, max=10, subtype='FACTOR', description="Number of leading zeros in render filename", )
    
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
