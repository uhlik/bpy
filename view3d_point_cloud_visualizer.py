bl_info = {"name": "Point Cloud Visualizer",
           "description": "Display colored point cloud PLY in Blender's 3d viewport. Works with binary point cloud PLY files in format (x, y, z, nx, ny, nz, r(8bit), g(8bit), b(8bit), (alpha)) exported from Agisoft PhotoScan or MeshLab.",
           "author": "Jakub Uhlik",
           "version": (0, 1, 0),
           "blender": (2, 78, 0),
           "location": "",
           "warning": "",
           "wiki_url": "",
           "tracker_url": "",
           "category": "3D View", }


import os
import math
import struct
import uuid

import bpy
import bgl
from mathutils import Matrix, Vector
from bpy.props import PointerProperty, BoolProperty, StringProperty
from bpy.types import PropertyGroup, Panel, Operator


def log(msg, indent=0):
    m = "{0}> {1}".format("    " * indent, msg)
    print(m)


class BinPlyPointCloudTypes():
    # (application name, data format, total number of verts header line index)
    PHOTOSCAN = ("PhotoScan", '<ffffffBBB', 2, )
    MESHLAB = ("MeshLab", '<ffffffBBBB', 3, )
    TEOPLIB = ("teoplib", '<ffffffBBB', 2, )
    ALL = [PHOTOSCAN, MESHLAB, TEOPLIB, ]


class BinPlyPointCloudInfo():
    def __init__(self, path, verbose=True, ):
        self.verbose = verbose
        if(self.verbose):
            log("BinPlyPointCloudInfo:", 0)
        
        if(os.path.exists(path) is False or os.path.isdir(path) is True):
            raise OSError("did you point me to an imaginary file?")
        self.path = path
        self._read()
    
    def _read(self):
        if(self.verbose):
            log("read ply from: {0}".format(self.path), 1)
        with open(self.path, 'rb') as ply:
            self._header(ply)
        
        self._guess()
        if(self.creator is not None):
            if(self.verbose):
                log("ply creator (guessed): {0}".format(self.creator[0]), 1)
        else:
            if(self.verbose):
                log("ply creator: unknown", 1)
        
        self.vertices = self.elements['vertex']['num']
        if(self.verbose):
            log("vertices: {0}".format(self.vertices), 1)
    
    def _header(self, ply):
        self.header = []
        for line in ply:
            ln = line.decode('ascii').rstrip()
            if(ln == "end_header"):
                self.header.append(ln)
                break
            self.header.append(ln)
        
        self.elements = {}
        elm = None
        for l in self.header:
            if(l == "ply"):
                pass
            elif(l[:7] == "format "):
                self.format = l
            elif(l[:8] == "element "):
                ls = l.split(" ")
                self.elements[ls[1]] = {"num": int(ls[2])}
                elm = self.elements[ls[1]]
                elm["props"] = []
            elif(l[:9] == "property "):
                elm["props"].append(l[9:])
            elif(l == "end_header"):
                pass
            else:
                if(l[:8] == "comment "):
                    pass
                else:
                    log("unknown header entry: {0}".format(l), 1)
    
    def _guess(self):
        self.creator = None
        
        fmt = ''
        if("binary_little_endian" in self.format):
            fmt = '<'
        else:
            log("supported only binary little endian", 1)
            return
        
        if(len(self.elements) == 1):
            try:
                props = self.elements['vertex']['props']
                for p in props:
                    sp = p.split(" ")
                    if(sp[0] == "float"):
                        fmt += ('f')
                    elif(sp[0] == "uchar"):
                        fmt += ('B')
                    else:
                        log("unimplemented format type: {0}".format(sp[0]), 1)
                        return
            except KeyError:
                log("seems like this is not just point cloud ply.. (no vertices??!?!??!)", 1)
                return
        else:
            if(len(self.elements) == 2 and 'face' in self.elements):
                try:
                    if(self.elements['face']['num'] == 0):
                        pass
                except:
                    log("seems like this is not just point cloud ply.. (not only vertices)", 1)
                
                try:
                    props = self.elements['vertex']['props']
                    for p in props:
                        sp = p.split(" ")
                        if(sp[0] == "float"):
                            fmt += ('f')
                        elif(sp[0] == "uchar"):
                            fmt += ('B')
                        else:
                            log("unimplemented format type: {0}".format(sp[0]), 1)
                            return
                except KeyError:
                    log("seems like this is not just point cloud ply.. (no vertices??!?!??!)", 1)
                    return
                
            else:
                log("contains more elements than expected.. is it a point cloud ply?", 1)
                return
        
        for i, l in enumerate(self.header):
            if(l[:8] == "element "):
                break
        vchl = i
        
        for a in BinPlyPointCloudTypes.ALL:
            if(a[1] == fmt and a[2] == vchl):
                self.creator = a
                break
        
        if(self.creator is None):
            log("unidentified ply cretor", 1)
    
    def __repr__(self):
        s = "BinPlyPointCloudInfo(path='{0}')"
        return s.format(self.path)
    
    def __str__(self):
        s = "BinPlyPointCloudInfo(path: '{0}', creator: {1}, vertices: {2}, )"
        return s.format(self.path, self.creator, self.vertices, )


class BinPlyPointCloudReader():
    def __init__(self, path, creator, read_from=0, read_length=0, ):
        log("BinPlyPointCloudReader:", 0)
        
        if(os.path.exists(path) is False or os.path.isdir(path) is True):
            raise OSError("did you point me to an imaginary file? ('{0}')".format(path))
        self.path = path
        
        # ply file creator application
        if(creator == BinPlyPointCloudTypes.PHOTOSCAN):
            self.creator = BinPlyPointCloudTypes.PHOTOSCAN
        elif(creator == BinPlyPointCloudTypes.MESHLAB):
            self.creator = BinPlyPointCloudTypes.MESHLAB
        elif(creator == BinPlyPointCloudTypes.TEOPLIB):
            self.creator = BinPlyPointCloudTypes.TEOPLIB
        else:
            raise ValueError("unknown ply creator")
        # ply data format
        self.format = self.creator[1]
        # ply header total number of vertices line number
        self.header_line = self.creator[2]
        
        # from which point start reading
        if(read_from < 0):
            raise ValueError("read_from '{0}' is less than zero".format(read_from))
        self.read_from = read_from
        
        # how many vertices to read from ply
        if(read_length < 0):
            raise ValueError("read_length '{0}' is less than zero".format(read_length))
        self.read_length = read_length
        
        # list of read vertices
        self.vertices = []
        # number of vertices in ply
        self.total_num_verts = 0
        
        self._read_ply()
    
    def _read_ply(self):
        log("read ply from: {0}".format(self.path), 1)
        with open(self.path, 'rb') as ply:
            self._header(ply)
            log("ply file creator: {0}".format(self.creator[0]), 1)
            log("total vertices: {0}".format(self.total_num_verts), 1)
            self._vertices(ply)
        log("done.", 1)
    
    def _header(self, ply):
        header = []
        for line in ply:
            if(line.decode('ascii') == "end_header\n"):
                header.append(line)
                break
            header.append(line)
        
        l = 0
        for i in header:
            l += len(i)
        ply.seek(0)
        h = ply.read(l)
        
        self.header_length = l
        self.total_num_verts = int(header[self.header_line].decode('ascii').split(" ")[2])
    
    def _vertices(self, ply):
        sz = struct.calcsize(self.format)
        n = self.total_num_verts
        l = self.read_length
        f = self.read_from
        
        if(f > n):
            raise ValueError("read_from is more than total_num_verts")
        if(f == n):
            raise ValueError("read_from is equal to total_num_verts")
        if(f + l > n):
            # overflow..
            log("no more than {0} verts available!".format(n - f), 1)
            l = n - f
            n = l
        else:
            # in range..
            # from: 0, length: 0 = all
            if(f == 0 and l == 0):
                n = n
            # from: a, length: 0 = all - a
            if(f != 0 and l == 0):
                n = n - f
            # from: 0, length: b = all - b
            if(f == 0 and l != 0):
                n = l
            # from: a, length: b = all - a - b
            if(f != 0 and l != 0):
                n = l
        
        # skip to read_from
        s = struct.calcsize(self.format[:1] + (self.format[1:] * self.read_from))
        ply.seek(self.header_length + s)
        
        # update self.read_length
        self.read_length = n
        
        log("reading {0} vertices starting at {1}:".format(n, f), 1)
        for i in range(n):
            d = ply.read(sz)
            v = struct.unpack(self.format, d)
            self.vertices.append(v)
    
    def __repr__(self):
        s = "BinPlyPointCloudReader(path='{0}', creator={1}, read_from={2}, read_length={3}, )"
        return s.format(self.path, self.creator, self.read_from, self.read_length, )
    
    def __str__(self):
        s = "BinPlyPointCloudInfo(path: '{0}')"
        return s.format(self.path)


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
        
        bgl.glDrawArrays(bgl.GL_POINTS, 0, c['length'])
        
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
        
        # load points
        points = BinPlyPointCloudReader(p, BinPlyPointCloudInfo(p, True).creator, 0, 0, ).vertices
        
        # process points
        vertices = []
        colors = []
        for i, p in enumerate(points):
            v = Vector(p[:3])
            vertices.extend(v.to_tuple())
            # ply from meshlab has also alpha value, throw it away..
            c = [v / 255 for v in p[6:9]]
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
        r = sub.row(align=True, )
        r.operator('point_cloud_visualizer.load')
        r.prop(pcv, 'auto', toggle=True, icon='AUTO', icon_only=True, )
        if(pcv.uuid != ""):
            if(pcv.draw or pcv.uuid in PCVCache.cache):
                sub.enabled = False
        
        c = l.column()
        r = c.row(align=True, )
        r.prop(pcv, 'draw', toggle=True, )
        r.prop(pcv, 'smooth', toggle=True, icon='ANTIALIASED', icon_only=True, )
        c.enabled = False
        
        if(pcv.uuid != "" and pcv.uuid in PCVCache.cache):
            if(PCVCache.cache[pcv.uuid]['ready']):
                c.enabled = True
        
        c = l.column()
        r = c.row()
        if(pcv.uuid in PCVCache.cache):
            
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
    
    filepath = StringProperty(name="PLY file", default="", description="", )
    auto = BoolProperty(name="Autoload", default=False, description="Load chosen file automatically", )
    uuid = StringProperty(default="", options={'HIDDEN', 'SKIP_SAVE', }, )
    smooth = BoolProperty(name="GL_POINT_SMOOTH", default=False, description="Use GL_POINT_SMOOTH", update=_smooth_update, )
    draw = BoolProperty(name="Draw", default=False, description="Enable/disable drawing", update=_draw_update, )
    
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
