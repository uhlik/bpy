bl_info = {"name": "Import ZBrush Wavefront OBJ",
           "description": "Import Wavefront OBJ exported from ZBrush with UV and polypaint (vertex colors).",
           "author": "Jakub Uhlik",
           "version": (0, 2, 0),
           "blender": (2, 78, 0),
           "location": "File > Import > ZBrush Wavefront (.obj)",
           "warning": "",
           "wiki_url": "",
           "tracker_url": "",
           "category": "Import-Export", }


import os
import sys
import time
import datetime

import bpy
from mathutils import Matrix
from bpy.props import BoolProperty, StringProperty, FloatProperty
from bpy.types import Operator
from bpy_extras.io_utils import ImportHelper, axis_conversion


def log(msg, indent=0, ):
    m = "{0}> {1}".format("    " * indent, msg)
    print(m)


class PercentDone():
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
        self.current += numdone
        self.percent = int(self.current / (self.total / 100))
        if(self.percent > self.last):
            sys.stdout.write(self.r)
            sys.stdout.write("{0}{1}{2}%".format(self.t * self.indent, self.prefix, self.percent))
            self.last = self.percent
        if(self.percent >= 100 or self.total == self.current):
            sys.stdout.write(self.r)
            sys.stdout.write("{0}{1}{2}%{3}".format(self.t * self.indent, self.prefix, 100, self.n))


def add_object(name, data, ):
    so = bpy.context.scene.objects
    for i in so:
        i.select = False
    o = bpy.data.objects.new(name, data)
    so.link(o)
    o.select = True
    if(so.active is None or so.active.mode == 'OBJECT'):
        so.active = o
    return o


def activate_object(obj):
    bpy.ops.object.select_all(action='DESELECT')
    sc = bpy.context.scene
    obj.select = True
    sc.objects.active = obj


class ZBrushOBJReader():
    def __init__(self, path, with_uv=True, with_polypaint=True, with_polygroups=True, report_progress=False, ):
        log("{}:".format(self.__class__.__name__), 0, )
        name = os.path.splitext(os.path.split(path)[1])[0]
        
        log("reading..", 1)
        ls = None
        with open(path, mode='r', encoding='utf-8') as f:
            ls = f.readlines()
        
        def v(l, pl=2, ):
            l = l[pl:-1]
            a = l.split(' ')
            return (float(a[0]), float(a[1]), float(a[2]))
        
        def vt(l, pl=3, ):
            l = l[pl:-1]
            a = l.split(' ')
            return (float(a[0]), float(a[1]))
        
        def f(l):
            l = l[2:-1]
            ls = l.split(' ')
            f = [int(i) - 1 for i in ls]
            return f
        
        def ft(l):
            l = l[2:-1]
            ls = l.split(' ')
            ls = [i.split('/') for i in ls]
            f = []
            t = []
            for i, p in enumerate(ls):
                f.append(int(p[0]) - 1)
                t.append(int(p[1]) - 1)
            return f, t
        
        def vc(l):
            r = []
            l = l[6:-1]
            for i in range(0, len(l), 8):
                v = l[i:i + 8]
                c = (int(v[2:4], 16) / 255, int(v[4:6], 16) / 255, int(v[6:8], 16) / 255)
                r.append(c)
            return r
        
        groups = {}
        verts = []
        tverts = []
        faces = []
        tfaces = []
        vcols = []
        
        log("parsing..", 1)
        if(report_progress):
            prgr = PercentDone(len(ls), 1)
        
        has_uv = None
        cg = None
        
        for l in ls:
            if(report_progress):
                prgr.step()
            if(l.startswith('g ')):
                g = l[2:-1]
                if(g not in groups):
                    groups[g] = []
                cg = g
            elif(l.startswith('v ')):
                verts.append(v(l))
            elif(l.startswith('vt ')):
                if(with_uv):
                    tverts.append(vt(l))
            elif(l.startswith('f ')):
                if(has_uv is None):
                    if('/' not in l):
                        has_uv = False
                    else:
                        has_uv = True
                if(has_uv):
                    a, b = ft(l)
                else:
                    a = f(l)
                faces.append(a)
                if(has_uv):
                    if(with_uv):
                        tfaces.append(b)
                if(cg is not None):
                    if(with_polygroups):
                        groups[cg].extend(a)
            elif(l.startswith('#MRGB ')):
                if(with_polypaint):
                    vcols.extend(vc(l))
            else:
                pass
        
        log("making mesh..", 1)
        me = bpy.data.meshes.new(name)
        me.from_pydata(verts, [], faces)
        
        if(len(tverts) > 0):
            log("making uv map..", 1)
            if(report_progress):
                prgr = PercentDone(len(tfaces), 1)
            
            me.uv_textures.new("UVMap")
            loops = me.uv_layers[0].data
            i = 0
            for j in range(len(tfaces)):
                if(report_progress):
                    prgr.step()
                f = tfaces[j]
                for k in range(len(f)):
                    loops[i + k].uv = tverts[f[k]]
                i += (k + 1)
        
        if(len(vcols) > 0):
            log("making vertex colors..", 1)
            if(report_progress):
                prgr = PercentDone(len(me.loops), 1)
            me.vertex_colors.new()
            vc = me.vertex_colors.active
            vcd = vc.data
            for l in me.loops:
                if(report_progress):
                    prgr.step()
                vcd[l.index].color = vcols[l.vertex_index]
        
        log("adding to scene..", 1)
        axis_forward = '-Z'
        axis_up = 'Y'
        cm = axis_conversion(from_forward=axis_forward, from_up=axis_up).to_4x4()
        me.transform(cm)
        
        self.name = name
        self.object = add_object(name, me)
        
        if(len(groups) > 0):
            log("making polygroups..", 1)
            o = self.object
            me = o.data
            for k, v in groups.items():
                o.vertex_groups.new(k)
                vg = o.vertex_groups[k]
                vg.add(list(set(v)), 1.0, 'REPLACE')


class ImportZBrushOBJ(Operator, ImportHelper):
    bl_idname = "import_mesh.zbrush_obj"
    bl_label = 'Import ZBrush Wavefront OBJ'
    bl_options = {'PRESET', 'UNDO', }
    
    filename_ext = ".obj"
    filter_glob = StringProperty(default="*.obj", options={'HIDDEN'}, )
    check_extension = True
    
    with_uv = BoolProperty(name="UV Coords", default=True, )
    with_polypaint = BoolProperty(name="Polypaint", default=True, )
    with_polygroups = BoolProperty(name="Polygroups", default=False, )
    scale = FloatProperty(name="Scale", description="", default=1.0, precision=3, )
    
    def execute(self, context):
        t = time.time()
        p = os.path.realpath(self.filepath)
        r = ZBrushOBJReader(p, with_uv=self.with_uv, with_polypaint=self.with_polypaint, with_polygroups=self.with_polygroups, report_progress=False, )
        o = r.object
        
        if(self.scale != 1.0):
            ms = Matrix.Scale(self.scale, 4)
            o.data.transform(ms)
        
        activate_object(o)
        o.data.update()
        
        d = datetime.timedelta(seconds=time.time() - t)
        log("import completed in {}".format(d), 1)
        
        return {'FINISHED'}


def menu_func_import(self, context):
    self.layout.operator(ImportZBrushOBJ.bl_idname, text="ZBrush Wavefront (.obj)")


def register():
    bpy.utils.register_module(__name__)
    bpy.types.INFO_MT_file_import.append(menu_func_import)


def unregister():
    bpy.utils.unregister_module(__name__)
    bpy.types.INFO_MT_file_import.remove(menu_func_import)


if __name__ == "__main__":
    register()
