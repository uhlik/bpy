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

# Copyright (c) 2023 Jakub Uhlik

bl_info = {"name": "Fast Wavefront^2 (.obj) (Cython)",
           "description": "Import/Export single mesh as Wavefront OBJ. Only active mesh is exported. Only single mesh is expected on import. Supported obj features: UVs, normals, vertex colors using MRGB format (ZBrush).",
           "author": "Jakub Uhlik",
           "version": (0, 3, 6),
           "blender": (2, 80, 0),
           "location": "File > Import/Export > Fast Wavefront (.obj) (Cython)",
           "warning": "work in progress, currently cythonized export only, binaries are not provided, you have to compile them by yourself",
           "wiki_url": "",
           "tracker_url": "",
           "category": "Import-Export", }

import os
import time
import datetime
import io
import shutil

import bpy
import bmesh
from mathutils import Matrix
from bpy_extras.io_utils import ExportHelper, ImportHelper, axis_conversion
from bpy.types import Operator
from bpy.props import StringProperty, BoolProperty, FloatProperty, IntProperty


USE_PY_EXPORT = False
try:
    from . import export_obj
except ImportError:
    USE_PY_EXPORT = True

DEBUG = True


def log(msg="", indent=0, prefix="> "):
    m = "{}{}{}".format("    " * indent, prefix, msg, )
    if(DEBUG):
        print(m)


class FastOBJReader():
    def __init__(self, path, convert_axes=True, with_uv=True, with_shading=True, with_vertex_colors=True, use_vcols_mrgb=True, use_m_as_vertex_group=False, use_vcols_ext=False, use_vcols_ext_with_gamma=False, with_polygroups=True, global_scale=1.0, apply_conversion=False, ):
        log("{}:".format(self.__class__.__name__), 0, )
        name = os.path.splitext(os.path.split(path)[1])[0]
        log("will import .obj at: {}".format(path), 1)
        log_args_align = 30
        
        # t = time.time()
        
        def add_object(name, data, ):
            so = bpy.context.scene.objects
            for i in so:
                i.select_set(False)
            
            o = bpy.data.objects.new(name, data)
            # so.link(o)
            
            context = bpy.context
            view_layer = context.view_layer
            collection = view_layer.active_layer_collection.collection
            collection.objects.link(o)
            
            view_layer.objects.active = o
            
            # o.select = True
            o.select_set(True)
            # if(so.active is None or so.active.mode == 'OBJECT'):
            #     so.active = o
            return o
        
        # def activate_object(obj, ):
        #     bpy.ops.object.select_all(action='DESELECT')
        #     sc = bpy.context.scene
        #     obj.select = True
        #     sc.objects.active = obj
        
        log("reading..", 1)
        ls = None
        with open(path, mode='r', encoding='utf-8') as f:
            ls = f.readlines()
        
        def v(l):
            a = l.split()[1:]
            return tuple(map(float, a))
        
        def vt(l):
            a = l.split()[1:]
            return tuple(map(float, a))
        
        def f(l):
            a = l.split()[1:]
            ls = map(int, a)
            return tuple([i - 1 for i in ls])
        
        def fn(l):
            a = l.split()[1:]
            ls = [i.split('/') for i in a]
            f = []
            for i, p in enumerate(ls):
                f.append(int(p[0]) - 1)
            return f
        
        def ftn(l):
            a = l.split()[1:]
            ls = [i.split('/') for i in a]
            f = []
            t = []
            for i, p in enumerate(ls):
                f.append(int(p[0]) - 1)
                t.append(int(p[1]) - 1)
            return f, t
        
        def vc_mrgb(l):
            r = []
            m = []
            l = l[6:]
            l = l.strip()
            for i in range(0, len(l), 8):
                v = l[i:i + 8]
                c = (int(v[2:4], 16) / 255, int(v[4:6], 16) / 255, int(v[6:8], 16) / 255)
                r.append(c)
                m.append(int(v[0:2], 16) / 255)
            return r, m
        
        def v_vc_ext(l):
            a = l.split()[1:]
            v = tuple(map(float, a))
            p = v[:3]
            c = v[3:]
            if(use_vcols_ext_with_gamma):
                g = 1 / 2.2
                c = tuple([i ** g for i in c])
            return p + c
        
        groups = {}
        verts = []
        tverts = []
        faces = []
        tfaces = []
        vcols = []
        shading = []
        shading_flag = None
        mask = []
        
        log("parsing..", 1)
        parsef = None
        has_uv = None
        cg = None
        
        for l in ls:
            if(l.startswith('s ')):
                if(with_shading):
                    if(l.lower() == 's off' or l.lower() == 's 0'):
                        shading_flag = False
                    else:
                        shading_flag = True
            elif(l.startswith('g ')):
                if(with_polygroups):
                    g = l[2:]
                    g = g.strip()
                    if(g not in groups):
                        groups[g] = []
                    cg = g
            elif(l.startswith('v ')):
                if(with_vertex_colors and use_vcols_ext):
                    a = v_vc_ext(l)
                    verts.append(a[:3])
                    vcols.append(a[3:])
                else:
                    verts.append(v(l))
            elif(l.startswith('vt ')):
                if(with_uv):
                    tverts.append(vt(l))
            elif(l.startswith('f ')):
                if(parsef is None):
                    if('//' in l):
                        parsef = fn
                    elif('/' not in l):
                        parsef = f
                    else:
                        parsef = ftn
                        has_uv = True
                if(has_uv):
                    a, b = parsef(l)
                else:
                    a = parsef(l)
                faces.append(a)
                if(with_shading):
                    shading.append(shading_flag)
                if(has_uv):
                    if(with_uv):
                        tfaces.append(b)
                if(with_polygroups):
                    if(cg is not None):
                        groups[cg].extend(a)
            elif(l.startswith('#MRGB ')):
                if(with_vertex_colors):
                    if(use_vcols_mrgb):
                        c, m = vc_mrgb(l)
                        vcols.extend(c)
                        if(use_m_as_vertex_group):
                            mask.extend(m)
            else:
                pass
        
        log("making mesh..", 1)
        me = bpy.data.meshes.new(name)
        me.from_pydata(verts, [], faces)
        
        log("{} {}".format("{}: ".format("with_uv").ljust(log_args_align, "."), with_uv), 1)
        if(len(tverts) > 0):
            log("making uv map..", 1)
            me.uv_layers.new(name="UVMap")
            loops = me.uv_layers[0].data
            i = 0
            for j in range(len(tfaces)):
                f = tfaces[j]
                for k in range(len(f)):
                    loops[i + k].uv = tverts[f[k]]
                i += (k + 1)
        
        log("{} {}".format("{}: ".format("with_vertex_colors").ljust(log_args_align, "."), with_vertex_colors), 1)
        log("{} {}".format("{}: ".format("use_vcols_mrgb").ljust(log_args_align, "."), use_vcols_mrgb), 1)
        log("{} {}".format("{}: ".format("use_m_as_vertex_group").ljust(log_args_align, "."), use_m_as_vertex_group), 1)
        log("{} {}".format("{}: ".format("use_vcols_ext").ljust(log_args_align, "."), use_vcols_ext), 1)
        if(len(vcols) > 0):
            log("making vertex colors..", 1)
            me.vertex_colors.new()
            vc = me.vertex_colors.active
            vcd = vc.data
            for l in me.loops:
                vcd[l.index].color = vcols[l.vertex_index] + (1.0, )
        
        log("{} {}".format("{}: ".format("convert_axes").ljust(log_args_align, "."), convert_axes), 1)
        log("{} {}".format("{}: ".format("apply_conversion").ljust(log_args_align, "."), apply_conversion), 1)
        if(convert_axes):
            if(apply_conversion):
                axis_forward = '-Z'
                axis_up = 'Y'
                cm = axis_conversion(from_forward=axis_forward, from_up=axis_up).to_4x4()
                me.transform(cm)
        log("{} {}".format("{}: ".format("global_scale").ljust(log_args_align, "."), global_scale), 1)
        if(global_scale != 1.0):
            sm = Matrix.Scale(global_scale, 4)
            me.transform(sm)
        me.update()
        
        log("adding to scene..", 1)
        self.name = name
        self.object = add_object(name, me)
        if(len(mask) > 0):
            log("making mask vertex group..", 1)
            g = self.object.vertex_groups.new(name="M")
            indexes = [i for i in range(len(me.vertices))]
            g.add(indexes, 0.0, 'REPLACE')
            ind = g.index
            for i, v in enumerate(me.vertices):
                v.groups[ind].weight = mask[i]
        
        if(convert_axes):
            if(not apply_conversion):
                axis_forward = '-Z'
                axis_up = 'Y'
                cm = axis_conversion(from_forward=axis_forward, from_up=axis_up).to_4x4()
                m = self.object.matrix_world
                mm = m @ cm
                self.object.matrix_world = mm
        
        log("{} {}".format("{}: ".format("with_shading").ljust(log_args_align, "."), with_shading), 1)
        if(with_shading):
            log("setting shading..", 1)
            for i, p in enumerate(me.polygons):
                p.use_smooth = shading[i]
        
        log("{} {}".format("{}: ".format("with_polygroups").ljust(log_args_align, "."), with_polygroups), 1)
        if(len(groups) > 0):
            log("making polygroups..", 1)
            o = self.object
            me = o.data
            for k, v in groups.items():
                o.vertex_groups.new(name=k)
                vg = o.vertex_groups[k]
                vg.add(list(set(v)), 1.0, 'REPLACE')
        
        log("imported object: '{}'".format(self.object.name), 1)
        
        # d = datetime.timedelta(seconds=time.time() - t)
        # log("completed in {}.".format(d), 1)


class FastOBJWriter():
    def __init__(self, context, o, path, apply_modifiers=False, apply_transformation=True, convert_axes=True, triangulate=False, use_uv=True, use_shading=False, use_vertex_colors=False, use_vcols_mrgb=True, use_vcols_ext=False, global_scale=1.0, precision=6, ):
        log("{}: {}".format(self.__class__.__name__, o.name))
        log("will write .obj at: {}".format(path), 1)
        
        log("prepare..", 1)
        log_args_align = 25
        
        log("{} {}".format("{}: ".format("apply_modifiers").ljust(log_args_align, "."), apply_modifiers), 1)
        owner = None
        if(apply_modifiers and o.modifiers):
            depsgraph = context.evaluated_depsgraph_get()
            owner = o.evaluated_get(depsgraph)
            me = owner.to_mesh()
        else:
            owner = o
            me = owner.to_mesh()
        
        bm = bmesh.new()
        bm.from_mesh(me)
        
        log("{} {}".format("{}: ".format("triangulate").ljust(log_args_align, "."), triangulate), 1)
        if(triangulate):
            bmesh.ops.triangulate(bm, faces=bm.faces)
        log("{} {}".format("{}: ".format("apply_transformation").ljust(log_args_align, "."), apply_transformation), 1)
        if(apply_transformation):
            m = o.matrix_world.copy()
            bm.transform(m)
        log("{} {}".format("{}: ".format("convert_axes").ljust(log_args_align, "."), convert_axes), 1)
        if(convert_axes):
            axis_forward = '-Z'
            axis_up = 'Y'
            cm = axis_conversion(to_forward=axis_forward, to_up=axis_up).to_4x4()
            bm.transform(cm)
        
        log("{} {}".format("{}: ".format("use_uv").ljust(log_args_align, "."), use_uv), 1)
        log("{} {}".format("{}: ".format("use_shading").ljust(log_args_align, "."), use_shading), 1)
        log("{} {}".format("{}: ".format("use_vertex_colors").ljust(log_args_align, "."), use_vertex_colors), 1)
        log("{} {}".format("{}: ".format("use_vcols_mrgb").ljust(log_args_align, "."), use_vcols_mrgb), 1)
        log("{} {}".format("{}: ".format("use_vcols_ext").ljust(log_args_align, "."), use_vcols_ext), 1)
        
        log("{} {}".format("{}: ".format("global_scale").ljust(log_args_align, "."), global_scale), 1)
        if(global_scale != 1.0):
            sm = Matrix.Scale(global_scale, 4)
            bm.transform(sm)
        
        log("{} {}".format("{}: ".format("precision").ljust(log_args_align, "."), precision), 1)
        
        # update normals after transforms
        bm.normal_update()
        
        sio = io.StringIO(initial_value='', newline='', )
        siov = io.StringIO(initial_value='', newline='', )
        siovn = io.StringIO(initial_value='', newline='', )
        siof = io.StringIO(initial_value='', newline='', )
        siovt = io.StringIO(initial_value='', newline='', )
        
        # sort-of-header
        sio.write("# Fast Wavefront^2 (.obj)\n")
        sio.write('#\n')
        sio.write('o %s_%s\n' % (o.name, o.data.name))
        
        # vertices
        vs = bm.verts
        bm.verts.ensure_lookup_table()
        
        # vertex colors
        col_layer = None
        if(use_vertex_colors):
            # try to find active vertex color layer
            col_layer = bm.loops.layers.color.active
            if(col_layer is None):
                # no vertex colors, turn them off
                use_vertex_colors = False
                use_vcols_mrgb = False
                use_vcols_ext = False
                log("no vertex colors found..", 1)
            if(col_layer is not None):
                if(use_vcols_mrgb and use_vcols_ext):
                    # if both True, use mrgb
                    log("using MRGB vertex colors..", 1)
                    use_vcols_ext = False
                if(not use_vcols_mrgb and not use_vcols_ext):
                    # if both False, use mrgb
                    log("using MRGB vertex colors..", 1)
                    use_vcols_mrgb = True
        
        fwv = siov.write
        vfmt = 'v %.{0}f %.{0}f %.{0}f\n'.format(precision)
        vfmtvcolext = 'v %.{0}f %.{0}f %.{0}f %.{0}f %.{0}f %.{0}f\n'.format(precision)
        if(use_vertex_colors and use_vcols_ext):
            # write vertices and vertex colors in extended format
            log("writing vertices and extended obj format vertex colors..", 1)
            cols = [[] for i in range(len(vs))]
            rgbf = []
            bm.faces.ensure_lookup_table()
            # get all colors for single vertex, ie, colors of all loops belonging to vertex
            if(col_layer is not None):
                fs = bm.faces
                for f in fs:
                    ls = f.loops
                    for l in ls:
                        vi = l.vert.index
                        # get rgb only
                        c = l[col_layer][:3]
                        cols[vi].append(c)
            # average color values
            for cl in cols:
                r = 0
                g = 0
                b = 0
                l = len(cl)
                for c in cl:
                    r += c[0]
                    g += c[1]
                    b += c[2]
                
                def limit(v):
                    if(v < 0.0):
                        v = 0.0
                    if(v > 1.0):
                        v = 1.0
                    return v
                
                rgbf.append((limit(r / l), limit(g / l), limit(b / l)))
            
            for v in vs:
                fwv(vfmtvcolext % tuple(v.co[:] + rgbf[v.index]))
        else:
            # no ext vcols, write regular vertices
            log("writing vertices..", 1)
            for v in vs:
                fwv(vfmt % v.co[:])
        
        if(use_vertex_colors and use_vcols_mrgb):
            # vertex colors in mrgb format
            log("writing mrgb vertex colors..", 1)
            # '#MRGB ' block polypaint and mask as 4hex values per vertex.
            # format is MMRRGGBB with up to 64 entries per line
            cols = [[] for i in range(len(vs))]
            hexc = []
            bm.faces.ensure_lookup_table()
            # get all colors from loops for each vertex
            if(col_layer is not None):
                fs = bm.faces
                for f in fs:
                    ls = f.loops
                    for l in ls:
                        vi = l.vert.index
                        # get rgb only
                        c = l[col_layer][:3]
                        cols[vi].append(c)
            
            # average colors and convert to 0-255 format and then to hexadecimal values, leave 'm' to ff
            for cl in cols:
                r = 0
                g = 0
                b = 0
                l = len(cl)
                for c in cl:
                    r += c[0]
                    g += c[1]
                    b += c[2]
                
                def limit(v):
                    if(v < 0):
                        v = 0
                    if(v > 255):
                        v = 255
                    return v
                
                rgb8 = (limit(int((r / l) * 255.0)), limit(int((g / l) * 255.0)), limit(int((b / l) * 255.0)))
                h = 'ff%02x%02x%02x' % rgb8
                hexc.append(h)
            
            for i in range(0, len(hexc), 64):
                # write in chunks of 64 entries per line as per specification directly after vertices
                ch = hexc[i:i + 64]
                s = "".join(ch)
                fwv('#MRGB %s\n' % s)
        
        # faces
        log("writing normals, faces and texture coordinates..", 1)
        fs = bm.faces
        fs.ensure_lookup_table()
        # texture coordinates stuff
        vtlocs = dict()
        vtmaps = dict()
        vtli = 0
        uvl = None
        if(use_uv):
            uvl = bm.loops.layers.uv.active
        if(uvl is None):
            use_uv = False
        # smoothing stuff
        fsmooth = None
        normap = dict()
        norlen = 0
        nori = 0
        # shortcuts
        fwvn = siovn.write
        vnfmt = 'vn %.{0}f %.{0}f %.{0}f\n'.format(precision)
        fwf = siof.write
        fwvt = siovt.write
        vtfmt = 'vt %.{0}f %.{0}f\n'.format(precision)
        
        if(use_uv):
            # if using texture coordinates, precalculate them first, each coordinate should be reused, no duplicates
            for f in fs:
                fl = f.loops
                for l in fl:
                    uv = l[uvl].uv[:]
                    vi = l.vert.index
                    li = l.index
                    fi = f.index
                    
                    if(uv not in vtlocs):
                        vtlocs[uv] = vtli
                        vtli += 1
                        fwvt(vtfmt % uv)
                    
                    e = [fi, vi, li, vtlocs[uv]]
                    if(fi not in vtmaps):
                        vtmaps[fi] = [e, ]
                    else:
                        vtmaps[fi].append(e)
        
        for f in fs:
            if(use_shading):
                # write shading flag, only change when flag is changed for following face
                if(fsmooth != f.smooth):
                    fsmooth = f.smooth
                    if(fsmooth):
                        fwf("s 1\n")
                    else:
                        fwf("s off\n")
            
            # start writing face
            fwf("f")
            fi = f.index
            vs = f.verts
            n = len(vs)
            
            for i in range(n):
                if(use_shading):
                    if(fsmooth):
                        # write smooth face with vertex normals
                        nor = vs[i].normal[:]
                    else:
                        # write flat face with face normal
                        nor = f.normal[:]
                    if(nor in normap):
                        # use already defined normal
                        nori = normap[nor]
                    else:
                        # create new normal
                        normap[nor] = norlen + 1
                        norlen += 1
                        nori = norlen
                        fwvn(vnfmt % nor)
                
                vi = vs[i].index
                if(use_uv):
                    # get correct texture coordinates
                    m = vtmaps[fi]
                    uvi = None
                    for j in m:
                        if(j[1] == vi):
                            uvi = j[3]
                            break
                    # write face vertex with uv and normal
                    fwf(" %d/%d/%d" % (vi + 1, uvi + 1, nori, ))
                else:
                    # write face vertex and normal
                    fwf(" %d//%d" % (vi + 1, nori))
            # finish face
            fwf("\n")
        
        # put it all to one string
        log("writing to disk..", 1)
        tp = "{}.tmp".format(path)
        sio.write(siov.getvalue())
        sio.write(siovt.getvalue())
        sio.write(siovn.getvalue())
        sio.write(siof.getvalue())
        # write to temporary file
        with open(tp, mode='w', encoding='utf-8', newline="\n", ) as of:
            of.write(sio.getvalue())
        sio.close()
        siov.close()
        siovn.close()
        siof.close()
        siovt.close()
        # remove existing file
        if(os.path.exists(path)):
            os.remove(path)
        # rename to final file name
        shutil.move(tp, path)
        
        log("cleanup..", 1)
        bm.free()
        owner.to_mesh_clear()


class ExportFastOBJ(Operator, ExportHelper):
    bl_idname = "export_mesh.fast_obj"
    bl_label = 'Export Fast OBJ (Cython)'
    bl_description = "Export single mesh as Wavefront OBJ. Only active mesh is exported. Supported obj features: UVs, normals, vertex colors using MRGB format (ZBrush)."
    bl_options = {'PRESET'}
    
    # filepath: StringProperty(name="File Path", description="Filepath used for exporting the file", maxlen=1024, subtype='FILE_PATH', )
    filename_ext = ".obj"
    filter_glob: StringProperty(default="*.obj", options={'HIDDEN'}, )
    check_extension = True
    
    apply_modifiers: BoolProperty(name="Apply Modifiers", default=False, description="Apply all modifiers.", )
    apply_transformation: BoolProperty(name="Apply Transformation", default=False, description="Zero-out mesh transformation.", )
    convert_axes: BoolProperty(name="Convert Axes", default=True, description="Convert from blender (y forward, z up) to forward -z, up y.", )
    triangulate: BoolProperty(name="Triangulate", default=False, description="Triangulate mesh before exporting.", )
    use_normals: BoolProperty(name="With Normals", default=True, description="Export vertex normals.", )
    use_uv: BoolProperty(name="With UV", default=True, description="Export active UV layout.", )
    use_vcols: BoolProperty(name="With Vertex Colors (#MRGB)", default=True, description="Export vertex colors, this is not part of official file format specification.", )
    global_scale: FloatProperty(name="Scale", default=1.0, precision=3, description="Uniform scale.", )
    precision: IntProperty(name="Precision", default=6, description="Floating point precision.", )
    
    @classmethod
    def poll(cls, context):
        o = context.active_object
        return (o and o.type == 'MESH')
    
    def draw(self, context):
        l = self.layout
        sub = l.column()
        
        if(USE_PY_EXPORT):
            sub.label(text='Cython module not found.', icon='ERROR', )
            sub.label(text='Using python implementation.', icon='ERROR', )
            sub.separator()
        
        sub.prop(self, 'apply_modifiers')
        sub.prop(self, 'apply_transformation')
        sub.prop(self, 'convert_axes')
        sub.prop(self, 'triangulate')
        sub.prop(self, 'use_normals')
        sub.prop(self, 'use_uv')
        sub.prop(self, 'use_vcols')
        sub.prop(self, 'global_scale')
        sub.prop(self, 'precision')
    
    def execute(self, context):
        t = time.time()
        
        o = context.active_object
        
        if(USE_PY_EXPORT):
            log("WARNING: cython module not found, using python implementation")
            d = {'context': context,
                 'o': o,
                 'path': self.filepath,
                 'apply_modifiers': self.apply_modifiers,
                 'apply_transformation': self.apply_transformation,
                 'convert_axes': self.convert_axes,
                 'triangulate': self.triangulate,
                 'use_uv': self.use_uv,
                 'use_shading': False,
                 'use_vertex_colors': self.use_vcols,
                 'use_vcols_mrgb': True,
                 'use_vcols_ext': False,
                 'global_scale': self.global_scale,
                 'precision': self.precision, }
            w = FastOBJWriter(**d)
            
            log("completed in {}.".format(datetime.timedelta(seconds=time.time() - t)))
            return {'FINISHED'}
        
        log("{}:".format(self.__class__.__name__), 0, )
        log("using Cython module: {}".format(not USE_PY_EXPORT), 1)
        log("will write .obj at: {}".format(self.filepath), 1)
        
        owner = None
        if(self.apply_modifiers and o.modifiers):
            depsgraph = context.evaluated_depsgraph_get()
            owner = o.evaluated_get(depsgraph)
            m = owner.to_mesh()
        else:
            owner = o
            m = owner.to_mesh()
        
        if(self.apply_transformation):
            mw = o.matrix_world.copy()
            m.transform(mw)
        if(self.convert_axes):
            axis_forward = '-Z'
            axis_up = 'Y'
            from bpy_extras.io_utils import axis_conversion
            cm = axis_conversion(to_forward=axis_forward, to_up=axis_up).to_4x4()
            m.transform(cm)
        if(self.triangulate):
            bm = bmesh.new()
            bm.from_mesh(m)
            bmesh.ops.triangulate(bm, faces=bm.faces)
            bm.to_mesh(m)
        if(self.global_scale != 1.0):
            sm = Matrix.Scale(self.global_scale, 4)
            m.transform(sm)
        
        has_uv = self.use_uv
        if(has_uv):
            if(not len(m.uv_layers)):
                has_uv = False
        has_vcols = self.use_vcols
        if(has_vcols):
            if(not len(m.vertex_colors)):
                has_vcols = False
        
        export_obj.export_obj(m.as_pointer(),
                              self.filepath,
                              "{}-{}".format(o.name, o.data.name),
                              use_normals=self.use_normals,
                              use_uv=has_uv,
                              use_vcols=has_vcols,
                              precision=self.precision,
                              debug=DEBUG, )
        
        owner.to_mesh_clear()
        
        log("completed in {}.".format(datetime.timedelta(seconds=time.time() - t)))
        return {'FINISHED'}


class ImportFastOBJ(Operator, ImportHelper):
    bl_idname = "import_mesh.fast_obj"
    bl_label = 'Import Fast OBJ'
    bl_description = "Import single mesh Wavefront OBJ. Only single mesh is expected on import. Supported obj features: UVs, normals, vertex colors using MRGB format (ZBrush)."
    bl_options = {'PRESET'}
    
    # filepath: StringProperty(name="File Path", description="Filepath used for exporting the file", maxlen=1024, subtype='FILE_PATH', )
    filename_ext = ".obj"
    filter_glob: StringProperty(default="*.obj", options={'HIDDEN'}, )
    check_extension = True
    
    convert_axes: BoolProperty(name="Convert Axes", default=True, description="Convert from blender (y forward, z up) to forward -z, up y.", )
    with_uv: BoolProperty(name="With UV", default=True, description="Import texture coordinates.", )
    
    def vcol_update_mrgb(self, context):
        if(self.with_vertex_colors_mrgb):
            if(self.with_vertex_colors_extended):
                self.with_vertex_colors_extended = False
    
    def vcol_update_ext(self, context):
        if(self.with_vertex_colors_extended):
            if(self.with_vertex_colors_mrgb):
                self.with_vertex_colors_mrgb = False
    
    with_vertex_colors_mrgb: BoolProperty(name="With Vertex Colors (#MRGB)", default=True, description="Import vertex colors, this is not part of official file format specification. ZBrush uses MRGB comments to write Polypaint to OBJ.", update=vcol_update_mrgb, )
    with_vertex_colors_extended: BoolProperty(name="With Vertex Colors (x,y,z,r,g,b)", default=False, description="Import vertex colors in 'extended' format where vertex is defined as (x, y, z, r, g, b), this is not part of official file format specification.", update=vcol_update_ext, )
    vcols_ext_with_gamma: BoolProperty(name="With Gamma Correction", default=False, description="Apply gamma correction to extended vertex colors.", )
    use_m_as_vertex_group: BoolProperty(name="M as Vertex Group", default=False, description="Import M from MRGB as vertex group.", )
    with_polygroups: BoolProperty(name="With Polygroups", default=False, description="Import ZBrush polygroups as vertex groups.", )
    global_scale: FloatProperty(name="Scale", default=1.0, precision=3, description="Uniform scale.", )
    apply_conversion: BoolProperty(name="Apply Axis Conversion", default=True, description="Apply new axes directly to mesh.", )
    
    @classmethod
    def poll(cls, context):
        return True
    
    def draw(self, context):
        # l = self.layout
        # sub = l.column()
        #
        # r = sub.row()
        # r.prop(self, 'convert_axes')
        # c = r.column()
        # c.prop(self, 'apply_conversion', toggle=True, text='', icon='EMPTY_AXIS')
        # c.enabled = self.convert_axes
        #
        # sub.prop(self, 'with_uv')
        #
        # r = sub.row()
        # r.prop(self, 'with_vertex_colors_mrgb')
        # c = r.column()
        # c.prop(self, 'use_m_as_vertex_group', toggle=True, text='', icon='GROUP_VERTEX')
        # c.enabled = self.with_vertex_colors_mrgb
        #
        # r = sub.row()
        # r.prop(self, 'with_vertex_colors_extended')
        # c = r.column()
        # c.prop(self, 'vcols_ext_with_gamma', toggle=True, text='', icon='FCURVE')
        # c.enabled = self.with_vertex_colors_extended
        #
        # sub.prop(self, 'with_polygroups')
        # sub.prop(self, 'global_scale')
        
        l = self.layout
        c = l.column()
        c.prop(self, 'convert_axes')
        e = c.column()
        e.prop(self, 'apply_conversion')
        e.enabled = self.convert_axes
        c.prop(self, 'with_uv')
        c.prop(self, 'with_vertex_colors_mrgb')
        e = c.column()
        e.prop(self, 'use_m_as_vertex_group')
        e.enabled = self.with_vertex_colors_mrgb
        c.prop(self, 'with_vertex_colors_extended')
        e = c.column()
        e.prop(self, 'vcols_ext_with_gamma')
        e.enabled = self.with_vertex_colors_extended
        c.prop(self, 'with_polygroups')
        c.prop(self, 'global_scale')
    
    def execute(self, context):
        t = time.time()
        
        vcols = False
        if(self.with_vertex_colors_mrgb or self.with_vertex_colors_extended):
            vcols = True
        
        d = {'path': self.filepath,
             'convert_axes': self.convert_axes,
             'with_uv': self.with_uv,
             'with_shading': False,
             'with_vertex_colors': vcols,
             'use_vcols_mrgb': self.with_vertex_colors_mrgb,
             'use_m_as_vertex_group': self.use_m_as_vertex_group,
             'use_vcols_ext': self.with_vertex_colors_extended,
             'use_vcols_ext_with_gamma': self.vcols_ext_with_gamma,
             'with_polygroups': self.with_polygroups,
             'global_scale': self.global_scale,
             'apply_conversion': self.apply_conversion, }
        r = FastOBJReader(**d)
        
        d = datetime.timedelta(seconds=time.time() - t)
        log("completed in {}.".format(d), 0)
        
        return {'FINISHED'}


def menu_func_export(self, context):
    self.layout.operator(ExportFastOBJ.bl_idname, text="Fast Wavefront^2 (.obj) (Cython)")


def menu_func_import(self, context):
    self.layout.operator(ImportFastOBJ.bl_idname, text="Fast Wavefront^2 (.obj)")


classes = (
    ExportFastOBJ,
    ImportFastOBJ,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)


def unregister():
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
