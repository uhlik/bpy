bl_info = {"name": "Fast Wavefront (.obj)",
           "description": "Import/Export single mesh as Wavefront OBJ. Only active mesh is exported. Only single mesh is expected on import. Supported obj features: UVs, normals, shading, vertex colors using MRGB format (ZBrush) or so called 'extended' format when each vertex is defined by 6 values (x, y, z, r, g, b).",
           "author": "Jakub Uhlik",
           "version": (0, 1, 1),
           "blender": (2, 78, 0),
           "location": "File > Import/Export > Fast Wavefront (.obj)",
           "warning": "",
           "wiki_url": "",
           "tracker_url": "",
           "category": "Import-Export", }


import os
import io
import sys
import time
import datetime
import shutil

import bpy
import bmesh
from mathutils import Matrix
from bpy_extras.io_utils import ExportHelper, ImportHelper, axis_conversion
from bpy.types import Operator
from bpy.props import StringProperty, BoolProperty, FloatProperty, IntProperty


def log(msg="", indent=0, prefix="> "):
    m = "{}{}{}".format("    " * indent, prefix, msg, )
    print(m)


class FastOBJWriter():
    def __init__(self, o, path, apply_modifiers=False, apply_transformation=True, convert_axes=True, triangulate=False, use_uv=True, use_shading=False, use_vertex_colors=False, use_vcols_mrgb=True, use_vcols_ext=False, global_scale=1.0, precision=6, ):
        log("{}: {}".format(self.__class__.__name__, o.name))
        log("will write .obj at: {}".format(path), 1)
        t = time.time()
        
        log("prepare..", 1)
        me = None
        if(apply_modifiers):
            me = o.to_mesh(bpy.context.scene, apply_modifiers, 'RENDER', calc_tessface=False, calc_undeformed=False, )
        
        bm = bmesh.new()
        if(me is not None):
            bm.from_mesh(me)
        else:
            bm.from_mesh(o.data)
        
        log_args_align = 25
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
        sio.write('# %s %s %s\n' % ('teoplib', self.__class__.__name__, 'Wavefront .OBJ file'))
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
                        c = l[col_layer]
                        cols[vi].append(c)
            # average color values
            for cl in cols:
                r = 0
                g = 0
                b = 0
                l = len(cl)
                for c in cl:
                    r += c.r
                    g += c.g
                    b += c.b
                
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
                        c = l[col_layer]
                        cols[vi].append(c)
            
            # average colors and convert to 0-255 format and then to hexadecimal values, leave 'm' to ff
            for cl in cols:
                r = 0
                g = 0
                b = 0
                l = len(cl)
                for c in cl:
                    r += c.r
                    g += c.g
                    b += c.b
                
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
        if(me is not None):
            bpy.data.meshes.remove(me)
        bm.free()
        
        d = datetime.timedelta(seconds=time.time() - t)
        log("completed in {}.".format(d), 1)


class FastOBJReader():
    def __init__(self, path, convert_axes=True, with_uv=True, with_shading=True, with_vertex_colors=True, use_vcols_mrgb=True, use_vcols_ext=False, with_polygroups=True, global_scale=1.0, apply_conversion=False, ):
        log("{}:".format(self.__class__.__name__), 0, )
        name = os.path.splitext(os.path.split(path)[1])[0]
        log("will import .obj at: {}".format(path), 1)
        log_args_align = 25
        
        t = time.time()
        
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
        
        def activate_object(obj, ):
            bpy.ops.object.select_all(action='DESELECT')
            sc = bpy.context.scene
            obj.select = True
            sc.objects.active = obj
        
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
        
        def fn(l):
            l = l[2:-1]
            ls = l.split(' ')
            ls = [i.split('/') for i in ls]
            f = []
            for i, p in enumerate(ls):
                f.append(int(p[0]) - 1)
            return f
        
        def ftn(l):
            l = l[2:-1]
            ls = l.split(' ')
            ls = [i.split('/') for i in ls]
            f = []
            t = []
            for i, p in enumerate(ls):
                f.append(int(p[0]) - 1)
                t.append(int(p[1]) - 1)
            return f, t
        
        def vc_mrgb(l):
            r = []
            l = l[6:-1]
            for i in range(0, len(l), 8):
                v = l[i:i + 8]
                c = (int(v[2:4], 16) / 255, int(v[4:6], 16) / 255, int(v[6:8], 16) / 255)
                r.append(c)
            return r
        
        def v_vc_ext(l, pl=2, ):
            l = l[pl:-1]
            a = l.split(' ')
            return (float(a[0]), float(a[1]), float(a[2]), float(a[3]), float(a[4]), float(a[5]))
        
        groups = {}
        verts = []
        tverts = []
        faces = []
        tfaces = []
        vcols = []
        shading = []
        shading_flag = None
        
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
                    g = l[2:-1]
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
                        vcols.extend(vc_mrgb(l))
            else:
                pass
        
        log("making mesh..", 1)
        me = bpy.data.meshes.new(name)
        me.from_pydata(verts, [], faces)
        
        log("{} {}".format("{}: ".format("with_uv").ljust(log_args_align, "."), with_uv), 1)
        if(len(tverts) > 0):
            log("making uv map..", 1)
            me.uv_textures.new("UVMap")
            loops = me.uv_layers[0].data
            i = 0
            for j in range(len(tfaces)):
                f = tfaces[j]
                for k in range(len(f)):
                    loops[i + k].uv = tverts[f[k]]
                i += (k + 1)
        
        log("{} {}".format("{}: ".format("with_vertex_colors").ljust(log_args_align, "."), with_vertex_colors), 1)
        log("{} {}".format("{}: ".format("use_vcols_mrgb").ljust(log_args_align, "."), use_vcols_mrgb), 1)
        log("{} {}".format("{}: ".format("use_vcols_ext").ljust(log_args_align, "."), use_vcols_ext), 1)
        if(len(vcols) > 0):
            log("making vertex colors..", 1)
            me.vertex_colors.new()
            vc = me.vertex_colors.active
            vcd = vc.data
            for l in me.loops:
                vcd[l.index].color = vcols[l.vertex_index]
        
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
        if(convert_axes):
            if(not apply_conversion):
                axis_forward = '-Z'
                axis_up = 'Y'
                cm = axis_conversion(from_forward=axis_forward, from_up=axis_up).to_4x4()
                self.object.matrix_world *= cm
        
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
                o.vertex_groups.new(k)
                vg = o.vertex_groups[k]
                vg.add(list(set(v)), 1.0, 'REPLACE')
        
        log("imported object: '{}'".format(self.object.name), 1)
        
        d = datetime.timedelta(seconds=time.time() - t)
        log("completed in {}.".format(d), 1)


class ExportFastOBJ(Operator, ExportHelper):
    bl_idname = "export_mesh.fast_obj"
    bl_label = 'Export Fast OBJ'
    bl_options = {'PRESET'}
    
    # filepath = StringProperty(name="File Path", description="Filepath used for exporting the file", maxlen=1024, subtype='FILE_PATH', )
    filename_ext = ".obj"
    filter_glob = StringProperty(default="*.obj", options={'HIDDEN'}, )
    check_extension = True
    
    apply_modifiers = BoolProperty(name="Apply Modifiers", default=False, description="Apply all modifiers.", )
    apply_transformation = BoolProperty(name="Apply Transformation", default=True, description="Zero-out mesh transformation.", )
    convert_axes = BoolProperty(name="Convert Axes", default=True, description="Convert from blender (y forward, z up) to forward -z, up y.", )
    triangulate = BoolProperty(name="Triangulate", default=False, description="", )
    use_uv = BoolProperty(name="With UV", default=True, description="Export active UV layout.", )
    use_shading = BoolProperty(name="With Shading", default=False, description="Export face shading.", )
    use_vertex_colors = BoolProperty(name="With Vertex Colors", default=False, description="Export vertex colors, this is not part of official file format specification.", )
    use_vcols_mrgb = BoolProperty(name="VCols MRGB", default=True, description="Use ZBrush vertex colors style.", )
    use_vcols_ext = BoolProperty(name="VCols Ext", default=False, description="Use 'Extended Vertex' vertex colors style.", )
    global_scale = FloatProperty(name="Scale", default=1.0, precision=3, description="", )
    precision = IntProperty(name="Precision", default=6, description="", )
    
    @classmethod
    def poll(cls, context):
        o = context.active_object
        return (o and o.type == 'MESH')
    
    def draw(self, context):
        l = self.layout
        sub = l.column()
        sub.prop(self, 'apply_modifiers')
        sub.prop(self, 'apply_transformation')
        sub.prop(self, 'convert_axes')
        sub.prop(self, 'triangulate')
        sub.prop(self, 'use_uv')
        sub.prop(self, 'use_shading')
        sub.prop(self, 'use_vertex_colors')
        
        c = sub.column()
        c.prop(self, 'use_vcols_mrgb')
        c.prop(self, 'use_vcols_ext')
        c.enabled = self.use_vertex_colors
        
        sub.prop(self, 'global_scale')
        sub.prop(self, 'precision')
    
    def execute(self, context):
        o = context.active_object
        d = {'o': o,
             'path': self.filepath,
             'apply_modifiers': self.apply_modifiers,
             'apply_transformation': self.apply_transformation,
             'convert_axes': self.convert_axes,
             'triangulate': self.triangulate,
             'use_uv': self.use_uv,
             'use_shading': self.use_shading,
             'use_vertex_colors': self.use_vertex_colors,
             'use_vcols_mrgb': self.use_vcols_mrgb,
             'use_vcols_ext': self.use_vcols_ext,
             'global_scale': self.global_scale,
             'precision': self.precision, }
        w = FastOBJWriter(**d)
        return {'FINISHED'}


class ImportFastOBJ(Operator, ImportHelper):
    bl_idname = "import_mesh.fast_obj"
    bl_label = 'Import Fast OBJ'
    bl_options = {'PRESET'}
    
    # filepath = StringProperty(name="File Path", description="Filepath used for exporting the file", maxlen=1024, subtype='FILE_PATH', )
    filename_ext = ".obj"
    filter_glob = StringProperty(default="*.obj", options={'HIDDEN'}, )
    check_extension = True
    
    convert_axes = BoolProperty(name="Convert Axes", default=True, description="Convert from blender (y forward, z up) to forward -z, up y.", )
    with_uv = BoolProperty(name="With UV", default=True, description="Import texture coordinates.", )
    with_shading = BoolProperty(name="With Shading", default=False, description="Import face shading.", )
    with_vertex_colors = BoolProperty(name="With Vertex Colors", default=False, description="Import vertex colors, this is not part of official file format specification.", )
    use_vcols_mrgb = BoolProperty(name="VCols MRGB", default=True, description="Use ZBrush vertex colors style.", )
    use_vcols_ext = BoolProperty(name="VCols Ext", default=False, description="Use 'Extended Vertex' vertex colors style.", )
    with_polygroups = BoolProperty(name="With Polygroups", default=False, description="", )
    global_scale = FloatProperty(name="Scale", default=1.0, precision=3, description="", )
    apply_conversion = BoolProperty(name="Apply Conversion", default=False, description="Apply new axes directly to mesh or only transform at object level.", )
    
    def draw(self, context):
        l = self.layout
        sub = l.column()
        sub.prop(self, 'convert_axes')
        sub.prop(self, 'with_uv')
        sub.prop(self, 'with_shading')
        sub.prop(self, 'with_vertex_colors')
        
        c = sub.column()
        c.prop(self, 'use_vcols_mrgb')
        c.prop(self, 'use_vcols_ext')
        c.enabled = self.with_vertex_colors
        
        sub.prop(self, 'with_polygroups')
        sub.prop(self, 'global_scale')
        sub.prop(self, 'apply_conversion')
    
    def execute(self, context):
        d = {'path': self.filepath,
             'convert_axes': self.convert_axes,
             'with_uv': self.with_uv,
             'with_shading': self.with_shading,
             'with_vertex_colors': self.with_vertex_colors,
             'use_vcols_mrgb': self.use_vcols_mrgb,
             'use_vcols_ext': self.use_vcols_ext,
             'with_polygroups': self.with_polygroups,
             'global_scale': self.global_scale,
             'apply_conversion': self.apply_conversion, }
        w = FastOBJReader(**d)
        return {'FINISHED'}


def menu_func_export(self, context):
    self.layout.operator(ExportFastOBJ.bl_idname, text="Fast Wavefront (.obj)")


def menu_func_import(self, context):
    self.layout.operator(ImportFastOBJ.bl_idname, text="Fast Wavefront (.obj)")


def setup():
    export_presets = {'photoscan_final': {'apply_modifiers': False, 'apply_transformation': False, 'convert_axes': False, 'triangulate': False, 'use_uv': True,
                                          'use_shading': False, 'use_vertex_colors': False, 'use_vcols_mrgb': False, 'use_vcols_ext': False, 'global_scale': 1.0,
                                          'precision': 6, },
                      'zbrush_cleanup': {'apply_modifiers': False, 'apply_transformation': True, 'convert_axes': True, 'triangulate': False, 'use_uv': True,
                                         'use_shading': False, 'use_vertex_colors': False, 'use_vcols_mrgb': False, 'use_vcols_ext': False, 'global_scale': 1.0,
                                         'precision': 6, }, }
    import_presets = {'photoscan_raw': {'convert_axes': True, 'with_uv': False, 'with_shading': False, 'with_vertex_colors': False, 'use_vcols_mrgb': False,
                                        'use_vcols_ext': False, 'with_polygroups': False, 'global_scale': 1.0, 'apply_conversion': False, },
                      'zbrush_cleanup': {'convert_axes': True, 'with_uv': True, 'with_shading': False, 'with_vertex_colors': False, 'use_vcols_mrgb': False,
                                         'use_vcols_ext': False, 'with_polygroups': False, 'global_scale': 1.0, 'apply_conversion': False, },
                      'zbrush_with_vcols': {'convert_axes': True, 'with_uv': True, 'with_shading': False, 'with_vertex_colors': True, 'use_vcols_mrgb': True,
                                            'use_vcols_ext': False, 'with_polygroups': False, 'global_scale': 1.0, 'apply_conversion': False, }, }
    defines = ['import bpy', 'op = bpy.context.active_operator', '', ]
    
    # TODO: do this only when all presets are missing, if only some, assume it is intentional.. also skip when all default are missing, but some user defined are present.
    
    def write_presets(pdir, presets, defs):
        pde = os.path.join(bpy.utils.user_resource('SCRIPTS'), "presets", "operator", pdir)
        if(not os.path.exists(pde)):
            os.makedirs(pde)
        for pname, pdata in presets.items():
            pp = os.path.join(pde, '{}.py'.format(pname))
            if(not os.path.exists(pp)):
                a = defs[:]
                for k, v in pdata.items():
                    a.append('op.{} = {}'.format(k, v))
                s = "\n".join(a)
                with open(pp, mode='w', encoding='utf-8') as f:
                    f.write(s)
    
    write_presets("export_mesh.fast_obj", export_presets, defines)
    write_presets("import_mesh.fast_obj", import_presets, defines)


def register():
    bpy.utils.register_module(__name__)
    bpy.types.INFO_MT_file_export.append(menu_func_export)
    bpy.types.INFO_MT_file_import.append(menu_func_import)
    
    setup()


def unregister():
    bpy.utils.unregister_module(__name__)
    bpy.types.INFO_MT_file_export.remove(menu_func_export)
    bpy.types.INFO_MT_file_import.remove(menu_func_import)


if __name__ == "__main__":
    register()
