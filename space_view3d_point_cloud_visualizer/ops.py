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
import time
import datetime
import numpy as np
import re

import bpy
import bmesh
from bpy.props import StringProperty
from bpy.types import Operator
from mathutils import Matrix, Vector, Quaternion, Color
from bpy_extras.io_utils import axis_conversion, ExportHelper
import mathutils.geometry

from .debug import log, debug_mode
from . import io_ply
from .machine import PCVManager, PCVSequence, preferences, PCVControl, load_shader_code


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
        if(context.object is None):
            return False
        
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
            ok = PCVManager.load_ply_to_cache(self, context)
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
        pcv = context.object.point_cloud_visualizer
        c = PCVManager.cache[pcv.uuid]
        c['draw'] = False
        
        context.area.tag_redraw()
        
        return {'FINISHED'}


class PCV_OT_load(Operator):
    bl_idname = "point_cloud_visualizer.load_ply_to_cache"
    bl_label = "Load PLY"
    bl_description = "Load PLY file"
    
    filename_ext = ".ply"
    filter_glob: StringProperty(default="*.ply", options={'HIDDEN'}, )
    filepath: StringProperty(name="File Path", default="", description="", maxlen=1024, subtype='FILE_PATH', )
    order = ["filepath", ]
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        
        pcv = context.object.point_cloud_visualizer
        if(not pcv.runtime):
            return True
        return False
    
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
        
        ok = PCVManager.load_ply_to_cache(self, context)
        
        if(not ok):
            return {'CANCELLED'}
        return {'FINISHED'}


class PCV_OT_export(Operator, ExportHelper):
    bl_idname = "point_cloud_visualizer.export"
    bl_label = "Export PLY"
    bl_description = "Export point cloud to ply file"
    
    filename_ext = ".ply"
    filter_glob: StringProperty(default="*.ply", options={'HIDDEN'}, )
    check_extension = True
    
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
    
    def draw(self, context):
        l = self.layout
        c = l.column()
        pcv = context.object.point_cloud_visualizer
        c.prop(pcv, 'export_apply_transformation')
        c.prop(pcv, 'export_convert_axes')
        c.prop(pcv, 'export_visible_only')
    
    def execute(self, context):
        log("Export:", 0)
        _t = time.time()
        
        pcv = context.object.point_cloud_visualizer
        c = PCVManager.cache[pcv.uuid]
        
        o = c['object']
        
        if(pcv.export_use_viewport):
            log("using viewport points..", 1)
            vs = c['vertices']
            ns = c['normals']
            cs = c['colors']
            
            if(pcv.export_visible_only):
                log("visible only..", 1)
                # points in cache are stored already shuffled (or not), so this should work the same as in viewport..
                l = c['display_length']
                vs = vs[:l]
                ns = ns[:l]
                cs = cs[:l]
            
            # TODO: viewport points have always some normals and colors, should i keep it how it was loaded or should i include also generic data created for viewing?
            normals = True
            colors = True
            
        else:
            log("using original loaded points..", 1)
            # get original loaded points
            points = c['points']
            # check for normals
            normals = True
            if(not set(('nx', 'ny', 'nz')).issubset(points.dtype.names)):
                normals = False
            # check for colors
            colors = True
            if(not set(('red', 'green', 'blue')).issubset(points.dtype.names)):
                colors = False
            # make vertices, normals, colors arrays, use None if data is not available, colors leave as they are
            vs = np.column_stack((points['x'], points['y'], points['z'], ))
            ns = None
            if(normals):
                ns = np.column_stack((points['nx'], points['ny'], points['nz'], ))
            cs = None
            if(colors):
                cs = np.column_stack((points['red'], points['green'], points['blue'], ))
        
        def apply_matrix(m, vs, ns=None, ):
            vs.shape = (-1, 3)
            vs = np.c_[vs, np.ones(vs.shape[0])]
            vs = np.dot(m, vs.T)[0:3].T.reshape((-1))
            vs.shape = (-1, 3)
            if(ns is not None):
                _, rot, _ = m.decompose()
                rmat = rot.to_matrix().to_4x4()
                ns.shape = (-1, 3)
                ns = np.c_[ns, np.ones(ns.shape[0])]
                ns = np.dot(rmat, ns.T)[0:3].T.reshape((-1))
                ns.shape = (-1, 3)
            return vs, ns
        
        # fabricate matrix
        m = Matrix.Identity(4)
        if(pcv.export_apply_transformation):
            if(o.matrix_world != Matrix.Identity(4)):
                log("apply transformation..", 1)
                vs, ns = apply_matrix(o.matrix_world.copy(), vs, ns, )
        if(pcv.export_convert_axes):
            log("convert axes..", 1)
            axis_forward = '-Z'
            axis_up = 'Y'
            cm = axis_conversion(to_forward=axis_forward, to_up=axis_up).to_4x4()
            vs, ns = apply_matrix(cm, vs, ns, )
        
        # TODO: make whole PCV data type agnostic, load anything, keep original, convert to what is needed for display (float32), use original for export if not set to use viewport/edited data. now i am forcing float32 for x, y, z, nx, ny, nz and uint8 for red, green, blue. 99% of ply files i've seen is like that, but specification is not that strict (read again the best resource: http://paulbourke.net/dataformats/ply/ )
        
        # somehow along the way i am getting double dtype, so correct that
        vs = vs.astype(np.float32)
        if(normals):
            ns = ns.astype(np.float32)
        if(colors):
            # cs = cs.astype(np.float32)
            
            if(pcv.export_use_viewport):
                # viewport colors are in float32 now, back to uint8 colors, loaded data should be in uint8, so no need for conversion
                cs = cs.astype(np.float32)
                cs = cs * 255
                cs = cs.astype(np.uint8)
        
        log("write..", 1)
        
        # combine back to points, using original dtype
        dt = (('x', vs[0].dtype.str, ),
              ('y', vs[1].dtype.str, ),
              ('z', vs[2].dtype.str, ), )
        if(normals):
            dt += (('nx', ns[0].dtype.str, ),
                   ('ny', ns[1].dtype.str, ),
                   ('nz', ns[2].dtype.str, ), )
        if(colors):
            dt += (('red', cs[0].dtype.str, ),
                   ('green', cs[1].dtype.str, ),
                   ('blue', cs[2].dtype.str, ), )
        log("dtype: {}".format(dt), 1)
        
        l = len(vs)
        dt = list(dt)
        a = np.empty(l, dtype=dt, )
        a['x'] = vs[:, 0]
        a['y'] = vs[:, 1]
        a['z'] = vs[:, 2]
        if(normals):
            a['nx'] = ns[:, 0]
            a['ny'] = ns[:, 1]
            a['nz'] = ns[:, 2]
        if(colors):
            a['red'] = cs[:, 0]
            a['green'] = cs[:, 1]
            a['blue'] = cs[:, 2]
        
        w = io_ply.BinPlyPointCloudWriter(self.filepath, a, )
        
        _d = datetime.timedelta(seconds=time.time() - _t)
        log("completed in {}.".format(_d), 1)
        
        return {'FINISHED'}


class PCV_OT_reload(Operator):
    bl_idname = "point_cloud_visualizer.reload"
    bl_label = "Reload"
    bl_description = "Reload points from file"
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        
        pcv = context.object.point_cloud_visualizer
        # if(pcv.filepath != '' and pcv.uuid != '' and not pcv.runtime):
        if(pcv.filepath != '' and pcv.uuid != ''):
            return True
        return False
    
    def execute(self, context):
        pcv = context.object.point_cloud_visualizer
        
        draw = False
        for k, v in PCVManager.cache.items():
            if(v['uuid'] == pcv.uuid):
                if(v['ready']):
                    if(v['draw']):
                        draw = True
                        bpy.ops.point_cloud_visualizer.erase()
        
        if(pcv.runtime):
            c = PCVManager.cache[pcv.uuid]
            points = c['points']
            vs = np.column_stack((points['x'], points['y'], points['z'], ))
            ns = np.column_stack((points['nx'], points['ny'], points['nz'], ))
            cs = c['colors_original']
            PCVManager.update(pcv.uuid, vs, ns, cs, )
        else:
            bpy.ops.point_cloud_visualizer.load_ply_to_cache(filepath=pcv.filepath)
        
        if(draw):
            bpy.ops.point_cloud_visualizer.draw()
        
        return {'FINISHED'}


class PCV_OT_sequence_preload(Operator):
    bl_idname = "point_cloud_visualizer.sequence_preload"
    bl_label = "Preload Sequence"
    bl_description = "Preload sequence of PLY files. Files should be numbered starting at 1. Missing files in sequence will be skipped."
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        
        pcv = context.object.point_cloud_visualizer
        if(pcv.uuid in PCVSequence.cache.keys()):
            return False
        ok = False
        for k, v in PCVManager.cache.items():
            if(v['uuid'] == pcv.uuid):
                if(v['ready']):
                    if(v['draw']):
                        ok = True
        return ok
    
    def execute(self, context):
        log('Preload Sequence..')
        pcv = context.object.point_cloud_visualizer
        
        # pcv.sequence_enabled = True
        
        dirpath = os.path.dirname(pcv.filepath)
        files = []
        for (p, ds, fs) in os.walk(dirpath):
            files.extend(fs)
            break
        
        fs = [f for f in files if f.lower().endswith('.ply')]
        f = os.path.split(pcv.filepath)[1]
        
        pattern = re.compile(r'(\d+)(?!.*(\d+))')
        
        m = re.search(pattern, f, )
        if(m is not None):
            prefix = f[:m.start()]
            suffix = f[m.end():]
        else:
            self.report({'ERROR'}, 'Filename does not contain any sequence number')
            return {'CANCELLED'}
        
        sel = []
        
        for n in fs:
            m = re.search(pattern, n, )
            if(m is not None):
                # some numbers present, lets compare with selected file prefix/suffix
                pre = n[:m.start()]
                if(pre == prefix):
                    # prefixes match
                    suf = n[m.end():]
                    if(suf == suffix):
                        # suffixes match, extract number
                        si = n[m.start():m.end()]
                        try:
                            # try convert it to integer
                            i = int(si)
                            # and store as selected file
                            sel.append((i, n))
                        except ValueError:
                            pass
        
        # sort by sequence number
        sel.sort()
        # fabricate list with missing sequence numbers as None
        sequence = [[None] for i in range(sel[-1][0])]
        for i, n in sel:
            sequence[i - 1] = (i, n)
        for i in range(len(sequence)):
            if(sequence[i][0] is None):
                sequence[i] = [i, None]
        
        log('found files:', 1)
        for i, n in sequence:
            log('{}: {}'.format(i, n), 2)
        
        log('preloading..', 1)
        # this is our sequence with matching filenames, sorted by numbers with missing as None, now load it all..
        cache = []
        for i, n in sequence:
            if(n is not None):
                p = os.path.join(dirpath, n)
                points = []
                try:
                    points = io_ply.PlyPointCloudReader(p).points
                except Exception as e:
                    self.report({'ERROR'}, str(e))
                if(len(points) == 0):
                    self.report({'ERROR'}, "No vertices loaded from file at {}".format(p))
                else:
                    if(not set(('x', 'y', 'z')).issubset(points.dtype.names)):
                        self.report({'ERROR'}, "Loaded data seems to miss vertex locations.")
                        return {'CANCELLED'}
                    
                    vs = np.column_stack((points['x'], points['y'], points['z'], ))
                    vs = vs.astype(np.float32)
                    
                    if(not set(('nx', 'ny', 'nz')).issubset(points.dtype.names)):
                        ns = None
                    else:
                        ns = np.column_stack((points['nx'], points['ny'], points['nz'], ))
                        ns = ns.astype(np.float32)
                    if(not set(('red', 'green', 'blue')).issubset(points.dtype.names)):
                        cs = None
                    else:
                        # cs = np.column_stack((points['red'] / 255, points['green'] / 255, points['blue'] / 255, np.ones(len(points), dtype=float, ), ))
                        # cs = cs.astype(np.float32)
                        
                        r_f32 = points['red'].astype(np.float32)
                        g_f32 = points['green'].astype(np.float32)
                        b_f32 = points['blue'].astype(np.float32)
                        r_f32 /= 255.0
                        g_f32 /= 255.0
                        b_f32 /= 255.0
                        cs = np.column_stack((r_f32, g_f32, b_f32, np.ones(len(points), dtype=np.float32, ), ))
                    
                    cache.append({'index': i,
                                  'name': n,
                                  'path': p,
                                  'vs': vs,
                                  'ns': ns,
                                  'cs': cs,
                                  'points': points, })
        
        log('...', 1)
        log('loaded {} item(s)'.format(len(cache)), 1)
        log('initializing..', 1)
        
        PCVSequence.init()
        
        ci = {'data': cache,
              'uuid': pcv.uuid,
              'pcv': pcv, }
        PCVSequence.cache[pcv.uuid] = ci
        
        log('force frame update..', 1)
        sc = bpy.context.scene
        cf = sc.frame_current
        sc.frame_current = cf
        log('done.', 1)
        
        return {'FINISHED'}


class PCV_OT_sequence_clear(Operator):
    bl_idname = "point_cloud_visualizer.sequence_clear"
    bl_label = "Clear Sequence"
    bl_description = "Clear preloaded sequence cache and reset all"
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        
        pcv = context.object.point_cloud_visualizer
        if(pcv.uuid in PCVSequence.cache.keys()):
            return True
        return False
        # ok = False
        # for k, v in PCVManager.cache.items():
        #     if(v['uuid'] == pcv.uuid):
        #         if(v['ready']):
        #             if(v['draw']):
        #                 ok = True
        # return ok
    
    def execute(self, context):
        pcv = context.object.point_cloud_visualizer
        
        del PCVSequence.cache[pcv.uuid]
        if(len(PCVSequence.cache.items()) == 0):
            PCVSequence.deinit()
        
        # c = PCVManager.cache[pcv.uuid]
        # vs = c['vertices']
        # ns = c['normals']
        # cs = c['colors']
        # PCVManager.update(pcv.uuid, vs, ns, cs, )
        
        bpy.ops.point_cloud_visualizer.reload()
        
        return {'FINISHED'}


class PCV_OT_seq_init(Operator):
    bl_idname = "point_cloud_visualizer.seq_init"
    bl_label = "seq_init"
    
    def execute(self, context):
        PCVSequence.init()
        context.area.tag_redraw()
        return {'FINISHED'}


class PCV_OT_seq_deinit(Operator):
    bl_idname = "point_cloud_visualizer.seq_deinit"
    bl_label = "seq_deinit"
    
    def execute(self, context):
        PCVSequence.deinit()
        context.area.tag_redraw()
        return {'FINISHED'}


class PCV_OT_reset_runtime(Operator):
    bl_idname = "point_cloud_visualizer.reset_runtime"
    bl_label = "Reset Runtime"
    bl_description = "Reset PCV to its default state if in runtime mode (displayed data is set with python and not with ui)"
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        pcv = context.object.point_cloud_visualizer
        if(pcv.runtime):
            return True
        return False
    
    def execute(self, context):
        o = context.object
        c = PCVControl(o)
        c.erase()
        c.reset()
        return {'FINISHED'}


class PCV_OT_clip_planes_from_bbox(Operator):
    bl_idname = "point_cloud_visualizer.clip_planes_from_bbox"
    bl_label = "Set Clip Planes From Object Bounding Box"
    bl_description = "Set clip planes from object bounding box"
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        
        pcv = context.object.point_cloud_visualizer
        ok = False
        
        bbo = pcv.clip_planes_from_bbox_object
        if(bbo is not None):
            ok = True
        
        return ok
    
    def execute(self, context):
        pcv = context.object.point_cloud_visualizer
        bbo = pcv.clip_planes_from_bbox_object
        
        mw = bbo.matrix_world
        vs = [mw @ Vector(v) for v in bbo.bound_box]
        
        # 0 front left down
        # 1 front left up
        # 2 back left up
        # 3 back left down
        # 4 front right down
        # 5 front right up
        # 6 back right up
        # 7 back right down
        #          2-----------6
        #         /           /|
        #       1-----------5  |
        #       |           |  |
        #       |           |  |
        #       |  3        |  7
        #       |           | /
        #       0-----------4
        
        fs = (
            (0, 4, 5, 1),  # front
            (3, 2, 6, 7),  # back
            (1, 5, 6, 2),  # top
            (0, 3, 7, 4),  # bottom
            (0, 1, 2, 3),  # left
            (4, 7, 6, 5),  # right
        )
        
        quads = [[vs[fs[i][j]] for j in range(4)] for i in range(6)]
        ns = [mathutils.geometry.normal(quads[i]) for i in range(6)]
        for i in range(6):
            # FIXME: if i need to do this, it is highly probable i have something wrong.. somewhere..
            ns[i].negate()
        
        ds = []
        for i in range(6):
            v = quads[i][0]
            n = ns[i]
            d = mathutils.geometry.distance_point_to_plane(Vector(), v, n)
            ds.append(d)
        
        a = [ns[i].to_tuple() + (ds[i], ) for i in range(6)]
        pcv.clip_plane0 = a[0]
        pcv.clip_plane1 = a[1]
        pcv.clip_plane2 = a[2]
        pcv.clip_plane3 = a[3]
        pcv.clip_plane4 = a[4]
        pcv.clip_plane5 = a[5]
        
        pcv.clip_shader_enabled = True
        pcv.clip_plane0_enabled = True
        pcv.clip_plane1_enabled = True
        pcv.clip_plane2_enabled = True
        pcv.clip_plane3_enabled = True
        pcv.clip_plane4_enabled = True
        pcv.clip_plane5_enabled = True
        
        return {'FINISHED'}


class PCV_OT_clip_planes_reset(Operator):
    bl_idname = "point_cloud_visualizer.clip_planes_reset"
    bl_label = "Reset Clip Planes"
    bl_description = "Reset all clip planes"
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        
        pcv = context.object.point_cloud_visualizer
        ok = True
        
        return ok
    
    def execute(self, context):
        pcv = context.object.point_cloud_visualizer
        
        pcv.clip_planes_from_bbox_object = None
        
        z = (0.0, 0.0, 0.0, 0.0, )
        pcv.clip_plane0 = z
        pcv.clip_plane1 = z
        pcv.clip_plane2 = z
        pcv.clip_plane3 = z
        pcv.clip_plane4 = z
        pcv.clip_plane5 = z
        
        pcv.clip_shader_enabled = False
        pcv.clip_plane0_enabled = False
        pcv.clip_plane1_enabled = False
        pcv.clip_plane2_enabled = False
        pcv.clip_plane3_enabled = False
        pcv.clip_plane4_enabled = False
        pcv.clip_plane5_enabled = False
        
        return {'FINISHED'}


class PCV_OT_clip_planes_from_camera_view(Operator):
    bl_idname = "point_cloud_visualizer.clip_planes_from_camera_view"
    bl_label = "Set Clip Planes From Camera View"
    bl_description = "Set clip planes from active camera view"
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        
        pcv = context.object.point_cloud_visualizer
        ok = False
        
        s = bpy.context.scene
        o = s.camera
        if(o):
            ok = True
        
        return ok
    
    def execute(self, context):
        pcv = context.object.point_cloud_visualizer
        
        o = bpy.context.object
        
        s = bpy.context.scene
        c = s.camera
        cd = c.data
        rx = s.render.resolution_x
        ry = s.render.resolution_y
        w = 0.5 * cd.sensor_width / cd.lens
        if(rx > ry):
            x = w
            y = w * ry / rx
        else:
            x = w * rx / ry
            y = w
        
        lr = Vector((x, -y, -1.0, ))
        ur = Vector((x, y, -1.0, ))
        ll = Vector((-x, -y, -1.0, ))
        ul = Vector((-x, y, -1.0, ))
        
        z = Vector()
        n0 = mathutils.geometry.normal((z, lr, ur))
        n1 = mathutils.geometry.normal((z, ur, ul))
        n2 = mathutils.geometry.normal((z, ul, ll))
        n3 = mathutils.geometry.normal((z, ll, lr))
        
        n0.negate()
        n1.negate()
        n2.negate()
        n3.negate()
        
        m = c.matrix_world
        l, r, _ = m.decompose()
        rm = r.to_matrix().to_4x4()
        lrm = Matrix.Translation(l).to_4x4() @ rm
        
        omi = o.matrix_world.inverted()
        _, r, _ = omi.decompose()
        orm = r.to_matrix().to_4x4()
        
        n0 = orm @ rm @ n0
        n1 = orm @ rm @ n1
        n2 = orm @ rm @ n2
        n3 = orm @ rm @ n3
        
        v0 = omi @ lrm @ lr
        v1 = omi @ lrm @ ur
        v2 = omi @ lrm @ ul
        v3 = omi @ lrm @ ll
        
        d0 = mathutils.geometry.distance_point_to_plane(Vector(), v0, n0)
        d1 = mathutils.geometry.distance_point_to_plane(Vector(), v1, n1)
        d2 = mathutils.geometry.distance_point_to_plane(Vector(), v2, n2)
        d3 = mathutils.geometry.distance_point_to_plane(Vector(), v3, n3)
        
        # TODO: add plane behind camera (not much needed anyway, but for consistency), but more important, add plane in clipping distance set on camera
        # TODO: ORTHO camera does not work
        
        pcv.clip_plane0 = n0.to_tuple() + (d0, )
        pcv.clip_plane1 = n1.to_tuple() + (d1, )
        pcv.clip_plane2 = n2.to_tuple() + (d2, )
        pcv.clip_plane3 = n3.to_tuple() + (d3, )
        pcv.clip_plane4 = (0.0, 0.0, 0.0, 0.0, )
        pcv.clip_plane5 = (0.0, 0.0, 0.0, 0.0, )
        
        pcv.clip_shader_enabled = True
        pcv.clip_plane0_enabled = True
        pcv.clip_plane1_enabled = True
        pcv.clip_plane2_enabled = True
        pcv.clip_plane3_enabled = True
        pcv.clip_plane4_enabled = False
        pcv.clip_plane5_enabled = False
        
        return {'FINISHED'}


classes = (
    PCV_OT_load, PCV_OT_draw, PCV_OT_erase,
    PCV_OT_reload, PCV_OT_export,
    PCV_OT_sequence_preload, PCV_OT_sequence_clear,
    PCV_OT_reset_runtime,
    PCV_OT_clip_planes_from_bbox, PCV_OT_clip_planes_reset, PCV_OT_clip_planes_from_camera_view,
)
classes_debug = (PCV_OT_init, PCV_OT_deinit, PCV_OT_gc, PCV_OT_seq_init, PCV_OT_seq_deinit, )
