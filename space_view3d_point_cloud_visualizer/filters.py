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
# (c) 2019 Jakub Uhlik

import os
import time
import datetime
import numpy as np

import bpy
import bmesh
from bpy.props import StringProperty
from bpy.types import Operator
from mathutils import Matrix, Vector, Quaternion, Color
from mathutils.kdtree import KDTree
from mathutils.geometry import barycentric_transform
from mathutils.interpolate import poly_3d_calc
from mathutils.bvhtree import BVHTree

from .debug import log, debug_mode, Progress
from . import io_ply
from .machine import PCVManager, preferences


class PCV_OT_filter_simplify(Operator):
    bl_idname = "point_cloud_visualizer.filter_simplify"
    bl_label = "Simplify"
    bl_description = "Simplify point cloud to exact number of evenly distributed samples, all loaded points are processed"
    
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
    
    def resample(self, context):
        scene = context.scene
        pcv = context.object.point_cloud_visualizer
        
        c = PCVManager.cache[pcv.uuid]
        vs = c['vertices']
        ns = c['normals']
        cs = c['colors']
        
        num_samples = pcv.filter_simplify_num_samples
        if(num_samples >= len(vs)):
            self.report({'ERROR'}, "Number of samples must be < number of points.")
            return False, []
        candidates = pcv.filter_simplify_num_candidates
        log("num_samples: {}, candidates: {}".format(num_samples, candidates), 1)
        
        l = len(vs)
        dt = [('x', '<f8'), ('y', '<f8'), ('z', '<f8'), ('nx', '<f8'), ('ny', '<f8'), ('nz', '<f8'), ('red', '<f8'), ('green', '<f8'), ('blue', '<f8'), ('alpha', '<f8'), ('index', '<i8')]
        pool = np.empty(l, dtype=dt, )
        pool['x'] = vs[:, 0]
        pool['y'] = vs[:, 1]
        pool['z'] = vs[:, 2]
        pool['nx'] = ns[:, 0]
        pool['ny'] = ns[:, 1]
        pool['nz'] = ns[:, 2]
        pool['red'] = cs[:, 0]
        pool['green'] = cs[:, 1]
        pool['blue'] = cs[:, 2]
        pool['alpha'] = cs[:, 3]
        pool['index'] = np.indices((l, ), dtype='<i8', )
        
        # to get random points, shuffle pool if not shuffled upon loading
        # preferences = bpy.context.preferences
        # addon_prefs = preferences.addons[__name__].preferences
        addon_prefs = preferences()
        if(not addon_prefs.shuffle_points):
            np.random.shuffle(pool)
            # and set indexes again
            pool['index'] = np.indices((l, ), dtype='<i8', )
        
        tree = KDTree(len(pool))
        samples = np.array(pool[0])
        last_used = 0
        # unused_pool = np.empty(0, dtype=dt, )
        unused_pool = []
        
        tree.insert([pool[0]['x'], pool[0]['y'], pool[0]['z']], 0)
        tree.balance()
        
        log("sampling:", 1)
        use_unused = False
        prgs = Progress(num_samples - 1, indent=2, prefix="> ")
        for i in range(num_samples - 1):
            prgs.step()
            # choose candidates
            cands = pool[last_used + 1:last_used + 1 + candidates]
            
            if(len(cands) <= 0):
                # lets pretend that last set of candidates was filled full..
                use_unused = True
                cands = unused_pool[:candidates]
                unused_pool = unused_pool[candidates:]
            
            if(len(cands) == 0):
                # out of candidates, nothing to do here now.. number of desired samples was too close to total of points
                return True, samples
            
            # get the most distant candidate
            dists = []
            inds = []
            for cl in cands:
                vec, ci, cd = tree.find((cl['x'], cl['y'], cl['z']))
                inds.append(ci)
                dists.append(cd)
            maxd = 0.0
            maxi = 0
            for j, d in enumerate(dists):
                if(d > maxd):
                    maxd = d
                    maxi = j
            # chosen candidate
            ncp = cands[maxi]
            # get unused candidates to recycle
            uncands = np.delete(cands, maxi)
            # unused_pool = np.append(unused_pool, uncands)
            unused_pool.extend(uncands)
            # store accepted sample
            samples = np.append(samples, ncp)
            # update kdtree
            tree.insert([ncp['x'], ncp['y'], ncp['z']], ncp['index'])
            tree.balance()
            # use next points
            last_used += 1 + candidates
        
        return True, samples
    
    def execute(self, context):
        log("Simplify:", 0)
        _t = time.time()
        
        # if(debug_mode()):
        #     import cProfile
        #     import pstats
        #     import io
        #     pr = cProfile.Profile()
        #     pr.enable()
        
        ok, a = self.resample(context)
        if(not ok):
            return {'CANCELLED'}
        
        # if(debug_mode()):
        #     pr.disable()
        #     s = io.StringIO()
        #     sortby = 'cumulative'
        #     ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        #     ps.print_stats()
        #     print(s.getvalue())
        
        vs = np.column_stack((a['x'], a['y'], a['z'], ))
        ns = np.column_stack((a['nx'], a['ny'], a['nz'], ))
        cs = np.column_stack((a['red'], a['green'], a['blue'], a['alpha'], ))
        vs = vs.astype(np.float32)
        ns = ns.astype(np.float32)
        cs = cs.astype(np.float32)
        
        # put to cache
        pcv = context.object.point_cloud_visualizer
        PCVManager.update(pcv.uuid, vs, ns, cs, )
        
        _d = datetime.timedelta(seconds=time.time() - _t)
        log("completed in {}.".format(_d), 1)
        
        return {'FINISHED'}


class PCV_OT_filter_project(Operator):
    bl_idname = "point_cloud_visualizer.filter_project"
    bl_label = "Project"
    bl_description = "Project points on mesh surface"
    
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
                        o = pcv.filter_project_object
                        if(o is not None):
                            if(o.type in ('MESH', 'CURVE', 'SURFACE', 'FONT', )):
                                ok = True
        return ok
    
    def execute(self, context):
        log("Project:", 0)
        _t = time.time()
        
        # if(debug_mode()):
        #     import cProfile
        #     import pstats
        #     import io
        #     pr = cProfile.Profile()
        #     pr.enable()
        
        log("preprocessing..", 1)
        
        pcv = context.object.point_cloud_visualizer
        o = pcv.filter_project_object
        
        if(o is None):
            raise Exception()
        
        c = PCVManager.cache[pcv.uuid]
        vs = c['vertices']
        ns = c['normals']
        cs = c['colors']
        
        # apply parent matrix to points
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
        
        m = c['object'].matrix_world.copy()
        vs, ns = apply_matrix(m, vs, ns, )
        vs = vs.astype(np.float32)
        ns = ns.astype(np.float32)
        
        # combine
        l = len(vs)
        dt = [('x', '<f8'), ('y', '<f8'), ('z', '<f8'), ('nx', '<f8'), ('ny', '<f8'), ('nz', '<f8'), ('red', '<f8'), ('green', '<f8'), ('blue', '<f8'), ('alpha', '<f8'), ('index', '<i8'), ('delete', '?')]
        points = np.empty(l, dtype=dt, )
        points['x'] = vs[:, 0]
        points['y'] = vs[:, 1]
        points['z'] = vs[:, 2]
        points['nx'] = ns[:, 0]
        points['ny'] = ns[:, 1]
        points['nz'] = ns[:, 2]
        points['red'] = cs[:, 0]
        points['green'] = cs[:, 1]
        points['blue'] = cs[:, 2]
        points['alpha'] = cs[:, 3]
        points['index'] = np.indices((l, ), dtype='<i8', )
        points['delete'] = np.zeros((l, ), dtype='?', )
        
        search_distance = pcv.filter_project_search_distance
        negative = pcv.filter_project_negative
        positive = pcv.filter_project_positive
        discard = pcv.filter_project_discard
        shift = pcv.filter_project_shift
        
        sc = context.scene
        depsgraph = bpy.context.evaluated_depsgraph_get()
        
        # make target
        tmp_mesh = o.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph, )
        target_mesh = tmp_mesh.copy()
        target_mesh.name = 'target_mesh_{}'.format(pcv.uuid)
        target_mesh.transform(o.matrix_world.copy())
        view_layer = context.view_layer
        collection = view_layer.active_layer_collection.collection
        target = bpy.data.objects.new('target_mesh_{}'.format(pcv.uuid), target_mesh)
        collection.objects.link(target)
        # TODO use BVHTree for ray_cast without need to add object to scene
        # still no idea, have to read about it..
        depsgraph.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        
        # this should be None if not available
        vcols = o.data.vertex_colors.active
        uvtex = o.data.uv_layers.active
        vgroup = o.vertex_groups.active
        # now check if color source is available, if not, cancel
        if(pcv.filter_project_colorize):
            
            # depsgraph = context.evaluated_depsgraph_get()
            # if(o.modifiers):
            #     owner = o.evaluated_get(depsgraph)
            #     me = owner.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph, )
            # else:
            #     owner = o
            #     me = owner.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph, )
            
            bm = bmesh.new()
            bm.from_mesh(target_mesh)
            bmesh.ops.triangulate(bm, faces=bm.faces)
            bm.verts.ensure_lookup_table()
            bm.faces.ensure_lookup_table()
            
            if(pcv.filter_project_colorize_from == 'VCOLS'):
                try:
                    col_layer = bm.loops.layers.color.active
                    if(col_layer is None):
                        raise Exception()
                except Exception:
                    self.report({'ERROR'}, "Cannot find active vertex colors", )
                    return {'CANCELLED'}
            elif(pcv.filter_project_colorize_from == 'UVTEX'):
                try:
                    if(o.active_material is None):
                        raise Exception("Cannot find active material")
                    uvtexnode = o.active_material.node_tree.nodes.active
                    if(uvtexnode is None):
                        raise Exception("Cannot find active image texture in active material")
                    if(uvtexnode.type != 'TEX_IMAGE'):
                        raise Exception("Cannot find active image texture in active material")
                    uvimage = uvtexnode.image
                    if(uvimage is None):
                        raise Exception("Cannot find active image texture with loaded image in active material")
                    uvimage.update()
                    uvarray = np.asarray(uvimage.pixels)
                    uvarray = uvarray.reshape((uvimage.size[1], uvimage.size[0], 4))
                    uvlayer = bm.loops.layers.uv.active
                    if(uvlayer is None):
                        raise Exception("Cannot find active UV layout")
                except Exception as e:
                    self.report({'ERROR'}, str(e), )
                    return {'CANCELLED'}
            elif(pcv.filter_project_colorize_from in ['GROUP_MONO', 'GROUP_COLOR', ]):
                try:
                    group_layer = bm.verts.layers.deform.active
                    if(group_layer is None):
                        raise Exception()
                    group_layer_index = o.vertex_groups.active.index
                except Exception:
                    self.report({'ERROR'}, "Cannot find active vertex group", )
                    return {'CANCELLED'}
            else:
                self.report({'ERROR'}, "Unsupported color source", )
                return {'CANCELLED'}
        
        def distance(a, b, ):
            return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5
        
        def shift_vert_along_normal(co, no, v):
            c = Vector(co)
            n = Vector(no)
            return c + (n.normalized() * v)
        
        def remap(v, min1, max1, min2, max2, ):
            def clamp(v, vmin, vmax):
                if(vmax <= vmin):
                    raise ValueError("Maximum value is smaller than or equal to minimum.")
                if(v <= vmin):
                    return vmin
                if(v >= vmax):
                    return vmax
                return v
            
            def normalize(v, vmin, vmax):
                return (v - vmin) / (vmax - vmin)
            
            def interpolate(nv, vmin, vmax):
                return vmin + (vmax - vmin) * nv
            
            if(max1 - min1 == 0):
                # handle zero division when min1 = max1
                return min2
            
            r = interpolate(normalize(v, min1, max1), min2, max2)
            return r
        
        def gen_color(bm, result, location, normal, index, distance, ):
            col = (0.0, 0.0, 0.0, )
            
            poly = bm.faces[index]
            v = location
            
            if(pcv.filter_project_colorize_from == 'VCOLS'):
                ws = poly_3d_calc([poly.verts[0].co, poly.verts[1].co, poly.verts[2].co, ], v)
                ac = poly.loops[0][col_layer][:3]
                bc = poly.loops[1][col_layer][:3]
                cc = poly.loops[2][col_layer][:3]
                r = ac[0] * ws[0] + bc[0] * ws[1] + cc[0] * ws[2]
                g = ac[1] * ws[0] + bc[1] * ws[1] + cc[1] * ws[2]
                b = ac[2] * ws[0] + bc[2] * ws[1] + cc[2] * ws[2]
                col = (r, g, b, )
            elif(pcv.filter_project_colorize_from == 'UVTEX'):
                uvtriangle = []
                for l in poly.loops:
                    uvtriangle.append(Vector(l[uvlayer].uv.to_tuple() + (0.0, )))
                uvpoint = barycentric_transform(v, poly.verts[0].co, poly.verts[1].co, poly.verts[2].co, *uvtriangle, )
                w, h = uvimage.size
                # x,y % 1.0 to wrap around if uv coordinate is outside 0.0-1.0 range
                x = int(round(remap(uvpoint.x % 1.0, 0.0, 1.0, 0, w - 1)))
                y = int(round(remap(uvpoint.y % 1.0, 0.0, 1.0, 0, h - 1)))
                col = tuple(uvarray[y][x][:3].tolist())
            elif(pcv.filter_project_colorize_from == 'GROUP_MONO'):
                ws = poly_3d_calc([poly.verts[0].co, poly.verts[1].co, poly.verts[2].co, ], v)
                aw = poly.verts[0][group_layer].get(group_layer_index, 0.0)
                bw = poly.verts[1][group_layer].get(group_layer_index, 0.0)
                cw = poly.verts[2][group_layer].get(group_layer_index, 0.0)
                m = aw * ws[0] + bw * ws[1] + cw * ws[2]
                col = (m, m, m, )
            elif(pcv.filter_project_colorize_from == 'GROUP_COLOR'):
                ws = poly_3d_calc([poly.verts[0].co, poly.verts[1].co, poly.verts[2].co, ], v)
                aw = poly.verts[0][group_layer].get(group_layer_index, 0.0)
                bw = poly.verts[1][group_layer].get(group_layer_index, 0.0)
                cw = poly.verts[2][group_layer].get(group_layer_index, 0.0)
                m = aw * ws[0] + bw * ws[1] + cw * ws[2]
                hue = remap(1.0 - m, 0.0, 1.0, 0.0, 1 / 1.5)
                c = Color()
                c.hsv = (hue, 1.0, 1.0, )
                col = (c.r, c.g, c.b, )
            
            return col
        
        a = np.empty(l, dtype=dt, )
        
        log("projecting:", 1)
        prgr = Progress(len(points), 2)
        for i, p in enumerate(points):
            prgr.step()
            
            v = Vector((p['x'], p['y'], p['z'], ))
            n = Vector((p['nx'], p['ny'], p['nz'], ))
            
            if(positive):
                p_result, p_location, p_normal, p_index = target.ray_cast(v, n, distance=search_distance, depsgraph=depsgraph, )
                p_distance = None
                if(p_result):
                    p_distance = distance(p_location, v[:])
            else:
                p_result = False
            
            nn = shift_vert_along_normal(v, n, -search_distance)
            ndir = Vector(nn - v).normalized()
            
            if(negative):
                n_result, n_location, n_normal, n_index = target.ray_cast(v, ndir, distance=search_distance, depsgraph=depsgraph, )
                n_distance = None
                if(n_result):
                    n_distance = distance(n_location, v[:])
            else:
                n_result = False
            
            rp = np.copy(p)
            
            # store ray_cast results which was used for current point and pass them to gen_color
            used = None
            
            if(p_result and n_result):
                if(p_distance < n_distance):
                    rp['x'] = p_location[0]
                    rp['y'] = p_location[1]
                    rp['z'] = p_location[2]
                    used = (p_result, p_location, p_normal, p_index, p_distance)
                else:
                    rp['x'] = n_location[0]
                    rp['y'] = n_location[1]
                    rp['z'] = n_location[2]
                    used = (n_result, n_location, n_normal, n_index, n_distance)
            elif(p_result):
                rp['x'] = p_location[0]
                rp['y'] = p_location[1]
                rp['z'] = p_location[2]
                used = (p_result, p_location, p_normal, p_index, p_distance)
            elif(n_result):
                rp['x'] = n_location[0]
                rp['y'] = n_location[1]
                rp['z'] = n_location[2]
                used = (n_result, n_location, n_normal, n_index, n_distance)
            else:
                rp['delete'] = 1
            
            if(pcv.filter_project_colorize):
                if(used is not None):
                    col = gen_color(bm, *used)
                    rp['red'] = col[0]
                    rp['green'] = col[1]
                    rp['blue'] = col[2]
            
            a[i] = rp
        
        if(discard):
            log("discarding:", 1)
            prgr = Progress(len(a), 2)
            indexes = []
            for i in a:
                prgr.step()
                if(i['delete']):
                    indexes.append(i['index'])
            a = np.delete(a, indexes)
        
        if(shift != 0.0):
            log("shifting:", 1)
            prgr = Progress(len(a), 2)
            for i in a:
                prgr.step()
                l = shift_vert_along_normal((i['x'], i['y'], i['z'], ), (i['nx'], i['ny'], i['nz'], ), shift, )
                i['x'] = l[0]
                i['y'] = l[1]
                i['z'] = l[2]
        
        log("postprocessing..", 1)
        # split back
        vs = np.column_stack((a['x'], a['y'], a['z'], ))
        ns = np.column_stack((a['nx'], a['ny'], a['nz'], ))
        cs = np.column_stack((a['red'], a['green'], a['blue'], a['alpha'], ))
        vs = vs.astype(np.float32)
        ns = ns.astype(np.float32)
        cs = cs.astype(np.float32)
        
        # unapply parent matrix to points
        m = c['object'].matrix_world.copy()
        m = m.inverted()
        vs, ns = apply_matrix(m, vs, ns, )
        vs = vs.astype(np.float32)
        ns = ns.astype(np.float32)
        
        # cleanup
        if(pcv.filter_project_colorize):
            bm.free()
            # owner.to_mesh_clear()
        
        collection.objects.unlink(target)
        bpy.data.objects.remove(target)
        bpy.data.meshes.remove(target_mesh)
        o.to_mesh_clear()
        
        # put to cache..
        pcv = context.object.point_cloud_visualizer
        PCVManager.update(pcv.uuid, vs, ns, cs, )
        
        # if(debug_mode()):
        #     pr.disable()
        #     s = io.StringIO()
        #     sortby = 'cumulative'
        #     ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        #     ps.print_stats()
        #     print(s.getvalue())
        
        _d = datetime.timedelta(seconds=time.time() - _t)
        log("completed in {}.".format(_d), 1)
        
        return {'FINISHED'}


class PCV_OT_filter_remove_color(Operator):
    bl_idname = "point_cloud_visualizer.filter_remove_color"
    bl_label = "Select Color"
    bl_description = "Select points with exact/similar color"
    
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
        log("Remove Color:", 0)
        _t = time.time()
        
        pcv = context.object.point_cloud_visualizer
        # cache item
        c = PCVManager.cache[pcv.uuid]
        vs = c['vertices']
        ns = c['normals']
        cs = c['colors']
        # join to indexed points
        l = len(vs)
        dt = [('x', '<f8'), ('y', '<f8'), ('z', '<f8'), ('nx', '<f8'), ('ny', '<f8'), ('nz', '<f8'), ('red', '<f8'), ('green', '<f8'), ('blue', '<f8'), ('alpha', '<f8'), ('index', '<i8')]
        points = np.empty(l, dtype=dt, )
        points['x'] = vs[:, 0]
        points['y'] = vs[:, 1]
        points['z'] = vs[:, 2]
        points['nx'] = ns[:, 0]
        points['ny'] = ns[:, 1]
        points['nz'] = ns[:, 2]
        points['red'] = cs[:, 0]
        points['green'] = cs[:, 1]
        points['blue'] = cs[:, 2]
        points['alpha'] = cs[:, 3]
        points['index'] = np.indices((l, ), dtype='<i8', )
        
        # evaluate
        indexes = []
        
        # black magic..
        c = [c ** (1 / 2.2) for c in pcv.filter_remove_color]
        c = [int(i * 256) for i in c]
        c = [i / 256 for i in c]
        rmcolor = Color(c)
        
        # take half of the value because 1/2 <- v -> 1/2, plus and minus => full range
        dh = pcv.filter_remove_color_delta_hue / 2
        # only for hue, because i take in consideration its radial nature
        ds = pcv.filter_remove_color_delta_saturation
        dv = pcv.filter_remove_color_delta_value
        uh = pcv.filter_remove_color_delta_hue_use
        us = pcv.filter_remove_color_delta_saturation_use
        uv = pcv.filter_remove_color_delta_value_use
        
        prgr = Progress(len(points), 1)
        for p in points:
            prgr.step()
            # get point color
            c = Color((p['red'], p['green'], p['blue']))
            # check for more or less same color, a few decimals should be more than enough, ply should have 8bit colors
            fpr = 5
            same = (round(c.r, fpr) == round(rmcolor.r, fpr),
                    round(c.g, fpr) == round(rmcolor.g, fpr),
                    round(c.b, fpr) == round(rmcolor.b, fpr))
            if(all(same)):
                indexes.append(p['index'])
                continue
            
            # check
            h = False
            s = False
            v = False
            if(uh):
                rm_hue = rmcolor.h
                hd = min(abs(rm_hue - c.h), 1.0 - abs(rm_hue - c.h))
                if(hd <= dh):
                    h = True
            if(us):
                if(rmcolor.s - ds < c.s < rmcolor.s + ds):
                    s = True
            if(uv):
                if(rmcolor.v - dv < c.v < rmcolor.v + dv):
                    v = True
            
            a = False
            if(uh and not us and not uv):
                if(h):
                    a = True
            elif(not uh and us and not uv):
                if(s):
                    a = True
            elif(not uh and not us and uv):
                if(v):
                    a = True
            elif(uh and us and not uv):
                if(h and s):
                    a = True
            elif(not uh and us and uv):
                if(s and v):
                    a = True
            elif(uh and not us and uv):
                if(h and v):
                    a = True
            elif(uh and us and uv):
                if(h and s and v):
                    a = True
            else:
                pass
            if(a):
                indexes.append(p['index'])
        
        log("selected: {} points".format(len(indexes)), 1)
        
        if(len(indexes) == 0):
            # self.report({'ERROR'}, "Nothing selected.")
            self.report({'INFO'}, "Nothing selected.")
        else:
            pcv.filter_remove_color_selection = True
            c = PCVManager.cache[pcv.uuid]
            c['selection_indexes'] = indexes
        
        context.area.tag_redraw()
        
        _d = datetime.timedelta(seconds=time.time() - _t)
        log("completed in {}.".format(_d), 1)
        
        return {'FINISHED'}


class PCV_OT_filter_remove_color_deselect(Operator):
    bl_idname = "point_cloud_visualizer.filter_remove_color_deselect"
    bl_label = "Deselect"
    bl_description = "Deselect points"
    
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
                        if(pcv.filter_remove_color_selection):
                            if('selection_indexes' in v.keys()):
                                ok = True
        return ok
    
    def execute(self, context):
        pcv = context.object.point_cloud_visualizer
        c = PCVManager.cache[pcv.uuid]
        
        pcv.filter_remove_color_selection = False
        del c['selection_indexes']
        
        context.area.tag_redraw()
        
        return {'FINISHED'}


class PCV_OT_filter_remove_color_delete_selected(Operator):
    bl_idname = "point_cloud_visualizer.filter_remove_color_delete_selected"
    bl_label = "Delete Selected"
    bl_description = "Remove selected points"
    
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
                        if(pcv.filter_remove_color_selection):
                            if('selection_indexes' in v.keys()):
                                ok = True
        return ok
    
    def execute(self, context):
        pcv = context.object.point_cloud_visualizer
        c = PCVManager.cache[pcv.uuid]
        vs = c['vertices']
        ns = c['normals']
        cs = c['colors']
        indexes = c['selection_indexes']
        vs = np.delete(vs, indexes, axis=0, )
        ns = np.delete(ns, indexes, axis=0, )
        cs = np.delete(cs, indexes, axis=0, )
        
        PCVManager.update(pcv.uuid, vs, ns, cs, )
        
        pcv.filter_remove_color_selection = False
        del c['selection_indexes']
        
        return {'FINISHED'}


class PCV_OT_filter_merge(Operator):
    bl_idname = "point_cloud_visualizer.filter_merge"
    bl_label = "Merge With Other PLY"
    bl_description = "Merge with other ply file"
    
    filename_ext = ".ply"
    filter_glob: StringProperty(default="*.ply", options={'HIDDEN'}, )
    filepath: StringProperty(name="File Path", default="", description="", maxlen=1024, subtype='FILE_PATH', )
    order = ["filepath", ]
    
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
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}
    
    def execute(self, context):
        pcv = context.object.point_cloud_visualizer
        addon_prefs = preferences()
        
        filepath = self.filepath
        h, t = os.path.split(filepath)
        n, e = os.path.splitext(t)
        if(e != '.ply'):
            self.report({'ERROR'}, "File at '{}' seems not to be a PLY file.".format(filepath))
            return {'CANCELLED'}
        
        points = []
        try:
            points = io_ply.PlyPointCloudReader(filepath).points
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        if(len(points) == 0):
            self.report({'ERROR'}, "No vertices loaded from file at {}".format(filepath))
            return {'CANCELLED'}
        
        if(not set(('x', 'y', 'z')).issubset(points.dtype.names)):
            # this is very unlikely..
            self.report({'ERROR'}, "Loaded data seems to miss vertex locations.")
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
            # preferences = bpy.context.preferences
            # addon_prefs = preferences.addons[__name__].preferences
            if(addon_prefs.convert_16bit_colors and points['red'].dtype == 'uint16'):
                r8 = (points['red'] / 256).astype('uint8')
                g8 = (points['green'] / 256).astype('uint8')
                b8 = (points['blue'] / 256).astype('uint8')
                if(addon_prefs.gamma_correct_16bit_colors):
                    cs = np.column_stack(((r8 / 255) ** (1 / 2.2),
                                          (g8 / 255) ** (1 / 2.2),
                                          (b8 / 255) ** (1 / 2.2),
                                          np.ones(len(points), dtype=float, ), ))
                else:
                    cs = np.column_stack((r8 / 255, g8 / 255, b8 / 255, np.ones(len(points), dtype=float, ), ))
                cs = cs.astype(np.float32)
            else:
                # 'uint8'
                cs = np.column_stack((points['red'] / 255, points['green'] / 255, points['blue'] / 255, np.ones(len(points), dtype=float, ), ))
                cs = cs.astype(np.float32)
        else:
            n = len(points)
            # preferences = bpy.context.preferences
            # addon_prefs = preferences.addons[__name__].preferences
            col = addon_prefs.default_vertex_color[:]
            col = tuple([c ** (1 / 2.2) for c in col]) + (1.0, )
            cs = np.column_stack((np.full(n, col[0], dtype=np.float32, ),
                                  np.full(n, col[1], dtype=np.float32, ),
                                  np.full(n, col[2], dtype=np.float32, ),
                                  np.ones(n, dtype=np.float32, ), ))
        
        # append
        c = PCVManager.cache[pcv.uuid]
        ovs = c['vertices']
        ons = c['normals']
        ocs = c['colors']
        vs = np.concatenate((ovs, vs, ))
        ns = np.concatenate((ons, ns, ))
        cs = np.concatenate((ocs, cs, ))
        
        # preferences = bpy.context.preferences
        # addon_prefs = preferences.addons[__name__].preferences
        if(addon_prefs.shuffle_points):
            l = len(vs)
            dt = [('x', '<f8'), ('y', '<f8'), ('z', '<f8'), ('nx', '<f8'), ('ny', '<f8'), ('nz', '<f8'), ('red', '<f8'), ('green', '<f8'), ('blue', '<f8'), ('alpha', '<f8'), ]
            a = np.empty(l, dtype=dt, )
            a['x'] = vs[:, 0]
            a['y'] = vs[:, 1]
            a['z'] = vs[:, 2]
            a['nx'] = ns[:, 0]
            a['ny'] = ns[:, 1]
            a['nz'] = ns[:, 2]
            a['red'] = cs[:, 0]
            a['green'] = cs[:, 1]
            a['blue'] = cs[:, 2]
            a['alpha'] = cs[:, 3]
            np.random.shuffle(a)
            vs = np.column_stack((a['x'], a['y'], a['z'], ))
            ns = np.column_stack((a['nx'], a['ny'], a['nz'], ))
            cs = np.column_stack((a['red'], a['green'], a['blue'], a['alpha'], ))
            vs = vs.astype(np.float32)
            ns = ns.astype(np.float32)
            cs = cs.astype(np.float32)
        
        PCVManager.update(pcv.uuid, vs, ns, cs, )
        
        return {'FINISHED'}


class PCV_OT_filter_join(Operator):
    bl_idname = "point_cloud_visualizer.filter_join"
    bl_label = "Join"
    bl_description = "Join with another PCV instance cloud"
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        
        pcv = context.object.point_cloud_visualizer
        pcv2 = None
        ok = False
        for k, v in PCVManager.cache.items():
            if(v['uuid'] == pcv.uuid):
                if(v['ready']):
                    if(v['draw']):
                        o = pcv.filter_join_object
                        if(o is not None):
                            pcv2 = o.point_cloud_visualizer
                break
        if(pcv2 is not None):
            for k, v in PCVManager.cache.items():
                if(v['uuid'] == pcv2.uuid):
                    if(v['ready']):
                        ok = True
                    break
        return ok
    
    def execute(self, context):
        pcv = context.object.point_cloud_visualizer
        pcv2 = pcv.filter_join_object.point_cloud_visualizer
        addon_prefs = preferences()
        
        c = PCVManager.cache[pcv.uuid]
        c2 = PCVManager.cache[pcv2.uuid]
        
        ovs = c['vertices']
        ons = c['normals']
        ocs = c['colors']
        nvs = c2['vertices']
        nns = c2['normals']
        ncs = c2['colors']
        
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
        
        nvs, nns = apply_matrix(pcv.filter_join_object.matrix_world, nvs, nns, )
        nvs, nns = apply_matrix(context.object.matrix_world.inverted(), nvs, nns, )
        
        vs = np.concatenate((ovs, nvs, ))
        ns = np.concatenate((ons, nns, ))
        cs = np.concatenate((ocs, ncs, ))
        
        # preferences = bpy.context.preferences
        # addon_prefs = preferences.addons[__name__].preferences
        if(addon_prefs.shuffle_points):
            l = len(vs)
            dt = [('x', '<f8'), ('y', '<f8'), ('z', '<f8'), ('nx', '<f8'), ('ny', '<f8'), ('nz', '<f8'), ('red', '<f8'), ('green', '<f8'), ('blue', '<f8'), ('alpha', '<f8'), ]
            a = np.empty(l, dtype=dt, )
            a['x'] = vs[:, 0]
            a['y'] = vs[:, 1]
            a['z'] = vs[:, 2]
            a['nx'] = ns[:, 0]
            a['ny'] = ns[:, 1]
            a['nz'] = ns[:, 2]
            a['red'] = cs[:, 0]
            a['green'] = cs[:, 1]
            a['blue'] = cs[:, 2]
            a['alpha'] = cs[:, 3]
            np.random.shuffle(a)
            vs = np.column_stack((a['x'], a['y'], a['z'], ))
            ns = np.column_stack((a['nx'], a['ny'], a['nz'], ))
            cs = np.column_stack((a['red'], a['green'], a['blue'], a['alpha'], ))
            vs = vs.astype(np.float32)
            ns = ns.astype(np.float32)
            cs = cs.astype(np.float32)
        
        PCVManager.update(pcv.uuid, vs, ns, cs, )
        
        c2['draw'] = False
        context.area.tag_redraw()
        
        return {'FINISHED'}


class PCV_OT_filter_boolean_intersect(Operator):
    bl_idname = "point_cloud_visualizer.filter_boolean_intersect"
    bl_label = "Intersect"
    bl_description = ""
    
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
                        o = pcv.filter_boolean_object
                        if(o is not None):
                            if(o.type in ('MESH', 'CURVE', 'SURFACE', 'FONT', )):
                                ok = True
        return ok
    
    def execute(self, context):
        log("Intersect:", 0)
        _t = time.time()
        
        pcv = context.object.point_cloud_visualizer
        o = pcv.filter_boolean_object
        if(o is None):
            raise Exception()
        
        def apply_matrix(vs, ns, matrix, ):
            matrot = matrix.decompose()[1].to_matrix().to_4x4()
            dtv = vs.dtype
            dtn = ns.dtype
            rvs = np.zeros(vs.shape, dtv)
            rns = np.zeros(ns.shape, dtn)
            for i in range(len(vs)):
                co = matrix @ Vector(vs[i])
                no = matrot @ Vector(ns[i])
                rvs[i] = np.array(co.to_tuple(), dtv)
                rns[i] = np.array(no.to_tuple(), dtn)
            return rvs, rns
        
        c = PCVManager.cache[pcv.uuid]
        vs = c['vertices']
        ns = c['normals']
        cs = c['colors']
        
        # apply parent matrix to points
        m = c['object'].matrix_world.copy()
        # vs, ns = apply_matrix(vs, ns, m)
        
        sc = context.scene
        depsgraph = bpy.context.evaluated_depsgraph_get()
        # make target
        tmp_mesh = o.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph, )
        target_mesh = tmp_mesh.copy()
        target_mesh.name = 'target_mesh_{}'.format(pcv.uuid)
        target_mesh.transform(o.matrix_world.copy())
        target_mesh.transform(m.inverted())
        view_layer = context.view_layer
        collection = view_layer.active_layer_collection.collection
        target = bpy.data.objects.new('target_mesh_{}'.format(pcv.uuid), target_mesh)
        collection.objects.link(target)
        # still no idea, have to read about it..
        depsgraph.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        
        # use object bounding box for fast check if point can even be inside/outside of mesh and then use ray casting etc..
        bounds = [bv[:] for bv in target.bound_box]
        xmin = min([v[0] for v in bounds])
        xmax = max([v[0] for v in bounds])
        ymin = min([v[1] for v in bounds])
        ymax = max([v[1] for v in bounds])
        zmin = min([v[2] for v in bounds])
        zmax = max([v[2] for v in bounds])
        
        def is_in_bound_box(v):
            x = False
            if(xmin < v[0] < xmax):
                x = True
            y = False
            if(ymin < v[1] < ymax):
                y = True
            z = False
            if(zmin < v[2] < zmax):
                z = True
            if(x and y and z):
                return True
            return False
        
        # v1 raycasting in three axes and counting hits
        def is_point_inside_mesh_v1(p, o, ):
            axes = [Vector((1.0, 0.0, 0.0)), Vector((0.0, 1.0, 0.0)), Vector((0.0, 0.0, 1.0)), ]
            r = False
            for a in axes:
                v = p
                c = 0
                while True:
                    _, l, n, i = o.ray_cast(v, v + a * 10000.0)
                    if(i == -1):
                        break
                    c += 1
                    v = l + a * 0.00001
                if(c % 2 == 0):
                    r = True
                    break
            return not r
        
        # v2 raycasting and counting hits
        def is_point_inside_mesh_v2(p, o, shift=0.000001, search_distance=10000.0, ):
            p = Vector(p)
            path = []
            hits = 0
            direction = Vector((0.0, 0.0, 1.0))
            opposite_direction = direction.copy()
            opposite_direction.negate()
            path.append(p)
            loc = p
            ok = True
            
            def shift_vector(co, no, v):
                return co + (no.normalized() * v)
            
            while(ok):
                end = shift_vector(loc, direction, search_distance)
                loc = shift_vector(loc, direction, shift)
                _, loc, nor, ind = o.ray_cast(loc, end)
                if(ind != -1):
                    a = shift_vector(loc, opposite_direction, shift)
                    path.append(a)
                    hits += 1
                else:
                    ok = False
            if(hits % 2 == 1):
                return True
            return False
        
        # v3 closest point on mesh normal checking
        def is_point_inside_mesh_v3(p, o, ):
            _, loc, nor, ind = o.closest_point_on_mesh(p)
            if(ind != -1):
                v = loc - p
                d = v.dot(nor)
                if(d >= 0):
                    return True
            return False
        
        # if(debug_mode()):
        #     import cProfile
        #     import pstats
        #     import io
        #     pr = cProfile.Profile()
        #     pr.enable()
        
        indexes = []
        prgs = Progress(len(vs), indent=1, prefix="> ")
        for i, v in enumerate(vs):
            prgs.step()
            vv = Vector(v)
            '''
            inside1 = is_point_inside_mesh_v1(vv, target, )
            inside2 = is_point_inside_mesh_v2(vv, target, )
            inside3 = is_point_inside_mesh_v3(vv, target, )
            # intersect
            if(not inside1 and not inside2 and not inside3):
                indexes.append(i)
            # # exclude
            # if(inside1 and inside2 and inside3):
            #     indexes.append(i)
            '''
            
            in_bb = is_in_bound_box(v)
            if(not in_bb):
                # is not in bounds i can skip completely
                indexes.append(i)
                continue
            
            inside3 = is_point_inside_mesh_v3(vv, target, )
            if(not inside3):
                indexes.append(i)
        
        # if(debug_mode()):
        #     pr.disable()
        #     s = io.StringIO()
        #     sortby = 'cumulative'
        #     ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        #     ps.print_stats()
        #     print(s.getvalue())
        
        c = PCVManager.cache[pcv.uuid]
        vs = c['vertices']
        ns = c['normals']
        cs = c['colors']
        vs = np.delete(vs, indexes, axis=0, )
        ns = np.delete(ns, indexes, axis=0, )
        cs = np.delete(cs, indexes, axis=0, )
        
        log("removed: {} points".format(len(indexes)), 1)
        
        # cleanup
        collection.objects.unlink(target)
        bpy.data.objects.remove(target)
        bpy.data.meshes.remove(target_mesh)
        o.to_mesh_clear()
        
        # put to cache..
        pcv = context.object.point_cloud_visualizer
        PCVManager.update(pcv.uuid, vs, ns, cs, )
        
        _d = datetime.timedelta(seconds=time.time() - _t)
        log("completed in {}.".format(_d), 1)
        
        return {'FINISHED'}


class PCV_OT_filter_boolean_exclude(Operator):
    bl_idname = "point_cloud_visualizer.filter_boolean_exclude"
    bl_label = "Exclude"
    bl_description = ""
    
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
                        o = pcv.filter_boolean_object
                        if(o is not None):
                            if(o.type in ('MESH', 'CURVE', 'SURFACE', 'FONT', )):
                                ok = True
        return ok
    
    def execute(self, context):
        log("Exclude:", 0)
        _t = time.time()
        
        pcv = context.object.point_cloud_visualizer
        o = pcv.filter_boolean_object
        if(o is None):
            raise Exception()
        
        def apply_matrix(vs, ns, matrix, ):
            matrot = matrix.decompose()[1].to_matrix().to_4x4()
            dtv = vs.dtype
            dtn = ns.dtype
            rvs = np.zeros(vs.shape, dtv)
            rns = np.zeros(ns.shape, dtn)
            for i in range(len(vs)):
                co = matrix @ Vector(vs[i])
                no = matrot @ Vector(ns[i])
                rvs[i] = np.array(co.to_tuple(), dtv)
                rns[i] = np.array(no.to_tuple(), dtn)
            return rvs, rns
        
        c = PCVManager.cache[pcv.uuid]
        vs = c['vertices']
        ns = c['normals']
        cs = c['colors']
        
        # apply parent matrix to points
        m = c['object'].matrix_world.copy()
        # vs, ns = apply_matrix(vs, ns, m)
        
        sc = context.scene
        depsgraph = bpy.context.evaluated_depsgraph_get()
        # make target
        tmp_mesh = o.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph, )
        target_mesh = tmp_mesh.copy()
        target_mesh.name = 'target_mesh_{}'.format(pcv.uuid)
        target_mesh.transform(o.matrix_world.copy())
        target_mesh.transform(m.inverted())
        view_layer = context.view_layer
        collection = view_layer.active_layer_collection.collection
        target = bpy.data.objects.new('target_mesh_{}'.format(pcv.uuid), target_mesh)
        collection.objects.link(target)
        # still no idea, have to read about it..
        depsgraph.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        
        # use object bounding box for fast check if point can even be inside/outside of mesh and then use ray casting etc..
        bounds = [bv[:] for bv in target.bound_box]
        xmin = min([v[0] for v in bounds])
        xmax = max([v[0] for v in bounds])
        ymin = min([v[1] for v in bounds])
        ymax = max([v[1] for v in bounds])
        zmin = min([v[2] for v in bounds])
        zmax = max([v[2] for v in bounds])
        
        def is_in_bound_box(v):
            x = False
            if(xmin < v[0] < xmax):
                x = True
            y = False
            if(ymin < v[1] < ymax):
                y = True
            z = False
            if(zmin < v[2] < zmax):
                z = True
            if(x and y and z):
                return True
            return False
        
        # v1 raycasting in three axes and counting hits
        def is_point_inside_mesh_v1(p, o, ):
            axes = [Vector((1.0, 0.0, 0.0)), Vector((0.0, 1.0, 0.0)), Vector((0.0, 0.0, 1.0)), ]
            r = False
            for a in axes:
                v = p
                c = 0
                while True:
                    _, l, n, i = o.ray_cast(v, v + a * 10000.0)
                    if(i == -1):
                        break
                    c += 1
                    v = l + a * 0.00001
                if(c % 2 == 0):
                    r = True
                    break
            return not r
        
        # v2 raycasting and counting hits
        def is_point_inside_mesh_v2(p, o, shift=0.000001, search_distance=10000.0, ):
            p = Vector(p)
            path = []
            hits = 0
            direction = Vector((0.0, 0.0, 1.0))
            opposite_direction = direction.copy()
            opposite_direction.negate()
            path.append(p)
            loc = p
            ok = True
            
            def shift_vector(co, no, v):
                return co + (no.normalized() * v)
            
            while(ok):
                end = shift_vector(loc, direction, search_distance)
                loc = shift_vector(loc, direction, shift)
                _, loc, nor, ind = o.ray_cast(loc, end)
                if(ind != -1):
                    a = shift_vector(loc, opposite_direction, shift)
                    path.append(a)
                    hits += 1
                else:
                    ok = False
            if(hits % 2 == 1):
                return True
            return False
        
        # v3 closest point on mesh normal checking
        def is_point_inside_mesh_v3(p, o, ):
            _, loc, nor, ind = o.closest_point_on_mesh(p)
            if(ind != -1):
                v = loc - p
                d = v.dot(nor)
                if(d >= 0):
                    return True
            return False
        
        indexes = []
        prgs = Progress(len(vs), indent=1, prefix="> ")
        for i, v in enumerate(vs):
            prgs.step()
            vv = Vector(v)
            '''
            inside1 = is_point_inside_mesh_v1(vv, target, )
            inside2 = is_point_inside_mesh_v2(vv, target, )
            inside3 = is_point_inside_mesh_v3(vv, target, )
            # # intersect
            # if(not inside1 and not inside2 and not inside3):
            #     indexes.append(i)
            # exclude
            if(inside1 and inside2 and inside3):
                indexes.append(i)
            '''
            
            in_bb = is_in_bound_box(v)
            if(in_bb):
                # indexes.append(i)
                # continue
                
                # is in bound so i can check further
                inside3 = is_point_inside_mesh_v3(vv, target, )
                if(inside3):
                    indexes.append(i)
            
            # inside3 = is_point_inside_mesh_v3(vv, target, )
            # if(inside3):
            #     indexes.append(i)
        
        c = PCVManager.cache[pcv.uuid]
        vs = c['vertices']
        ns = c['normals']
        cs = c['colors']
        vs = np.delete(vs, indexes, axis=0, )
        ns = np.delete(ns, indexes, axis=0, )
        cs = np.delete(cs, indexes, axis=0, )
        
        log("removed: {} points".format(len(indexes)), 1)
        
        # cleanup
        collection.objects.unlink(target)
        bpy.data.objects.remove(target)
        bpy.data.meshes.remove(target_mesh)
        o.to_mesh_clear()
        
        # put to cache..
        pcv = context.object.point_cloud_visualizer
        PCVManager.update(pcv.uuid, vs, ns, cs, )
        
        _d = datetime.timedelta(seconds=time.time() - _t)
        log("completed in {}.".format(_d), 1)
        
        return {'FINISHED'}


class PCV_OT_color_adjustment_shader_reset(Operator):
    bl_idname = "point_cloud_visualizer.color_adjustment_shader_reset"
    bl_label = "Reset"
    bl_description = "Reset color adjustment values"
    
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
                        if(pcv.color_adjustment_shader_enabled):
                            ok = True
        return ok
    
    def execute(self, context):
        pcv = context.object.point_cloud_visualizer
        
        pcv.color_adjustment_shader_exposure = 0.0
        pcv.color_adjustment_shader_gamma = 1.0
        pcv.color_adjustment_shader_brightness = 0.0
        pcv.color_adjustment_shader_contrast = 1.0
        pcv.color_adjustment_shader_hue = 0.0
        pcv.color_adjustment_shader_saturation = 0.0
        pcv.color_adjustment_shader_value = 0.0
        pcv.color_adjustment_shader_invert = False
        
        for area in bpy.context.screen.areas:
            if(area.type == 'VIEW_3D'):
                area.tag_redraw()
        
        return {'FINISHED'}


class PCV_OT_color_adjustment_shader_apply(Operator):
    bl_idname = "point_cloud_visualizer.color_adjustment_shader_apply"
    bl_label = "Apply"
    bl_description = "Apply color adjustments to points, reset and exit"
    
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
                        if(pcv.color_adjustment_shader_enabled):
                            ok = True
        return ok
    
    def execute(self, context):
        pcv = context.object.point_cloud_visualizer
        
        c = PCVManager.cache[pcv.uuid]
        vs = c['vertices']
        ns = c['normals']
        cs = c['colors']
        
        cs = cs * (2 ** pcv.color_adjustment_shader_exposure)
        cs = np.clip(cs, 0.0, 1.0, )
        cs = cs ** (1 / pcv.color_adjustment_shader_gamma)
        cs = np.clip(cs, 0.0, 1.0, )
        cs = (cs - 0.5) * pcv.color_adjustment_shader_contrast + 0.5 + pcv.color_adjustment_shader_brightness
        cs = np.clip(cs, 0.0, 1.0, )
        
        h = pcv.color_adjustment_shader_hue
        s = pcv.color_adjustment_shader_saturation
        v = pcv.color_adjustment_shader_value
        if(h > 1.0):
            h = h % 1.0
        for _i, ca in enumerate(cs):
            col = Color(ca[:3])
            _h, _s, _v = col.hsv
            _h = (_h + h) % 1.0
            _s += s
            _v += v
            col.hsv = (_h, _s, _v)
            cs[_i][0] = col.r
            cs[_i][1] = col.g
            cs[_i][2] = col.b
        cs = np.clip(cs, 0.0, 1.0, )
        
        if(pcv.color_adjustment_shader_invert):
            cs = 1.0 - cs
        cs = np.clip(cs, 0.0, 1.0, )
        
        bpy.ops.point_cloud_visualizer.color_adjustment_shader_reset()
        pcv.color_adjustment_shader_enabled = False
        
        if('extra' in c.keys()):
            del c['extra']
        
        PCVManager.update(pcv.uuid, vs, ns, cs, )
        
        return {'FINISHED'}
