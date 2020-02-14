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

import time
import datetime
import math
import numpy as np
import sys
import random
import statistics

import bpy
import bmesh
from bpy.types import Operator
from mathutils import Matrix, Vector, Quaternion, Color
from mathutils.kdtree import KDTree
from mathutils.geometry import barycentric_transform
from mathutils.interpolate import poly_3d_calc
from mathutils.bvhtree import BVHTree

from .debug import log, debug_mode, Progress
from .machine import PCVManager, PCVControl


class PCVTriangleSurfaceSampler():
    def __init__(self, context, o, num_samples, rnd, colorize=None, constant_color=None, vcols=None, uvtex=None, vgroup=None, exact_number_of_points=False, ):
        log("{}:".format(self.__class__.__name__), 0)
        
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
        
        def distance(a, b, ):
            return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2) ** 0.5
        
        def random_point_in_triangle(a, b, c, ):
            r1 = rnd.random()
            r2 = rnd.random()
            p = (1 - math.sqrt(r1)) * a + (math.sqrt(r1) * (1 - r2)) * b + (math.sqrt(r1) * r2) * c
            return p
        
        depsgraph = context.evaluated_depsgraph_get()
        if(o.modifiers):
            owner = o.evaluated_get(depsgraph)
            me = owner.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph, )
        else:
            owner = o
            me = owner.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph, )
        
        bm = bmesh.new()
        bm.from_mesh(me)
        bmesh.ops.triangulate(bm, faces=bm.faces)
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        
        if(len(bm.faces) == 0):
            raise Exception("Mesh has no faces")
        
        areas = tuple([p.calc_area() for p in bm.faces])
        if(sum(areas) == 0.0):
            raise Exception("Mesh surface area is zero")
        area_min = min(areas)
        area_max = max(areas)
        avg_ppf = num_samples / len(areas)
        area_med = statistics.median(areas)
        nums = []
        for p in bm.faces:
            r = p.calc_area() / area_med
            nums.append(avg_ppf * r)
        
        max_ppf = max(nums)
        min_ppf = min(nums)
        
        vs = []
        ns = []
        cs = []
        
        if(colorize == 'UVTEX'):
            try:
                if(o.active_material is None):
                    raise Exception("Cannot find active material")
                uvtexnode = o.active_material.node_tree.nodes.active
                if(uvtexnode is None):
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
                raise Exception(str(e))
        if(colorize == 'VCOLS'):
            try:
                col_layer = bm.loops.layers.color.active
                if(col_layer is None):
                    raise Exception()
            except Exception:
                raise Exception("Cannot find active vertex colors")
        if(colorize in ('GROUP_MONO', 'GROUP_COLOR')):
            try:
                group_layer = bm.verts.layers.deform.active
                if(group_layer is None):
                    raise Exception()
                group_layer_index = o.vertex_groups.active.index
            except Exception:
                raise Exception("Cannot find active vertex group")
        
        def generate(poly, vs, ns, cs, override_num=None, ):
            ps = poly.verts
            tri = (ps[0].co, ps[1].co, ps[2].co)
            # if num is 0, it can happen when mesh has large and very small polygons, increase number of samples and eventually all polygons gets covered
            num = int(round(remap(poly.calc_area(), area_min, area_max, min_ppf, max_ppf)))
            if(override_num is not None):
                num = override_num
            for i in range(num):
                v = random_point_in_triangle(*tri)
                vs.append(v.to_tuple())
                if(poly.smooth):
                    a = poly.verts[0].normal
                    b = poly.verts[1].normal
                    c = poly.verts[2].normal
                    nws = poly_3d_calc([a, b, c, ], v)
                    
                    nx = a.x * nws[0] + b.x * nws[1] + c.x * nws[2]
                    ny = a.y * nws[0] + b.y * nws[1] + c.y * nws[2]
                    nz = a.z * nws[0] + b.z * nws[1] + c.z * nws[2]
                    normal = Vector((nx, ny, nz)).normalized()
                    ns.append(normal.to_tuple())
                    
                    # n = Vector((0.0, 0.0, 0.0, ))
                    # n += nws[0] * a
                    # n += nws[1] * b
                    # n += nws[2] * c
                    # n = n / 3
                    # n = n.normalized()
                    # ns.append(n.to_tuple())
                    
                else:
                    ns.append(poly.normal.to_tuple())
                
                if(colorize is None):
                    cs.append((1.0, 0.0, 0.0, ))
                elif(colorize == 'CONSTANT'):
                    cs.append(constant_color)
                elif(colorize == 'VCOLS'):
                    ws = poly_3d_calc([poly.verts[0].co, poly.verts[1].co, poly.verts[2].co, ], v)
                    ac = poly.loops[0][col_layer][:3]
                    bc = poly.loops[1][col_layer][:3]
                    cc = poly.loops[2][col_layer][:3]
                    r = ac[0] * ws[0] + bc[0] * ws[1] + cc[0] * ws[2]
                    g = ac[1] * ws[0] + bc[1] * ws[1] + cc[1] * ws[2]
                    b = ac[2] * ws[0] + bc[2] * ws[1] + cc[2] * ws[2]
                    cs.append((r, g, b, ))
                elif(colorize == 'UVTEX'):
                    uvtriangle = []
                    for l in poly.loops:
                        uvtriangle.append(Vector(l[uvlayer].uv.to_tuple() + (0.0, )))
                    uvpoint = barycentric_transform(v, poly.verts[0].co, poly.verts[1].co, poly.verts[2].co, *uvtriangle, )
                    w, h = uvimage.size
                    # x,y % 1.0 to wrap around if uv coordinate is outside 0.0-1.0 range
                    x = int(round(remap(uvpoint.x % 1.0, 0.0, 1.0, 0, w - 1)))
                    y = int(round(remap(uvpoint.y % 1.0, 0.0, 1.0, 0, h - 1)))
                    cs.append(tuple(uvarray[y][x][:3].tolist()))
                elif(colorize == 'GROUP_MONO'):
                    ws = poly_3d_calc([poly.verts[0].co, poly.verts[1].co, poly.verts[2].co, ], v)
                    aw = poly.verts[0][group_layer].get(group_layer_index, 0.0)
                    bw = poly.verts[1][group_layer].get(group_layer_index, 0.0)
                    cw = poly.verts[2][group_layer].get(group_layer_index, 0.0)
                    m = aw * ws[0] + bw * ws[1] + cw * ws[2]
                    cs.append((m, m, m, ))
                elif(colorize == 'GROUP_COLOR'):
                    ws = poly_3d_calc([poly.verts[0].co, poly.verts[1].co, poly.verts[2].co, ], v)
                    aw = poly.verts[0][group_layer].get(group_layer_index, 0.0)
                    bw = poly.verts[1][group_layer].get(group_layer_index, 0.0)
                    cw = poly.verts[2][group_layer].get(group_layer_index, 0.0)
                    m = aw * ws[0] + bw * ws[1] + cw * ws[2]
                    hue = remap(1.0 - m, 0.0, 1.0, 0.0, 1 / 1.5)
                    c = Color()
                    c.hsv = (hue, 1.0, 1.0, )
                    cs.append((c.r, c.g, c.b, ))
        
        log("generating {} samples:".format(num_samples), 1)
        progress = Progress(len(bm.faces), indent=2, )
        for poly in bm.faces:
            progress.step()
            generate(poly, vs, ns, cs, )
        
        if(exact_number_of_points):
            if(len(vs) < num_samples):
                log("generating extra samples..", 1)
                while(len(vs) < num_samples):
                    # generate one sample in random face until full
                    poly = bm.faces[rnd.randrange(len(bm.faces))]
                    generate(poly, vs, ns, cs, override_num=1, )
            if(len(vs) > num_samples):
                log("throwing out extra samples..", 1)
                a = np.concatenate((vs, ns, cs), axis=1, )
                np.random.shuffle(a)
                a = a[:num_samples]
                # vs = np.column_stack((a[:, 0], a[:, 1], a[:, 2], ))
                # ns = np.column_stack((a[:, 3], a[:, 4], a[:, 5], ))
                # cs = np.column_stack((a[:, 6], a[:, 7], a[:, 8], ))
                vs = a[:, :3]
                ns = a[:, 3:6]
                cs = a[:, 6:]
        
        if(len(vs) == 0):
            raise Exception("No points generated, increase number of points or decrease minimal distance")
        
        # and shuffle..
        a = np.concatenate((vs, ns, cs), axis=1, )
        np.random.shuffle(a)
        vs = a[:, :3]
        ns = a[:, 3:6]
        cs = a[:, 6:]
        
        self.vs = vs[:]
        self.ns = ns[:]
        self.cs = cs[:]
        
        bm.free()
        owner.to_mesh_clear()


class PCVPoissonDiskSurfaceSampler():
    def __init__(self, context, o, rnd, minimal_distance, sampling_exponent=10, colorize=None, constant_color=None, vcols=None, uvtex=None, vgroup=None, ):
        log("{}:".format(self.__class__.__name__), 0)
        # pregenerate samples, normals and colors will be handled too
        num_presamples = int((sum([p.area for p in o.data.polygons]) / (minimal_distance ** 2)) * sampling_exponent)
        presampler = PCVTriangleSurfaceSampler(context, o, num_presamples, rnd, colorize=colorize, constant_color=constant_color, vcols=vcols, uvtex=uvtex, vgroup=vgroup, exact_number_of_points=False, )
        pre_vs = presampler.vs
        pre_ns = presampler.ns
        pre_cs = presampler.cs
        # join to points
        ppoints = []
        for i in range(len(pre_vs)):
            ppoints.append(tuple(pre_vs[i].tolist()) + tuple(pre_ns[i].tolist()) + tuple(pre_cs[i].tolist()))
        # keep track of used/unused points
        pbools = [True] * len(ppoints)
        
        def random_point():
            while(True):
                i = rnd.randint(0, len(ppoints) - 1)
                if(pbools[i]):
                    pbools[i] = False
                    return ppoints[i]
        
        accepted = []
        candidates = []
        # get starting point
        candidates.append(random_point())
        
        # make kdtree of all points for getting annulus points
        ptree = KDTree(len(ppoints))
        for i in range(len(ppoints)):
            ptree.insert(ppoints[i][:3], i)
        ptree.balance()
        
        def annulus_points(c):
            # generate k point with candidate annulus
            pool = []
            # points within radius minimal_distance
            ball = ptree.find_range(c[:3], minimal_distance)
            ball_indexes = set([i for _, i, _ in ball])
            # points within radius minimal_distance * 2
            ball2 = ptree.find_range(c[:3], minimal_distance * 2)
            ball2_indexes = set([i for _, i, _ in ball2])
            # difference between those two are points in annulus
            s = ball2_indexes.difference(ball_indexes)
            pool = []
            for i in s:
                if(pbools[i]):
                    # pool.append(tuple(ptree.data[i]))
                    pool.append(tuple(pre_vs[i].tolist()) + tuple(pre_ns[i].tolist()) + tuple(pre_cs[i].tolist()))
                    # don't reuse points, if point is ok, will be added to candidate, if not will not be used anymore
                    pbools[i] = False
                    # # isn't the 'hole' problem here?
                    # # this might leave a gap on mesh
                    # # because i am no generating new samples and with this i cut out pre generated neighbours?
                    # if(len(pool) >= k):
                    #     break
            return pool
        
        log("sampling..", 1)
        
        loop = True
        while(loop):
            # choose one random candidate
            c = candidates[rnd.randint(0, len(candidates) - 1)]
            # generate k points in candidate annulus
            ap = annulus_points(c)
            na = False
            for i, a in enumerate(ap):
                # this is not quite right to create kdtree each iteration..
                tps = candidates + accepted
                tpsl = len(tps)
                tree = KDTree(tpsl)
                for j in range(tpsl):
                    tree.insert(tps[j][:3], j)
                tree.balance()
                
                ball = tree.find_range(a[:3], minimal_distance)
                ball_indexes = set([i for _, i, _ in ball])
                if(len(ball) == 0):
                    # add if annulus points is acceptable as candidate - no other candidates or accepted are within minimal_distance
                    candidates.append(a)
                    na = True
            if(not na):
                # if no candidate or accepted is within all annulus points minimal_distance accept candidate and remove from candidates
                accepted.append(c)
                candidates.remove(c)
            # no candidates left, we end here
            if(len(candidates) == 0):
                loop = False
            
            if(debug_mode()):
                sys.stdout.write("\r")
                sys.stdout.write("        accepted: {0} | candidates: {1}  ".format(len(accepted), len(candidates)))
                sys.stdout.write("\b")
                sys.stdout.flush()
        
        if(debug_mode()):
            sys.stdout.write("\n")
        log("done..", 1)
        
        # split point data back
        vs = []
        ns = []
        cs = []
        for i in range(len(accepted)):
            vs.append(accepted[i][:3])
            ns.append(accepted[i][3:6])
            cs.append(accepted[i][6:])
        self.vs = vs
        self.ns = ns
        self.cs = cs


class PCVVertexSampler():
    def __init__(self, context, o, colorize=None, constant_color=None, vcols=None, uvtex=None, vgroup=None, ):
        log("{}:".format(self.__class__.__name__), 0)
        
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
        
        depsgraph = context.evaluated_depsgraph_get()
        if(o.modifiers):
            owner = o.evaluated_get(depsgraph)
            me = owner.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph, )
        else:
            owner = o
            me = owner.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph, )
        
        bm = bmesh.new()
        bm.from_mesh(me)
        bmesh.ops.triangulate(bm, faces=bm.faces)
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        
        if(len(bm.verts) == 0):
            raise Exception("Mesh has no vertices")
        if(colorize in ('UVTEX', 'VCOLS', )):
            if(len(bm.faces) == 0):
                raise Exception("Mesh has no faces")
        
        vs = []
        ns = []
        cs = []
        
        if(colorize == 'UVTEX'):
            try:
                if(o.active_material is None):
                    raise Exception("Cannot find active material")
                uvtexnode = o.active_material.node_tree.nodes.active
                if(uvtexnode is None):
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
                raise Exception(str(e))
        if(colorize == 'VCOLS'):
            try:
                col_layer = bm.loops.layers.color.active
                if(col_layer is None):
                    raise Exception()
            except Exception:
                raise Exception("Cannot find active vertex colors")
        if(colorize in ('GROUP_MONO', 'GROUP_COLOR')):
            try:
                group_layer = bm.verts.layers.deform.active
                if(group_layer is None):
                    raise Exception()
                group_layer_index = o.vertex_groups.active.index
            except Exception:
                raise Exception("Cannot find active vertex group")
        
        vs = []
        ns = []
        cs = []
        for v in bm.verts:
            vs.append(v.co.to_tuple())
            ns.append(v.normal.to_tuple())
            
            if(colorize is None):
                cs.append((1.0, 0.0, 0.0, ))
            elif(colorize == 'CONSTANT'):
                cs.append(constant_color)
            elif(colorize == 'VCOLS'):
                ls = v.link_loops
                r = 0.0
                g = 0.0
                b = 0.0
                for l in ls:
                    c = l[col_layer][:3]
                    r += c[0]
                    g += c[1]
                    b += c[2]
                r /= len(ls)
                g /= len(ls)
                b /= len(ls)
                cs.append((r, g, b, ))
            elif(colorize == 'UVTEX'):
                ls = v.link_loops
                w, h = uvimage.size
                r = 0.0
                g = 0.0
                b = 0.0
                for l in ls:
                    uvloc = l[uvlayer].uv.to_tuple()
                    # x,y % 1.0 to wrap around if uv coordinate is outside 0.0-1.0 range
                    x = int(round(remap(uvloc[0] % 1.0, 0.0, 1.0, 0, w - 1)))
                    y = int(round(remap(uvloc[1] % 1.0, 0.0, 1.0, 0, h - 1)))
                    c = tuple(uvarray[y][x][:3].tolist())
                    r += c[0]
                    g += c[1]
                    b += c[2]
                r /= len(ls)
                g /= len(ls)
                b /= len(ls)
                cs.append((r, g, b, ))
            elif(colorize == 'GROUP_MONO'):
                w = v[group_layer].get(group_layer_index, 0.0)
                cs.append((w, w, w, ))
            elif(colorize == 'GROUP_COLOR'):
                w = v[group_layer].get(group_layer_index, 0.0)
                hue = remap(1.0 - w, 0.0, 1.0, 0.0, 1 / 1.5)
                c = Color()
                c.hsv = (hue, 1.0, 1.0, )
                cs.append((c.r, c.g, c.b, ))
        
        # and shuffle..
        a = np.concatenate((vs, ns, cs), axis=1, )
        np.random.shuffle(a)
        vs = a[:, :3]
        ns = a[:, 3:6]
        cs = a[:, 6:]
        
        self.vs = vs[:]
        self.ns = ns[:]
        self.cs = cs[:]
        
        bm.free()
        owner.to_mesh_clear()


class PCVParticleSystemSampler():
    def __init__(self, context, o, alive_only=True, colorize=None, constant_color=None, vcols=None, uvtex=None, vgroup=None, ):
        log("{}:".format(self.__class__.__name__), 0)
        
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
        
        vs = []
        ns = []
        cs = []
        
        depsgraph = context.evaluated_depsgraph_get()
        if(o.modifiers):
            owner = o.evaluated_get(depsgraph)
            me = owner.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph, )
        else:
            owner = o
            me = owner.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph, )
        
        o = owner
        
        bm = bmesh.new()
        bm.from_mesh(me)
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        
        psys = o.particle_systems.active
        if(psys is None):
            raise Exception("Cannot find active particle system")
        if(len(psys.particles) == 0):
            raise Exception("Active particle system has 0 particles")
        if(alive_only):
            ok = False
            for p in psys.particles:
                if(p.alive_state == "ALIVE"):
                    ok = True
                    break
            if(not ok):
                raise Exception("Active particle system has 0 alive particles")
        
        mod = None
        uv_no = None
        if(colorize in ('VCOLS', 'UVTEX', 'GROUP_MONO', 'GROUP_COLOR', )):
            if(uvtex is None):
                raise Exception("Cannot find active uv layout on emitter")
            for m in o.modifiers:
                if(m.type == 'PARTICLE_SYSTEM'):
                    if(m.particle_system == psys):
                        mod = m
                        break
            uv_no = o.data.uv_layers.active_index
        
        if(colorize == 'UVTEX'):
            try:
                if(o.active_material is None):
                    raise Exception("Cannot find active material")
                uvtexnode = o.active_material.node_tree.nodes.active
                if(uvtexnode is None):
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
                raise Exception(str(e))
        if(colorize == 'VCOLS'):
            try:
                col_layer = bm.loops.layers.color.active
                if(col_layer is None):
                    raise Exception()
            except Exception:
                raise Exception("Cannot find active vertex colors")
            # for vertex groups, uv layout is required.. non-overlapping for best results
            uvlayer = bm.loops.layers.uv.active
            if(uvlayer is None):
                raise Exception("Cannot find active UV layout")
        if(colorize in ('GROUP_MONO', 'GROUP_COLOR')):
            try:
                group_layer = bm.verts.layers.deform.active
                if(group_layer is None):
                    raise Exception()
                group_layer_index = o.vertex_groups.active.index
            except Exception:
                raise Exception("Cannot find active vertex group")
            # for vertex groups, uv layout is required.. non-overlapping for best results
            uvlayer = bm.loops.layers.uv.active
            if(uvlayer is None):
                raise Exception("Cannot find active UV layout")
        
        # flatten mesh by uv and add original face indexes as int layer
        def simple_flatten_uv_mesh(bm):
            bm.faces.index_update()
            bm.faces.ensure_lookup_table()
            r = bmesh.new()
            ilayer = r.faces.layers.int.new('face_indexes')
            for f in bm.faces:
                fvs = []
                for i, l in enumerate(f.loops):
                    uv = l[uvlayer].uv
                    rv = r.verts.new((uv.x, uv.y, 0.0))
                    fvs.append(rv)
                rf = r.faces.new(fvs)
                rf[ilayer] = f.index
            r.faces.index_update()
            r.faces.ensure_lookup_table()
            return r
        
        if(colorize in ('VCOLS', 'GROUP_MONO', 'GROUP_COLOR', )):
            # i do not need extra flat mesh each time..
            bmf = simple_flatten_uv_mesh(bm)
            bmf_il = bmf.faces.layers.int['face_indexes']
            # with that i can ray_cast uv location and get original index of face i hit
            bmf_bvh = BVHTree.FromBMesh(bmf)
        
        for i, p in enumerate(psys.particles):
            if(p.alive_state != "ALIVE" and alive_only):
                continue
            
            vs.append(p.location.to_tuple())
            
            n = Vector((1.0, 0.0, 0.0, ))
            n.rotate(p.rotation)
            ns.append(n.normalized().to_tuple())
            
            if(colorize is None):
                cs.append((1.0, 0.0, 0.0, ))
            elif(colorize == 'CONSTANT'):
                cs.append(constant_color)
            elif(colorize == 'VCOLS'):
                uv = p.uv_on_emitter(mod)
                # intersect with flattened mesh and get original index of face from emitter mesh
                fl, fn, fi, di = bmf_bvh.ray_cast(Vector((uv.x, uv.y, -0.1)), Vector((0.0, 0.0, 1.0)), 0.2, )
                fpoly = bmf.faces[fi]
                oi = fpoly[bmf_il]
                poly = bm.faces[oi]
                # get final color from barycentric weights
                ws = poly_3d_calc([fv.co for fv in fpoly.verts], p.location)
                cols = [l[col_layer][:3] for l in poly.loops]
                r = sum([cc[0] * ws[ci] for ci, cc in enumerate(cols)])
                g = sum([cc[1] * ws[ci] for ci, cc in enumerate(cols)])
                b = sum([cc[2] * ws[ci] for ci, cc in enumerate(cols)])
                cs.append((r, g, b, ))
            elif(colorize == 'UVTEX'):
                uv = p.uv_on_emitter(mod)
                w, h = uvimage.size
                # x,y % 1.0 to wrap around if uv coordinate is outside 0.0-1.0 range
                x = int(round(remap(uv.x % 1.0, 0.0, 1.0, 0, w - 1)))
                y = int(round(remap(uv.y % 1.0, 0.0, 1.0, 0, h - 1)))
                cs.append(tuple(uvarray[y][x][:3].tolist()))
            elif(colorize == 'GROUP_MONO'):
                uv = p.uv_on_emitter(mod)
                fl, fn, fi, di = bmf_bvh.ray_cast(Vector((uv.x, uv.y, -0.1)), Vector((0.0, 0.0, 1.0)), 0.2, )
                fpoly = bmf.faces[fi]
                oi = fpoly[bmf_il]
                poly = bm.faces[oi]
                ws = poly_3d_calc([fv.co for fv in fpoly.verts], p.location)
                weights = [pv[group_layer].get(group_layer_index, 0.0) for pv in poly.verts]
                w = sum([ww * ws[wi] for wi, ww in enumerate(weights)])
                cs.append((w, w, w, ))
            elif(colorize == 'GROUP_COLOR'):
                uv = p.uv_on_emitter(mod)
                fl, fn, fi, di = bmf_bvh.ray_cast(Vector((uv.x, uv.y, -0.1)), Vector((0.0, 0.0, 1.0)), 0.2, )
                fpoly = bmf.faces[fi]
                oi = fpoly[bmf_il]
                poly = bm.faces[oi]
                ws = poly_3d_calc([fv.co for fv in fpoly.verts], p.location)
                weights = [pv[group_layer].get(group_layer_index, 0.0) for pv in poly.verts]
                w = sum([ww * ws[wi] for wi, ww in enumerate(weights)])
                hue = remap(1.0 - w, 0.0, 1.0, 0.0, 1 / 1.5)
                c = Color()
                c.hsv = (hue, 1.0, 1.0, )
                cs.append((c.r, c.g, c.b, ))
        
        a = np.concatenate((vs, ns, cs), axis=1, )
        np.random.shuffle(a)
        vs = a[:, :3]
        ns = a[:, 3:6]
        cs = a[:, 6:]
        
        self.vs = vs[:]
        self.ns = ns[:]
        self.cs = cs[:]
        
        if(colorize in ('VCOLS', 'GROUP_MONO', 'GROUP_COLOR', )):
            bmf.free()
        
        bm.free()
        owner.to_mesh_clear()


class PCVRandomVolumeSampler():
    def __init__(self, ob, num_samples, rnd, ):
        log("{}:".format(self.__class__.__name__), 0)
        
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
        
        def is_point_inside_mesh_v1(point, ob, mwi, ):
            axes = [Vector((1.0, 0.0, 0.0)), Vector((0.0, 1.0, 0.0)), Vector((0.0, 0.0, 1.0)), ]
            r = False
            for a in axes:
                o = mwi @ point
                c = 0
                while True:
                    _, l, n, i = ob.ray_cast(o, o + a * 10000.0)
                    if(i == -1):
                        break
                    c += 1
                    o = l + a * 0.00001
                if(c % 2 == 0):
                    r = True
                    break
            return not r
        
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
        
        def is_point_inside_mesh_v3(p, o):
            _, loc, nor, ind = o.closest_point_on_mesh(p)
            if(ind != -1):
                v = loc - p
                d = v.dot(nor)
                if(d >= 0):
                    return True, loc, nor, ind
            return False, None, None, None
        
        def max_values(o):
            bbox = [Vector(b) for b in o.bound_box]
            nx = sorted(bbox, key=lambda c: c.x, reverse=False)[0].x
            ny = sorted(bbox, key=lambda c: c.y, reverse=False)[0].y
            nz = sorted(bbox, key=lambda c: c.z, reverse=False)[0].z
            x = sorted(bbox, key=lambda c: c.x, reverse=True)[0].x
            y = sorted(bbox, key=lambda c: c.y, reverse=True)[0].y
            z = sorted(bbox, key=lambda c: c.z, reverse=True)[0].z
            return nx, ny, nz, x, y, z
        
        xmin, ymin, zmin, xmax, ymax, zmax = max_values(ob)
        
        def sample():
            x = remap(rnd.random(), 0.0, 1.0, xmin, xmax)
            y = remap(rnd.random(), 0.0, 1.0, ymin, ymax)
            z = remap(rnd.random(), 0.0, 1.0, zmin, zmax)
            return Vector((x, y, z))
        
        vs = []
        ns = []
        cs = []
        progress = Progress(num_samples, indent=1, )
        while(len(vs) < num_samples):
            a = sample()
            ok1 = is_point_inside_mesh_v1(a, ob, Matrix())
            ok2 = is_point_inside_mesh_v2(a, ob)
            ok3, location, normal, index = is_point_inside_mesh_v3(a, ob)
            if(ok1 and ok2 and ok3):
                vs.append(a.to_tuple())
                ns.append(normal.to_tuple())
                cs.append((1.0, 1.0, 1.0, ))
                progress.step()
        
        self.vs = vs[:]
        self.ns = ns[:]
        self.cs = cs[:]


class PCV_OT_generate_point_cloud(Operator):
    bl_idname = "point_cloud_visualizer.generate_from_mesh"
    bl_label = "Generate"
    bl_description = "Generate colored point cloud from mesh (or object convertible to mesh)"
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        
        # pcv = context.object.point_cloud_visualizer
        # if(pcv.uuid in PCVSequence.cache.keys()):
        #     return False
        # ok = False
        # for k, v in PCVManager.cache.items():
        #     if(v['uuid'] == pcv.uuid):
        #         if(v['ready']):
        #             if(v['draw']):
        #                 ok = True
        # return ok
        
        return True
    
    def execute(self, context):
        log("Generate From Mesh:", 0)
        _t = time.time()
        
        o = context.object
        
        if(o.type not in ('MESH', 'CURVE', 'SURFACE', 'FONT', )):
            self.report({'ERROR'}, "Object does not have geometry data.")
            return {'CANCELLED'}
        
        pcv = o.point_cloud_visualizer
        
        if(pcv.generate_source not in ('SURFACE', 'VERTICES', 'PARTICLES', )):
            self.report({'ERROR'}, "Source not implemented.")
            return {'CANCELLED'}
        
        n = pcv.generate_number_of_points
        r = random.Random(pcv.generate_seed)
        
        if(pcv.generate_colors in ('CONSTANT', 'UVTEX', 'VCOLS', 'GROUP_MONO', 'GROUP_COLOR', )):
            if(o.type in ('CURVE', 'SURFACE', 'FONT', ) and pcv.generate_colors != 'CONSTANT'):
                self.report({'ERROR'}, "Object type does not support UV textures, vertex colors or vertex groups.")
                return {'CANCELLED'}
        else:
            self.report({'ERROR'}, "Color generation not implemented.")
            return {'CANCELLED'}
        
        if(o.type != 'MESH'):
            vcols = None
            uvtex = None
            vgroup = None
        else:
            # all of following should return None is not available, at least for mesh object
            vcols = o.data.vertex_colors.active
            uvtex = o.data.uv_layers.active
            vgroup = o.vertex_groups.active
        
        generate_constant_color = tuple([c ** (1 / 2.2) for c in pcv.generate_constant_color]) + (1.0, )
        
        if(pcv.generate_source == 'VERTICES'):
            try:
                sampler = PCVVertexSampler(context, o,
                                           colorize=pcv.generate_colors,
                                           constant_color=generate_constant_color,
                                           vcols=vcols, uvtex=uvtex, vgroup=vgroup, )
            except Exception as e:
                self.report({'ERROR'}, str(e), )
                return {'CANCELLED'}
        elif(pcv.generate_source == 'SURFACE'):
            if(pcv.generate_algorithm == 'WEIGHTED_RANDOM_IN_TRIANGLE'):
                try:
                    sampler = PCVTriangleSurfaceSampler(context, o, n, r,
                                                        colorize=pcv.generate_colors,
                                                        constant_color=generate_constant_color,
                                                        vcols=vcols, uvtex=uvtex, vgroup=vgroup,
                                                        exact_number_of_points=pcv.generate_exact_number_of_points, )
                except Exception as e:
                    self.report({'ERROR'}, str(e), )
                    return {'CANCELLED'}
            elif(pcv.generate_algorithm == 'POISSON_DISK_SAMPLING'):
                try:
                    sampler = PCVPoissonDiskSurfaceSampler(context, o, r, minimal_distance=pcv.generate_minimal_distance,
                                                           sampling_exponent=pcv.generate_sampling_exponent,
                                                           colorize=pcv.generate_colors,
                                                           constant_color=generate_constant_color,
                                                           vcols=vcols, uvtex=uvtex, vgroup=vgroup, )
                except Exception as e:
                    self.report({'ERROR'}, str(e), )
                    return {'CANCELLED'}
            else:
                self.report({'ERROR'}, "Algorithm not implemented.")
                return {'CANCELLED'}
            
        elif(pcv.generate_source == 'PARTICLES'):
            try:
                alive_only = True
                if(pcv.generate_source_psys == 'ALL'):
                    alive_only = False
                sampler = PCVParticleSystemSampler(context, o, alive_only=alive_only,
                                                   colorize=pcv.generate_colors,
                                                   constant_color=generate_constant_color,
                                                   vcols=vcols, uvtex=uvtex, vgroup=vgroup, )
            except Exception as e:
                self.report({'ERROR'}, str(e), )
                return {'CANCELLED'}
        else:
            self.report({'ERROR'}, "Source type not implemented.")
            return {'CANCELLED'}
        
        vs = sampler.vs
        ns = sampler.ns
        cs = sampler.cs
        
        log("generated {} points.".format(len(vs)), 1)
        
        ok = False
        for k, v in PCVManager.cache.items():
            if(v['uuid'] == pcv.uuid):
                if(v['ready']):
                    if(v['draw']):
                        ok = True
        if(ok):
            bpy.ops.point_cloud_visualizer.erase()
        
        c = PCVControl(o)
        c.draw(vs, ns, cs)
        
        if(debug_mode()):
            o.display_type = 'BOUNDS'
        
        _d = datetime.timedelta(seconds=time.time() - _t)
        log("completed in {}.".format(_d), 1)
        
        return {'FINISHED'}


class PCV_OT_generate_volume_point_cloud(Operator):
    bl_idname = "point_cloud_visualizer.generate_volume_from_mesh"
    bl_label = "Generate Volume"
    bl_description = "Generate colored point cloud in mesh (or object convertible to mesh) volume"
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        
        # pcv = context.object.point_cloud_visualizer
        # if(pcv.uuid in PCVSequence.cache.keys()):
        #     return False
        # ok = False
        # for k, v in PCVManager.cache.items():
        #     if(v['uuid'] == pcv.uuid):
        #         if(v['ready']):
        #             if(v['draw']):
        #                 ok = True
        # return ok
        
        return True
    
    def execute(self, context):
        log("Generate From Mesh:", 0)
        _t = time.time()
        
        o = context.object
        pcv = o.point_cloud_visualizer
        n = pcv.generate_number_of_points
        r = random.Random(pcv.generate_seed)
        g = PCVRandomVolumeSampler(o, n, r, )
        vs = g.vs
        ns = g.ns
        cs = g.cs
        
        log("generated {} points.".format(len(vs)), 1)
        
        ok = False
        for k, v in PCVManager.cache.items():
            if(v['uuid'] == pcv.uuid):
                if(v['ready']):
                    if(v['draw']):
                        ok = True
        if(ok):
            bpy.ops.point_cloud_visualizer.erase()
        
        c = PCVControl(o)
        c.draw(vs, ns, cs)
        
        if(debug_mode()):
            o.display_type = 'BOUNDS'
        
        _d = datetime.timedelta(seconds=time.time() - _t)
        log("completed in {}.".format(_d), 1)
        
        return {'FINISHED'}
