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

bl_info = {"name": "Tube UV Unwrap",
           "description": "UV unwrap tube-like meshes (all quads, no caps, fixed number of vertices in each ring)",
           "author": "Jakub Uhlik",
           "version": (0, 3, 0),
           "blender": (2, 80, 0),
           "location": "Edit mode > UV > Tube UV Unwrap",
           "warning": "",
           "wiki_url": "",
           "tracker_url": "",
           "category": "UV", }

import bpy
import bmesh
from bpy.types import Operator
from bpy.props import BoolProperty
from mathutils import Vector

# notes:
#   - Works only on tube-like parts of mesh defined by selection and active vertex
#   (therefore you must be in vertex selection mode) and the selection must have a start
#   and an end ring. Tube-like mesh is: all quads, no caps, fixed number of vertices
#   in each ring. (Best example of such mesh is mesh circle extruded several times
#   or beveled curve (not cyclic) converted to mesh.) There must be an active vertex
#   on one of the boundary loops in selection. This active vertex define place where
#   mesh will be 'cut' - where seam will be placed.
#   - Result is rectangular UV for easy texturing, scaled to fit square, horizontal
#   and vertical distances between vertices are averaged and proportional to each other.

# usage:
#   1 tab to Edit mode
#   2 select part of mesh you want to unwrap, tube type explained above
#   3 make sure your selection has boundaries and there is an active vertex on one border of selection
#   4 hit "U" and select "Tube UV Unwrap"
#   5 optionally check/uncheck 'Mark Seams' or 'Flip' in operator properties

# changelog:
# 2018.12.16 updated to blender 2.8
# 2014.10.08 removed 'Rectangular' option, it was buggy and now i really don't
#            see why would anyone need this.. sorry if you used it, but i guess
#            nobody will miss that..
# 2014.08.29 clarified docs (i hope so)
# 2014.08.28 fixed maximum recursion depth exceeded error on large meshes
# 2014.08.27 new option, 'Rectangular': if true, all faces will be rectangular,
#            if false, horizontal edges will be scaled proportionally and whole
#            island will be centered in layout (just a by-product)
# 2014.08.27 almost full rewrite, now it works on selection only,
#            any mesh will work, if selection comply to requirements
# 2014.06.16 fail nicely when encountered 2 ring cylinder
# 2014.06.16 got rid of changing edit/object mode
# 2014.06.13 fixed accidental freeze on messy geometry
#            fixed first loop vertex order (also on messy geometry)
#            uv creation part completely rewritten from scratch
# 2014.06.12 first release


class UnsuitableMeshError(Exception):
    pass


class ActiveVertexError(Exception):
    pass


class SelectionError(Exception):
    pass


def tube_unwrap(operator, context, mark_seams, flip, ):
    ob = context.active_object
    me = ob.data
    bm = bmesh.from_edit_mesh(me)
    
    # make a copy and remove unselected, this will be 'working' bmesh
    bm2 = bm.copy()
    selected_verts = [v for v in bm2.verts if v.select is True]
    not_selected_verts = [v for v in bm2.verts if v.select is False]
    for v in not_selected_verts:
        bm2.verts.remove(v)
    
    # now i have to determine, if this is continuous cylinder from quads
    active2 = bm2.select_history.active
    if(active2 is None):
        raise ActiveVertexError("No active vertex found.")
    
    # verts checks
    if(not active2.is_boundary):
        raise SelectionError("Active vertex is not on selection boundary.")
    boundary_verts = [v for v in bm2.verts if v.is_boundary is True]
    if(len(boundary_verts) == 0):
        # no faces = no boundary verts
        raise UnsuitableMeshError("Unsuitable mesh or selection.")
    if(len(boundary_verts) % 2 != 0):
        raise UnsuitableMeshError("Unsuitable mesh or selection.")
    if(len(bm2.verts) % (len(boundary_verts) / 2) != 0):
        raise UnsuitableMeshError("Unsuitable mesh or selection.")
    num_rings = int(len(bm2.verts) / (len(boundary_verts) / 2))
    verts_per_ring = int(len(boundary_verts) / 2)
    # if(len(bm2.verts) - len(boundary_verts) == 0):
    #     # this should be handled by special function
    #     raise UnsuitableMeshError("only 2 rings selected")
    # edges checks
    if(len(bm2.edges) != (num_rings * verts_per_ring) + ((num_rings - 1) * verts_per_ring)):
        raise UnsuitableMeshError("Unexpected number of edges.")
    # polygon checks
    not_quads = [f for f in bm2.faces if len(f.verts) != 4]
    if(len(not_quads) != 0):
        raise UnsuitableMeshError("Mesh is not quad only.")
    # all linked a bit more sophisticated check, but maybe it is already checked above..
    # but, this kind of recursion is good as an exercise
    linked = []
    
    def get_neighbours(v):
        r = []
        for le in v.link_edges:
            # a = le.verts[0]
            # b = le.verts[1]
            # if(a == v):
            #     r.append(b)
            # else:
            #     r.append(a)
            # hmm, better to read docs thoroughly, didn't know about this until now..
            r.append(le.other_vert(v))
        return r
    
    # changed to iteration, it is a bit slow i think, searching for element in list twice in row and removing from list by value..
    def walk(v, linked):
        ok = True
        other = [v, ]
        while(ok):
            v = other[0]
            linked.append(v)
            other.remove(v)
            ns = get_neighbours(v)
            for n in ns:
                if(n not in linked and n not in other):
                    other.append(n)
            if(len(other) == 0):
                ok = False
    
    walk(active2, linked)
    
    if(len(linked) != len(bm2.verts)):
        raise UnsuitableMeshError("Mesh or selection is not continuous.")
    
    def get_seam_and_rings(vert):
        def decide_direction(v, a, b, ):
            if(flip):
                return b
            return a
        
        # get ring from active vertex around selection edge
        def get_boundary_edge_loop(vert):
            def is_boundary(v):
                if(v.is_boundary):
                    return True
                le = v.link_edges
                stats = [False] * len(le)
                for i, e in enumerate(le):
                    a = e.verts[0]
                    b = e.verts[1]
                    if(v == a):
                        if(b.select):
                            stats[i] = True
                    else:
                        if(a.select):
                            stats[i] = True
                if(sum(stats) != len(stats) - 1):
                    return False
                return True
            
            def get_next_boundary_vertices(vert):
                lf = vert.link_faces
                fs = []
                for f in lf:
                    if(f.select):
                        fs.append(f)
                if(len(fs) != 2):
                    raise SelectionError("Selection is not continuous. Select all rings you want to unwrap without gaps.")
                fa = fs[0]
                fb = fs[1]
                a = None
                b = None
                for i, v in enumerate(fa.verts):
                    if(is_boundary(v) and v is not vert):
                        a = v
                for i, v in enumerate(fb.verts):
                    if(is_boundary(v) and v is not vert):
                        b = v
                return a, b
            
            def walk_verts(v, path):
                path.append(v)
                a, b = get_next_boundary_vertices(v)
                if(len(path) == 1):
                    # i need a second vert, decide one direction..
                    # path = walk_verts(a, path)
                    nv = decide_direction(v, a, b)
                    path = walk_verts(nv, path)
                if(a in path):
                    if(b not in path):
                        path = walk_verts(b, path)
                    else:
                        return path
                elif(b in path):
                    if(a not in path):
                        path = walk_verts(a, path)
                    else:
                        return path
                # else:
                #     raise UnsuitableMeshError("Selection with only two rings both boundary detected. Add a loop cut between or select more loops in order to make unwrap work.")
                return path
            
            verts = walk_verts(vert, [])
            return verts
        
        def get_seam_and_rings_2ring_mesh(vert):
            # got vert - active vertex, go by link_edges, and use only e.is_boundary = True
            # choose one and walk around until start vertex is reached..
            # walk next ring and now i have rings, and seam (both start point)
            # now i can skip straight to uv creation
            def get_next_boundary_vertices(vert):
                le = [e for e in vert.link_edges if e.is_boundary is True]
                vs = []
                for e in le:
                    vs.extend(e.verts)
                r = [v for v in vs if v is not vert]
                return r
            
            def walk_verts(v, path):
                path.append(v)
                a, b = get_next_boundary_vertices(v)
                if(len(path) == 1):
                    # i need a second vert, decide one direction..
                    # path = walk_verts(a, path)
                    nv = decide_direction(v, a, b)
                    path = walk_verts(nv, path)
                if(a in path):
                    if(b not in path):
                        path = walk_verts(b, path)
                    else:
                        return path
                elif(b in path):
                    if(a not in path):
                        path = walk_verts(a, path)
                    else:
                        return path
                return path
            
            ring = walk_verts(vert, [])
            e = [e for e in vert.link_edges if e.is_boundary is not True][0]
            if(e.verts[0] == vert):
                vert2 = e.verts[1]
            else:
                vert2 = e.verts[0]
            ring2 = []
            for i, v in enumerate(ring):
                e = [e for e in v.link_edges if e.is_boundary is not True][0]
                if(e.verts[0] == v):
                    a = e.verts[1]
                else:
                    a = e.verts[0]
                ring2.append(a)
            
            return [vert, vert2, ], [ring, ring2, ]
        
        if(num_rings == 2):
            seam, rings = get_seam_and_rings_2ring_mesh(vert)
            # skip right to uv creation
            return (seam, rings)
        else:
            boundary_ring = get_boundary_edge_loop(vert)
        
        # if(len(selected_verts) % len(boundary_ring) != 0):
        #     raise UnsuitableMeshError("Number of vertices != number of rings * number of ring vertices.")
        # num_loops = int(len(selected_verts) / len(boundary_ring))
        
        # old code, just swap names
        num_loops = num_rings
        
        # get all rings
        def is_in_rings(vert, rings):
            for r in rings:
                for v in r:
                    if(v == vert):
                        return True
            return False
        
        def get_next_ring(rings):
            prev_ring = rings[len(rings) - 1]
            nr = []
            for v in prev_ring:
                le = v.link_edges
                for e in le:
                    for v in e.verts:
                        if(v not in prev_ring and is_in_rings(v, rings) is False and v.select):
                            nr.append(v)
            return nr
        
        rings = [boundary_ring, ]
        for i in range(num_loops - 1):
            r = get_next_ring(rings)
            rings.append(r)
        
        # and seam vertices
        def get_seam():
            seam = [vert, ]
            for i in range(num_loops - 1):
                for v in rings[i + 1]:
                    sle = seam[i].link_edges
                    for e in sle:
                        if(v in e.verts):
                            if(e.verts[0] == seam[i]):
                                seam.append(e.verts[1])
                            else:
                                seam.append(e.verts[0])
            return seam
        seam = get_seam()
        
        return (seam, rings)
    
    seam, rings = get_seam_and_rings(active2)
    
    # sum all seam edges lengths
    def calc_seam_length(seam):
        l = 0
        for i in range(len(seam) - 1):
            v = seam[i]
            le = v.link_edges
            for e in le:
                if(seam[i + 1] in e.verts):
                    l += e.calc_length()
        return l
    
    seam_length = calc_seam_length(seam)
    
    # sum all ring edges lengths
    def calc_circumference(r):
        def get_edge(av, bv):
            for e in bm2.edges:
                if(av in e.verts and bv in e.verts):
                    return e
            return None
        
        l = 0
        for i in range(len(r)):
            ei = i + 1
            if(ei >= len(r)):
                ei = 0
            e = get_edge(r[i], r[ei])
            l += e.calc_length()
        return l
    
    # ideal uv layout width and height, and scale_ratio to fit
    def calc_sizes(rings, seam_length, seam):
        ac = 0
        for r in rings:
            ac += calc_circumference(r)
        ac = ac / len(rings)
        
        if(ac > seam_length):
            scale_ratio = 1 / ac
            w = 0
            h = (seam_length / len(seam)) * scale_ratio
        else:
            scale_ratio = 1 / seam_length
            w = (ac / len(rings[0])) * scale_ratio
            h = 0
        return scale_ratio, w, h
    
    scale_ratio, w, h = calc_sizes(rings, seam_length, seam)
    
    # create uv
    def make_uvmap(bm, name):
        uvs = bm.loops.layers.uv
        if(uvs.active is None):
            uvs.new(name)
        uv_lay = uvs.active
        return uv_lay
    
    uv_lay = make_uvmap(bm, "UVMap")
    
    # convert verts from bm2 to bm
    if(bpy.app.version >= (2, 73, 0)):
        bm.verts.ensure_lookup_table()
    seam = [bm.verts[v.index] for v in seam]
    rs = []
    for ring in rings:
        r = [bm.verts[v.index] for v in ring]
        rs.append(r)
    rings2 = rings
    rings = rs
    
    # make uv, scale it correctly
    def make_uvs(uv_lay, scale_ratio, w, h, rings, seam, ):
        def get_edge(av, bv):
            for e in bm.edges:
                if(av in e.verts and bv in e.verts):
                    return e
            return None
        
        def get_face(verts):
            a = set(verts[0].link_faces)
            b = a.intersection(verts[1].link_faces, verts[2].link_faces, verts[3].link_faces)
            return list(b)[0]
        
        def get_face_loops(f, vo):
            lo = []
            for i, v in enumerate(vo):
                for j, l in enumerate(f.loops):
                    if(l.vert == v):
                        lo.append(j)
            return lo
        
        x = 0
        y = 0
        for ir, ring in enumerate(rings):
            if(len(rings) > ir + 1):
                if(w == 0):
                    # circumference <= length
                    fw = 1 / len(rings[0])
                    fh = get_edge(seam[ir], seam[ir + 1]).calc_length() * scale_ratio
                else:
                    # circumference > length
                    fw = w
                    fh = get_edge(seam[ir], seam[ir + 1]).calc_length() * scale_ratio
            
            for iv, vert in enumerate(ring):
                if(len(rings) > ir + 1):
                    next_ring = rings[ir + 1]
                    # d - c
                    # |   |
                    # a - b
                    if(len(ring) == iv + 1):
                        poly = (vert, ring[0], next_ring[0], next_ring[iv])
                    else:
                        poly = (vert, ring[iv + 1], next_ring[iv + 1], next_ring[iv])
                    
                    face = get_face(poly)
                    loops = get_face_loops(face, poly)
                    
                    luv = face.loops[loops[0]][uv_lay]
                    luv.uv = Vector((x, y))
                    
                    x += fw
                    luv = face.loops[loops[1]][uv_lay]
                    luv.uv = Vector((x, y))
                    
                    y += fh
                    luv = face.loops[loops[2]][uv_lay]
                    luv.uv = Vector((x, y))
                    
                    x -= fw
                    luv = face.loops[loops[3]][uv_lay]
                    luv.uv = Vector((x, y))
                    
                x += fw
                y -= fh
            x = 0
            y += fh
            fw = 0
            fh = 0
    
    make_uvs(uv_lay, scale_ratio, w, h, rings, seam, )
    
    def remap(v, min1, max1, min2, max2):
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
        
        v = clamp(v, min1, max1)
        r = interpolate(normalize(v, min1, max1), min2, max2)
        r = clamp(r, min2, max2)
        return r
    
    # mark seams, both boundary rings and seam between them
    if(mark_seams):
        def get_edge(av, bv):
            for e in bm.edges:
                if(av in e.verts and bv in e.verts):
                    return e
            return None
        
        def mark_seam(seam):
            for i, v in enumerate(seam):
                if(i < len(seam) - 1):
                    nv = seam[i + 1]
                    e = get_edge(v, nv)
                    e.seam = True
        
        mark_seam(seam)
        # me.show_edge_seams = True
        
        def mark_additional_seams(r):
            for i in range(len(r) - 1):
                a = r[i]
                b = r[i + 1]
                e = get_edge(a, b)
                e.seam = True
            a = r[0]
            b = r[len(r) - 1]
            e = get_edge(a, b)
            e.seam = True
        
        mark_additional_seams(rings[0])
        mark_additional_seams(rings[len(rings) - 1])
    
    # put back
    bmesh.update_edit_mesh(me)
    
    # cleanup
    bm2.free()
    
    return True


class TUVUW_OT_tube_uv_unwrap(Operator):
    bl_idname = "uv.tube_uv_unwrap"
    bl_label = "Tube UV Unwrap"
    bl_description = "UV unwrap tube-like mesh selection. Selection must be all quads, no caps, fixed number of vertices in each ring."
    bl_options = {'REGISTER', 'UNDO'}
    
    mark_seams: BoolProperty(name="Mark seams", description="Marks seams around all island edges.", default=True, )
    flip: BoolProperty(name="Flip", description="Flip unwrapped island.", default=False, )
    
    @classmethod
    def poll(cls, context):
        ob = context.active_object
        msm = context.scene.tool_settings.mesh_select_mode
        return (ob and ob.type == 'MESH' and context.mode == 'EDIT_MESH' and msm[0])
    
    def execute(self, context):
        r = False
        
        import traceback
        print_errors = False
        
        try:
            r = tube_unwrap(self, context, self.mark_seams, self.flip, )
        except UnsuitableMeshError as e:
            self.report({'ERROR'}, str(e))
            if(print_errors):
                tb = traceback.print_exc()
                print(tb)
        except ActiveVertexError as e:
            self.report({'ERROR'}, str(e))
            if(print_errors):
                tb = traceback.print_exc()
                print(tb)
        except SelectionError as e:
            self.report({'ERROR'}, str(e))
            if(print_errors):
                tb = traceback.print_exc()
                print(tb)
        if(not r):
            return {'CANCELLED'}
        return {'FINISHED'}
    
    def draw(self, context):
        layout = self.layout
        c = layout.column()
        r = c.row()
        r.prop(self, "mark_seams")
        r = c.row()
        r.prop(self, "flip")


def menu_func(self, context):
    l = self.layout
    l.separator()
    l.operator(TUVUW_OT_tube_uv_unwrap.bl_idname, text=TUVUW_OT_tube_uv_unwrap.bl_label)


classes = (TUVUW_OT_tube_uv_unwrap, )


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    
    bpy.types.IMAGE_MT_uvs.append(menu_func)
    bpy.types.VIEW3D_MT_uv_map.append(menu_func)


def unregister():
    bpy.types.IMAGE_MT_uvs.remove(menu_func)
    bpy.types.VIEW3D_MT_uv_map.remove(menu_func)
    
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
