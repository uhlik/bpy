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

bl_info = {"name": "Tube UV Unwrap",
           "description": "UV unwrap tube like meshes (all quads, no caps, fixed number of vertices in each ring)",
           "author": "Jakub Uhlik",
           "version": (0, 1, 2),
           "blender": (2, 69, 0),
           "location": "Edit mode > Mesh > UV Unwrap... > Tube UV Unwrap",
           "warning": "",
           "wiki_url": "",
           "tracker_url": "",
           "category": "UV", }

import bpy
import bmesh
from mathutils import Vector

# notes:
#   - Works only on tube like meshes, all quads, no caps, fixed number of vertices
#     in each ring. Best example of such mesh is mesh circle extruded several times
#     or beveled curve converted to mesh.
#   - Result is right-angled UV for easy texturing
#   - Single selected vertex on boundary ring is required before running operator.
#     This vertex marks loop, along which tube will be cut.
#   - Distances of vertices in next tube ring are averaged.
#   - UV is scaled to fit area.
#   - Seam will be marked on mesh automatically.

# usage:
#   1 tab to Edit mode
#   2 select single vertex on boundary ring
#   3 hit "U" and select "Tube UV Unwrap"

# changelog:
# 2014.06.16 got rid of changing edit/object mode
# 2014.06.13 fixed accidental freeze on messy geometry
#            fixed first loop vertex order (also on messy geometry)
#            uv creation part completely rewritten from scratch
# 2014.06.12 first release


def tube_unwrap(operator, context):
    ob = context.active_object
    me = ob.data
    bm = bmesh.from_edit_mesh(me)
    
    vert = bm.select_history.active
    if(not vert):
        operator.report({'ERROR'}, "Select one boundary vertex. Seam will be placed there.")
        return False
    else:
        if(vert.is_boundary is False):
            operator.report({'ERROR'}, "Select one boundary vertex. Seam will be placed there.")
            return False
    
    def get_seam_and_rings(vert):
        if(vert.is_boundary):
            def get_boundary_edge_loop(vert):
                def get_next_boundary_vertices(vert):
                    lf = vert.link_faces
                    fa = lf[0]
                    fb = lf[1]
                    a = None
                    b = None
                    for i, v in enumerate(fa.verts):
                        if(v.is_boundary and v is not vert):
                            a = v
                    for i, v in enumerate(fb.verts):
                        if(v.is_boundary and v is not vert):
                            b = v
                    return a, b
                
                def walk_verts(v, path):
                    path.append(v)
                    a, b = get_next_boundary_vertices(v)
                    if(len(path) == 1):
                        # i need a second vert, decide one direction..
                        path = walk_verts(a, path)
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
                    else:
                        raise RuntimeError("Something very bad happened. Please contact support immediately.")
                    return path
                
                verts = walk_verts(vert, [])
                return verts
            
            boundary_ring = get_boundary_edge_loop(vert)
            
            if(len(bm.verts) % len(boundary_ring) != 0):
                # abort
                operator.report({'ERROR'}, "This is not a simple tube. Number of vertices != number of rings * number of ring vertices.")
                return (None, None)
            num_loops = int(len(bm.verts) / len(boundary_ring))
            
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
                            if(v not in prev_ring and is_in_rings(v, rings) is False):
                                nr.append(v)
                return nr
            
            rings = [boundary_ring, ]
            for i in range(num_loops - 1):
                r = get_next_ring(rings)
                rings.append(r)
            
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
            return (seam, rings)
    
    seam, rings = get_seam_and_rings(vert)
    if(seam is None or rings is None):
        # abort
        return False
    
    def walk_face_ring(vert, ring, next_vert, next_ring):
        edges = []
        for i, v in enumerate(ring):
            le = v.link_edges
            for e in le:
                if(next_ring[i] in e.verts):
                    break
            edges.append(e)
        faces = []
        for i, e in enumerate(edges):
            lf = e.link_faces
            for f in lf:
                ni = i + 1
                if(ni >= len(edges)):
                    ni = 0
                if(f in edges[ni].link_faces and f not in faces):
                    faces.append(f)
                    # here i have to decide in first iteration in which direction walk through faces
                    # i do not know yet how to do it. so i am taking the first face
                    # in hope the second (last) will be get in next iteration..
                    break
        return faces
    
    def make_face_rings(seam, rings):
        face_rings = []
        for i, v in enumerate(seam):
            if(i < len(seam) - 1):
                next_vert = seam[i + 1]
                fr = walk_face_ring(v, rings[i], next_vert, rings[i + 1])
                face_rings.append(fr)
        return face_rings
    
    face_rings = make_face_rings(seam, rings)
    
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
    
    def calc_circumference(r):
        def get_edge(av, bv):
            for e in bm.edges:
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
    
    def make_uvmap(bm, name):
        uvs = bm.loops.layers.uv
        if(uvs.active is None):
            uvs.new(name)
        uv_lay = uvs.active
        return uv_lay
    
    uv_lay = make_uvmap(bm, "UVMap")
    
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
                    
                    face.loops[loops[0]][uv_lay].uv = Vector((x, y))
                    x += fw
                    face.loops[loops[1]][uv_lay].uv = Vector((x, y))
                    y += fh
                    face.loops[loops[2]][uv_lay].uv = Vector((x, y))
                    x -= fw
                    face.loops[loops[3]][uv_lay].uv = Vector((x, y))
                    
                x += fw
                y -= fh
            x = 0
            y += fh
            fw = 0
            fh = 0
    
    make_uvs(uv_lay, scale_ratio, w, h, rings, seam, )
    
    def mark_seam(seam):
        def get_edge(av, bv):
            for e in bm.edges:
                if(av in e.verts and bv in e.verts):
                    return e
            return None
        
        for i, v in enumerate(seam):
            if(i < len(seam) - 1):
                nv = seam[i + 1]
                e = get_edge(v, nv)
                e.seam = True
    
    mark_seam(seam)
    me.show_edge_seams = True
    
    bmesh.update_edit_mesh(me)
    
    return True


class TubeUVUnwrapOperator(bpy.types.Operator):
    bl_idname = "uv.tube_uv_unwrap"
    bl_label = "Tube UV Unwrap"
    bl_description = "UV unwrap tube like meshes. Mesh have to be all quads and cannot have caps."
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return (ob and ob.type == 'MESH' and context.mode == 'EDIT_MESH')
    
    def execute(self, context):
        r = tube_unwrap(self, context)
        if(r is False):
            return {'CANCELLED'}
        return {'FINISHED'}


def menu_func(self, context):
    l = self.layout
    l.separator()
    l.operator(TubeUVUnwrapOperator.bl_idname, text=TubeUVUnwrapOperator.bl_label)


def register():
    bpy.utils.register_module(__name__)
    bpy.types.IMAGE_MT_uvs.append(menu_func)
    bpy.types.VIEW3D_MT_uv_map.append(menu_func)


def unregister():
    bpy.utils.unregister_module(__name__)
    bpy.types.IMAGE_MT_uvs.remove(menu_func)
    bpy.types.VIEW3D_MT_uv_map.remove(menu_func)


if __name__ == "__main__":
    register()
