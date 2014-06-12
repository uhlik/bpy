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
           "version": (0, 1, 0),
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


def tube_unwrap(operator, context):
    bpy.ops.object.mode_set(mode='OBJECT')
    
    ob = context.active_object
    me = ob.data
    bm = bmesh.new()
    bm.from_mesh(me)
    
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
            def get_next_boundary_vertex_clockwise_along_seam(vert):
                lf = vert.link_faces
                fa = lf[0]
                fb = lf[1]
                a = 0
                b = 0
                for i, v in enumerate(fa.verts):
                    if(v.is_boundary and v is not vert):
                        a = (i, v)
                for i, v in enumerate(fb.verts):
                    if(v.is_boundary and v is not vert):
                        b = (i, v)
                if(a > b):
                    return a[1]
                return b[1]
            
            # here i need to decide in which direction loop over vertices
            # how to do it?
            # get adjacent faces, by position of start vertex and other boundary vertices, determine in which side face normal points, and then decide..
            
            def get_boundary_edge_loop(vert):
                if(vert.is_boundary):
                    def get_boundary_neighbours(v):
                        es = [e for e in v.link_edges]
                        be = []
                        for e in es:
                            if(e.is_boundary):
                                be.append(e)
                        vs = []
                        for e in be:
                            for ev in e.verts:
                                if(ev.index != v.index):
                                    vs.append(ev)
                        vs = list(set(vs))
                        return vs
                    
                    def walk_verts(v, path):
                        path.append(v)
                        bn = get_next_boundary_vertex_clockwise_along_seam(v)
                        if(bn != path[0] and bn not in path):
                            path = walk_verts(bn, path)
                        return path
                    
                    verts = walk_verts(vert, [])
                    return verts
                return []
            
            boundary_ring = get_boundary_edge_loop(vert)
            
            if(len(bm.verts) % len(boundary_ring) != 0):
                operator.report({'ERROR'}, "This is not a simple tube. Number of vertices != number of rings * number of ring vertices.")
                return False
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
    
    def make_uvs(face_rings, w, h, scale_ratio, uv_lay, vert, seam, rings):
        def common_edge(fa, fb):
            for e in fa.edges:
                if(e in fb.edges):
                    return e
            return None
        
        def opposite_edge(f, e):
            for le in f.edges:
                if(e.verts[0] not in le.verts and e.verts[1] not in le.verts):
                    return le
            return None
        
        def get_edge(av, bv):
            for e in bm.edges:
                if(av in e.verts and bv in e.verts):
                    return e
            return None
        
        # get starting vertex and the next one in rings
        # get edge between them
        # get the only face with this edge (is boundary)
        # on this face determine order of loops..
        # put into list of indices and use them in assigning loop uv coordinates
        def get_first_poly_order():
            a = vert
            b = rings[0][1]
            c = None
            d = seam[1]
            
            e = get_edge(a, b)
            # there should be just one..
            f = e.link_faces[0]
            for v in f.verts:
                if(v != a and v != b and v != d):
                    c = v
                    break
            vo = [a, b, c, d]
            lo = []
            for i, v in enumerate(vo):
                for j, l in enumerate(f.loops):
                    if(l.vert == v):
                        lo.append(j)
            return lo
        
        lo = get_first_poly_order()
        
        def get_lo(vo, f):
            lo = []
            for i, v in enumerate(vo):
                for j, l in enumerate(f.loops):
                    if(l.vert == v):
                        lo.append(j)
            return lo
        
        x = 0
        y = 0
        for j, fr in enumerate(face_rings):
            # not sure if following is 100% correct..
            if(w == 0):
                # circumference <= length
                fw = 1 / len(rings[0])
                fh = get_edge(seam[j], seam[j + 1]).calc_length() * scale_ratio
            else:
                # circumference > length
                fw = w
                fh = get_edge(seam[j], seam[j + 1]).calc_length() * scale_ratio
            
            for i in range(len(fr)):
                f = fr[i]
                fi = i + 1
                if(fi >= len(fr)):
                    fi = 0
                
                f.loops[lo[0]][uv_lay].uv = Vector((x, y))
                x += fw
                f.loops[lo[1]][uv_lay].uv = Vector((x, y))
                y += fh
                f.loops[lo[2]][uv_lay].uv = Vector((x, y))
                x -= fw
                f.loops[lo[3]][uv_lay].uv = Vector((x, y))
                
                # next face loops order:
                nextr = (i == len(fr) - 1)
                if(nextr):
                    # next ring, first face, common edge is on the top of current face
                    if(j != len(face_rings) - 1):
                        # it is not the last ring
                        
                        # get faces and common edge
                        ff = face_rings[j][0]
                        nf = face_rings[j + 1][0]
                        ce = common_edge(ff, nf)
                        # sort current face vertices by desired uv order
                        sv = []
                        for l in lo:
                            sv.append(ff.verts[l])
                        # build vertex order from next face
                        nfsv = [None] * 4
                        nfsv[0] = sv[3]
                        nfsv[1] = sv[2]
                        # get connecting edge from second vertex in order
                        cne = None
                        for e in nf.edges:
                            if(sv[2] in e.verts and sv[3] not in e.verts):
                                cne = e
                                break
                        for v in cne.verts:
                            if(v not in nfsv):
                                nfsv[2] = v
                        # put last remaining vertex
                        for v in nf.verts:
                            if(v not in nfsv):
                                nfsv[3] = v
                        # convert vertex order to loop order
                        nlo = get_lo(nfsv, nf)
                        # swap
                        lo = nlo
                        
                    else:
                        # it is the last ring, no need to do anything
                        pass
                else:
                    # same ring, just next face, common edge is on right in current face
                    # lets presume that loop order is the same in all faces in ring
                    # we'll see if thas's right decision..
                    pass
                
                x += fw
                y -= fh
            x = 0
            y += fh
            fw = 0
            fh = 0
    
    make_uvs(face_rings, w, h, scale_ratio, uv_lay, vert, seam, rings)
    
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
    
    bm.to_mesh(me)
    bm.free()
    
    bpy.ops.object.mode_set(mode='EDIT')
    
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
            bpy.ops.object.mode_set(mode='EDIT')
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
