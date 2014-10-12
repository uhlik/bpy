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

bl_info = {"name": "UV Equalize",
           "description": "Equalizes scale of UVs of selected objects to active object.",
           "author": "Jakub Uhlik",
           "version": (0, 1, 2),
           "blender": (2, 70, 0),
           "location": "View3d > Object > UV Equalize",
           "warning": "",
           "wiki_url": "",
           "tracker_url": "",
           "category": "UV", }


# - Use when tileable texture needs to be applied on all objects and its scale should be the same across them.
# - Beware, active UV on each object will be repacked, in active object as well.
# - Available in Object menu of 3d view while in object mode.
# - To enable, more than two mesh objects must be selected, one must be active.


# changelog:
# 2014.10.12 fixed different uv names bug
# 2014.06.16 uuid windows workaround
# 2014.06.12 first release


import bpy
import bmesh
from bpy.props import FloatProperty, BoolProperty
import math


# http://blender.stackexchange.com/a/7670

# uuid module causes an error messagebox on windows.
#
# using a dirty workaround to preload uuid without ctypes,
# until blender gets compiled with vs2013
def uuid_workaround():
    import platform
    if platform.system() == "Windows":
        import ctypes
        CDLL = ctypes.CDLL
        ctypes.CDLL = None
        import uuid
        ctypes.CDLL = CDLL


uuid_workaround()
import uuid


def add_object(n, d):
    """Add object of name n and with data d."""
    so = bpy.context.scene.objects
    for i in so:
        i.select = False
    o = bpy.data.objects.new(n, d)
    so.link(o)
    o.select = True
    if(so.active is None or so.active.mode == 'OBJECT'):
        so.active = o
    return o


def activate_object(o):
    """Set object o as active."""
    bpy.ops.object.select_all(action='DESELECT')
    sc = bpy.context.scene
    o.select = True
    sc.objects.active = o


def duplicate(o):
    """Duplicate object and return it, how cools is that?"""
    activate_object(o)
    p = [ob.name for ob in bpy.data.objects]
    bpy.ops.object.duplicate(linked=False)
    a = [ob.name for ob in bpy.data.objects]
    d = list(set(a) - set(p))
    c = bpy.data.objects[d[0]]
    return c


def delete_mesh_object(o):
    """Delete mesh object and remove mesh data."""
    m = o.data
    bpy.context.scene.objects.unlink(o)
    bpy.data.objects.remove(o)
    m.user_clear()
    bpy.data.meshes.remove(m)


def equalize(operator, context, rotate, margin, use_active, ):
    # objects we will operate on..
    so = [o for o in context.selected_objects]
    aco = context.active_object
    
    # some more sophisticated checks
    for o in so:
        if(o.type != 'MESH'):
            operator.report({'ERROR'}, "Object {} is not a mesh.".format(o.name))
            return False
        if(len(o.data.uv_layers) < 1):
            operator.report({'ERROR'}, "Object {} has no uv map.".format(o.name))
            return False
    
    # uuid as name for one quad test object
    uid = str(uuid.uuid1())
    
    # determine some reasonably small size like 1% of average area
    def get_sc(o):
        bm = bmesh.new()
        bm.from_mesh(o.data)
        a = sum([f.calc_area() for f in bm.faces]) / len(bm.faces)
        sc = (a / 100) * 1
        bm.free()
        return sc
    
    if(use_active):
        # just for active object
        sc = get_sc(aco)
    else:
        # average all..
        sc = sum([get_sc(o) for o in so]) / len(so)
    # add object of known size, one quad polygon
    # with so small area that will not have much influence..
    pme = bpy.data.meshes.new(uid)
    pdt = ([(1 * sc, 1 * sc, 0),
            (1 * sc, -1 * sc, 0),
            (-1 * sc, -1 * sc, 0),
            (-1 * sc, 1 * sc, 0)],
           [],
           [(0, 3, 2, 1, ), ], )
    pme.from_pydata(*pdt)
    po = add_object(uid, pme)
    activate_object(po)
    # and add vertex group which will become very handy soon
    vg = po.vertex_groups.new(uid)
    vg.add([0, 1, 2, 3], 1.0, 'REPLACE')
    # add uv, since it is one quad, it should add something "fully unwrapped" :)
    # no need to mess with it some more than this..
    pme.uv_textures.new("UVMap")
    
    # 1) join copy of test object to all
    # 2) uv average scale and pack
    # 3) calculate how long is one edge of the polygon just added
    db = []
    for o in so:
        if(o != po):
            c = duplicate(po)
            activate_object(o)
            c.data.uv_textures.active.name = o.data.uv_textures.active.name
            c.select = True
            bpy.ops.object.join()
            
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.uv.select_all(action='SELECT')
            bpy.ops.uv.average_islands_scale()
            bpy.ops.uv.pack_islands(rotate=rotate, margin=margin, )
            bpy.ops.object.mode_set(mode='OBJECT')
            
            # calculate length of side of test polygon
            # after edit mode it is pure magic..
            me = o.data
            vg = o.vertex_groups[uid]
            pv = []
            for v in me.vertices:
                if(len(v.groups) > 0):
                    gs = [g.group for g in v.groups]
                    if(vg.index in gs):
                        pv.append(v)
            pi = [v.index for v in pv]
            es = [e for e in me.edges if e.vertices[0] in pi]
            a = me.vertices[es[0].vertices[0]]
            b = me.vertices[es[0].vertices[1]]
            ls = me.loops
            for l in ls:
                if(l.vertex_index == a.index):
                    al = l
                if(l.vertex_index == b.index):
                    bl = l
            uvd = me.uv_layers.active.data
            a = uvd[al.index].uv
            b = uvd[bl.index].uv
            d = math.sqrt(((a[0] - b[0]) ** 2) + ((a[1] - b[1]) ** 2))
            
            e = [o, d, ]
            db.append(e)
    
    # get d of active object
    for o, d in db:
        if(o == aco):
            e = d
            break
    
    if(not use_active):
        # restore old functionality
        db.sort(key=lambda v: v[1])
        aco = db[len(db) - 1][0]
        e = db[len(db) - 1][1]
    
    # transform UVs and cleanup
    for o, d in db:
        activate_object(o)
        # ugly operators ahead..
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.select_all(action='SELECT')
        
        original_type = bpy.context.area.type
        bpy.context.area.type = "IMAGE_EDITOR"
        
        if(o != aco):
            # only not active objects
            v = 1 / (d / e)
            bpy.ops.transform.resize(value=(v, v, v))
        
        bpy.context.area.type = original_type
        bpy.ops.mesh.select_all(action='DESELECT')
        
        # remove test polygon, find it by vertex group
        # then remove vertex group as well
        i = None
        for g in o.vertex_groups:
            if(g.name == uid):
                i = g.index
                break
        o.vertex_groups.active_index = i
        bpy.ops.object.vertex_group_select()
        bpy.ops.object.vertex_group_remove(all=False)
        bpy.ops.mesh.delete(type='VERT')
        # select all for convenience..
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.select_all(action='SELECT')
        
        bpy.ops.object.mode_set(mode='OBJECT')
    
    # remove test poly source
    activate_object(po)
    delete_mesh_object(po)
    
    # activate the one which was not changed
    activate_object(aco)
    # reselect objects for convenience
    for o in so:
        o.select = True
    
    return True


class UVEqualize(bpy.types.Operator):
    bl_idname = "uv.uv_equalize"
    bl_label = "UV Equalize"
    bl_description = "Equalizes scale of UVs of selected objects to active object."
    bl_options = {'REGISTER', 'UNDO'}
    
    rotate = BoolProperty(name="Pack Islands Rotate",
                          description="Rotate islands for best fit",
                          default=True, )
    margin = FloatProperty(name="Pack Islands Margin",
                           description="Space between islands",
                           min=0.0, max=1.0,
                           default=0.001, )
    use_active = BoolProperty(name="Use Active",
                              description="Use active object as scale specimen. Otherwise will be used object with largest polygons after packing. This object will be packed to fit bounds.",
                              default=True, )
    
    @classmethod
    def poll(cls, context):
        ao = context.active_object
        so = bpy.context.selected_objects
        return (ao and ao.type == 'MESH' and len(so) > 1 and context.mode == 'OBJECT')
    
    def execute(self, context):
        r = equalize(self, context, self.rotate, self.margin, self.use_active, )
        if(r is False):
            return {'CANCELLED'}
        return {'FINISHED'}
    
    def invoke(self, context, event):
        wm = context.window_manager
        return wm.invoke_props_dialog(self)
    
    def draw(self, context):
        layout = self.layout
        c = layout.column()
        r = c.row()
        r.prop(self, "rotate")
        r = c.row()
        r.prop(self, "margin")
        r = c.row()
        r.prop(self, "use_active")


def menu_func(self, context):
    l = self.layout
    l.separator()
    l.operator(UVEqualize.bl_idname, text=UVEqualize.bl_label)


def register():
    bpy.utils.register_module(__name__)
    bpy.types.VIEW3D_MT_object.append(menu_func)


def unregister():
    bpy.utils.unregister_module(__name__)
    bpy.types.VIEW3D_MT_object.remove(menu_func)


if __name__ == "__main__":
    register()
