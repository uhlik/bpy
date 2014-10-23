# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
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
           "version": (0, 2, 2),
           "blender": (2, 70, 0),
           "location": "View3d > Object > UV Equalize",
           "warning": "",
           "wiki_url": "",
           "tracker_url": "",
           "category": "UV", }


# - Use when tileable texture needs to be applied on all objects and its scale should be the same across them.
# - Available in Object menu of 3d view while in object mode.
# - To enable, more than two mesh objects must be selected, one must be active.


# changelog:
# 2014.10.23 fixed bug which prevented script to work, operators are used for transforming uvs,
#            but when in image editor is loaded 'Render Result', UV will not be displayed
#            and therefore operators will not work.. it's one line fix, just set displayed
#            image to None..
# 2014.10.22 auto deselect non mesh objects
# 2014.10.13 complete rewrite, now it is pure math
# 2014.10.12 fixed different uv names bug
# 2014.06.16 uuid windows workaround
# 2014.06.12 first release


import bpy
import bmesh
from bpy.props import FloatProperty, BoolProperty
from mathutils import Vector
import math


def equalize(operator, context, use_pack, rotate, margin, use_active, ):
    def activate_object(o):
        bpy.ops.object.select_all(action='DESELECT')
        sc = bpy.context.scene
        o.select = True
        sc.objects.active = o
    
    ao = context.scene.objects.active
    # obs = [ob for ob in context.scene.objects if ob.name != ao.name and ob.select]
    # make it easier to select all, exclude non-mesh objects from list
    obs = [ob for ob in context.scene.objects if ob.name != ao.name and ob.select and ob.type == 'MESH']
    
    # some checks
    for o in obs:
        if(o.type != 'MESH'):
            operator.report({'ERROR'}, "Object {} is not a mesh.".format(o.name))
            return False
        if(len(o.data.uv_layers) < 1):
            operator.report({'ERROR'}, "Object {} has no uv map.".format(o.name))
            return False
    
    cache = {}
    
    def calc_areas(o):
        # cache
        k = o.name
        try:
            mesh_area = cache[k][0]
            uv_area = cache[k][1]
            return mesh_area, uv_area
        except:
            pass
        # prepare
        bm = bmesh.new()
        # bm.from_mesh(o.data)
        # this way modifiers are taken into count, like mirror etc..
        me = o.to_mesh(context.scene, True, 'PREVIEW', )
        bm.from_mesh(me)
        #
        bm.transform(o.matrix_world)
        bmesh.ops.triangulate(bm, faces=bm.faces)
        # mesh
        mesh_area = sum([f.calc_area() for f in bm.faces])
        # uv
        uv_layer = bm.loops.layers.uv.active
        tas = []
        for f in bm.faces:
            locs = []
            for l in f.loops:
                x, y = l[uv_layer].uv
                locs.append((x, y, ))
            a = Vector((locs[0][0], locs[0][1], 0.0))
            b = Vector((locs[1][0], locs[1][1], 0.0))
            c = Vector((locs[2][0], locs[2][1], 0.0))
            ab = b - a
            ac = c - a
            cr = ab.cross(ac)
            a = cr.length * 0.5
            tas.append(a)
        uv_area = sum(tas)
        # cleanup
        bm.free()
        # also remove temp mesh
        bpy.data.meshes.remove(me)
        # cache
        cache[k] = (mesh_area, uv_area, )
        return mesh_area, uv_area
    
    if(not use_active):
        obs.append(ao)
        oms = []
        ouvs = []
        for o in obs:
            om, ouv = calc_areas(o)
            oms.append(om)
            ouvs.append(ouv)
        aom = sum(oms) / len(oms)
        aouv = sum(ouvs) / len(ouvs)
    else:
        aom, aouv = calc_areas(ao)
    
    for o in obs:
        activate_object(o)
        # average and pack islands
        if(use_pack):
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.uv.select_all(action='SELECT')
            bpy.ops.uv.average_islands_scale()
            bpy.ops.uv.pack_islands(rotate=rotate, margin=margin, )
            bpy.ops.object.mode_set(mode='OBJECT')
        # transform uv
        bpy.ops.object.mode_set(mode='EDIT')
        if(not use_pack):
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.uv.select_all(action='SELECT')
        
        original_type = bpy.context.area.type
        bpy.context.area.type = "IMAGE_EDITOR"
        # reset image inside editor, it might be Render Result and in this case,
        # UV operators will not work because UVs will not be displayed..
        bpy.context.area.spaces[0].image = None
        
        om, ouv = calc_areas(o)
        x = (aouv / aom) * om
        v = x / ouv
        v = math.sqrt(v)
        
        bpy.ops.transform.resize(value=(v, v, v), )
        bpy.context.area.type = original_type
        
        bpy.ops.object.mode_set(mode='OBJECT')
    
    # activate the one which was not changed
    activate_object(ao)
    # reselect objects for convenience
    for o in obs:
        o.select = True
    
    return True


class UVEqualize(bpy.types.Operator):
    bl_idname = "uv.uv_equalize"
    bl_label = "UV Equalize"
    bl_description = "Equalizes scale of UVs of selected objects to active object."
    bl_options = {'REGISTER', 'UNDO'}
    
    use_active = BoolProperty(name="Use Active",
                              description="Use active object as scale specimen. Otherwise will be used object with largest polygons after packing. This object will be packed to fit bounds.",
                              default=True, )
    use_pack = BoolProperty(name="Pack Islands",
                            description="Average island scale and pack",
                            default=False, )
    rotate = BoolProperty(name="Pack Islands Rotate",
                          description="Rotate islands for best fit",
                          default=True, )
    margin = FloatProperty(name="Pack Islands Margin",
                           description="Space between islands",
                           min=0.0,
                           max=1.0,
                           default=0.001, )
    
    @classmethod
    def poll(cls, context):
        ao = context.active_object
        so = bpy.context.selected_objects
        return (ao and ao.type == 'MESH' and len(so) > 1 and context.mode == 'OBJECT')
    
    def execute(self, context):
        r = equalize(self, context, self.use_pack, self.rotate, self.margin, self.use_active, )
        if(r is False):
            return {'CANCELLED'}
        return {'FINISHED'}
    
    def draw(self, context):
        l = self.layout
        
        r = l.row()
        r.prop(self, "use_active")
        
        r = l.row()
        r.prop(self, "use_pack")
        r = l.row()
        r.prop(self, "rotate")
        r.enabled = self.use_pack
        r = l.row()
        r.prop(self, "margin")
        r.enabled = self.use_pack


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
