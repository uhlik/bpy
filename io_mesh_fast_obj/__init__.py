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

bl_info = {"name": "Fast Wavefront^2 (.obj) (Cython)",
           "description": "Import/Export single mesh as Wavefront OBJ. Only active mesh is exported. Only single mesh is expected on import. Supported obj features: UVs, normals, vertex colors using MRGB format (ZBrush).",
           "author": "Jakub Uhlik",
           "version": (0, 3, 0),
           "blender": (2, 80, 0),
           "location": "File > Import/Export > Fast Wavefront (.obj) (Cython)",
           "warning": "work in progress, currently export only, binaries are not provided, you have to compile them by yourself",
           "wiki_url": "",
           "tracker_url": "",
           "category": "Import-Export", }

import os
import time
import datetime
from mathutils import Matrix

import bpy
from bpy_extras.io_utils import ExportHelper, ImportHelper, axis_conversion
from bpy.types import Operator
from bpy.props import StringProperty, BoolProperty, FloatProperty, IntProperty

from . import export


# note for myself:
# $ pycodestyle --ignore=W293,E501,E741,E402 --exclude='io_mesh_fast_obj/blender' .


DEBUG = True


def log(msg="", indent=0, prefix="> "):
    m = "{}{}{}".format("    " * indent, prefix, msg, )
    if(DEBUG):
        print(m)


class ExportFastOBJ(Operator, ExportHelper):
    bl_idname = "export_mesh.fast_obj"
    bl_label = 'Export Fast OBJ (Cython)'
    bl_description = "Export single mesh as Wavefront OBJ. Only active mesh is exported. Supported obj features: UVs, normals, vertex colors using MRGB format (ZBrush)."
    bl_options = {'PRESET'}
    
    # filepath = StringProperty(name="File Path", description="Filepath used for exporting the file", maxlen=1024, subtype='FILE_PATH', )
    filename_ext = ".obj"
    filter_glob: StringProperty(default="*.obj", options={'HIDDEN'}, )
    check_extension = True
    
    apply_modifiers: BoolProperty(name="Apply Modifiers", default=False, description="Apply all modifiers.", )
    apply_transformation: BoolProperty(name="Apply Transformation", default=True, description="Zero-out mesh transformation.", )
    convert_axes: BoolProperty(name="Convert Axes", default=True, description="Convert from blender (y forward, z up) to forward -z, up y.", )
    triangulate: BoolProperty(name="Triangulate", default=False, description="", )
    use_normals: BoolProperty(name="With Normals", default=True, description="", )
    use_uv: BoolProperty(name="With UV", default=True, description="Export active UV layout.", )
    use_vcols: BoolProperty(name="With Vertex Colors", default=False, description="Export vertex colors, this is not part of official file format specification.", )
    global_scale: FloatProperty(name="Scale", default=1.0, precision=3, description="", )
    precision: IntProperty(name="Precision", default=6, description="", )
    
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
        sub.prop(self, 'use_normals')
        sub.prop(self, 'use_uv')
        sub.prop(self, 'use_vcols')
        sub.prop(self, 'global_scale')
        sub.prop(self, 'precision')
    
    def execute(self, context):
        t = time.time()
        
        o = context.active_object
        m = o.to_mesh(bpy.context.depsgraph, self.apply_modifiers, )
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
        
        export.export_obj(m.as_pointer(),
                          self.filepath,
                          "{}-{}".format(o.name, o.data.name),
                          use_normals=self.use_normals,
                          use_uv=self.use_uv,
                          use_vcols=self.use_vcols,
                          precision=self.precision,
                          debug=DEBUG, )
        
        bpy.data.meshes.remove(m)
        
        log("completed in {}.".format(datetime.timedelta(seconds=time.time() - t)))
        return {'FINISHED'}


class ImportFastOBJ(Operator, ImportHelper):
    bl_idname = "import_mesh.fast_obj"
    bl_label = 'Import Fast OBJ (Cython)'
    bl_description = "Import single mesh Wavefront OBJ. Only single mesh is expected on import. Supported obj features: UVs, normals, vertex colors using MRGB format (ZBrush)."
    bl_options = {'PRESET'}
    
    # filepath = StringProperty(name="File Path", description="Filepath used for exporting the file", maxlen=1024, subtype='FILE_PATH', )
    filename_ext = ".obj"
    filter_glob: StringProperty(default="*.obj", options={'HIDDEN'}, )
    check_extension = True
    
    convert_axes: BoolProperty(name="Convert Axes", default=True, description="Convert from blender (y forward, z up) to forward -z, up y.", )
    with_uv: BoolProperty(name="With UV", default=True, description="Import texture coordinates.", )
    with_vertex_colors: BoolProperty(name="With Vertex Colors", default=False, description="Import vertex colors, this is not part of official file format specification.", )
    use_mask_as_vertex_group: BoolProperty(name="Mask as Vertex Group", default=False, description="Import ZBrush mask as vertex group.", )
    with_polygroups: BoolProperty(name="With Polygroups", default=False, description="", )
    global_scale: FloatProperty(name="Scale", default=1.0, precision=3, description="", )
    apply_conversion: BoolProperty(name="Apply Conversion", default=False, description="Apply new axes directly to mesh or only transform at object level.", )
    
    @classmethod
    def poll(cls, context):
        return False
    
    def draw(self, context):
        l = self.layout
        sub = l.column()
        sub.prop(self, 'convert_axes')
        sub.prop(self, 'with_uv')
        sub.prop(self, 'with_vertex_colors')
        sub.prop(self, 'use_mask_as_vertex_group')
        sub.prop(self, 'with_polygroups')
        sub.prop(self, 'global_scale')
        sub.prop(self, 'apply_conversion')
    
    def execute(self, context):
        return {'FINISHED'}


def menu_func_export(self, context):
    self.layout.operator(ExportFastOBJ.bl_idname, text="Fast Wavefront^2 (.obj) (Cython)")


def menu_func_import(self, context):
    self.layout.operator(ImportFastOBJ.bl_idname, text="Fast Wavefront^2 (.obj) (Cython)")


classes = (
    ExportFastOBJ,
    # ImportFastOBJ,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)
    # bpy.types.TOPBAR_MT_file_import.append(menu_func_import)
    
    # setup()


def unregister():
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)
    # bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
