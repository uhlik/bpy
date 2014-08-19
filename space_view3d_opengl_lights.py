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

bl_info = {"name": "OpenGL Lights",
           "description": "Quick access to Solid OpenGL Lights with preset functionality",
           "author": "Jakub Uhlik",
           "version": (0, 1, 0),
           "blender": (2, 70, 0),
           "location": "View3d > Properties > OpenGL Lights",
           "warning": "",
           "wiki_url": "",
           "tracker_url": "",
           "category": "3D View", }


import bpy
from bl_operators.presets import AddPresetBase


class OpenGLLightsProperties(bpy.types.PropertyGroup):
    @classmethod
    def register(cls):
        bpy.types.Scene.opengl_lights_properties = bpy.props.PointerProperty(type=cls)
        cls.edit = bpy.props.BoolProperty(name="Edit OpenGL Lights presets", description="", default=False, )
    
    @classmethod
    def unregister(cls):
        del bpy.types.Scene.opengl_lights_properties


class VIEW3D_OT_opengl_lights_preset_add(AddPresetBase, bpy.types.Operator):
    # http://blender.stackexchange.com/a/2509/2555
    bl_idname = 'scene.opengl_lights_preset_add'
    bl_label = 'Add OpenGL Lights Preset'
    bl_options = {'REGISTER', 'UNDO'}
    preset_menu = 'VIEW3D_MT_opengl_lights_presets'
    preset_subdir = 'opengl_lights_presets'
    
    preset_defines = [
        "l0 = bpy.context.user_preferences.system.solid_lights[0]",
        "l1 = bpy.context.user_preferences.system.solid_lights[1]",
        "l2 = bpy.context.user_preferences.system.solid_lights[2]",
    ]
    preset_values = [
        "l0.use", "l0.diffuse_color", "l0.specular_color", "l0.direction",
        "l1.use", "l1.diffuse_color", "l1.specular_color", "l1.direction",
        "l2.use", "l2.diffuse_color", "l2.specular_color", "l2.direction",
    ]


class VIEW3D_MT_opengl_lights_presets(bpy.types.Menu):
    # http://blender.stackexchange.com/a/2509/2555
    bl_label = "OpenGL Lights Presets"
    bl_idname = "VIEW3D_MT_opengl_lights_presets"
    preset_subdir = "opengl_lights_presets"
    preset_operator = "script.execute_preset"
    
    draw = bpy.types.Menu.draw_preset


class VIEW3D_PT_opengl_lights_panel(bpy.types.Panel):
    bl_label = 'OpenGL Lights'
    bl_space_type = 'VIEW_3D'
    bl_context = "scene"
    bl_region_type = 'UI'
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        system = bpy.context.user_preferences.system
        
        def opengl_lamp_buttons(column, lamp):
            # from space_userpref.py
            split = column.split(percentage=0.1)
            split.prop(lamp, "use", text="", icon='OUTLINER_OB_LAMP' if lamp.use else 'LAMP_DATA')
            col = split.column()
            col.active = lamp.use
            row = col.row()
            row.label(text="Diffuse:")
            row.prop(lamp, "diffuse_color", text="")
            row = col.row()
            row.label(text="Specular:")
            row.prop(lamp, "specular_color", text="")
            col = split.column()
            col.active = lamp.use
            col.prop(lamp, "direction", text="")
        
        layout = self.layout
        
        p = context.scene.opengl_lights_properties
        layout.prop(p, "edit")
        
        if(p.edit):
            column = layout.column()
            
            split = column.split(percentage=0.1)
            split.label()
            split.label(text="Colors:")
            split.label(text="Direction:")
            
            lamp = system.solid_lights[0]
            opengl_lamp_buttons(column, lamp)
            
            lamp = system.solid_lights[1]
            opengl_lamp_buttons(column, lamp)
            
            lamp = system.solid_lights[2]
            opengl_lamp_buttons(column, lamp)
        
        col = layout.column_flow(align=True)
        row = col.row(align=True)
        row.menu("VIEW3D_MT_opengl_lights_presets", text=bpy.types.VIEW3D_MT_opengl_lights_presets.bl_label)
        row.operator("scene.opengl_lights_preset_add", text="", icon='ZOOMIN')
        row.operator("scene.opengl_lights_preset_add", text="", icon='ZOOMOUT').remove_active = True


def register():
    bpy.utils.register_class(OpenGLLightsProperties)
    bpy.utils.register_class(VIEW3D_OT_opengl_lights_preset_add)
    bpy.utils.register_class(VIEW3D_MT_opengl_lights_presets)
    bpy.utils.register_class(VIEW3D_PT_opengl_lights_panel)


def unregister():
    bpy.utils.unregister_class(OpenGLLightsProperties)
    bpy.utils.unregister_class(VIEW3D_OT_opengl_lights_preset_add)
    bpy.utils.unregister_class(VIEW3D_MT_opengl_lights_presets)
    bpy.utils.unregister_class(VIEW3D_PT_opengl_lights_panel)


if __name__ == "__main__":
    register()
