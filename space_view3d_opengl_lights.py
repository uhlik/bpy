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
           "version": (0, 2, 0),
           "blender": (2, 70, 0),
           "location": "View3d > Properties > OpenGL Lights",
           "warning": "",
           "wiki_url": "",
           "tracker_url": "",
           "category": "3D View", }

# changelog:
# 2014.08.24 added defaults, created when no presets are available
# 2014.08.19 first release

import os

import bpy
from bl_operators.presets import AddPresetBase


def get_default_presets():
    presets = {
        'blender_default': [
            (True, (0.800000011920929, 0.800000011920929, 0.800000011920929), (0.5, 0.5, 0.5), (-0.8920000791549683, 0.30000001192092896, 0.8999999761581421), ),
            (True, (0.4980020225048065, 0.500000536441803, 0.6000001430511475), (0.20000000298023224, 0.20000000298023224, 0.20000000298023224), (0.5880000591278076, 0.46000003814697266, 0.24800002574920654), ),
            (True, (0.7980005145072937, 0.8379999399185181, 1.0), (0.06599999219179153, 0.0, 0.0), (0.21599984169006348, -0.3920000195503235, -0.21599996089935303), ),
        ],
        'mshade3': [
            (True, (0.43621575832366943, 0.33640265464782715, 0.31929048895835876), (0.9113942980766296, 0.8065879940986633, 0.5334088206291199), (-0.012820512987673283, 0.44871795177459717, 0.8935815095901489), ),
            (True, (0.5633640289306641, 0.35928845405578613, 0.24127797782421112), (0.0, 0.0, 0.0), (0.0, 0.9230769276618958, 0.38461539149284363), ),
            (True, (0.36491984128952026, 0.206924706697464, 0.17659659683704376), (0.5704712867736816, 0.5164262652397156, 0.41887882351875305), (0.012177534401416779, -0.9742027521133423, -0.22534571588039398), ),
        ],
        'mshade5': [
            (True, (0.4753793776035309, 0.4753793776035309, 0.4753793776035309), (0.5, 0.5, 0.5), (-0.012820512987673283, 0.44871795177459717, 0.8935815095901489), ),
            (True, (0.7233469486236572, 0.7233469486236572, 0.7233469486236572), (0.0, 0.0, 0.0), (0.0, 0.9230769276618958, 0.38461539149284363), ),
            (True, (0.1267620027065277, 0.12150080502033234, 0.11475051194429398), (0.0, 0.0, 0.0), (0.012177534401416779, -0.9742027521133423, -0.22534571588039398), ),
        ],
        'carbon': [
            (True, (0.800000011920929, 0.800000011920929, 0.800000011920929), (0.5, 0.5, 0.5), (-0.8920000195503235, 0.30000001192092896, 0.8999999761581421), ),
            (True, (0.43799999356269836, 0.503000020980835, 0.6000000238418579), (0.17000000178813934, 0.18199999630451202, 0.20000000298023224), (0.5879999995231628, 0.46000000834465027, 0.24799999594688416), ),
            (True, (0.6000000238418579, 0.49000000953674316, 0.36000001430511475), (0.15000000596046448, 0.08100000023841858, 0.0), (0.2160000056028366, -0.3919999897480011, -0.2160000056028366), ),
        ],
    }
    return presets


def setup():
    # make sure there is directory for presets
    preset_subdir = "opengl_lights_presets"
    preset_directory = os.path.join(bpy.utils.user_resource('SCRIPTS'), "presets", preset_subdir)
    preset_paths = bpy.utils.preset_paths(preset_subdir)
    if(preset_directory not in preset_paths):
        if(not os.path.exists(preset_directory)):
            os.makedirs(preset_directory)
    
    # search for presets, .py file is considered as preset
    def walk(p):
        r = {'files': [], 'dirs': [], }
        for(root, dirs, files) in os.walk(p):
            r['files'].extend(files)
            r['dirs'].extend(dirs)
            break
        return r
    
    found = []
    for p in preset_paths:
        c = walk(p)
        for f in c['files']:
            if(f.endswith(".py")):
                found.append(f[:-3])
    
    if(len(found) == 0):
        # nothing found, write default presets
        default_presets = get_default_presets()
        e = "\n"
        for n, p in default_presets.items():
            s = ""
            s += "import bpy" + e
            for i in range(3):
                s += "l{0} = bpy.context.user_preferences.system.solid_lights[{0}]".format(i) + e
            s += e
            for i in range(3):
                s += "l{}.use = {}{}".format(i, p[i][0], e)
                s += "l{}.diffuse_color = {}{}".format(i, p[i][1], e)
                s += "l{}.specular_color = {}{}".format(i, p[i][2], e)
                s += "l{}.direction = {}{}".format(i, p[i][3], e)
            
            with open(os.path.join(preset_directory, "{}.py".format(n)), mode='w', encoding='utf-8') as f:
                f.write(s)


class OpenGLLightsProperties(bpy.types.PropertyGroup):
    @classmethod
    def register(cls):
        bpy.types.Scene.opengl_lights_properties = bpy.props.PointerProperty(type=cls)
        cls.edit = bpy.props.BoolProperty(name="Edit", description="", default=False, )
    
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
    setup()
    
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
