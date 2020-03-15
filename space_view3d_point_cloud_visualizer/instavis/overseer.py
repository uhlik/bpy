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

import bpy

from .debug import debug_mode, log
from .mechanist import PCVIVMechanist


class PCVIVOverseer():
    @classmethod
    def init(cls):
        PCVIVMechanist.init()
    
    @classmethod
    def deinit(cls):
        PCVIVMechanist.deinit()
        bpy.app.handlers.load_post.remove(auto_init)


class SCUtils():
    @classmethod
    def collect(cls, scene=False, ):
        prefix = "SCATTER"
        sc_mods = []
        if(scene):
            sc_mods = [m for o in bpy.context.scene.objects for m in o.modifiers if m.name.startswith(prefix)]
        else:
            o = bpy.context.scene.C_Slots_settings.Terrain_pointer
            sc_mods = [m for m in o.modifiers if m.name.startswith(prefix)]
        user_sel = [m for m in sc_mods if m.particle_system.settings.scatter_ui.is_selected]
        return tuple(sc_mods), tuple(user_sel)


def pcviv_draw_sc_ui(context, uilayout, ):
    addon_prefs = bpy.context.preferences.addons["Scatter"].preferences
    terrain = bpy.context.scene.C_Slots_settings.Terrain_pointer
    scatter_particles, scatter_selected = SCUtils.collect()
    
    def flags():
        use_batch = False
        use_disable = False
        ext = ''
        if((addon_prefs.A_instavis_influence == 'SELECTED' and len(scatter_selected) not in [0, 1]) or addon_prefs.A_instavis_influence == 'SCENE'):
            ext = ' [Batch]'
            use_batch = True
        else:
            ext = ''
        if(addon_prefs.A_instavis_influence == 'SELECTED' and len(scatter_particles) == 0):
            name = 'No Created System(s) Yet'
            use_batch = True
            use_disable = True
        elif(addon_prefs.A_instavis_influence == 'SELECTED' and len(scatter_selected) == 0):
            name = 'No Selected System(s)'
            use_batch = True
            use_disable = True
        else:
            name = 'Instavis:' + ext
        return use_batch, use_disable, name
    
    use_batch, use_disable, name = flags()
    
    tab = uilayout.box()
    h = tab.box()
    h.operator("scatter.general_panel_toggle", emboss=False, text=name, icon="NONE", ).pref = "addon_prefs.A_instavis_controls_is_open"
    if(addon_prefs.A_instavis_controls_is_open):
        tab.label(text='hello')
    
    tab = uilayout.box()
    h = tab.box()
    h.operator("scatter.general_panel_toggle", emboss=False, text="Global Settings", icon='NONE', ).pref = "addon_prefs.A_instavis_global_is_open"
    if(addon_prefs.A_instavis_global_is_open):
        c = tab.column()
        pcviv_prefs = context.scene.pcv_instavis
        c.prop(pcviv_prefs, 'quality')
        c.prop(pcviv_prefs, 'update_method')
        c.separator()
        c.prop(pcviv_prefs, 'use_exit_display')
        cc = c.column()
        cc.prop(pcviv_prefs, 'exit_object_display_type')
        cc.prop(pcviv_prefs, 'exit_psys_display_method')
        cc.enabled = pcviv_prefs.use_exit_display
        c.separator()
        c.label(text="Auto Switch To Origins Only:")
        c.prop(pcviv_prefs, 'switch_origins_only', text='Enabled', )
        c.prop(pcviv_prefs, 'switch_origins_only_threshold')


@bpy.app.handlers.persistent
def auto_init(undefined):
    PCVIVOverseer.init()


# auto initialize, this will be called once when blend file is loaded, even startup file
bpy.app.handlers.load_post.append(auto_init)
