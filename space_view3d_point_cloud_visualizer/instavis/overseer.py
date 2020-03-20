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
from bpy.types import Operator
from bpy.types import PropertyGroup, UIList
from bpy.props import PointerProperty, BoolProperty, StringProperty, FloatProperty, IntProperty, FloatVectorProperty, EnumProperty, CollectionProperty

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
    
    @classmethod
    def apply_settings_psys(cls, source, destinations, ):
        psystems = [mod.particle_system for mod in destinations]
        psettings = [p.settings for p in psystems]
        
        update = False
        apsys = source.particle_system
        apset = apsys.settings
        apcviv = apset.pcv_instavis
        
        for pset in psettings:
            if(apset is pset):
                continue
            pcviv = pset.pcv_instavis
            pcviv.point_scale = apcviv.point_scale
            pcviv.draw = apcviv.draw
            pcviv.display = apcviv.display
            pcviv.use_origins_only = apcviv.use_origins_only
            update = True
        
        if(update):
            PCVIVMechanist.force_update(with_caches=False, )
    
    @classmethod
    def apply_settings_instances(cls, source, destinations, ):
        pass


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
        last_sel = None
        if(len(user_sel)):
            ts = [(m.particle_system.settings.scatter_ui.is_selected_time, i, ) for i, m in enumerate(user_sel)]
            ts.sort()
            tsl, tsli = ts[len(ts) - 1]
            if(tsl > 0.0):
                last_sel = user_sel[tsli]
        
        return tuple(sc_mods), tuple(user_sel), last_sel


def pcviv_draw_sc_ui(context, uilayout, ):
    addon_prefs = bpy.context.preferences.addons["Scatter"].preferences
    terrain = bpy.context.scene.C_Slots_settings.Terrain_pointer
    scatter_particles, scatter_selected, last_sel = SCUtils.collect()
    
    # big enable button
    tab = uilayout.box()
    r = tab.row()
    r.prop(addon_prefs, 'A_instavis_enabled', toggle=True, )
    r.scale_y = 1.5
    
    # last selected particles
    tab = uilayout.box()
    h = tab.box()
    h.operator("scatter.general_panel_toggle", emboss=False, text='Particle Visualization Options', icon="NONE", ).pref = "addon_prefs.A_instavis_controls_is_open"
    if(addon_prefs.A_instavis_controls_is_open):
        if(last_sel is not None):
            pset_pcviv = last_sel.particle_system.settings.pcv_instavis
            cc = tab.column()
            cc.prop(pset_pcviv, 'display')
            r = cc.row()
            r.prop(pset_pcviv, 'point_scale')
            if(pset_pcviv.use_origins_only):
                r.enabled = False
            r = cc.row()
            r.prop(pset_pcviv, 'use_origins_only')
            ccc = r.column(align=True)
            pcviv_prefs = context.scene.pcv_instavis
            if(pcviv_prefs.quality == 'BASIC'):
                ccc.prop(pcviv_prefs, 'origins_point_size')
            else:
                ccc.prop(pcviv_prefs, 'origins_point_size_f')
            if(not pset_pcviv.use_origins_only):
                ccc.enabled = False
        else:
            cc = tab.column()
            cc.label(text="No Selected System(s)", icon='ERROR', )
        
        tab.separator()
        
        c = tab.column()
        r = c.row(align=True)
        r.label(text="{}:".format(addon_prefs.bl_rna.properties['A_instavis_influence'].name))
        r.prop(addon_prefs, 'A_instavis_influence', expand=True, )
        t = "{} to {}".format(PCVIV_OT_sc_apply_settings_psys.bl_label, addon_prefs.A_instavis_influence, )
        if(addon_prefs.A_instavis_influence in ('SCENE', )):
            t = "{} [Batch]".format(t)
        r = tab.row()
        r.operator('pcviv.sc_apply_settings_psys', text=t, )
        if(len(scatter_selected) <= 1):
            r.enabled = False
        if(addon_prefs.A_instavis_influence in ('SCENE', )):
            r.enabled = True
        if(len(scatter_selected) == 0):
            r.enabled = False
    
    # last selected particles instanced objects
    tab = uilayout.box()
    h = tab.box()
    h.operator("scatter.general_panel_toggle", emboss=False, text='Instance Visualization Options', icon="NONE", ).pref = "addon_prefs.A_instavis_instances_is_open"
    if(addon_prefs.A_instavis_instances_is_open):
        if(last_sel is not None):
            c = tab.column()
            pset = last_sel.particle_system.settings
            if(pset.render_type == 'COLLECTION' and pset.instance_collection is not None):
                c.label(text='{}: Instanced Collection Objects:'.format(pset.name))
                
                col = pset.instance_collection
                pcvcol = col.pcv_instavis
                c.template_list("PCVIV_UL_instances", "", col, "objects", pcvcol, "active_index", rows=5, )
                
                co = col.objects[col.objects.keys()[pcvcol.active_index]]
                pcvco = co.pcv_instavis
                
                c.label(text='Base Object "{}" Settings:'.format(co.name), )
                
                pcviv_prefs = context.scene.pcv_instavis
                if(pcviv_prefs.quality == 'BASIC'):
                    c.prop(pcvco, 'point_size')
                else:
                    c.prop(pcvco, 'point_size_f')
                
                c.prop(pcvco, 'source', )
                c.prop(pcvco, 'max_points')
                
                if(pcvco.source == 'VERTICES'):
                    r = c.row()
                    r.prop(pcvco, 'color_constant', )
                else:
                    c.prop(pcvco, 'color_source', )
                    if(pcvco.color_source == 'CONSTANT'):
                        r = c.row()
                        r.prop(pcvco, 'color_constant', )
                    else:
                        c.prop(pcvco, 'use_face_area')
                        c.prop(pcvco, 'use_material_factors')
                
                if(pcvco.use_material_factors):
                    b = c.box()
                    cc = b.column(align=True)
                    for slot in co.material_slots:
                        if(slot.material is not None):
                            cc.prop(slot.material.pcv_instavis, 'factor', text=slot.material.name)
            elif(pset.render_type == 'OBJECT' and pset.instance_object is not None):
                c.label(text='{}: Instanced Object:'.format(pset.name))
                
                co = pset.instance_object
                b = c.box()
                b.label(text=co.name, icon='OBJECT_DATA', )
                
                c.label(text='Base Object "{}" Settings:'.format(co.name), )
                
                pcvco = co.pcv_instavis
                
                pcviv_prefs = context.scene.pcv_instavis
                if(pcviv_prefs.quality == 'BASIC'):
                    c.prop(pcvco, 'point_size')
                else:
                    c.prop(pcvco, 'point_size_f')
                
                c.prop(pcvco, 'source', )
                c.prop(pcvco, 'max_points')
                
                if(pcvco.source == 'VERTICES'):
                    r = c.row()
                    r.prop(pcvco, 'color_constant', )
                else:
                    c.prop(pcvco, 'color_source', )
                    if(pcvco.color_source == 'CONSTANT'):
                        r = c.row()
                        r.prop(pcvco, 'color_constant', )
                    else:
                        c.prop(pcvco, 'use_face_area')
                        c.prop(pcvco, 'use_material_factors')
                
                if(pcvco.use_material_factors):
                    b = c.box()
                    cc = b.column(align=True)
                    for slot in co.material_slots:
                        if(slot.material is not None):
                            cc.prop(slot.material.pcv_instavis, 'factor', text=slot.material.name)
            else:
                c.label(text="No collection/object found.", icon='ERROR', )
        else:
            cc = tab.column()
            cc.label(text="No Selected System(s)", icon='ERROR', )
        
        tab.separator()
        
        c = tab.column()
        r = c.row(align=True)
        r.label(text="{}:".format(addon_prefs.bl_rna.properties['A_instavis_influence_instances'].name))
        r.prop(addon_prefs, 'A_instavis_influence_instances', expand=True, )
        t = "{} to {}".format(PCVIV_OT_sc_apply_settings_instances.bl_label, addon_prefs.A_instavis_influence_instances, )
        if(addon_prefs.A_instavis_influence_instances in ('SCENE', 'SELECTED', )):
            t = "{} [Batch]".format(t)
        r = tab.row()
        r.operator('pcviv.sc_apply_settings_instances', text=t, )
        if(len(scatter_selected) <= 1):
            r.enabled = False
        if(addon_prefs.A_instavis_influence_instances in ('SCENE', )):
            r.enabled = True
        if(len(scatter_selected) == 0):
            r.enabled = False
    
    # global settings
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


class PCVIV_OT_sc_apply_settings_psys(Operator):
    bl_idname = "pcviv.sc_apply_settings_psys"
    bl_label = "Apply Settings"
    bl_description = "Apply settings from active to selected or all in scene"
    
    @classmethod
    def poll(cls, context):
        if(PCVIVMechanist.initialized):
            return True
        return False
    
    def execute(self, context):
        scatter_particles, scatter_selected, last_sel = SCUtils.collect()
        addon_prefs = bpy.context.preferences.addons["Scatter"].preferences
        if(addon_prefs.A_instavis_influence == 'SCENE'):
            destinations = scatter_particles
        else:
            destinations = scatter_selected
        PCVIVOverseer.apply_settings_psys(last_sel, destinations, )
        return {'FINISHED'}


class PCVIV_OT_sc_apply_settings_instances(Operator):
    bl_idname = "pcviv.sc_apply_settings_instances"
    bl_label = "Apply Settings"
    bl_description = "Apply settings from active to selected or all in scene"
    
    @classmethod
    def poll(cls, context):
        if(PCVIVMechanist.initialized):
            return True
        return False
    
    def execute(self, context):
        scatter_particles, scatter_selected, last_sel = SCUtils.collect()
        addon_prefs = bpy.context.preferences.addons["Scatter"].preferences
        if(addon_prefs.A_instavis_influence == 'SCENE'):
            destinations = scatter_particles
        else:
            destinations = scatter_selected
        # TODO: and get active object in collection to copy settings
        PCVIVOverseer.apply_settings_instances(last_sel, destinations, )
        return {'FINISHED'}


@bpy.app.handlers.persistent
def auto_init(undefined):
    PCVIVOverseer.init()


# auto initialize, this will be called once when blend file is loaded, even startup file
bpy.app.handlers.load_post.append(auto_init)

classes = (PCVIV_OT_sc_apply_settings_psys, PCVIV_OT_sc_apply_settings_instances, )
classes_debug = ()
