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
from bpy.props import PointerProperty, BoolProperty, StringProperty, FloatProperty, IntProperty, FloatVectorProperty, EnumProperty, CollectionProperty
from bpy.types import PropertyGroup

from .debug import debug_mode, log
from .mechanist import PCVIVMechanist


class PCVIVOverseer():
    @classmethod
    def init(cls):
        PCVIVMechanist.init()
    
    @classmethod
    def deinit(cls):
        PCVIVMechanist.deinit()
        if(auto_init in bpy.app.handlers.load_post):
            bpy.app.handlers.load_post.remove(auto_init)
    
    @classmethod
    def sc_enable(cls, destinations, enable, ):
        PCVIVMechanist.init()
        
        psettings = [mod.particle_system.settings for mod in destinations]
        for pset in psettings:
            pset.pcv_instavis.use = enable
        PCVIVMechanist.force_update()
    
    @classmethod
    def sc_draw_type(cls, destinations, draw_type, ):
        psettings = [mod.particle_system.settings for mod in destinations]
        for pset in psettings:
            if(draw_type == 'ORIGINS'):
                pset.pcv_instavis.use_origins_only = True
            else:
                pset.pcv_instavis.use_origins_only = False
        PCVIVMechanist.force_update()
    
    sc_psys_prop_map = {
        'point_scale': 'point_scale',
        # 'point_percentage': 'display',
        'origins_point_size': 'origins_point_size',
        'origins_point_size_f': 'origins_point_size_f',
    }
    
    @classmethod
    def sc_apply_psys_prop(cls, context, destinations, prop_name, ):
        scp = context.scene.pcv_instavis_sc_props
        try:
            v = scp[prop_name]
        except Exception as e:
            v = scp.bl_rna.properties[prop_name].default
        n = cls.sc_psys_prop_map[prop_name]
        psettings = [mod.particle_system.settings for mod in destinations]
        for pset in psettings:
            pset.pcv_instavis[n] = v
        PCVIVMechanist.force_update()


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


'''
def pcviv_draw_sc_ui_obsolete(context, uilayout, ):
    sc_prefs = bpy.context.preferences.addons["Scatter"].preferences
    terrain = bpy.context.scene.C_Slots_settings.Terrain_pointer
    scatter_particles, scatter_selected, last_sel = SCUtils.collect()
    
    # big enable button
    tab = uilayout.box()
    r = tab.row()
    r.prop(sc_prefs, 'A_instavis_enabled', toggle=True, )
    r.scale_y = 1.5
    
    # last selected particles
    tab = uilayout.box()
    h = tab.box()
    h.operator("scatter.general_panel_toggle", emboss=False, text='Particle Visualization Options', icon='PARTICLES', ).pref = "sc_prefs.A_instavis_controls_is_open"
    if(sc_prefs.A_instavis_controls_is_open):
        if(last_sel is not None):
            pset_pcviv = last_sel.particle_system.settings.pcv_instavis
            cc = tab.column()
            cc.label(text='{} > {} > {}'.format(terrain.name, last_sel.particle_system.name, last_sel.particle_system.settings.name))
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
        r.label(text="{}:".format(sc_prefs.bl_rna.properties['A_instavis_influence'].name))
        r.prop(sc_prefs, 'A_instavis_influence', expand=True, )
        t = "{} to {}".format(PCVIV_OT_sc_apply_settings_psys.bl_label, sc_prefs.A_instavis_influence, )
        if(sc_prefs.A_instavis_influence in ('SCENE', )):
            t = "{} [Batch]".format(t)
        r = tab.row()
        r.operator('pcviv.sc_apply_settings_psys', text=t, )
    
    # last selected particles instanced objects
    tab = uilayout.box()
    h = tab.box()
    h.operator("scatter.general_panel_toggle", emboss=False, text='Instance Visualization Options', icon='OUTLINER_OB_GROUP_INSTANCE', ).pref = "sc_prefs.A_instavis_instances_is_open"
    if(sc_prefs.A_instavis_instances_is_open):
        if(last_sel is not None):
            c = tab.column()
            pset = last_sel.particle_system.settings
            if(pset.render_type == 'COLLECTION' and pset.instance_collection is not None):
                c.label(text='{} > {}'.format(last_sel.particle_system.settings.name, pset.instance_collection.name))
                
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
                c.label(text='{} > {}'.format(last_sel.particle_system.settings.name, pset.instance_object.name))
                
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
        r.label(text="{}:".format(sc_prefs.bl_rna.properties['A_instavis_influence_instances'].name))
        r.prop(sc_prefs, 'A_instavis_influence_instances', expand=True, )
        t = "{} to {}".format(PCVIV_OT_sc_apply_settings_instances.bl_label, sc_prefs.A_instavis_influence_instances, )
        if(sc_prefs.A_instavis_influence_instances in ('SCENE', 'SELECTED', )):
            t = "{} [Batch]".format(t)
        r = tab.row()
        r.operator('pcviv.sc_apply_settings_instances', text=t, )
    
    # global settings
    tab = uilayout.box()
    h = tab.box()
    h.operator("scatter.general_panel_toggle", emboss=False, text="Global Settings", icon='SETTINGS', ).pref = "sc_prefs.A_instavis_global_is_open"
    if(sc_prefs.A_instavis_global_is_open):
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
'''


def pcviv_draw_sc_ui(context, uilayout, ):
    sc_prefs = bpy.context.preferences.addons["Scatter"].preferences
    scatter_particles, scatter_selected, last_sel = SCUtils.collect()
    
    tab = uilayout.box()
    
    c = tab.column()
    r = c.row(align=True)
    r.label(text="{}:".format(sc_prefs.bl_rna.properties['A_instavis_influence'].name))
    r.prop(sc_prefs, 'A_instavis_influence', expand=True, )
    
    c.label(text="Visualization:", )
    
    r = c.row(align=True)
    r.operator('pcviv.sc_enable', text="Start", ).enable = True
    r.operator('pcviv.sc_enable', text="Stop", ).enable = False
    
    scp = context.scene.pcv_instavis_sc_props
    
    c.label(text="Display:", )
    cdt = c.column()
    
    cc = cdt.column(align=True)
    cc.operator('pcviv.sc_draw_type', text="Full", ).draw_type = 'FULL'
    r = cc.row(align=True)
    r.prop(scp, 'point_scale')
    r.operator('pcviv.sc_apply_psys_prop', ).prop_name = 'point_scale'
    
    cc = cdt.column(align=True)
    cc.operator('pcviv.sc_draw_type', text="Origins", ).draw_type = 'ORIGINS'
    r = cc.row(align=True)
    pcviv_prefs = context.scene.pcv_instavis
    if(pcviv_prefs.quality == 'BASIC'):
        r.prop(scp, 'origins_point_size')
        r.operator('pcviv.sc_apply_psys_prop', ).prop_name = 'origins_point_size'
    else:
        r.prop(scp, 'origins_point_size_f')
        r.operator('pcviv.sc_apply_psys_prop', ).prop_name = 'origins_point_size_f'
    
    if(sc_prefs.A_instavis_influence == 'SELECTED'):
        if(not len(scatter_selected)):
            cdt.enabled = False


'''
class PCVIV_OT_sc_apply_settings_psys(Operator):
    bl_idname = "pcviv.sc_apply_settings_psys"
    bl_label = "Apply Settings"
    bl_description = "Apply settings from active to selected or all in scene"
    
    @classmethod
    def poll(cls, context):
        if(PCVIVMechanist.initialized):
            sc_prefs = bpy.context.preferences.addons["Scatter"].preferences
            scatter_particles, scatter_selected, last_sel = SCUtils.collect()
            if(len(scatter_selected) > 0):
                if(sc_prefs.A_instavis_influence in ('SELECTED', )):
                    if(len(scatter_selected) <= 1):
                        return False
                return True
        return False
    
    def execute(self, context):
        scatter_particles, scatter_selected, last_sel = SCUtils.collect()
        sc_prefs = bpy.context.preferences.addons["Scatter"].preferences
        if(sc_prefs.A_instavis_influence == 'SCENE'):
            destinations = scatter_particles
        else:
            destinations = scatter_selected
        if(last_sel is not None):
            PCVIVOverseer.apply_settings_psys(last_sel, destinations, )
        return {'FINISHED'}


class PCVIV_OT_sc_apply_settings_instances(Operator):
    bl_idname = "pcviv.sc_apply_settings_instances"
    bl_label = "Apply Settings"
    bl_description = "Apply settings from active to selected or all in scene"
    
    @classmethod
    def poll(cls, context):
        if(PCVIVMechanist.initialized):
            scatter_particles, scatter_selected, last_sel = SCUtils.collect()
            if(len(scatter_selected) > 0):
                return True
        return False
    
    def execute(self, context):
        scatter_particles, scatter_selected, last_sel = SCUtils.collect()
        sc_prefs = bpy.context.preferences.addons["Scatter"].preferences
        if(sc_prefs.A_instavis_influence == 'SCENE'):
            destinations = scatter_particles
        elif(sc_prefs.A_instavis_influence == 'SELECTED'):
            destinations = scatter_selected
        else:
            destinations = last_sel
        pset = last_sel.particle_system.settings
        if(pset.render_type == 'COLLECTION' and pset.instance_collection is not None):
            col = pset.instance_collection
            pcvcol = col.pcv_instavis
            source = col.objects[col.objects.keys()[pcvcol.active_index]]
        elif(pset.render_type == 'OBJECT' and pset.instance_object is not None):
            source.pset.instance_object
        if(source is not None):
            PCVIVOverseer.apply_settings_instances(source, destinations, )
        return {'FINISHED'}
'''


class PCVIV_sc_properties(PropertyGroup):
    point_scale: FloatProperty(name="Point Scale", default=1.0, min=0.001, max=10.0, precision=2, description="Adjust point size of all points", )
    # point_percentage: FloatProperty(name="Point Percentage", default=100.0, min=0.0, max=100.0, precision=0, subtype='PERCENTAGE', description="Adjust percentage of displayed points", )
    origins_point_size: IntProperty(name="Origin Size", default=6, min=1, max=10, subtype='PIXEL', description="Point size", )
    origins_point_size_f: FloatProperty(name="Origin Size", default=0.05, min=0.001, max=1.0, precision=2, description="Point size", )
    
    @classmethod
    def register(cls):
        bpy.types.Scene.pcv_instavis_sc_props = PointerProperty(type=cls)
    
    @classmethod
    def unregister(cls):
        del bpy.types.Scene.pcv_instavis_sc_props


class PCVIV_OT_sc_base(Operator):
    bl_idname = "pcviv.sc_base"
    bl_label = "Base Operator"
    bl_description = ""
    
    @classmethod
    def poll(cls, context):
        sc_prefs = bpy.context.preferences.addons["Scatter"].preferences
        if(sc_prefs.A_instavis_influence in ('SCENE', )):
            # whole scene influence, can be run anytime
            return True
        # selected influence
        scatter_particles, scatter_selected, last_sel = SCUtils.collect()
        if(len(scatter_selected) > 0):
            # at least one particle system selected to operate on
            return True
        return False
    
    def destinations(self):
        scatter_particles, scatter_selected, last_sel = SCUtils.collect()
        sc_prefs = bpy.context.preferences.addons["Scatter"].preferences
        r = scatter_selected
        if(sc_prefs.A_instavis_influence == 'SCENE'):
            r = scatter_particles
        return r


class PCVIV_OT_sc_enable(PCVIV_OT_sc_base):
    bl_idname = "pcviv.sc_enable"
    bl_label = "Start/Stop"
    bl_description = "Start/stop visualization for all/selected particle system(s)"
    
    enable: BoolProperty(default=True, options={'HIDDEN', 'SKIP_SAVE', }, )
    
    def execute(self, context):
        PCVIVOverseer.sc_enable(self.destinations(), self.enable, )
        return {'FINISHED'}


class PCVIV_OT_sc_draw_type(PCVIV_OT_sc_base):
    bl_idname = "pcviv.sc_draw_type"
    bl_label = "Draw Type"
    bl_description = ""
    
    draw_type: EnumProperty(items=[('FULL', '', "", ), ('ORIGINS', '', "", ), ], default='FULL', options={'HIDDEN', 'SKIP_SAVE', }, )
    
    def execute(self, context):
        PCVIVOverseer.sc_draw_type(self.destinations(), self.draw_type, )
        return {'FINISHED'}


class PCVIV_OT_sc_apply_psys_prop(PCVIV_OT_sc_base):
    bl_idname = "pcviv.sc_apply_psys_prop"
    bl_label = "Apply"
    bl_description = ""
    
    prop_name: StringProperty(default='', options={'HIDDEN', 'SKIP_SAVE', }, )
    
    def execute(self, context):
        PCVIVOverseer.sc_apply_psys_prop(context, self.destinations(), self.prop_name, )
        return {'FINISHED'}


# NOTE: somehow it conflicts with Scatter, clouds are drawn twice, viewport display icon has no effect, msgbus notifications are lost (?) and deinit throws errors, like handlers were not set, disabling auto start and adding init() call into Start Operator.. will see in future how it goes..
# @bpy.app.handlers.persistent
# def auto_init(undefined):
#     PCVIVOverseer.init()
#
#
# # auto initialize, this will be called once when blend file is loaded, even startup file
# bpy.app.handlers.load_post.append(auto_init)

classes = (PCVIV_sc_properties, PCVIV_OT_sc_enable, PCVIV_OT_sc_draw_type, PCVIV_OT_sc_apply_psys_prop, )
classes_debug = ()
