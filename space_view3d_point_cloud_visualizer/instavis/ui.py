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
from bpy.types import Panel, UIList

from .debug import debug_mode, log
from .mechanist import PCVIVMechanist


class PCVIV_PT_base(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "PCVIV"
    bl_label = "PCVIV base"
    bl_options = {'DEFAULT_CLOSED'}
    
    def prop_name(self, cls, prop, colon=False, ):
        for p in cls.bl_rna.properties:
            if(p.identifier == prop):
                if(colon):
                    return "{}:".format(p.name)
                return p.name
        return ''
    
    def third_label_two_thirds_prop(self, cls, prop, uil, ):
        f = 0.33
        r = uil.row()
        s = r.split(factor=f)
        s.label(text=self.prop_name(cls, prop, True, ))
        s = s.split(factor=1.0)
        r = s.row()
        r.prop(cls, prop, text='', )
    
    @classmethod
    def poll(cls, context):
        o = context.active_object
        if(o is None):
            return False
        return True


class PCVIV_PT_main(PCVIV_PT_base):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    # bl_category = "View"
    bl_category = "PCVIV"
    bl_label = "PCV Instance Visualizer"
    # bl_parent_id = "PCV_PT_panel"
    # bl_options = {'DEFAULT_CLOSED'}
    bl_options = set()
    
    @classmethod
    def poll(cls, context):
        # o = context.active_object
        # if(o is None):
        #     return False
        return True
    
    def draw(self, context):
        o = context.active_object
        if(not o):
            self.layout.label(text='Select an object..', icon='ERROR', )
            return
        
        l = self.layout
        c = l.column()
        
        # active object
        c.label(text="Active Object:")
        r = c.row(align=True)
        cc = r.column(align=True)
        if(not o.pcv_instavis.target):
            cc.alert = True
        cc.prop(o.pcv_instavis, 'target', text='Target', toggle=True, )
        
        # manager
        c.label(text='PCVIV Mechanist:')
        r = c.row(align=True)
        cc = r.column(align=True)
        if(not PCVIVMechanist.initialized):
            cc.alert = True
        cc.operator('point_cloud_visualizer.pcviv_init')
        cc = r.column(align=True)
        if(PCVIVMechanist.initialized):
            cc.alert = True
        cc.operator('point_cloud_visualizer.pcviv_deinit')
        r = c.row()
        if(PCVIVMechanist.initialized):
            r.alert = True
        r.operator('point_cloud_visualizer.pcviv_force_update')


class PCVIV_PT_particles(PCVIV_PT_base):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    # bl_category = "View"
    bl_category = "PCVIV"
    bl_label = "Particle Systems"
    bl_parent_id = "PCVIV_PT_main"
    # bl_options = {'DEFAULT_CLOSED'}
    bl_options = set()
    
    @classmethod
    def poll(cls, context):
        o = context.active_object
        if(o is None):
            return False
        return True
    
    def draw(self, context):
        l = self.layout
        c = l.column()
        
        if(context.object is not None):
            o = context.object
            c.label(text='{}: Particle Systems:'.format(o.name))
            c.template_list("PARTICLE_UL_particle_systems", "particle_systems", o, "particle_systems", o.particle_systems, "active_index", rows=3, )
        
        # psys if there is any..
        n = 'n/a'
        if(context.object is not None):
            o = context.object
            if(o.particle_systems.active is not None):
                n = o.particle_systems.active.name
        c.label(text='Active Particle System: {}'.format(n))
        
        ok = False
        if(context.object is not None):
            o = context.object
            if(o.particle_systems.active is not None):
                ok = True
        if(ok):
            pset_pcviv = o.particle_systems.active.settings.pcv_instavis
            r = c.row()
            r.prop(pset_pcviv, 'draw', toggle=True, )
            r.scale_y = 1.5
            r = c.row()
            r.prop(pset_pcviv, 'display')
            r = c.row()
            r.prop(pset_pcviv, 'point_scale')
            
            if(pset_pcviv.use_origins_only):
                r.enabled = False
            c.prop(pset_pcviv, 'use_origins_only')
            
            cc = c.column(align=True)
            pcviv_prefs = context.scene.pcv_instavis
            if(pcviv_prefs.quality == 'BASIC'):
                cc.prop(pcviv_prefs, 'origins_point_size')
            else:
                cc.prop(pcviv_prefs, 'origins_point_size_f')
            if(not pset_pcviv.use_origins_only):
                cc.enabled = False


class PCVIV_UL_instances(UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, ):
        pcviv = context.object.pcv_instavis
        layout.label(text=item.name, icon='OBJECT_DATA', )


class PCVIV_PT_instances(PCVIV_PT_base):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    # bl_category = "View"
    bl_category = "PCVIV"
    bl_label = "Instance Options"
    bl_parent_id = "PCVIV_PT_main"
    bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        o = context.active_object
        if(o is None):
            return False
        return True
    
    def draw(self, context):
        l = self.layout
        c = l.column()
        
        ok = False
        if(context.object is not None):
            o = context.object
            c.label(text='{}: Particle Systems:'.format(o.name))
            c.template_list("PARTICLE_UL_particle_systems", "particle_systems", o, "particle_systems", o.particle_systems, "active_index", rows=3, )
            if(o.particle_systems.active is not None):
                pset = o.particle_systems.active.settings
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
                    
                    self.third_label_two_thirds_prop(pcvco, 'source', c, )
                    c.prop(pcvco, 'max_points')
                    
                    if(pcvco.source == 'VERTICES'):
                        r = c.row()
                        self.third_label_two_thirds_prop(pcvco, 'color_constant', r, )
                    else:
                        self.third_label_two_thirds_prop(pcvco, 'color_source', c, )
                        if(pcvco.color_source == 'CONSTANT'):
                            r = c.row()
                            self.third_label_two_thirds_prop(pcvco, 'color_constant', r, )
                        else:
                            c.prop(pcvco, 'use_face_area')
                            c.prop(pcvco, 'use_material_factors')
                    
                    if(pcvco.use_material_factors):
                        b = c.box()
                        cc = b.column(align=True)
                        for slot in co.material_slots:
                            if(slot.material is not None):
                                cc.prop(slot.material.pcv_instavis, 'factor', text=slot.material.name)
                    c.operator('point_cloud_visualizer.pcviv_apply_generator_settings')
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
                    
                    self.third_label_two_thirds_prop(pcvco, 'source', c, )
                    c.prop(pcvco, 'max_points')
                    
                    if(pcvco.source == 'VERTICES'):
                        r = c.row()
                        self.third_label_two_thirds_prop(pcvco, 'color_constant', r, )
                    else:
                        self.third_label_two_thirds_prop(pcvco, 'color_source', c, )
                        if(pcvco.color_source == 'CONSTANT'):
                            r = c.row()
                            self.third_label_two_thirds_prop(pcvco, 'color_constant', r, )
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
                c.label(text="No particle systems found.", icon='ERROR', )


class PCVIV_PT_preferences(PCVIV_PT_base):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    # bl_category = "View"
    bl_category = "PCVIV"
    bl_label = "Preferences"
    bl_parent_id = "PCVIV_PT_main"
    bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        o = context.active_object
        if(o is None):
            return False
        return True
    
    def draw(self, context):
        pcviv = context.object.pcv_instavis
        l = self.layout
        c = l.column()
        
        pcviv_prefs = context.scene.pcv_instavis
        c.label(text="Global Settings:")
        self.third_label_two_thirds_prop(pcviv_prefs, 'quality', c, )
        self.third_label_two_thirds_prop(pcviv_prefs, 'update_method', c, )
        c.separator()
        c.label(text="Exit Display Settings:")
        self.third_label_two_thirds_prop(pcviv_prefs, 'exit_object_display_type', c, )
        self.third_label_two_thirds_prop(pcviv_prefs, 'exit_psys_display_method', c, )
        c.separator()
        c.label(text="Auto Switch To Origins Only:")
        c.prop(pcviv_prefs, 'switch_origins_only', text='Enabled', )
        c.prop(pcviv_prefs, 'switch_origins_only_threshold')


class PCVIV_PT_debug(PCVIV_PT_base):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    # bl_category = "View"
    bl_category = "PCVIV"
    bl_label = "Debug"
    bl_parent_id = "PCVIV_PT_main"
    bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        o = context.active_object
        if(o is not None):
            if(debug_mode()):
                return True
        return False
    
    def draw(self, context):
        pcviv = context.object.pcv_instavis
        l = self.layout
        c = l.column()
        
        tab = '    '
        
        targets = [o for o in context.scene.objects if o.pcv_instavis.target]
        b = c.box()
        b.scale_y = 0.333
        b.label(text='targets: ({})'.format(len(targets)))
        for t in targets:
            b.label(text='{}o: {}'.format(tab, t.name))
            for p in t.particle_systems:
                b.label(text='{}ps: {}'.format(tab * 2, p.name))
        
        b = c.box()
        b.scale_y = 0.333
        b.label(text='cache: ({})'.format(len(PCVIVMechanist.cache.keys())))
        for k, v in PCVIVMechanist.cache.items():
            b.label(text='{}{}'.format(tab, k))
        
        def human_readable_number(num, suffix='', ):
            f = 1000.0
            for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z', ]:
                if(abs(num) < f):
                    return "{:3.1f}{}{}".format(num, unit, suffix)
                num /= f
            return "{:.1f}{}{}".format(num, 'Y', suffix)
        
        b = c.box()
        b.scale_y = 0.5
        f = 0.5
        cc = b.column()
        
        def table_row(uil, ct1, ct2, fac, ):
            r = uil.row()
            s = r.split(factor=fac)
            s.label(text=ct1)
            s = s.split(factor=1.0)
            s.alignment = 'RIGHT'
            s.label(text=ct2)
        
        if(PCVIVMechanist.stats_enabled):
            table_row(cc, 'points: ', '{}'.format(human_readable_number(PCVIVMechanist.stats_num_points)), f, )
            table_row(cc, 'instances: ', '{}'.format(human_readable_number(PCVIVMechanist.stats_num_instances)), f, )
            table_row(cc, 'draws: ', '{}'.format(human_readable_number(PCVIVMechanist.stats_num_draws)), f, )
        else:
            table_row(cc, 'points: ', 'n/a', f, )
            table_row(cc, 'instances: ', 'n/a', f, )
            table_row(cc, 'draws: ', 'n/a', f, )
        
        c.separator()
        c.operator('point_cloud_visualizer.pcviv_reset_viewport_draw')
        c.operator('point_cloud_visualizer.pcviv_invalidate_caches')
        c.separator()
        r = c.row()
        r.alert = True
        r.operator('script.reload')


classes = ()
classes_debug = (PCVIV_UL_instances, PCVIV_PT_main, PCVIV_PT_particles, PCVIV_PT_instances, PCVIV_PT_preferences, PCVIV_PT_debug, )
