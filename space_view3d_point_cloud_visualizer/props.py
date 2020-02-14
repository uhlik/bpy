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
from bpy.props import PointerProperty, BoolProperty, StringProperty, FloatProperty, IntProperty, FloatVectorProperty, EnumProperty, CollectionProperty
from bpy.types import PropertyGroup

from .debug import log, debug_mode
from .machine import PCVManager, preferences


class PCV_properties(PropertyGroup):
    filepath: StringProperty(name="PLY File", default="", description="", )
    uuid: StringProperty(default="", options={'HIDDEN', }, )
    
    def _instance_visualizer_active_get(self, ):
        return self.instance_visualizer_active_hidden_value
    
    def _instance_visualizer_active_set(self, value, ):
        pass
    
    # for setting value, there are handlers for save, pre which sets to False, and post which sets back to True if it was True before, instance visualizer have to be activated at runtime and this value should not be saved, this way it works ok.. if only there was a way to specify which properties should not save, and/or save only as default value..
    instance_visualizer_active_hidden_value: BoolProperty(default=False, options={'HIDDEN', }, )
    # for display, read-only
    instance_visualizer_active: BoolProperty(name="Instance Visualizer Active", default=False, get=_instance_visualizer_active_get, set=_instance_visualizer_active_set, )
    
    runtime: BoolProperty(default=False, options={'HIDDEN', }, )
    
    # TODO: add some prefix to global props, like global_size, global_display_percent, .. leave unprefixed only essentials, like uuid, runtime, ..
    point_size: IntProperty(name="Size", default=3, min=1, max=10, subtype='PIXEL', description="Point size", )
    alpha_radius: FloatProperty(name="Radius", default=1.0, min=0.001, max=1.0, precision=3, subtype='FACTOR', description="Adjust point circular discard radius", )
    
    def _display_percent_update(self, context, ):
        if(self.uuid not in PCVManager.cache):
            return
        d = PCVManager.cache[self.uuid]
        dp = self.display_percent
        vl = d['length']
        l = int((vl / 100) * dp)
        if(dp >= 99):
            l = vl
        d['display_length'] = l
    
    display_percent: FloatProperty(name="Display", default=100.0, min=0.0, max=100.0, precision=0, subtype='PERCENTAGE', update=_display_percent_update, description="Adjust percentage of points displayed", )
    global_alpha: FloatProperty(name="Alpha", default=1.0, min=0.0, max=1.0, precision=2, subtype='FACTOR', description="Adjust alpha of points displayed", )
    
    vertex_normals: BoolProperty(name="Normals", description="Draw normals of points", default=False, )
    vertex_normals_size: FloatProperty(name="Length", description="Length of point normal line", default=0.01, min=0.00001, max=1.0, soft_min=0.001, soft_max=0.2, step=1, precision=3, )
    vertex_normals_alpha: FloatProperty(name="Alpha", description="Alpha of point normal line", default=0.5, min=0.0, max=1.0, soft_min=0.0, soft_max=1.0, step=1, precision=3, )
    
    render_point_size: IntProperty(name="Size", default=3, min=1, max=100, subtype='PIXEL', description="Point size", )
    render_display_percent: FloatProperty(name="Count", default=100.0, min=0.0, max=100.0, precision=0, subtype='PERCENTAGE', description="Adjust percentage of points rendered", )
    render_path: StringProperty(name="Output Path", default="//pcv_render_###.png", description="Directory/name to save rendered images, # characters defines the position and length of frame numbers, filetype is always png", subtype='FILE_PATH', )
    render_resolution_x: IntProperty(name="Resolution X", default=1920, min=4, max=65536, description="Number of horizontal pixels in rendered image", subtype='PIXEL', )
    render_resolution_y: IntProperty(name="Resolution Y", default=1080, min=4, max=65536, description="Number of vertical pixels in rendered image", subtype='PIXEL', )
    render_resolution_percentage: IntProperty(name="Resolution %", default=100, min=1, max=100, description="Percentage scale for render resolution", subtype='PERCENTAGE', )
    render_smoothstep: BoolProperty(name="Smooth Circles", default=False, description="Currently works only for basic shader with/without illumination and generally is much slower than Supersampling, use only when Supersampling fails", )
    render_supersampling: IntProperty(name="Supersampling", default=1, soft_min=1, soft_max=4, min=1, max=10, description="Render larger image and then resize back, 1 - disabled, 2 - render 200%, 3 - render 300%, ...", )
    
    def _render_resolution_linked_update(self, context, ):
        if(not self.render_resolution_linked):
            # now it is False, so it must have been True, so for convenience, copy values
            r = context.scene.render
            self.render_resolution_x = r.resolution_x
            self.render_resolution_y = r.resolution_y
            self.render_resolution_percentage = r.resolution_percentage
    
    render_resolution_linked: BoolProperty(name="Resolution Linked", description="Link resolution settings to scene", default=True, update=_render_resolution_linked_update, )
    
    has_normals: BoolProperty(default=False, options={'HIDDEN', }, )
    # TODO: rename to 'has_colors'
    has_vcols: BoolProperty(default=False, options={'HIDDEN', }, )
    illumination: BoolProperty(name="Illumination", description="Enable extra illumination on point cloud", default=False, )
    illumination_edit: BoolProperty(name="Edit", description="Edit illumination properties", default=False, )
    light_direction: FloatVectorProperty(name="Light Direction", description="Light direction", default=(0.0, 1.0, 0.0), subtype='DIRECTION', size=3, )
    # light_color: FloatVectorProperty(name="Light Color", description="", default=(0.2, 0.2, 0.2), min=0, max=1, subtype='COLOR', size=3, )
    light_intensity: FloatProperty(name="Light Intensity", description="Light intensity", default=0.3, min=0, max=1, subtype='FACTOR', )
    shadow_intensity: FloatProperty(name="Shadow Intensity", description="Shadow intensity", default=0.2, min=0, max=1, subtype='FACTOR', )
    # show_normals: BoolProperty(name="Colorize By Vertex Normals", description="", default=False, )
    
    mesh_type: EnumProperty(name="Type", items=[('VERTEX', "Vertex", ""),
                                                ('TRIANGLE', "Equilateral Triangle", ""),
                                                ('TETRAHEDRON', "Tetrahedron", ""),
                                                ('CUBE', "Cube", ""),
                                                ('ICOSPHERE', "Ico Sphere", ""),
                                                ('INSTANCER', "Instancer", ""),
                                                ('PARTICLES', "Particle System", ""), ], default='CUBE', description="Instance mesh type", )
    mesh_size: FloatProperty(name="Size", description="Mesh instance size, instanced mesh has size 1.0", default=0.01, min=0.000001, precision=4, max=100.0, )
    mesh_normal_align: BoolProperty(name="Align To Normal", description="Align instance to point normal", default=True, )
    mesh_vcols: BoolProperty(name="Colors", description="Assign point color to instance vertex colors", default=True, )
    mesh_all: BoolProperty(name="All", description="Convert all points", default=True, )
    mesh_percentage: FloatProperty(name="Subset", default=100.0, min=0.0, max=100.0, precision=0, subtype='PERCENTAGE', description="Convert random subset of points by given percentage", )
    mesh_base_sphere_subdivisions: IntProperty(name="Sphere Subdivisions", default=2, min=1, max=6, description="Particle instance (Ico Sphere) subdivisions, instance mesh can be change later", )
    mesh_use_instancer2: BoolProperty(name="Use Faster Conversion", description="Faster (especially with icosphere) Numpy implementation, use if you don't mind all triangles in result", default=False, )
    
    export_use_viewport: BoolProperty(name="Use Viewport Points", default=True, description="When checked, export points currently displayed in viewport or when unchecked, export data loaded from original ply file", )
    export_apply_transformation: BoolProperty(name="Apply Transformation", default=False, description="Apply parent object transformation to points", )
    export_convert_axes: BoolProperty(name="Convert Axes", default=False, description="Convert from blender (y forward, z up) to forward -z, up y axes", )
    export_visible_only: BoolProperty(name="Visible Points Only", default=False, description="Export currently visible points only (controlled by 'Display' on main panel)", )
    
    filter_simplify_num_samples: IntProperty(name="Samples", default=10000, min=1, subtype='NONE', description="Number of points in simplified point cloud, best result when set to less than 20% of points, when samples has value close to total expect less points in result", )
    filter_simplify_num_candidates: IntProperty(name="Candidates", default=10, min=3, max=100, subtype='NONE', description="Number of candidates used during resampling, the higher value, the slower calculation, but more even", )
    
    filter_remove_color: FloatVectorProperty(name="Color", default=(1.0, 1.0, 1.0, ), min=0, max=1, subtype='COLOR', size=3, description="Color to remove from point cloud", )
    filter_remove_color_delta_hue: FloatProperty(name="Δ Hue", default=0.1, min=0.0, max=1.0, precision=3, subtype='FACTOR', description="", )
    filter_remove_color_delta_hue_use: BoolProperty(name="Use Δ Hue", description="", default=True, )
    filter_remove_color_delta_saturation: FloatProperty(name="Δ Saturation", default=0.1, min=0.0, max=1.0, precision=3, subtype='FACTOR', description="", )
    filter_remove_color_delta_saturation_use: BoolProperty(name="Use Δ Saturation", description="", default=True, )
    filter_remove_color_delta_value: FloatProperty(name="Δ Value", default=0.1, min=0.0, max=1.0, precision=3, subtype='FACTOR', description="", )
    filter_remove_color_delta_value_use: BoolProperty(name="Use Δ Value", description="", default=True, )
    filter_remove_color_selection: BoolProperty(default=False, options={'HIDDEN', }, )
    
    def _project_positive_radio_update(self, context):
        if(not self.filter_project_negative and not self.filter_project_positive):
            self.filter_project_negative = True
    
    def _project_negative_radio_update(self, context):
        if(not self.filter_project_negative and not self.filter_project_positive):
            self.filter_project_positive = True
    
    def _filter_project_object_poll(self, o, ):
        if(o and o.type in ('MESH', 'CURVE', 'SURFACE', 'FONT', )):
            return True
        return False
    
    filter_project_object: PointerProperty(type=bpy.types.Object, name="Object", description="", poll=_filter_project_object_poll, )
    filter_project_search_distance: FloatProperty(name="Search Distance", default=0.1, min=0.0, max=10000.0, precision=3, subtype='DISTANCE', description="Maximum search distance in which to search for surface", )
    filter_project_positive: BoolProperty(name="Positive", description="Search along point normal forwards", default=True, update=_project_positive_radio_update, )
    filter_project_negative: BoolProperty(name="Negative", description="Search along point normal backwards", default=True, update=_project_negative_radio_update, )
    filter_project_discard: BoolProperty(name="Discard Unprojectable", description="Discard points which didn't hit anything", default=False, )
    filter_project_colorize: BoolProperty(name="Colorize", description="Colorize projected points", default=False, )
    filter_project_colorize_from: EnumProperty(name="Source", items=[('VCOLS', "Vertex Colors", "Use active vertex colors from target"),
                                                                     ('UVTEX', "UV Texture", "Use colors from active image texture node in active material using active UV layout from target"),
                                                                     ('GROUP_MONO', "Vertex Group Monochromatic", "Use active vertex group from target, result will be shades of grey"),
                                                                     ('GROUP_COLOR', "Vertex Group Colorized", "Use active vertex group from target, result will be colored from red (1.0) to blue (0.0) like in weight paint viewport"),
                                                                     ], default='UVTEX', description="Color source for projected point cloud", )
    filter_project_shift: FloatProperty(name="Shift", default=0.0, precision=3, subtype='DISTANCE', description="Shift points after projection above (positive) or below (negative) surface", )
    
    def _filter_boolean_object_poll(self, o, ):
        if(o and o.type in ('MESH', 'CURVE', 'SURFACE', 'FONT', )):
            return True
        return False
    
    filter_boolean_object: PointerProperty(type=bpy.types.Object, name="Object", description="", poll=_filter_boolean_object_poll, )
    
    def _filter_join_object_poll(self, o, ):
        ok = False
        if(o):
            pcv = o.point_cloud_visualizer
            if(pcv.uuid != ''):
                for k, v in PCVManager.cache.items():
                    if(v['uuid'] == pcv.uuid):
                        if(v['ready']):
                            # if(v['draw']):
                            #     ok = True
                            ok = True
                        break
        return ok
    
    filter_join_object: PointerProperty(type=bpy.types.Object, name="Object", description="", poll=_filter_join_object_poll, )
    
    edit_initialized: BoolProperty(default=False, options={'HIDDEN', }, )
    edit_is_edit_mesh: BoolProperty(default=False, options={'HIDDEN', }, )
    edit_is_edit_uuid: StringProperty(default="", options={'HIDDEN', }, )
    edit_pre_edit_alpha: FloatProperty(default=0.5, options={'HIDDEN', }, )
    edit_pre_edit_display: FloatProperty(default=100.0, options={'HIDDEN', }, )
    edit_pre_edit_size: IntProperty(default=3, options={'HIDDEN', }, )
    
    def _edit_overlay_alpha_update(self, context, ):
        o = context.object
        p = o.parent
        pcv = p.point_cloud_visualizer
        pcv.global_alpha = self.edit_overlay_alpha
    
    def _edit_overlay_size_update(self, context, ):
        o = context.object
        p = o.parent
        pcv = p.point_cloud_visualizer
        pcv.point_size = self.edit_overlay_size
    
    edit_overlay_alpha: FloatProperty(name="Overlay Alpha", default=0.5, min=0.0, max=1.0, precision=2, subtype='FACTOR', description="Overlay point alpha", update=_edit_overlay_alpha_update, )
    edit_overlay_size: IntProperty(name="Overlay Size", default=3, min=1, max=10, subtype='PIXEL', description="Overlay point size", update=_edit_overlay_size_update, )
    
    # sequence_enabled: BoolProperty(default=False, options={'HIDDEN', }, )
    # sequence_frame_duration: IntProperty(name="Frames", default=1, min=1, description="", )
    # sequence_frame_start: IntProperty(name="Start Frame", default=1, description="", )
    # sequence_frame_offset: IntProperty(name="Offset", default=0, description="", )
    sequence_use_cyclic: BoolProperty(name="Cycle Forever", default=True, description="Cycle preloaded point clouds (ply_index = (current_frame % len(ply_files)) - 1)", )
    
    generate_source: EnumProperty(name="Source", items=[('VERTICES', "Vertices", "Use mesh vertices"),
                                                        ('SURFACE', "Surface", "Use triangulated mesh surface"),
                                                        ('PARTICLES', "Particle System", "Use active particle system"),
                                                        ], default='SURFACE', description="Points generation source", )
    generate_source_psys: EnumProperty(name="Particles", items=[('ALL', "All", "Use all particles"),
                                                                ('ALIVE', "Alive", "Use alive particles"),
                                                                ], default='ALIVE', description="Particles source", )
    generate_algorithm: EnumProperty(name="Algorithm", items=[('WEIGHTED_RANDOM_IN_TRIANGLE', "Weighted Random In Triangle", "Average triangle areas to approximate number of random points in each to get even distribution of points. If some very small polygons are left without points, increase number of samples. Mesh is triangulated before processing, on non-planar polygons, points will not be exactly on original polygon surface."),
                                                              ('POISSON_DISK_SAMPLING', "Poisson Disk Sampling", "Warning: slow, very slow indeed.. Uses Weighted Random In Triangle algorithm to pregenerate samples with all its inconveniences."),
                                                              ], default='WEIGHTED_RANDOM_IN_TRIANGLE', description="Point generating algorithm", )
    generate_number_of_points: IntProperty(name="Approximate Number Of Points", default=100000, min=1, description="Number of points to generate, some algorithms may not generate exact number of points.", )
    generate_seed: IntProperty(name="Seed", default=0, min=0, description="Random number generator seed", )
    generate_colors: EnumProperty(name="Colors", items=[('CONSTANT', "Constant Color", "Use constant color value"),
                                                        ('VCOLS', "Vertex Colors", "Use active vertex colors"),
                                                        ('UVTEX', "UV Texture", "Generate colors from active image texture node in active material using active UV layout"),
                                                        ('GROUP_MONO', "Vertex Group Monochromatic", "Use active vertex group, result will be shades of grey"),
                                                        ('GROUP_COLOR', "Vertex Group Colorized", "Use active vertex group, result will be colored from red (1.0) to blue (0.0) like in weight paint viewport"),
                                                        ], default='CONSTANT', description="Color source for generated point cloud", )
    generate_constant_color: FloatVectorProperty(name="Color", description="Constant color", default=(0.7, 0.7, 0.7, ), min=0, max=1, subtype='COLOR', size=3, )
    generate_exact_number_of_points: BoolProperty(name="Exact Number of Samples", default=False, description="Generate exact number of points, if selected algorithm result is less points, more points will be calculated on random polygons at the end, if result is more points, points will be shuffled and sliced to match exact value", )
    generate_minimal_distance: FloatProperty(name="Minimal Distance", default=0.1, precision=3, subtype='DISTANCE', description="Poisson Disk minimal distance between points, the smaller value, the slower calculation", )
    generate_sampling_exponent: IntProperty(name="Sampling Exponent", default=5, min=1, description="Poisson Disk presampling exponent, lower values are faster but less even, higher values are slower exponentially", )
    
    override_default_shader: BoolProperty(default=False, options={'HIDDEN', }, )
    
    def _update_dev_depth(self, context, ):
        if(self.dev_depth_enabled):
            # FIXME: this is really getting ridiculous
            self.dev_normal_colors_enabled = False
            self.dev_position_colors_enabled = False
            
            self.dev_minimal_shader_enabled = False
            self.dev_minimal_shader_variable_size_enabled = False
            self.dev_minimal_shader_variable_size_and_depth_enabled = False
            self.dev_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_no_depth_enabled = False
            self.dev_phong_shader_enabled = False
            self.clip_shader_enabled = False
            self.billboard_phong_enabled = False
            self.skip_point_shader_enabled = False
            
            self.override_default_shader = True
        else:
            self.override_default_shader = False
    
    def _update_dev_normal(self, context, ):
        if(self.dev_normal_colors_enabled):
            # FIXME: this is really getting ridiculous
            self.dev_depth_enabled = False
            self.dev_position_colors_enabled = False
            
            self.dev_minimal_shader_enabled = False
            self.dev_minimal_shader_variable_size_enabled = False
            self.dev_minimal_shader_variable_size_and_depth_enabled = False
            self.dev_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_no_depth_enabled = False
            self.dev_phong_shader_enabled = False
            self.clip_shader_enabled = False
            self.billboard_phong_enabled = False
            self.skip_point_shader_enabled = False
            
            self.override_default_shader = True
        else:
            self.override_default_shader = False
    
    def _update_dev_position(self, context, ):
        if(self.dev_position_colors_enabled):
            # FIXME: this is really getting ridiculous
            self.dev_depth_enabled = False
            self.dev_normal_colors_enabled = False
            
            self.dev_minimal_shader_enabled = False
            self.dev_minimal_shader_variable_size_enabled = False
            self.dev_minimal_shader_variable_size_and_depth_enabled = False
            self.dev_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_no_depth_enabled = False
            self.dev_phong_shader_enabled = False
            self.clip_shader_enabled = False
            self.billboard_phong_enabled = False
            self.skip_point_shader_enabled = False
            
            self.override_default_shader = True
        else:
            self.override_default_shader = False
    
    # dev_depth_enabled: BoolProperty(name="Depth", default=False, description="", update=_update_override_default_shader, )
    dev_depth_enabled: BoolProperty(name="Depth", default=False, description="Enable depth debug shader", update=_update_dev_depth, )
    # dev_depth_edit: BoolProperty(name="Edit", description="Edit depth shader properties", default=False, )
    dev_depth_brightness: FloatProperty(name="Brightness", description="Depth shader color brightness", default=0.0, min=-10.0, max=10.0, )
    dev_depth_contrast: FloatProperty(name="Contrast", description="Depth shader color contrast", default=1.0, min=-10.0, max=10.0, )
    dev_depth_false_colors: BoolProperty(name="False Colors", default=False, description="Display depth shader in false colors", )
    dev_depth_color_a: FloatVectorProperty(name="Color A", description="Depth shader false colors front color", default=(0.0, 1.0, 0.0, ), min=0, max=1, subtype='COLOR', size=3, )
    dev_depth_color_b: FloatVectorProperty(name="Color B", description="Depth shader false colors back color", default=(0.0, 0.0, 1.0, ), min=0, max=1, subtype='COLOR', size=3, )
    # dev_normal_colors_enabled: BoolProperty(name="Normal", default=False, description="", update=_update_override_default_shader, )
    dev_normal_colors_enabled: BoolProperty(name="Normal", default=False, description="Enable normal debug shader", update=_update_dev_normal, )
    # dev_position_colors_enabled: BoolProperty(name="Position", default=False, description="", update=_update_override_default_shader, )
    dev_position_colors_enabled: BoolProperty(name="Position", default=False, description="Enable position debug shader", update=_update_dev_position, )
    
    # NOTE: icon for bounding box 'SHADING_BBOX' ?
    dev_bbox_enabled: BoolProperty(name="Bounding Box", default=False, description="", )
    dev_bbox_color: FloatVectorProperty(name="Color", description="", default=(0.7, 0.7, 0.7), min=0, max=1, subtype='COLOR', size=3, )
    dev_bbox_size: FloatProperty(name="Size", description="", default=0.3, min=0.1, max=0.9, subtype='FACTOR', )
    dev_bbox_alpha: FloatProperty(name="Alpha", description="", default=0.7, min=0.0, max=1.0, subtype='FACTOR', )
    
    def _dev_sel_color_update(self, context, ):
        preferences().selection_color = self.dev_selection_shader_color
    
    dev_selection_shader_display: BoolProperty(name="Selection", default=False, description="", )
    dev_selection_shader_color: FloatVectorProperty(name="Color", description="", default=(1.0, 0.0, 0.0, 0.5), min=0, max=1, subtype='COLOR', size=4, update=_dev_sel_color_update, )
    
    def _update_color_adjustment(self, context, ):
        if(self.color_adjustment_shader_enabled):
            # FIXME: this is really getting ridiculous
            self.dev_depth_enabled = False
            self.dev_normal_colors_enabled = False
            self.dev_position_colors_enabled = False
            
            self.dev_minimal_shader_enabled = False
            self.dev_minimal_shader_variable_size_enabled = False
            self.dev_minimal_shader_variable_size_and_depth_enabled = False
            self.dev_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_no_depth_enabled = False
            self.dev_phong_shader_enabled = False
            self.clip_shader_enabled = False
            self.billboard_phong_enabled = False
            self.skip_point_shader_enabled = False
            
            self.illumination = False
            self.override_default_shader = True
        else:
            self.override_default_shader = False
    
    color_adjustment_shader_enabled: BoolProperty(name="Enabled", default=False, description="Enable color adjustment shader, other shaders will be overrided until disabled", update=_update_color_adjustment, )
    color_adjustment_shader_exposure: FloatProperty(name="Exposure", description="formula: color = color * (2 ** value)", default=0.0, min=-5.0, max=5.0, )
    color_adjustment_shader_gamma: FloatProperty(name="Gamma", description="formula: color = color ** (1 / value)", default=1.0, min=0.01, max=9.99, )
    color_adjustment_shader_brightness: FloatProperty(name="Brightness", description="formula: color = (color - 0.5) * contrast + 0.5 + brightness", default=0.0, min=-5.0, max=5.0, )
    color_adjustment_shader_contrast: FloatProperty(name="Contrast", description="formula: color = (color - 0.5) * contrast + 0.5 + brightness", default=1.0, min=0.0, max=10.0, )
    color_adjustment_shader_hue: FloatProperty(name="Hue", description="formula: color.h = (color.h + (value % 1.0)) % 1.0", default=0.0, min=0.0, max=1.0, )
    color_adjustment_shader_saturation: FloatProperty(name="Saturation", description="formula: color.s += value", default=0.0, min=-1.0, max=1.0, )
    color_adjustment_shader_value: FloatProperty(name="Value", description="formula: color.v += value", default=0.0, min=-1.0, max=1.0, )
    color_adjustment_shader_invert: BoolProperty(name="Invert", description="formula: color = 1.0 - color", default=False, )
    
    def _update_minimal_shader(self, context, ):
        if(self.dev_minimal_shader_enabled):
            # FIXME: this is really getting ridiculous
            self.illumination = False
            self.dev_depth_enabled = False
            self.dev_normal_colors_enabled = False
            self.dev_position_colors_enabled = False
            self.color_adjustment_shader_enabled = False
            self.dev_minimal_shader_variable_size_enabled = False
            self.dev_minimal_shader_variable_size_and_depth_enabled = False
            self.dev_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_no_depth_enabled = False
            self.dev_phong_shader_enabled = False
            self.clip_shader_enabled = False
            self.billboard_phong_enabled = False
            self.skip_point_shader_enabled = False
            
            self.override_default_shader = True
        else:
            self.override_default_shader = False
    
    def _update_minimal_shader_variable_size(self, context, ):
        if(self.dev_minimal_shader_variable_size_enabled):
            # FIXME: this is really getting ridiculous
            self.illumination = False
            self.dev_depth_enabled = False
            self.dev_normal_colors_enabled = False
            self.dev_position_colors_enabled = False
            self.color_adjustment_shader_enabled = False
            self.dev_minimal_shader_enabled = False
            self.dev_minimal_shader_variable_size_and_depth_enabled = False
            self.dev_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_no_depth_enabled = False
            self.dev_phong_shader_enabled = False
            self.clip_shader_enabled = False
            self.billboard_phong_enabled = False
            self.skip_point_shader_enabled = False
            
            self.override_default_shader = True
        else:
            self.override_default_shader = False
    
    dev_minimal_shader_enabled: BoolProperty(name="Enabled", default=False, description="Enable minimal shader", update=_update_minimal_shader, )
    dev_minimal_shader_variable_size_enabled: BoolProperty(name="Enabled", default=False, description="Enable minimal shader with variable size", update=_update_minimal_shader_variable_size, )
    
    def _update_minimal_shader_variable_size_with_depth(self, context, ):
        if(self.dev_minimal_shader_variable_size_and_depth_enabled):
            # FIXME: this is really getting ridiculous
            self.illumination = False
            self.dev_depth_enabled = False
            self.dev_normal_colors_enabled = False
            self.dev_position_colors_enabled = False
            self.color_adjustment_shader_enabled = False
            self.dev_minimal_shader_enabled = False
            self.dev_minimal_shader_variable_size_enabled = False
            self.dev_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_no_depth_enabled = False
            self.dev_phong_shader_enabled = False
            self.clip_shader_enabled = False
            self.billboard_phong_enabled = False
            self.skip_point_shader_enabled = False
            
            self.override_default_shader = True
        else:
            self.override_default_shader = False
    
    dev_minimal_shader_variable_size_and_depth_enabled: BoolProperty(name="Enabled", default=False, description="Enable minimal shader with variable size with depth", update=_update_minimal_shader_variable_size_with_depth, )
    dev_minimal_shader_variable_size_and_depth_brightness: FloatProperty(name="Brightness", default=0.25, min=-10.0, max=10.0, description="Depth shader color brightness", )
    dev_minimal_shader_variable_size_and_depth_contrast: FloatProperty(name="Contrast", default=0.5, min=-10.0, max=10.0, description="Depth shader color contrast", )
    dev_minimal_shader_variable_size_and_depth_blend: FloatProperty(name="Blend", default=0.75, min=0.0, max=1.0, subtype='FACTOR', description="Depth shader blending with original colors", )
    
    def _update_dev_billboard_point_cloud_enabled(self, context, ):
        if(self.dev_billboard_point_cloud_enabled):
            # FIXME: this is really getting ridiculous
            self.illumination = False
            self.dev_depth_enabled = False
            self.dev_normal_colors_enabled = False
            self.dev_position_colors_enabled = False
            self.color_adjustment_shader_enabled = False
            self.dev_minimal_shader_enabled = False
            self.dev_minimal_shader_variable_size_enabled = False
            self.dev_minimal_shader_variable_size_and_depth_enabled = False
            self.dev_rich_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_no_depth_enabled = False
            self.dev_phong_shader_enabled = False
            self.clip_shader_enabled = False
            self.billboard_phong_enabled = False
            self.skip_point_shader_enabled = False
            
            self.override_default_shader = True
        else:
            self.override_default_shader = False
    
    dev_billboard_point_cloud_enabled: BoolProperty(name="Enabled", default=False, description="Enable Billboard Shader", update=_update_dev_billboard_point_cloud_enabled, )
    dev_billboard_point_cloud_size: FloatProperty(name="Size", default=0.002, min=0.0001, max=0.2, description="", precision=6, )
    
    def _update_dev_rich_billboard_point_cloud_enabled(self, context):
        if(self.dev_rich_billboard_point_cloud_enabled):
            # FIXME: this is really getting ridiculous
            self.illumination = False
            self.dev_depth_enabled = False
            self.dev_normal_colors_enabled = False
            self.dev_position_colors_enabled = False
            self.color_adjustment_shader_enabled = False
            self.dev_minimal_shader_enabled = False
            self.dev_minimal_shader_variable_size_enabled = False
            self.dev_minimal_shader_variable_size_and_depth_enabled = False
            self.dev_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_no_depth_enabled = False
            self.dev_phong_shader_enabled = False
            self.clip_shader_enabled = False
            self.billboard_phong_enabled = False
            self.skip_point_shader_enabled = False
            
            self.override_default_shader = True
        else:
            self.override_default_shader = False
    
    dev_rich_billboard_point_cloud_enabled: BoolProperty(name="Enabled", default=False, description="Enable Rich Billboard Shader", update=_update_dev_rich_billboard_point_cloud_enabled, )
    dev_rich_billboard_point_cloud_size: FloatProperty(name="Size", default=0.01, min=0.0001, max=1.0, description="", precision=6, )
    dev_rich_billboard_depth_brightness: FloatProperty(name="Brightness", default=0.25, min=-10.0, max=10.0, description="Depth shader color brightness", )
    dev_rich_billboard_depth_contrast: FloatProperty(name="Contrast", default=0.5, min=-10.0, max=10.0, description="Depth shader color contrast", )
    dev_rich_billboard_depth_blend: FloatProperty(name="Blend", default=0.75, min=0.0, max=1.0, subtype='FACTOR', description="Depth shader blending with original colors", )
    
    def _update_dev_rich_billboard_point_cloud_no_depth_enabled(self, context):
        if(self.dev_rich_billboard_point_cloud_no_depth_enabled):
            # FIXME: this is really getting ridiculous
            self.illumination = False
            self.dev_depth_enabled = False
            self.dev_normal_colors_enabled = False
            self.dev_position_colors_enabled = False
            self.color_adjustment_shader_enabled = False
            self.dev_minimal_shader_enabled = False
            self.dev_minimal_shader_variable_size_enabled = False
            self.dev_minimal_shader_variable_size_and_depth_enabled = False
            self.dev_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_enabled = False
            self.dev_phong_shader_enabled = False
            self.clip_shader_enabled = False
            self.billboard_phong_enabled = False
            self.skip_point_shader_enabled = False
            
            self.override_default_shader = True
        else:
            self.override_default_shader = False
    
    dev_rich_billboard_point_cloud_no_depth_enabled: BoolProperty(name="Enabled", default=False, description="Enable Rich Billboard Shader Without Depth", update=_update_dev_rich_billboard_point_cloud_no_depth_enabled, )
    
    def _update_dev_phong_shader_enabled(self, context):
        if(self.dev_phong_shader_enabled):
            # FIXME: this is really getting ridiculous
            self.illumination = False
            self.dev_depth_enabled = False
            self.dev_normal_colors_enabled = False
            self.dev_position_colors_enabled = False
            self.color_adjustment_shader_enabled = False
            self.dev_minimal_shader_enabled = False
            self.dev_minimal_shader_variable_size_enabled = False
            self.dev_minimal_shader_variable_size_and_depth_enabled = False
            self.dev_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_no_depth_enabled = False
            self.clip_shader_enabled = False
            self.billboard_phong_enabled = False
            self.skip_point_shader_enabled = False
            
            self.override_default_shader = True
        else:
            self.override_default_shader = False
    
    dev_phong_shader_enabled: BoolProperty(name="Enabled", default=False, description="", update=_update_dev_phong_shader_enabled, )
    dev_phong_shader_ambient_strength: FloatProperty(name="ambient_strength", default=0.5, min=0.0, max=1.0, description="", )
    dev_phong_shader_specular_strength: FloatProperty(name="specular_strength", default=0.5, min=0.0, max=1.0, description="", )
    dev_phong_shader_specular_exponent: FloatProperty(name="specular_exponent", default=8.0, min=1.0, max=512.0, description="", )
    
    debug_panel_show_properties: BoolProperty(default=False, options={'HIDDEN', }, )
    debug_panel_show_manager: BoolProperty(default=False, options={'HIDDEN', }, )
    debug_panel_show_sequence: BoolProperty(default=False, options={'HIDDEN', }, )
    debug_panel_show_cache_items: BoolProperty(default=False, options={'HIDDEN', }, )
    
    # store info how long was last draw call, ie get points from cache, join, draw
    pcviv_debug_draw: StringProperty(default="", )
    pcviv_debug_panel_show_info: BoolProperty(default=False, options={'HIDDEN', }, )
    # have to provide prop for indexing, not needed for anything in this case
    pcviv_material_list_active_index: IntProperty(name="Index", default=0, description="", options={'HIDDEN', }, )
    
    # testing / development stuff
    def _dev_transform_normals_target_object_poll(self, o, ):
        if(o and o.type in ('MESH', )):
            return True
        return False
    
    dev_transform_normals_target_object: PointerProperty(type=bpy.types.Object, name="Object", description="", poll=_dev_transform_normals_target_object_poll, )
    
    # dev
    def _clip_shader_enabled(self, context):
        if(self.clip_shader_enabled):
            # FIXME: this is really getting ridiculous
            self.illumination = False
            self.dev_depth_enabled = False
            self.dev_normal_colors_enabled = False
            self.dev_position_colors_enabled = False
            self.color_adjustment_shader_enabled = False
            self.dev_minimal_shader_enabled = False
            self.dev_minimal_shader_variable_size_enabled = False
            self.dev_minimal_shader_variable_size_and_depth_enabled = False
            self.dev_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_no_depth_enabled = False
            self.dev_phong_shader_enabled = False
            self.billboard_phong_enabled = False
            self.skip_point_shader_enabled = False
            
            self.override_default_shader = True
        else:
            self.override_default_shader = False
    
    clip_shader_enabled: BoolProperty(name="Enabled", default=False, description="", update=_clip_shader_enabled, )
    clip_plane0_enabled: BoolProperty(name="Enabled", default=False, description="", )
    clip_plane1_enabled: BoolProperty(name="Enabled", default=False, description="", )
    clip_plane2_enabled: BoolProperty(name="Enabled", default=False, description="", )
    clip_plane3_enabled: BoolProperty(name="Enabled", default=False, description="", )
    clip_plane4_enabled: BoolProperty(name="Enabled", default=False, description="", )
    clip_plane5_enabled: BoolProperty(name="Enabled", default=False, description="", )
    clip_plane0: FloatVectorProperty(name="Plane 0", default=(0.0, 0.0, 0.0, 0.0), subtype='NONE', size=4, description="", )
    clip_plane1: FloatVectorProperty(name="Plane 1", default=(0.0, 0.0, 0.0, 0.0), subtype='NONE', size=4, description="", )
    clip_plane2: FloatVectorProperty(name="Plane 2", default=(0.0, 0.0, 0.0, 0.0), subtype='NONE', size=4, description="", )
    clip_plane3: FloatVectorProperty(name="Plane 3", default=(0.0, 0.0, 0.0, 0.0), subtype='NONE', size=4, description="", )
    clip_plane4: FloatVectorProperty(name="Plane 4", default=(0.0, 0.0, 0.0, 0.0), subtype='NONE', size=4, description="", )
    clip_plane5: FloatVectorProperty(name="Plane 5", default=(0.0, 0.0, 0.0, 0.0), subtype='NONE', size=4, description="", )
    
    def _clip_planes_from_bbox_object_poll(self, o, ):
        if(o and o.type in ('MESH', )):
            return True
        return False
    
    clip_planes_from_bbox_object: PointerProperty(type=bpy.types.Object, name="Object", description="", poll=_clip_planes_from_bbox_object_poll, )
    
    def _billboard_phong_enabled(self, context):
        if(self.billboard_phong_enabled):
            # FIXME: this is really getting ridiculous
            self.illumination = False
            self.dev_depth_enabled = False
            self.dev_normal_colors_enabled = False
            self.dev_position_colors_enabled = False
            self.color_adjustment_shader_enabled = False
            self.dev_minimal_shader_enabled = False
            self.dev_minimal_shader_variable_size_enabled = False
            self.dev_minimal_shader_variable_size_and_depth_enabled = False
            self.dev_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_no_depth_enabled = False
            self.dev_phong_shader_enabled = False
            self.clip_shader_enabled = False
            self.skip_point_shader_enabled = False
            
            self.override_default_shader = True
        else:
            self.override_default_shader = False
    
    billboard_phong_enabled: BoolProperty(name="Enabled", default=False, description="", update=_billboard_phong_enabled, )
    billboard_phong_circles: BoolProperty(name="Circles (slower)", default=False, description="", )
    billboard_phong_size: FloatProperty(name="Size", default=0.002, min=0.0001, max=0.2, description="", precision=6, )
    billboard_phong_ambient_strength: FloatProperty(name="Ambient", default=0.5, min=0.0, max=1.0, description="", )
    billboard_phong_specular_strength: FloatProperty(name="Specular", default=0.5, min=0.0, max=1.0, description="", )
    billboard_phong_specular_exponent: FloatProperty(name="Hardness", default=8.0, min=1.0, max=512.0, description="", )
    
    def _skip_point_shader_enabled(self, context):
        if(self.skip_point_shader_enabled):
            # FIXME: this is really getting ridiculous
            self.illumination = False
            self.dev_depth_enabled = False
            self.dev_normal_colors_enabled = False
            self.dev_position_colors_enabled = False
            self.color_adjustment_shader_enabled = False
            self.dev_minimal_shader_enabled = False
            self.dev_minimal_shader_variable_size_enabled = False
            self.dev_minimal_shader_variable_size_and_depth_enabled = False
            self.dev_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_no_depth_enabled = False
            self.dev_phong_shader_enabled = False
            self.clip_shader_enabled = False
            self.billboard_phong_enabled = False
            
            self.override_default_shader = True
        else:
            self.override_default_shader = False
    
    skip_point_shader_enabled: BoolProperty(name="Enabled", default=False, description="", update=_skip_point_shader_enabled, )
    skip_point_percentage: FloatProperty(name="Skip Percentage", default=100.0, min=0.0, max=100.0, precision=3, description="", )
    
    def _fresnel_shader_enabled(self, context):
        if(self.fresnel_shader_enabled):
            # FIXME: this is really getting ridiculous
            self.illumination = False
            self.dev_depth_enabled = False
            self.dev_normal_colors_enabled = False
            self.dev_position_colors_enabled = False
            self.color_adjustment_shader_enabled = False
            self.dev_minimal_shader_enabled = False
            self.dev_minimal_shader_variable_size_enabled = False
            self.dev_minimal_shader_variable_size_and_depth_enabled = False
            self.dev_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_enabled = False
            self.dev_rich_billboard_point_cloud_no_depth_enabled = False
            self.dev_phong_shader_enabled = False
            self.clip_shader_enabled = False
            self.billboard_phong_enabled = False
            self.skip_point_shader_enabled = False
            
            self.override_default_shader = True
        else:
            self.override_default_shader = False
    
    fresnel_shader_enabled: BoolProperty(name="Enabled", default=False, description="", update=_fresnel_shader_enabled, )
    fresnel_shader_sharpness: FloatProperty(name="Sharpness", default=0.2, min=0.0, max=1.0, precision=3, description="", )
    fresnel_shader_invert: BoolProperty(name="Invert", default=False, description="", )
    fresnel_shader_colors: BoolProperty(name="Colors", default=False, description="", )
    
    @classmethod
    def register(cls):
        bpy.types.Object.point_cloud_visualizer = PointerProperty(type=cls)
    
    @classmethod
    def unregister(cls):
        del bpy.types.Object.point_cloud_visualizer
