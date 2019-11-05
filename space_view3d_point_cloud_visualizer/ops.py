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
# (c) 2019 Jakub Uhlik

import os
import time
import datetime
import numpy as np
import re
import random

import bpy
import bmesh
from bpy.props import PointerProperty, BoolProperty, StringProperty, FloatProperty, IntProperty, FloatVectorProperty, EnumProperty, CollectionProperty
from bpy.types import Operator
from mathutils import Matrix, Vector, Quaternion, Color
from bpy_extras.object_utils import world_to_camera_view
from bpy_extras.io_utils import axis_conversion, ExportHelper
from mathutils.kdtree import KDTree
from mathutils.geometry import barycentric_transform
from mathutils.interpolate import poly_3d_calc
from mathutils.bvhtree import BVHTree
import mathutils.geometry
import gpu
from gpu.types import GPUOffScreen, GPUShader, GPUBatch, GPUVertBuf, GPUVertFormat
from gpu_extras.batch import batch_for_shader
import bgl

from .debug import log, debug_mode
from . import debug
from . import io
from . import shaders
from .machine import PCVManager, PCVSequence, preferences, PCVControl
from . import convert
from . import sample


class PCV_OT_init(Operator):
    bl_idname = "point_cloud_visualizer.init"
    bl_label = "init"
    
    def execute(self, context):
        PCVManager.init()
        context.area.tag_redraw()
        return {'FINISHED'}


class PCV_OT_deinit(Operator):
    bl_idname = "point_cloud_visualizer.deinit"
    bl_label = "deinit"
    
    def execute(self, context):
        PCVManager.deinit()
        context.area.tag_redraw()
        return {'FINISHED'}


class PCV_OT_gc(Operator):
    bl_idname = "point_cloud_visualizer.gc"
    bl_label = "gc"
    
    def execute(self, context):
        PCVManager.gc()
        return {'FINISHED'}


class PCV_OT_draw(Operator):
    bl_idname = "point_cloud_visualizer.draw"
    bl_label = "Draw"
    bl_description = "Draw point cloud to viewport"
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        
        pcv = context.object.point_cloud_visualizer
        ok = False
        cached = False
        for k, v in PCVManager.cache.items():
            if(v['uuid'] == pcv.uuid):
                if(v['ready']):
                    cached = True
                    if(not v['draw']):
                        ok = True
        if(not ok and pcv.filepath != "" and pcv.uuid != "" and not cached):
            ok = True
        return ok
    
    def execute(self, context):
        PCVManager.init()
        
        pcv = context.object.point_cloud_visualizer
        
        if(pcv.uuid not in PCVManager.cache):
            pcv.uuid = ""
            ok = PCVManager.load_ply_to_cache(self, context)
            if(not ok):
                return {'CANCELLED'}
        
        c = PCVManager.cache[pcv.uuid]
        c['draw'] = True
        
        context.area.tag_redraw()
        
        return {'FINISHED'}


class PCV_OT_erase(Operator):
    bl_idname = "point_cloud_visualizer.erase"
    bl_label = "Erase"
    bl_description = "Erase point cloud from viewport"
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        
        pcv = context.object.point_cloud_visualizer
        ok = False
        for k, v in PCVManager.cache.items():
            if(v['uuid'] == pcv.uuid):
                if(v['ready']):
                    if(v['draw']):
                        ok = True
        return ok
    
    def execute(self, context):
        pcv = context.object.point_cloud_visualizer
        c = PCVManager.cache[pcv.uuid]
        c['draw'] = False
        
        context.area.tag_redraw()
        
        return {'FINISHED'}


class PCV_OT_load(Operator):
    bl_idname = "point_cloud_visualizer.load_ply_to_cache"
    bl_label = "Load PLY"
    bl_description = "Load PLY file"
    
    filename_ext = ".ply"
    filter_glob: StringProperty(default="*.ply", options={'HIDDEN'}, )
    filepath: StringProperty(name="File Path", default="", description="", maxlen=1024, subtype='FILE_PATH', )
    order = ["filepath", ]
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        
        pcv = context.object.point_cloud_visualizer
        if(not pcv.runtime):
            return True
        return False
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}
    
    def execute(self, context):
        pcv = context.object.point_cloud_visualizer
        ok = True
        h, t = os.path.split(self.filepath)
        n, e = os.path.splitext(t)
        if(e != '.ply'):
            ok = False
        if(not ok):
            self.report({'ERROR'}, "File at '{}' seems not to be a PLY file.".format(self.filepath))
            return {'CANCELLED'}
        
        pcv.filepath = self.filepath
        
        if(pcv.uuid != ""):
            if(pcv.uuid in PCVManager.cache):
                PCVManager.cache[pcv.uuid]['kill'] = True
                PCVManager.gc()
        
        ok = PCVManager.load_ply_to_cache(self, context)
        
        if(not ok):
            return {'CANCELLED'}
        return {'FINISHED'}


class PCV_OT_render(Operator):
    bl_idname = "point_cloud_visualizer.render"
    bl_label = "Render"
    bl_description = "Render displayed point cloud from active camera view to image"
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        
        pcv = context.object.point_cloud_visualizer
        ok = False
        for k, v in PCVManager.cache.items():
            if(v['uuid'] == pcv.uuid):
                if(v['ready']):
                    if(v['draw']):
                        ok = True
        return ok
    
    def execute(self, context):
        _t = time.time()
        
        bgl.glEnable(bgl.GL_PROGRAM_POINT_SIZE)
        bgl.glEnable(bgl.GL_DEPTH_TEST)
        bgl.glEnable(bgl.GL_BLEND)
        
        scene = context.scene
        render = scene.render
        image_settings = render.image_settings
        
        original_depth = image_settings.color_depth
        image_settings.color_depth = '8'
        
        pcv = context.object.point_cloud_visualizer
        
        if(pcv.render_resolution_linked):
            scale = render.resolution_percentage / 100
            
            if(pcv.render_supersampling > 1):
                scale *= pcv.render_supersampling
            
            width = int(render.resolution_x * scale)
            height = int(render.resolution_y * scale)
        else:
            scale = pcv.render_resolution_percentage / 100
            
            if(pcv.render_supersampling > 1):
                scale *= pcv.render_supersampling
            
            width = int(pcv.render_resolution_x * scale)
            height = int(pcv.render_resolution_y * scale)
        
        cloud = PCVManager.cache[pcv.uuid]
        cam = scene.camera
        if(cam is None):
            self.report({'ERROR'}, "No camera found.")
            return {'CANCELLED'}
        
        def get_output_path():
            p = bpy.path.abspath(pcv.render_path)
            ok = True
            
            # ensure soem directory and filename
            h, t = os.path.split(p)
            if(h == ''):
                # next to blend file
                h = bpy.path.abspath('//')
            
            if(not os.path.exists(h)):
                self.report({'ERROR'}, "Directory does not exist ('{}').".format(h))
                ok = False
            if(not os.path.isdir(h)):
                self.report({'ERROR'}, "Not a directory ('{}').".format(h))
                ok = False
            
            if(t == ''):
                # default name
                t = 'pcv_render_###.png'
            
            # ensure extension
            p = os.path.join(h, t)
            p = bpy.path.ensure_ext(p, '.png', )
            h, t = os.path.split(p)
            
            # ensure frame number
            if('#' not in t):
                n, e = os.path.splitext(t)
                n = '{}_###'.format(n)
                t = '{}{}'.format(n, e)
            
            # return with a bit of luck valid path
            p = os.path.join(h, t)
            return ok, p
        
        def swap_frame_number(p):
            fn = scene.frame_current
            h, t = os.path.split(p)
            
            # swap all occurences of # with zfilled frame number
            pattern = r'#+'
            
            def repl(m):
                l = len(m.group(0))
                return str(fn).zfill(l)
            
            t = re.sub(pattern, repl, t, )
            p = os.path.join(h, t)
            return p
        
        bd = context.blend_data
        if(bd.filepath == "" and not bd.is_saved):
            self.report({'ERROR'}, "Save .blend file first.")
            return {'CANCELLED'}
        
        ok, output_path = get_output_path()
        if(not ok):
            return {'CANCELLED'}
        
        # path is ok, write it back..
        pcv.render_path = output_path
        
        # swap frame number
        output_path = swap_frame_number(output_path)
        
        # TODO: on my machine, maximum size is 16384, is it max in general or just my hardware? check it, and add some warning, or hadle this: RuntimeError: gpu.offscreen.new(...) failed with 'GPUFrameBuffer: framebuffer status GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT somehow..
        offscreen = GPUOffScreen(width, height)
        offscreen.bind()
        try:
            gpu.matrix.load_matrix(Matrix.Identity(4))
            gpu.matrix.load_projection_matrix(Matrix.Identity(4))
            
            bgl.glClear(bgl.GL_COLOR_BUFFER_BIT)
            bgl.glClear(bgl.GL_DEPTH_BUFFER_BIT)
            
            o = cloud['object']
            vs = cloud['vertices']
            cs = cloud['colors']
            ns = cloud['normals']
            
            dp = pcv.render_display_percent
            l = int((len(vs) / 100) * dp)
            if(dp >= 99):
                l = len(vs)
            vs = vs[:l]
            cs = cs[:l]
            ns = ns[:l]
            
            use_smoothstep = False
            if(pcv.render_smoothstep):
                # for anti-aliasing basic shader should be enabled
                # if(not pcv.illumination and not pcv.override_default_shader):
                if(not pcv.override_default_shader):
                    use_smoothstep = True
                    # sort by depth
                    mw = o.matrix_world
                    depth = []
                    for i, v in enumerate(vs):
                        vw = mw @ Vector(v)
                        depth.append(world_to_camera_view(scene, cam, vw)[2])
                    zps = zip(depth, vs, cs, ns)
                    sps = sorted(zps, key=lambda a: a[0])
                    # split and reverse
                    vs = [a for _, a, b, c in sps][::-1]
                    cs = [b for _, a, b, c in sps][::-1]
                    ns = [c for _, a, b, c in sps][::-1]
            
            if(pcv.dev_depth_enabled):
                if(pcv.illumination):
                    shader_data_vert, shader_data_frag, shader_data_geom = shaders.load('depth_illumination')
                    shader = GPUShader(shader_data_vert, shader_data_frag)
                    batch = batch_for_shader(shader, 'POINTS', {"position": vs, "normal": ns, })
                elif(pcv.dev_depth_false_colors):
                    shader_data_vert, shader_data_frag, shader_data_geom = shaders.load('depth_false_colors')
                    shader = GPUShader(shader_data_vert, shader_data_frag)
                    batch = batch_for_shader(shader, 'POINTS', {"position": vs, })
                else:
                    shader_data_vert, shader_data_frag, shader_data_geom = shaders.load('depth_simple')
                    shader = GPUShader(shader_data_vert, shader_data_frag)
                    batch = batch_for_shader(shader, 'POINTS', {"position": vs, })
            elif(pcv.dev_normal_colors_enabled):
                shader_data_vert, shader_data_frag, shader_data_geom = shaders.load('normal_colors')
                shader = GPUShader(shader_data_vert, shader_data_frag)
                batch = batch_for_shader(shader, 'POINTS', {"position": vs, "normal": ns, })
            elif(pcv.dev_position_colors_enabled):
                shader_data_vert, shader_data_frag, shader_data_geom = shaders.load('position_colors')
                shader = GPUShader(shader_data_vert, shader_data_frag)
                batch = batch_for_shader(shader, 'POINTS', {"position": vs, })
            elif(pcv.illumination):
                if(use_smoothstep):
                    shader_data_vert, shader_data_frag, shader_data_geom = shaders.load('render_illumination_smooth')
                    shader = GPUShader(shader_data_vert, shader_data_frag)
                    batch = batch_for_shader(shader, 'POINTS', {"position": vs, "color": cs, "normal": ns, })
                else:
                    shader_data_vert, shader_data_frag, shader_data_geom = shaders.load('illumination')
                    shader = GPUShader(shader_data_vert, shader_data_frag)
                    batch = batch_for_shader(shader, 'POINTS', {"position": vs, "color": cs, "normal": ns, })
            else:
                if(use_smoothstep):
                    shader_data_vert, shader_data_frag, shader_data_geom = shaders.load('render_simple_smooth')
                    shader = GPUShader(shader_data_vert, shader_data_frag)
                    batch = batch_for_shader(shader, 'POINTS', {"position": vs, "color": cs, })
                else:
                    shader_data_vert, shader_data_frag, shader_data_geom = shaders.load('simple')
                    shader = GPUShader(shader_data_vert, shader_data_frag)
                    batch = batch_for_shader(shader, 'POINTS', {"position": vs, "color": cs, })
            
            shader.bind()
            
            view_matrix = cam.matrix_world.inverted()
            depsgraph = bpy.context.evaluated_depsgraph_get()
            camera_matrix = cam.calc_matrix_camera(depsgraph, x=render.resolution_x, y=render.resolution_y, scale_x=render.pixel_aspect_x, scale_y=render.pixel_aspect_y, )
            perspective_matrix = camera_matrix @ view_matrix
            
            shader.uniform_float("perspective_matrix", perspective_matrix)
            shader.uniform_float("object_matrix", o.matrix_world)
            if(pcv.render_supersampling > 1):
                shader.uniform_float("point_size", pcv.render_point_size * pcv.render_supersampling)
            else:
                shader.uniform_float("point_size", pcv.render_point_size)
            shader.uniform_float("alpha_radius", pcv.alpha_radius)
            shader.uniform_float("global_alpha", pcv.global_alpha)
            
            if(pcv.dev_depth_enabled):
                # pm = bpy.context.region_data.perspective_matrix
                # shader.uniform_float("perspective_matrix", pm)
                # shader.uniform_float("object_matrix", o.matrix_world)
                
                # NOTE: precalculating and storing following should speed up things a bit, but then it won't reflect edits..
                cx = np.sum(vs[:, 0]) / len(vs)
                cy = np.sum(vs[:, 1]) / len(vs)
                cz = np.sum(vs[:, 2]) / len(vs)
                _, _, s = o.matrix_world.decompose()
                l = s.length
                maxd = abs(np.max(vs))
                mind = abs(np.min(vs))
                maxdist = maxd
                if(mind > maxd):
                    maxdist = mind
                shader.uniform_float("maxdist", float(maxdist) * l)
                shader.uniform_float("center", (cx, cy, cz, ))
                
                shader.uniform_float("brightness", pcv.dev_depth_brightness)
                shader.uniform_float("contrast", pcv.dev_depth_contrast)
                
                # shader.uniform_float("point_size", pcv.point_size)
                # shader.uniform_float("alpha_radius", pcv.alpha_radius)
                # shader.uniform_float("global_alpha", pcv.global_alpha)
                
                if(pcv.illumination):
                    cm = Matrix(((-1.0, 0.0, 0.0, 0.0, ), (0.0, -0.0, 1.0, 0.0, ), (0.0, -1.0, -0.0, 0.0, ), (0.0, 0.0, 0.0, 1.0, ), ))
                    _, obrot, _ = o.matrix_world.decompose()
                    mr = obrot.to_matrix().to_4x4()
                    mr.invert()
                    direction = cm @ pcv.light_direction
                    direction = mr @ direction
                    shader.uniform_float("light_direction", direction)
                    inverted_direction = direction.copy()
                    inverted_direction.negate()
                    c = pcv.light_intensity
                    shader.uniform_float("light_intensity", (c, c, c, ))
                    shader.uniform_float("shadow_direction", inverted_direction)
                    c = pcv.shadow_intensity
                    shader.uniform_float("shadow_intensity", (c, c, c, ))
                    if(pcv.dev_depth_false_colors):
                        shader.uniform_float("color_a", pcv.dev_depth_color_a)
                        shader.uniform_float("color_b", pcv.dev_depth_color_b)
                    else:
                        shader.uniform_float("color_a", (1.0, 1.0, 1.0))
                        shader.uniform_float("color_b", (0.0, 0.0, 0.0))
                else:
                    if(pcv.dev_depth_false_colors):
                        shader.uniform_float("color_a", pcv.dev_depth_color_a)
                        shader.uniform_float("color_b", pcv.dev_depth_color_b)
            elif(pcv.dev_normal_colors_enabled):
                pass
            elif(pcv.dev_position_colors_enabled):
                pass
            elif(pcv.illumination and pcv.has_normals and cloud['illumination']):
                cm = Matrix(((-1.0, 0.0, 0.0, 0.0, ), (0.0, -0.0, 1.0, 0.0, ), (0.0, -1.0, -0.0, 0.0, ), (0.0, 0.0, 0.0, 1.0, ), ))
                _, obrot, _ = o.matrix_world.decompose()
                mr = obrot.to_matrix().to_4x4()
                mr.invert()
                direction = cm @ pcv.light_direction
                direction = mr @ direction
                shader.uniform_float("light_direction", direction)
                
                inverted_direction = direction.copy()
                inverted_direction.negate()
                
                c = pcv.light_intensity
                shader.uniform_float("light_intensity", (c, c, c, ))
                shader.uniform_float("shadow_direction", inverted_direction)
                c = pcv.shadow_intensity
                shader.uniform_float("shadow_intensity", (c, c, c, ))
                # shader.uniform_float("show_normals", float(pcv.show_normals))
                # shader.uniform_float("show_illumination", float(pcv.illumination))
            else:
                pass
            
            batch.draw(shader)
            
            buffer = bgl.Buffer(bgl.GL_BYTE, width * height * 4)
            bgl.glReadBuffer(bgl.GL_BACK)
            bgl.glReadPixels(0, 0, width, height, bgl.GL_RGBA, bgl.GL_UNSIGNED_BYTE, buffer)
            
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
            
        finally:
            bgl.glDisable(bgl.GL_PROGRAM_POINT_SIZE)
            bgl.glDisable(bgl.GL_DEPTH_TEST)
            bgl.glDisable(bgl.GL_BLEND)
            offscreen.unbind()
            offscreen.free()
        
        # image from buffer
        image_name = "pcv_output"
        if(image_name not in bpy.data.images):
            bpy.data.images.new(image_name, width, height)
        image = bpy.data.images[image_name]
        image.pixels = [v / 255 for v in buffer]
        # image.scale(width, height)
        
        if(pcv.render_supersampling > 1):
            width = int(width / pcv.render_supersampling)
            height = int(height / pcv.render_supersampling)
            image.scale(width, height)
        
        # save as image file
        def save_render(operator, scene, image, output_path, ):
            rs = scene.render
            s = rs.image_settings
            ff = s.file_format
            cm = s.color_mode
            cd = s.color_depth
            
            vs = scene.view_settings
            vsvt = vs.view_transform
            vsl = vs.look
            vs.view_transform = 'Standard'
            vs.look = 'None'
            
            s.file_format = 'PNG'
            s.color_mode = 'RGBA'
            s.color_depth = '8'
            
            try:
                image.save_render(output_path)
                log("image '{}' saved".format(output_path))
            except Exception as e:
                s.file_format = ff
                s.color_mode = cm
                s.color_depth = cd
                vs.view_transform = vsvt
                vs.look = vsl
                
                log("error: {}".format(e))
                operator.report({'ERROR'}, "Unable to save render image, see console for details.")
                return
            
            s.file_format = ff
            s.color_mode = cm
            s.color_depth = cd
            vs.view_transform = vsvt
            vs.look = vsl
        
        save_render(self, scene, image, output_path, )
        
        # restore
        image_settings.color_depth = original_depth
        # cleanup
        bpy.data.images.remove(image)
        
        _d = datetime.timedelta(seconds=time.time() - _t)
        print("PCV: Frame completed in {}.".format(_d))
        
        return {'FINISHED'}


class PCV_OT_render_animation(Operator):
    bl_idname = "point_cloud_visualizer.render_animation"
    bl_label = "Animation"
    bl_description = "Render displayed point cloud from active camera view to animation frames"
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        
        pcv = context.object.point_cloud_visualizer
        ok = False
        for k, v in PCVManager.cache.items():
            if(v['uuid'] == pcv.uuid):
                if(v['ready']):
                    if(v['draw']):
                        ok = True
        return ok
    
    def execute(self, context):
        scene = context.scene
        
        if(scene.camera is None):
            self.report({'ERROR'}, "No camera found.")
            return {'CANCELLED'}
        
        def rm_ms(d):
            return d - datetime.timedelta(microseconds=d.microseconds)
        
        user_frame = scene.frame_current
        
        _t = time.time()
        log_format = 'PCV: Frame: {} ({}/{}) | Time: {} | Remaining: {}'
        frames = [i for i in range(scene.frame_start, scene.frame_end + 1, 1)]
        num_frames = len(frames)
        times = []
        
        for i, n in enumerate(frames):
            t = time.time()
            
            scene.frame_set(n)
            bpy.ops.point_cloud_visualizer.render()
            
            d = time.time() - t
            times.append(d)
            print(log_format.format(scene.frame_current, i + 1, num_frames,
                                    rm_ms(datetime.timedelta(seconds=d)),
                                    rm_ms(datetime.timedelta(seconds=(sum(times) / len(times)) * (num_frames - i - 1)), ), ))
            
        scene.frame_set(user_frame)
        
        _d = datetime.timedelta(seconds=time.time() - _t)
        print("PCV: Animation completed in {}.".format(_d))
        
        return {'FINISHED'}


class PCV_OT_convert(Operator):
    bl_idname = "point_cloud_visualizer.convert"
    bl_label = "Convert"
    bl_description = "Convert point cloud to mesh"
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        
        pcv = context.object.point_cloud_visualizer
        ok = False
        for k, v in PCVManager.cache.items():
            if(v['uuid'] == pcv.uuid):
                if(v['ready']):
                    if(v['draw']):
                        ok = True
        return ok
    
    def execute(self, context):
        _t = time.time()
        
        scene = context.scene
        pcv = context.object.point_cloud_visualizer
        o = context.object
        
        cache = PCVManager.cache[pcv.uuid]
        
        l = cache['stats']
        if(not pcv.mesh_all):
            nump = l
            mps = pcv.mesh_percentage
            l = int((nump / 100) * mps)
            if(mps >= 99):
                l = nump
        
        vs = cache['vertices'][:l]
        ns = cache['normals'][:l]
        cs = cache['colors'][:l]
        
        points = []
        for i in range(l):
            c = tuple([int(255 * cs[i][j]) for j in range(3)])
            points.append(tuple(vs[i]) + tuple(ns[i]) + c)
        
        def apply_matrix(points, matrix):
            matrot = matrix.decompose()[1].to_matrix().to_4x4()
            r = [None] * len(points)
            for i, p in enumerate(points):
                co = matrix @ Vector((p[0], p[1], p[2]))
                no = matrot @ Vector((p[3], p[4], p[5]))
                r[i] = (co.x, co.y, co.z, no.x, no.y, no.z, p[6], p[7], p[8])
            return r
        
        _, t = os.path.split(pcv.filepath)
        n, _ = os.path.splitext(t)
        m = o.matrix_world.copy()
        
        points = apply_matrix(points, m)
        
        g = None
        if(pcv.mesh_type in ('INSTANCER', 'PARTICLES', )):
            # TODO: if normals are missing or align to normal is not required, make just vertices instead of triangles and use that as source for particles and instances, will be a bit faster
            g = convert.PCMeshInstancerMeshGenerator(mesh_type='TRIANGLE', )
        else:
            g = convert.PCMeshInstancerMeshGenerator(mesh_type=pcv.mesh_type, )
        
        names = {'VERTEX': "{}-vertices",
                 'TRIANGLE': "{}-triangles",
                 'TETRAHEDRON': "{}-tetrahedrons",
                 'CUBE': "{}-cubes",
                 'ICOSPHERE': "{}-icospheres",
                 'INSTANCER': "{}-instancer",
                 'PARTICLES': "{}-particles", }
        n = names[pcv.mesh_type].format(n)
        
        # hide the point cloud, 99% of time i go back, hide cloud and then go to conversion product again..
        bpy.ops.point_cloud_visualizer.erase()
        
        s = pcv.mesh_size
        a = pcv.mesh_normal_align
        c = pcv.mesh_vcols
        if(not pcv.has_normals):
            a = False
        if(not pcv.has_vcols):
            c = False
        d = {'name': n, 'points': points, 'generator': g, 'matrix': Matrix(),
             'size': s, 'normal_align': a, 'vcols': c, }
        if(pcv.mesh_type == 'VERTEX'):
            # faster than instancer.. single vertices can't have normals and colors, so no need for instancer
            bm = bmesh.new()
            for p in points:
                bm.verts.new(p[:3])
            me = bpy.data.meshes.new(n)
            bm.to_mesh(me)
            bm.free()
            o = bpy.data.objects.new(n, me)
            view_layer = context.view_layer
            collection = view_layer.active_layer_collection.collection
            collection.objects.link(o)
            bpy.ops.object.select_all(action='DESELECT')
            o.select_set(True)
            view_layer.objects.active = o
        else:
            
            if(pcv.mesh_use_instancer2):
                # import cProfile
                # import pstats
                # import io
                # pr = cProfile.Profile()
                # pr.enable()
                
                d = {
                    'name': n,
                    'vs': vs,
                    'ns': ns,
                    'cs': cs,
                    'generator': g,
                    'matrix': Matrix(),
                    'size': s,
                    'with_normal_align': a,
                    'with_vertex_colors': c,
                }
                instancer = convert.PCMeshInstancer2(**d)
                
                # pr.disable()
                # s = io.StringIO()
                # sortby = 'cumulative'
                # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
                # ps.print_stats()
                # print(s.getvalue())
            else:
                instancer = convert.PCMeshInstancer(**d)
            
            o = instancer.object
        
        me = o.data
        me.transform(m.inverted())
        o.matrix_world = m
        
        if(pcv.mesh_type == 'INSTANCER'):
            pci = convert.PCInstancer(o, pcv.mesh_size, pcv.mesh_base_sphere_subdivisions, )
        if(pcv.mesh_type == 'PARTICLES'):
            pcp = convert.PCParticles(o, pcv.mesh_size, pcv.mesh_base_sphere_subdivisions, )
        
        _d = datetime.timedelta(seconds=time.time() - _t)
        log("completed in {}.".format(_d))
        
        return {'FINISHED'}


class PCV_OT_export(Operator, ExportHelper):
    bl_idname = "point_cloud_visualizer.export"
    bl_label = "Export PLY"
    bl_description = "Export point cloud to ply file"
    
    filename_ext = ".ply"
    filter_glob: StringProperty(default="*.ply", options={'HIDDEN'}, )
    check_extension = True
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        
        pcv = context.object.point_cloud_visualizer
        ok = False
        for k, v in PCVManager.cache.items():
            if(v['uuid'] == pcv.uuid):
                if(v['ready']):
                    if(v['draw']):
                        ok = True
        return ok
    
    def draw(self, context):
        l = self.layout
        c = l.column()
        pcv = context.object.point_cloud_visualizer
        c.prop(pcv, 'export_apply_transformation')
        c.prop(pcv, 'export_convert_axes')
        c.prop(pcv, 'export_visible_only')
    
    def execute(self, context):
        log("Export:", 0)
        _t = time.time()
        
        pcv = context.object.point_cloud_visualizer
        c = PCVManager.cache[pcv.uuid]
        
        o = c['object']
        
        if(pcv.export_use_viewport):
            log("using viewport points..", 1)
            vs = c['vertices']
            ns = c['normals']
            cs = c['colors']
            
            if(pcv.export_visible_only):
                log("visible only..", 1)
                # points in cache are stored already shuffled (or not), so this should work the same as in viewport..
                l = c['display_length']
                vs = vs[:l]
                ns = ns[:l]
                cs = cs[:l]
            
            # TODO: viewport points have always some normals and colors, should i keep it how it was loaded or should i include also generic data created for viewing?
            normals = True
            colors = True
            
        else:
            log("using original loaded points..", 1)
            # get original loaded points
            points = c['points']
            # check for normals
            normals = True
            if(not set(('nx', 'ny', 'nz')).issubset(points.dtype.names)):
                normals = False
            # check for colors
            colors = True
            if(not set(('red', 'green', 'blue')).issubset(points.dtype.names)):
                colors = False
            # make vertices, normals, colors arrays, use None if data is not available, colors leave as they are
            vs = np.column_stack((points['x'], points['y'], points['z'], ))
            ns = None
            if(normals):
                ns = np.column_stack((points['nx'], points['ny'], points['nz'], ))
            cs = None
            if(colors):
                cs = np.column_stack((points['red'], points['green'], points['blue'], ))
        
        def apply_matrix(m, vs, ns=None, ):
            vs.shape = (-1, 3)
            vs = np.c_[vs, np.ones(vs.shape[0])]
            vs = np.dot(m, vs.T)[0:3].T.reshape((-1))
            vs.shape = (-1, 3)
            if(ns is not None):
                _, rot, _ = m.decompose()
                rmat = rot.to_matrix().to_4x4()
                ns.shape = (-1, 3)
                ns = np.c_[ns, np.ones(ns.shape[0])]
                ns = np.dot(rmat, ns.T)[0:3].T.reshape((-1))
                ns.shape = (-1, 3)
            return vs, ns
        
        # fabricate matrix
        m = Matrix.Identity(4)
        if(pcv.export_apply_transformation):
            if(o.matrix_world != Matrix.Identity(4)):
                log("apply transformation..", 1)
                vs, ns = apply_matrix(o.matrix_world.copy(), vs, ns, )
        if(pcv.export_convert_axes):
            log("convert axes..", 1)
            axis_forward = '-Z'
            axis_up = 'Y'
            cm = axis_conversion(to_forward=axis_forward, to_up=axis_up).to_4x4()
            vs, ns = apply_matrix(cm, vs, ns, )
        
        # TODO: make whole PCV data type agnostic, load anything, keep original, convert to what is needed for display (float32), use original for export if not set to use viewport/edited data. now i am forcing float32 for x, y, z, nx, ny, nz and uint8 for red, green, blue. 99% of ply files i've seen is like that, but specification is not that strict (read again the best resource: http://paulbourke.net/dataformats/ply/ )
        
        # somehow along the way i am getting double dtype, so correct that
        vs = vs.astype(np.float32)
        if(normals):
            ns = ns.astype(np.float32)
        if(colors):
            # cs = cs.astype(np.float32)
            
            if(pcv.export_use_viewport):
                # viewport colors are in float32 now, back to uint8 colors, loaded data should be in uint8, so no need for conversion
                cs = cs.astype(np.float32)
                cs = cs * 255
                cs = cs.astype(np.uint8)
        
        log("write..", 1)
        
        # combine back to points, using original dtype
        dt = (('x', vs[0].dtype.str, ),
              ('y', vs[1].dtype.str, ),
              ('z', vs[2].dtype.str, ), )
        if(normals):
            dt += (('nx', ns[0].dtype.str, ),
                   ('ny', ns[1].dtype.str, ),
                   ('nz', ns[2].dtype.str, ), )
        if(colors):
            dt += (('red', cs[0].dtype.str, ),
                   ('green', cs[1].dtype.str, ),
                   ('blue', cs[2].dtype.str, ), )
        log("dtype: {}".format(dt), 1)
        
        l = len(vs)
        dt = list(dt)
        a = np.empty(l, dtype=dt, )
        a['x'] = vs[:, 0]
        a['y'] = vs[:, 1]
        a['z'] = vs[:, 2]
        if(normals):
            a['nx'] = ns[:, 0]
            a['ny'] = ns[:, 1]
            a['nz'] = ns[:, 2]
        if(colors):
            a['red'] = cs[:, 0]
            a['green'] = cs[:, 1]
            a['blue'] = cs[:, 2]
        
        w = io.BinPlyPointCloudWriter(self.filepath, a, )
        
        _d = datetime.timedelta(seconds=time.time() - _t)
        log("completed in {}.".format(_d), 1)
        
        return {'FINISHED'}


class PCV_OT_filter_simplify(Operator):
    bl_idname = "point_cloud_visualizer.filter_simplify"
    bl_label = "Simplify"
    bl_description = "Simplify point cloud to exact number of evenly distributed samples, all loaded points are processed"
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        
        pcv = context.object.point_cloud_visualizer
        ok = False
        for k, v in PCVManager.cache.items():
            if(v['uuid'] == pcv.uuid):
                if(v['ready']):
                    if(v['draw']):
                        ok = True
        return ok
    
    def resample(self, context):
        scene = context.scene
        pcv = context.object.point_cloud_visualizer
        
        c = PCVManager.cache[pcv.uuid]
        vs = c['vertices']
        ns = c['normals']
        cs = c['colors']
        
        num_samples = pcv.filter_simplify_num_samples
        if(num_samples >= len(vs)):
            self.report({'ERROR'}, "Number of samples must be < number of points.")
            return False, []
        candidates = pcv.filter_simplify_num_candidates
        log("num_samples: {}, candidates: {}".format(num_samples, candidates), 1)
        
        l = len(vs)
        dt = [('x', '<f8'), ('y', '<f8'), ('z', '<f8'), ('nx', '<f8'), ('ny', '<f8'), ('nz', '<f8'), ('red', '<f8'), ('green', '<f8'), ('blue', '<f8'), ('alpha', '<f8'), ('index', '<i8')]
        pool = np.empty(l, dtype=dt, )
        pool['x'] = vs[:, 0]
        pool['y'] = vs[:, 1]
        pool['z'] = vs[:, 2]
        pool['nx'] = ns[:, 0]
        pool['ny'] = ns[:, 1]
        pool['nz'] = ns[:, 2]
        pool['red'] = cs[:, 0]
        pool['green'] = cs[:, 1]
        pool['blue'] = cs[:, 2]
        pool['alpha'] = cs[:, 3]
        pool['index'] = np.indices((l, ), dtype='<i8', )
        
        # to get random points, shuffle pool if not shuffled upon loading
        # preferences = bpy.context.preferences
        # addon_prefs = preferences.addons[__name__].preferences
        addon_prefs = preferences()
        if(not addon_prefs.shuffle_points):
            np.random.shuffle(pool)
            # and set indexes again
            pool['index'] = np.indices((l, ), dtype='<i8', )
        
        tree = KDTree(len(pool))
        samples = np.array(pool[0])
        last_used = 0
        # unused_pool = np.empty(0, dtype=dt, )
        unused_pool = []
        
        tree.insert([pool[0]['x'], pool[0]['y'], pool[0]['z']], 0)
        tree.balance()
        
        log("sampling:", 1)
        use_unused = False
        prgs = debug.Progress(num_samples - 1, indent=2, prefix="> ")
        for i in range(num_samples - 1):
            prgs.step()
            # choose candidates
            cands = pool[last_used + 1:last_used + 1 + candidates]
            
            if(len(cands) <= 0):
                # lets pretend that last set of candidates was filled full..
                use_unused = True
                cands = unused_pool[:candidates]
                unused_pool = unused_pool[candidates:]
            
            if(len(cands) == 0):
                # out of candidates, nothing to do here now.. number of desired samples was too close to total of points
                return True, samples
            
            # get the most distant candidate
            dists = []
            inds = []
            for cl in cands:
                vec, ci, cd = tree.find((cl['x'], cl['y'], cl['z']))
                inds.append(ci)
                dists.append(cd)
            maxd = 0.0
            maxi = 0
            for j, d in enumerate(dists):
                if(d > maxd):
                    maxd = d
                    maxi = j
            # chosen candidate
            ncp = cands[maxi]
            # get unused candidates to recycle
            uncands = np.delete(cands, maxi)
            # unused_pool = np.append(unused_pool, uncands)
            unused_pool.extend(uncands)
            # store accepted sample
            samples = np.append(samples, ncp)
            # update kdtree
            tree.insert([ncp['x'], ncp['y'], ncp['z']], ncp['index'])
            tree.balance()
            # use next points
            last_used += 1 + candidates
        
        return True, samples
    
    def execute(self, context):
        log("Simplify:", 0)
        _t = time.time()
        
        # if(debug_mode()):
        #     import cProfile
        #     import pstats
        #     import io
        #     pr = cProfile.Profile()
        #     pr.enable()
        
        ok, a = self.resample(context)
        if(not ok):
            return {'CANCELLED'}
        
        # if(debug_mode()):
        #     pr.disable()
        #     s = io.StringIO()
        #     sortby = 'cumulative'
        #     ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        #     ps.print_stats()
        #     print(s.getvalue())
        
        vs = np.column_stack((a['x'], a['y'], a['z'], ))
        ns = np.column_stack((a['nx'], a['ny'], a['nz'], ))
        cs = np.column_stack((a['red'], a['green'], a['blue'], a['alpha'], ))
        vs = vs.astype(np.float32)
        ns = ns.astype(np.float32)
        cs = cs.astype(np.float32)
        
        # put to cache
        pcv = context.object.point_cloud_visualizer
        PCVManager.update(pcv.uuid, vs, ns, cs, )
        
        _d = datetime.timedelta(seconds=time.time() - _t)
        log("completed in {}.".format(_d), 1)
        
        return {'FINISHED'}


class PCV_OT_filter_project(Operator):
    bl_idname = "point_cloud_visualizer.filter_project"
    bl_label = "Project"
    bl_description = "Project points on mesh surface"
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        
        pcv = context.object.point_cloud_visualizer
        ok = False
        for k, v in PCVManager.cache.items():
            if(v['uuid'] == pcv.uuid):
                if(v['ready']):
                    if(v['draw']):
                        o = pcv.filter_project_object
                        if(o is not None):
                            if(o.type in ('MESH', 'CURVE', 'SURFACE', 'FONT', )):
                                ok = True
        return ok
    
    def execute(self, context):
        log("Project:", 0)
        _t = time.time()
        
        # if(debug_mode()):
        #     import cProfile
        #     import pstats
        #     import io
        #     pr = cProfile.Profile()
        #     pr.enable()
        
        log("preprocessing..", 1)
        
        pcv = context.object.point_cloud_visualizer
        o = pcv.filter_project_object
        
        if(o is None):
            raise Exception()
        
        c = PCVManager.cache[pcv.uuid]
        vs = c['vertices']
        ns = c['normals']
        cs = c['colors']
        
        # apply parent matrix to points
        def apply_matrix(m, vs, ns=None, ):
            vs.shape = (-1, 3)
            vs = np.c_[vs, np.ones(vs.shape[0])]
            vs = np.dot(m, vs.T)[0:3].T.reshape((-1))
            vs.shape = (-1, 3)
            if(ns is not None):
                _, rot, _ = m.decompose()
                rmat = rot.to_matrix().to_4x4()
                ns.shape = (-1, 3)
                ns = np.c_[ns, np.ones(ns.shape[0])]
                ns = np.dot(rmat, ns.T)[0:3].T.reshape((-1))
                ns.shape = (-1, 3)
            return vs, ns
        
        m = c['object'].matrix_world.copy()
        vs, ns = apply_matrix(m, vs, ns, )
        vs = vs.astype(np.float32)
        ns = ns.astype(np.float32)
        
        # combine
        l = len(vs)
        dt = [('x', '<f8'), ('y', '<f8'), ('z', '<f8'), ('nx', '<f8'), ('ny', '<f8'), ('nz', '<f8'), ('red', '<f8'), ('green', '<f8'), ('blue', '<f8'), ('alpha', '<f8'), ('index', '<i8'), ('delete', '?')]
        points = np.empty(l, dtype=dt, )
        points['x'] = vs[:, 0]
        points['y'] = vs[:, 1]
        points['z'] = vs[:, 2]
        points['nx'] = ns[:, 0]
        points['ny'] = ns[:, 1]
        points['nz'] = ns[:, 2]
        points['red'] = cs[:, 0]
        points['green'] = cs[:, 1]
        points['blue'] = cs[:, 2]
        points['alpha'] = cs[:, 3]
        points['index'] = np.indices((l, ), dtype='<i8', )
        points['delete'] = np.zeros((l, ), dtype='?', )
        
        search_distance = pcv.filter_project_search_distance
        negative = pcv.filter_project_negative
        positive = pcv.filter_project_positive
        discard = pcv.filter_project_discard
        shift = pcv.filter_project_shift
        
        sc = context.scene
        depsgraph = bpy.context.evaluated_depsgraph_get()
        
        # make target
        tmp_mesh = o.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph, )
        target_mesh = tmp_mesh.copy()
        target_mesh.name = 'target_mesh_{}'.format(pcv.uuid)
        target_mesh.transform(o.matrix_world.copy())
        view_layer = context.view_layer
        collection = view_layer.active_layer_collection.collection
        target = bpy.data.objects.new('target_mesh_{}'.format(pcv.uuid), target_mesh)
        collection.objects.link(target)
        # TODO use BVHTree for ray_cast without need to add object to scene
        # still no idea, have to read about it..
        depsgraph.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        
        # this should be None if not available
        vcols = o.data.vertex_colors.active
        uvtex = o.data.uv_layers.active
        vgroup = o.vertex_groups.active
        # now check if color source is available, if not, cancel
        if(pcv.filter_project_colorize):
            
            # depsgraph = context.evaluated_depsgraph_get()
            # if(o.modifiers):
            #     owner = o.evaluated_get(depsgraph)
            #     me = owner.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph, )
            # else:
            #     owner = o
            #     me = owner.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph, )
            
            bm = bmesh.new()
            bm.from_mesh(target_mesh)
            bmesh.ops.triangulate(bm, faces=bm.faces)
            bm.verts.ensure_lookup_table()
            bm.faces.ensure_lookup_table()
            
            if(pcv.filter_project_colorize_from == 'VCOLS'):
                try:
                    col_layer = bm.loops.layers.color.active
                    if(col_layer is None):
                        raise Exception()
                except Exception:
                    self.report({'ERROR'}, "Cannot find active vertex colors", )
                    return {'CANCELLED'}
            elif(pcv.filter_project_colorize_from == 'UVTEX'):
                try:
                    if(o.active_material is None):
                        raise Exception("Cannot find active material")
                    uvtexnode = o.active_material.node_tree.nodes.active
                    if(uvtexnode is None):
                        raise Exception("Cannot find active image texture in active material")
                    if(uvtexnode.type != 'TEX_IMAGE'):
                        raise Exception("Cannot find active image texture in active material")
                    uvimage = uvtexnode.image
                    if(uvimage is None):
                        raise Exception("Cannot find active image texture with loaded image in active material")
                    uvimage.update()
                    uvarray = np.asarray(uvimage.pixels)
                    uvarray = uvarray.reshape((uvimage.size[1], uvimage.size[0], 4))
                    uvlayer = bm.loops.layers.uv.active
                    if(uvlayer is None):
                        raise Exception("Cannot find active UV layout")
                except Exception as e:
                    self.report({'ERROR'}, str(e), )
                    return {'CANCELLED'}
            elif(pcv.filter_project_colorize_from in ['GROUP_MONO', 'GROUP_COLOR', ]):
                try:
                    group_layer = bm.verts.layers.deform.active
                    if(group_layer is None):
                        raise Exception()
                    group_layer_index = o.vertex_groups.active.index
                except Exception:
                    self.report({'ERROR'}, "Cannot find active vertex group", )
                    return {'CANCELLED'}
            else:
                self.report({'ERROR'}, "Unsupported color source", )
                return {'CANCELLED'}
        
        def distance(a, b, ):
            return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5
        
        def shift_vert_along_normal(co, no, v):
            c = Vector(co)
            n = Vector(no)
            return c + (n.normalized() * v)
        
        def remap(v, min1, max1, min2, max2, ):
            def clamp(v, vmin, vmax):
                if(vmax <= vmin):
                    raise ValueError("Maximum value is smaller than or equal to minimum.")
                if(v <= vmin):
                    return vmin
                if(v >= vmax):
                    return vmax
                return v
            
            def normalize(v, vmin, vmax):
                return (v - vmin) / (vmax - vmin)
            
            def interpolate(nv, vmin, vmax):
                return vmin + (vmax - vmin) * nv
            
            if(max1 - min1 == 0):
                # handle zero division when min1 = max1
                return min2
            
            r = interpolate(normalize(v, min1, max1), min2, max2)
            return r
        
        def gen_color(bm, result, location, normal, index, distance, ):
            col = (0.0, 0.0, 0.0, )
            
            poly = bm.faces[index]
            v = location
            
            if(pcv.filter_project_colorize_from == 'VCOLS'):
                ws = poly_3d_calc([poly.verts[0].co, poly.verts[1].co, poly.verts[2].co, ], v)
                ac = poly.loops[0][col_layer][:3]
                bc = poly.loops[1][col_layer][:3]
                cc = poly.loops[2][col_layer][:3]
                r = ac[0] * ws[0] + bc[0] * ws[1] + cc[0] * ws[2]
                g = ac[1] * ws[0] + bc[1] * ws[1] + cc[1] * ws[2]
                b = ac[2] * ws[0] + bc[2] * ws[1] + cc[2] * ws[2]
                col = (r, g, b, )
            elif(pcv.filter_project_colorize_from == 'UVTEX'):
                uvtriangle = []
                for l in poly.loops:
                    uvtriangle.append(Vector(l[uvlayer].uv.to_tuple() + (0.0, )))
                uvpoint = barycentric_transform(v, poly.verts[0].co, poly.verts[1].co, poly.verts[2].co, *uvtriangle, )
                w, h = uvimage.size
                # x,y % 1.0 to wrap around if uv coordinate is outside 0.0-1.0 range
                x = int(round(remap(uvpoint.x % 1.0, 0.0, 1.0, 0, w - 1)))
                y = int(round(remap(uvpoint.y % 1.0, 0.0, 1.0, 0, h - 1)))
                col = tuple(uvarray[y][x][:3].tolist())
            elif(pcv.filter_project_colorize_from == 'GROUP_MONO'):
                ws = poly_3d_calc([poly.verts[0].co, poly.verts[1].co, poly.verts[2].co, ], v)
                aw = poly.verts[0][group_layer].get(group_layer_index, 0.0)
                bw = poly.verts[1][group_layer].get(group_layer_index, 0.0)
                cw = poly.verts[2][group_layer].get(group_layer_index, 0.0)
                m = aw * ws[0] + bw * ws[1] + cw * ws[2]
                col = (m, m, m, )
            elif(pcv.filter_project_colorize_from == 'GROUP_COLOR'):
                ws = poly_3d_calc([poly.verts[0].co, poly.verts[1].co, poly.verts[2].co, ], v)
                aw = poly.verts[0][group_layer].get(group_layer_index, 0.0)
                bw = poly.verts[1][group_layer].get(group_layer_index, 0.0)
                cw = poly.verts[2][group_layer].get(group_layer_index, 0.0)
                m = aw * ws[0] + bw * ws[1] + cw * ws[2]
                hue = remap(1.0 - m, 0.0, 1.0, 0.0, 1 / 1.5)
                c = Color()
                c.hsv = (hue, 1.0, 1.0, )
                col = (c.r, c.g, c.b, )
            
            return col
        
        a = np.empty(l, dtype=dt, )
        
        log("projecting:", 1)
        prgr = debug.Progress(len(points), 2)
        for i, p in enumerate(points):
            prgr.step()
            
            v = Vector((p['x'], p['y'], p['z'], ))
            n = Vector((p['nx'], p['ny'], p['nz'], ))
            
            if(positive):
                p_result, p_location, p_normal, p_index = target.ray_cast(v, n, distance=search_distance, depsgraph=depsgraph, )
                p_distance = None
                if(p_result):
                    p_distance = distance(p_location, v[:])
            else:
                p_result = False
            
            nn = shift_vert_along_normal(v, n, -search_distance)
            ndir = Vector(nn - v).normalized()
            
            if(negative):
                n_result, n_location, n_normal, n_index = target.ray_cast(v, ndir, distance=search_distance, depsgraph=depsgraph, )
                n_distance = None
                if(n_result):
                    n_distance = distance(n_location, v[:])
            else:
                n_result = False
            
            rp = np.copy(p)
            
            # store ray_cast results which was used for current point and pass them to gen_color
            used = None
            
            if(p_result and n_result):
                if(p_distance < n_distance):
                    rp['x'] = p_location[0]
                    rp['y'] = p_location[1]
                    rp['z'] = p_location[2]
                    used = (p_result, p_location, p_normal, p_index, p_distance)
                else:
                    rp['x'] = n_location[0]
                    rp['y'] = n_location[1]
                    rp['z'] = n_location[2]
                    used = (n_result, n_location, n_normal, n_index, n_distance)
            elif(p_result):
                rp['x'] = p_location[0]
                rp['y'] = p_location[1]
                rp['z'] = p_location[2]
                used = (p_result, p_location, p_normal, p_index, p_distance)
            elif(n_result):
                rp['x'] = n_location[0]
                rp['y'] = n_location[1]
                rp['z'] = n_location[2]
                used = (n_result, n_location, n_normal, n_index, n_distance)
            else:
                rp['delete'] = 1
            
            if(pcv.filter_project_colorize):
                if(used is not None):
                    col = gen_color(bm, *used)
                    rp['red'] = col[0]
                    rp['green'] = col[1]
                    rp['blue'] = col[2]
            
            a[i] = rp
        
        if(discard):
            log("discarding:", 1)
            prgr = debug.Progress(len(a), 2)
            indexes = []
            for i in a:
                prgr.step()
                if(i['delete']):
                    indexes.append(i['index'])
            a = np.delete(a, indexes)
        
        if(shift != 0.0):
            log("shifting:", 1)
            prgr = debug.Progress(len(a), 2)
            for i in a:
                prgr.step()
                l = shift_vert_along_normal((i['x'], i['y'], i['z'], ), (i['nx'], i['ny'], i['nz'], ), shift, )
                i['x'] = l[0]
                i['y'] = l[1]
                i['z'] = l[2]
        
        log("postprocessing..", 1)
        # split back
        vs = np.column_stack((a['x'], a['y'], a['z'], ))
        ns = np.column_stack((a['nx'], a['ny'], a['nz'], ))
        cs = np.column_stack((a['red'], a['green'], a['blue'], a['alpha'], ))
        vs = vs.astype(np.float32)
        ns = ns.astype(np.float32)
        cs = cs.astype(np.float32)
        
        # unapply parent matrix to points
        m = c['object'].matrix_world.copy()
        m = m.inverted()
        vs, ns = apply_matrix(m, vs, ns, )
        vs = vs.astype(np.float32)
        ns = ns.astype(np.float32)
        
        # cleanup
        if(pcv.filter_project_colorize):
            bm.free()
            # owner.to_mesh_clear()
        
        collection.objects.unlink(target)
        bpy.data.objects.remove(target)
        bpy.data.meshes.remove(target_mesh)
        o.to_mesh_clear()
        
        # put to cache..
        pcv = context.object.point_cloud_visualizer
        PCVManager.update(pcv.uuid, vs, ns, cs, )
        
        # if(debug_mode()):
        #     pr.disable()
        #     s = io.StringIO()
        #     sortby = 'cumulative'
        #     ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        #     ps.print_stats()
        #     print(s.getvalue())
        
        _d = datetime.timedelta(seconds=time.time() - _t)
        log("completed in {}.".format(_d), 1)
        
        return {'FINISHED'}


class PCV_OT_filter_remove_color(Operator):
    bl_idname = "point_cloud_visualizer.filter_remove_color"
    bl_label = "Select Color"
    bl_description = "Select points with exact/similar color"
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        
        pcv = context.object.point_cloud_visualizer
        ok = False
        for k, v in PCVManager.cache.items():
            if(v['uuid'] == pcv.uuid):
                if(v['ready']):
                    if(v['draw']):
                        ok = True
        return ok
    
    def execute(self, context):
        log("Remove Color:", 0)
        _t = time.time()
        
        pcv = context.object.point_cloud_visualizer
        # cache item
        c = PCVManager.cache[pcv.uuid]
        vs = c['vertices']
        ns = c['normals']
        cs = c['colors']
        # join to indexed points
        l = len(vs)
        dt = [('x', '<f8'), ('y', '<f8'), ('z', '<f8'), ('nx', '<f8'), ('ny', '<f8'), ('nz', '<f8'), ('red', '<f8'), ('green', '<f8'), ('blue', '<f8'), ('alpha', '<f8'), ('index', '<i8')]
        points = np.empty(l, dtype=dt, )
        points['x'] = vs[:, 0]
        points['y'] = vs[:, 1]
        points['z'] = vs[:, 2]
        points['nx'] = ns[:, 0]
        points['ny'] = ns[:, 1]
        points['nz'] = ns[:, 2]
        points['red'] = cs[:, 0]
        points['green'] = cs[:, 1]
        points['blue'] = cs[:, 2]
        points['alpha'] = cs[:, 3]
        points['index'] = np.indices((l, ), dtype='<i8', )
        
        # evaluate
        indexes = []
        
        # black magic..
        c = [c ** (1 / 2.2) for c in pcv.filter_remove_color]
        c = [int(i * 256) for i in c]
        c = [i / 256 for i in c]
        rmcolor = Color(c)
        
        # take half of the value because 1/2 <- v -> 1/2, plus and minus => full range
        dh = pcv.filter_remove_color_delta_hue / 2
        # only for hue, because i take in consideration its radial nature
        ds = pcv.filter_remove_color_delta_saturation
        dv = pcv.filter_remove_color_delta_value
        uh = pcv.filter_remove_color_delta_hue_use
        us = pcv.filter_remove_color_delta_saturation_use
        uv = pcv.filter_remove_color_delta_value_use
        
        prgr = debug.Progress(len(points), 1)
        for p in points:
            prgr.step()
            # get point color
            c = Color((p['red'], p['green'], p['blue']))
            # check for more or less same color, a few decimals should be more than enough, ply should have 8bit colors
            fpr = 5
            same = (round(c.r, fpr) == round(rmcolor.r, fpr),
                    round(c.g, fpr) == round(rmcolor.g, fpr),
                    round(c.b, fpr) == round(rmcolor.b, fpr))
            if(all(same)):
                indexes.append(p['index'])
                continue
            
            # check
            h = False
            s = False
            v = False
            if(uh):
                rm_hue = rmcolor.h
                hd = min(abs(rm_hue - c.h), 1.0 - abs(rm_hue - c.h))
                if(hd <= dh):
                    h = True
            if(us):
                if(rmcolor.s - ds < c.s < rmcolor.s + ds):
                    s = True
            if(uv):
                if(rmcolor.v - dv < c.v < rmcolor.v + dv):
                    v = True
            
            a = False
            if(uh and not us and not uv):
                if(h):
                    a = True
            elif(not uh and us and not uv):
                if(s):
                    a = True
            elif(not uh and not us and uv):
                if(v):
                    a = True
            elif(uh and us and not uv):
                if(h and s):
                    a = True
            elif(not uh and us and uv):
                if(s and v):
                    a = True
            elif(uh and not us and uv):
                if(h and v):
                    a = True
            elif(uh and us and uv):
                if(h and s and v):
                    a = True
            else:
                pass
            if(a):
                indexes.append(p['index'])
        
        log("selected: {} points".format(len(indexes)), 1)
        
        if(len(indexes) == 0):
            # self.report({'ERROR'}, "Nothing selected.")
            self.report({'INFO'}, "Nothing selected.")
        else:
            pcv.filter_remove_color_selection = True
            c = PCVManager.cache[pcv.uuid]
            c['selection_indexes'] = indexes
        
        context.area.tag_redraw()
        
        _d = datetime.timedelta(seconds=time.time() - _t)
        log("completed in {}.".format(_d), 1)
        
        return {'FINISHED'}


class PCV_OT_filter_remove_color_deselect(Operator):
    bl_idname = "point_cloud_visualizer.filter_remove_color_deselect"
    bl_label = "Deselect"
    bl_description = "Deselect points"
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        
        pcv = context.object.point_cloud_visualizer
        ok = False
        for k, v in PCVManager.cache.items():
            if(v['uuid'] == pcv.uuid):
                if(v['ready']):
                    if(v['draw']):
                        if(pcv.filter_remove_color_selection):
                            if('selection_indexes' in v.keys()):
                                ok = True
        return ok
    
    def execute(self, context):
        pcv = context.object.point_cloud_visualizer
        c = PCVManager.cache[pcv.uuid]
        
        pcv.filter_remove_color_selection = False
        del c['selection_indexes']
        
        context.area.tag_redraw()
        
        return {'FINISHED'}


class PCV_OT_filter_remove_color_delete_selected(Operator):
    bl_idname = "point_cloud_visualizer.filter_remove_color_delete_selected"
    bl_label = "Delete Selected"
    bl_description = "Remove selected points"
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        
        pcv = context.object.point_cloud_visualizer
        ok = False
        for k, v in PCVManager.cache.items():
            if(v['uuid'] == pcv.uuid):
                if(v['ready']):
                    if(v['draw']):
                        if(pcv.filter_remove_color_selection):
                            if('selection_indexes' in v.keys()):
                                ok = True
        return ok
    
    def execute(self, context):
        pcv = context.object.point_cloud_visualizer
        c = PCVManager.cache[pcv.uuid]
        vs = c['vertices']
        ns = c['normals']
        cs = c['colors']
        indexes = c['selection_indexes']
        vs = np.delete(vs, indexes, axis=0, )
        ns = np.delete(ns, indexes, axis=0, )
        cs = np.delete(cs, indexes, axis=0, )
        
        PCVManager.update(pcv.uuid, vs, ns, cs, )
        
        pcv.filter_remove_color_selection = False
        del c['selection_indexes']
        
        return {'FINISHED'}


class PCV_OT_edit_start(Operator):
    bl_idname = "point_cloud_visualizer.edit_start"
    bl_label = "Start"
    bl_description = "Start edit mode, create helper object and switch to it"
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        
        pcv = context.object.point_cloud_visualizer
        ok = False
        for k, v in PCVManager.cache.items():
            if(v['uuid'] == pcv.uuid):
                if(v['ready']):
                    if(v['draw']):
                        ok = True
        return ok
    
    def execute(self, context):
        pcv = context.object.point_cloud_visualizer
        c = PCVManager.cache[pcv.uuid]
        vs = c['vertices']
        ns = c['normals']
        cs = c['colors']
        
        # ensure object mode
        bpy.ops.object.mode_set(mode='OBJECT')
        
        # prepare mesh
        bm = bmesh.new()
        for v in vs:
            bm.verts.new(v)
        bm.verts.ensure_lookup_table()
        l = bm.verts.layers.int.new('pcv_indexes')
        for i in range(len(vs)):
            bm.verts[i][l] = i
        # add mesh to scene, activate
        nm = 'pcv_edit_mesh_{}'.format(pcv.uuid)
        me = bpy.data.meshes.new(nm)
        bm.to_mesh(me)
        bm.free()
        o = bpy.data.objects.new(nm, me)
        view_layer = context.view_layer
        collection = view_layer.active_layer_collection.collection
        collection.objects.link(o)
        p = context.object
        o.parent = p
        o.matrix_world = p.matrix_world.copy()
        bpy.ops.object.select_all(action='DESELECT')
        o.select_set(True)
        view_layer.objects.active = o
        # and set edit mode
        bpy.ops.object.mode_set(mode='EDIT')
        # set vertex select mode..
        bpy.ops.mesh.select_mode(use_extend=False, use_expand=False, type='VERT', )
        
        o.point_cloud_visualizer.edit_is_edit_uuid = pcv.uuid
        o.point_cloud_visualizer.edit_is_edit_mesh = True
        pcv.edit_initialized = True
        pcv.edit_pre_edit_alpha = pcv.global_alpha
        pcv.global_alpha = pcv.edit_overlay_alpha
        pcv.edit_pre_edit_display = pcv.display_percent
        pcv.display_percent = 100.0
        pcv.edit_pre_edit_size = pcv.point_size
        pcv.point_size = pcv.edit_overlay_size
        
        return {'FINISHED'}


class PCV_OT_edit_update(Operator):
    bl_idname = "point_cloud_visualizer.edit_update"
    bl_label = "Update"
    bl_description = "Update displayed cloud from edited mesh"
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        
        pcv = context.object.point_cloud_visualizer
        ok = False
        if(pcv.edit_is_edit_mesh):
            for k, v in PCVManager.cache.items():
                if(v['uuid'] == pcv.edit_is_edit_uuid):
                    if(v['ready']):
                        if(v['draw']):
                            if(context.mode == 'EDIT_MESH'):
                                ok = True
        return ok
    
    def execute(self, context):
        # get current data
        uuid = context.object.point_cloud_visualizer.edit_is_edit_uuid
        c = PCVManager.cache[uuid]
        vs = c['vertices']
        ns = c['normals']
        cs = c['colors']
        # extract edited data
        o = context.object
        bm = bmesh.from_edit_mesh(o.data)
        bm.verts.ensure_lookup_table()
        l = bm.verts.layers.int['pcv_indexes']
        edit_vs = []
        edit_indexes = []
        for v in bm.verts:
            edit_vs.append(v.co.to_tuple())
            edit_indexes.append(v[l])
        # combine
        u_vs = []
        u_ns = []
        u_cs = []
        for i, indx in enumerate(edit_indexes):
            u_vs.append(edit_vs[i])
            u_ns.append(ns[indx])
            u_cs.append(cs[indx])
        # display
        vs = np.array(u_vs, dtype=np.float32, )
        ns = np.array(u_ns, dtype=np.float32, )
        cs = np.array(u_cs, dtype=np.float32, )
        PCVManager.update(uuid, vs, ns, cs, )
        # update indexes
        bm.verts.ensure_lookup_table()
        l = bm.verts.layers.int['pcv_indexes']
        for i in range(len(vs)):
            bm.verts[i][l] = i
        
        return {'FINISHED'}


class PCV_OT_edit_end(Operator):
    bl_idname = "point_cloud_visualizer.edit_end"
    bl_label = "End"
    bl_description = "Update displayed cloud from edited mesh, stop edit mode and remove helper object"
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        
        pcv = context.object.point_cloud_visualizer
        ok = False
        if(pcv.edit_is_edit_mesh):
            for k, v in PCVManager.cache.items():
                if(v['uuid'] == pcv.edit_is_edit_uuid):
                    if(v['ready']):
                        if(v['draw']):
                            if(context.mode == 'EDIT_MESH'):
                                ok = True
        return ok
    
    def execute(self, context):
        # update
        bpy.ops.point_cloud_visualizer.edit_update()
        
        # cleanup
        bpy.ops.object.mode_set(mode='EDIT')
        o = context.object
        p = o.parent
        me = o.data
        view_layer = context.view_layer
        collection = view_layer.active_layer_collection.collection
        collection.objects.unlink(o)
        bpy.data.objects.remove(o)
        bpy.data.meshes.remove(me)
        # go back
        bpy.ops.object.select_all(action='DESELECT')
        p.select_set(True)
        view_layer.objects.active = p
        
        p.point_cloud_visualizer.edit_initialized = False
        p.point_cloud_visualizer.global_alpha = p.point_cloud_visualizer.edit_pre_edit_alpha
        p.point_cloud_visualizer.display_percent = p.point_cloud_visualizer.edit_pre_edit_display
        p.point_cloud_visualizer.point_size = p.point_cloud_visualizer.edit_pre_edit_size
        
        return {'FINISHED'}


class PCV_OT_edit_cancel(Operator):
    bl_idname = "point_cloud_visualizer.edit_cancel"
    bl_label = "Cancel"
    bl_description = "Stop edit mode, try to remove helper object and reload original point cloud"
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        
        pcv = context.object.point_cloud_visualizer
        if(pcv.edit_initialized):
            return True
        return False
    
    def execute(self, context):
        pcv = context.object.point_cloud_visualizer
        po = context.object
        nm = 'pcv_edit_mesh_{}'.format(pcv.uuid)
        view_layer = context.view_layer
        collection = view_layer.active_layer_collection.collection
        for o in po.children:
            if(o.name == nm):
                me = o.data
                collection.objects.unlink(o)
                bpy.data.objects.remove(o)
                bpy.data.meshes.remove(me)
                break
        
        pcv.edit_initialized = False
        pcv.global_alpha = pcv.edit_pre_edit_alpha
        pcv.edit_pre_edit_alpha = 0.5
        pcv.display_percent = pcv.edit_pre_edit_display
        pcv.edit_pre_edit_display = 100.0
        pcv.point_size = pcv.edit_pre_edit_size
        pcv.edit_pre_edit_size = 3
        
        # also beware, this changes uuid
        bpy.ops.point_cloud_visualizer.reload()
        
        return {'FINISHED'}


class PCV_OT_filter_merge(Operator):
    bl_idname = "point_cloud_visualizer.filter_merge"
    bl_label = "Merge With Other PLY"
    bl_description = "Merge with other ply file"
    
    filename_ext = ".ply"
    filter_glob: StringProperty(default="*.ply", options={'HIDDEN'}, )
    filepath: StringProperty(name="File Path", default="", description="", maxlen=1024, subtype='FILE_PATH', )
    order = ["filepath", ]
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        
        pcv = context.object.point_cloud_visualizer
        ok = False
        for k, v in PCVManager.cache.items():
            if(v['uuid'] == pcv.uuid):
                if(v['ready']):
                    if(v['draw']):
                        ok = True
        return ok
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}
    
    def execute(self, context):
        pcv = context.object.point_cloud_visualizer
        addon_prefs = preferences()
        
        filepath = self.filepath
        h, t = os.path.split(filepath)
        n, e = os.path.splitext(t)
        if(e != '.ply'):
            self.report({'ERROR'}, "File at '{}' seems not to be a PLY file.".format(filepath))
            return {'CANCELLED'}
        
        points = []
        try:
            points = io.PlyPointCloudReader(filepath).points
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        if(len(points) == 0):
            self.report({'ERROR'}, "No vertices loaded from file at {}".format(filepath))
            return {'CANCELLED'}
        
        if(not set(('x', 'y', 'z')).issubset(points.dtype.names)):
            # this is very unlikely..
            self.report({'ERROR'}, "Loaded data seems to miss vertex locations.")
            return False
        normals = True
        if(not set(('nx', 'ny', 'nz')).issubset(points.dtype.names)):
            normals = False
        pcv.has_normals = normals
        if(not pcv.has_normals):
            pcv.illumination = False
        vcols = True
        if(not set(('red', 'green', 'blue')).issubset(points.dtype.names)):
            vcols = False
        pcv.has_vcols = vcols
        
        vs = np.column_stack((points['x'], points['y'], points['z'], ))
        
        if(normals):
            ns = np.column_stack((points['nx'], points['ny'], points['nz'], ))
        else:
            n = len(points)
            ns = np.column_stack((np.full(n, 0.0, dtype=np.float32, ),
                                  np.full(n, 0.0, dtype=np.float32, ),
                                  np.full(n, 1.0, dtype=np.float32, ), ))
        
        if(vcols):
            # preferences = bpy.context.preferences
            # addon_prefs = preferences.addons[__name__].preferences
            if(addon_prefs.convert_16bit_colors and points['red'].dtype == 'uint16'):
                r8 = (points['red'] / 256).astype('uint8')
                g8 = (points['green'] / 256).astype('uint8')
                b8 = (points['blue'] / 256).astype('uint8')
                if(addon_prefs.gamma_correct_16bit_colors):
                    cs = np.column_stack(((r8 / 255) ** (1 / 2.2),
                                          (g8 / 255) ** (1 / 2.2),
                                          (b8 / 255) ** (1 / 2.2),
                                          np.ones(len(points), dtype=float, ), ))
                else:
                    cs = np.column_stack((r8 / 255, g8 / 255, b8 / 255, np.ones(len(points), dtype=float, ), ))
                cs = cs.astype(np.float32)
            else:
                # 'uint8'
                cs = np.column_stack((points['red'] / 255, points['green'] / 255, points['blue'] / 255, np.ones(len(points), dtype=float, ), ))
                cs = cs.astype(np.float32)
        else:
            n = len(points)
            # preferences = bpy.context.preferences
            # addon_prefs = preferences.addons[__name__].preferences
            col = addon_prefs.default_vertex_color[:]
            col = tuple([c ** (1 / 2.2) for c in col]) + (1.0, )
            cs = np.column_stack((np.full(n, col[0], dtype=np.float32, ),
                                  np.full(n, col[1], dtype=np.float32, ),
                                  np.full(n, col[2], dtype=np.float32, ),
                                  np.ones(n, dtype=np.float32, ), ))
        
        # append
        c = PCVManager.cache[pcv.uuid]
        ovs = c['vertices']
        ons = c['normals']
        ocs = c['colors']
        vs = np.concatenate((ovs, vs, ))
        ns = np.concatenate((ons, ns, ))
        cs = np.concatenate((ocs, cs, ))
        
        # preferences = bpy.context.preferences
        # addon_prefs = preferences.addons[__name__].preferences
        if(addon_prefs.shuffle_points):
            l = len(vs)
            dt = [('x', '<f8'), ('y', '<f8'), ('z', '<f8'), ('nx', '<f8'), ('ny', '<f8'), ('nz', '<f8'), ('red', '<f8'), ('green', '<f8'), ('blue', '<f8'), ('alpha', '<f8'), ]
            a = np.empty(l, dtype=dt, )
            a['x'] = vs[:, 0]
            a['y'] = vs[:, 1]
            a['z'] = vs[:, 2]
            a['nx'] = ns[:, 0]
            a['ny'] = ns[:, 1]
            a['nz'] = ns[:, 2]
            a['red'] = cs[:, 0]
            a['green'] = cs[:, 1]
            a['blue'] = cs[:, 2]
            a['alpha'] = cs[:, 3]
            np.random.shuffle(a)
            vs = np.column_stack((a['x'], a['y'], a['z'], ))
            ns = np.column_stack((a['nx'], a['ny'], a['nz'], ))
            cs = np.column_stack((a['red'], a['green'], a['blue'], a['alpha'], ))
            vs = vs.astype(np.float32)
            ns = ns.astype(np.float32)
            cs = cs.astype(np.float32)
        
        PCVManager.update(pcv.uuid, vs, ns, cs, )
        
        return {'FINISHED'}


class PCV_OT_filter_join(Operator):
    bl_idname = "point_cloud_visualizer.filter_join"
    bl_label = "Join"
    bl_description = "Join with another PCV instance cloud"
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        
        pcv = context.object.point_cloud_visualizer
        pcv2 = None
        ok = False
        for k, v in PCVManager.cache.items():
            if(v['uuid'] == pcv.uuid):
                if(v['ready']):
                    if(v['draw']):
                        o = pcv.filter_join_object
                        if(o is not None):
                            pcv2 = o.point_cloud_visualizer
                break
        if(pcv2 is not None):
            for k, v in PCVManager.cache.items():
                if(v['uuid'] == pcv2.uuid):
                    if(v['ready']):
                        ok = True
                    break
        return ok
    
    def execute(self, context):
        pcv = context.object.point_cloud_visualizer
        pcv2 = pcv.filter_join_object.point_cloud_visualizer
        addon_prefs = preferences()
        
        c = PCVManager.cache[pcv.uuid]
        c2 = PCVManager.cache[pcv2.uuid]
        
        ovs = c['vertices']
        ons = c['normals']
        ocs = c['colors']
        nvs = c2['vertices']
        nns = c2['normals']
        ncs = c2['colors']
        
        def apply_matrix(m, vs, ns=None, ):
            vs.shape = (-1, 3)
            vs = np.c_[vs, np.ones(vs.shape[0])]
            vs = np.dot(m, vs.T)[0:3].T.reshape((-1))
            vs.shape = (-1, 3)
            if(ns is not None):
                _, rot, _ = m.decompose()
                rmat = rot.to_matrix().to_4x4()
                ns.shape = (-1, 3)
                ns = np.c_[ns, np.ones(ns.shape[0])]
                ns = np.dot(rmat, ns.T)[0:3].T.reshape((-1))
                ns.shape = (-1, 3)
            return vs, ns
        
        nvs, nns = apply_matrix(pcv.filter_join_object.matrix_world, nvs, nns, )
        nvs, nns = apply_matrix(context.object.matrix_world.inverted(), nvs, nns, )
        
        vs = np.concatenate((ovs, nvs, ))
        ns = np.concatenate((ons, nns, ))
        cs = np.concatenate((ocs, ncs, ))
        
        # preferences = bpy.context.preferences
        # addon_prefs = preferences.addons[__name__].preferences
        if(addon_prefs.shuffle_points):
            l = len(vs)
            dt = [('x', '<f8'), ('y', '<f8'), ('z', '<f8'), ('nx', '<f8'), ('ny', '<f8'), ('nz', '<f8'), ('red', '<f8'), ('green', '<f8'), ('blue', '<f8'), ('alpha', '<f8'), ]
            a = np.empty(l, dtype=dt, )
            a['x'] = vs[:, 0]
            a['y'] = vs[:, 1]
            a['z'] = vs[:, 2]
            a['nx'] = ns[:, 0]
            a['ny'] = ns[:, 1]
            a['nz'] = ns[:, 2]
            a['red'] = cs[:, 0]
            a['green'] = cs[:, 1]
            a['blue'] = cs[:, 2]
            a['alpha'] = cs[:, 3]
            np.random.shuffle(a)
            vs = np.column_stack((a['x'], a['y'], a['z'], ))
            ns = np.column_stack((a['nx'], a['ny'], a['nz'], ))
            cs = np.column_stack((a['red'], a['green'], a['blue'], a['alpha'], ))
            vs = vs.astype(np.float32)
            ns = ns.astype(np.float32)
            cs = cs.astype(np.float32)
        
        PCVManager.update(pcv.uuid, vs, ns, cs, )
        
        c2['draw'] = False
        context.area.tag_redraw()
        
        return {'FINISHED'}


class PCV_OT_filter_boolean_intersect(Operator):
    bl_idname = "point_cloud_visualizer.filter_boolean_intersect"
    bl_label = "Intersect"
    bl_description = ""
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        
        pcv = context.object.point_cloud_visualizer
        ok = False
        for k, v in PCVManager.cache.items():
            if(v['uuid'] == pcv.uuid):
                if(v['ready']):
                    if(v['draw']):
                        o = pcv.filter_boolean_object
                        if(o is not None):
                            if(o.type in ('MESH', 'CURVE', 'SURFACE', 'FONT', )):
                                ok = True
        return ok
    
    def execute(self, context):
        log("Intersect:", 0)
        _t = time.time()
        
        pcv = context.object.point_cloud_visualizer
        o = pcv.filter_boolean_object
        if(o is None):
            raise Exception()
        
        def apply_matrix(vs, ns, matrix, ):
            matrot = matrix.decompose()[1].to_matrix().to_4x4()
            dtv = vs.dtype
            dtn = ns.dtype
            rvs = np.zeros(vs.shape, dtv)
            rns = np.zeros(ns.shape, dtn)
            for i in range(len(vs)):
                co = matrix @ Vector(vs[i])
                no = matrot @ Vector(ns[i])
                rvs[i] = np.array(co.to_tuple(), dtv)
                rns[i] = np.array(no.to_tuple(), dtn)
            return rvs, rns
        
        c = PCVManager.cache[pcv.uuid]
        vs = c['vertices']
        ns = c['normals']
        cs = c['colors']
        
        # apply parent matrix to points
        m = c['object'].matrix_world.copy()
        # vs, ns = apply_matrix(vs, ns, m)
        
        sc = context.scene
        depsgraph = bpy.context.evaluated_depsgraph_get()
        # make target
        tmp_mesh = o.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph, )
        target_mesh = tmp_mesh.copy()
        target_mesh.name = 'target_mesh_{}'.format(pcv.uuid)
        target_mesh.transform(o.matrix_world.copy())
        target_mesh.transform(m.inverted())
        view_layer = context.view_layer
        collection = view_layer.active_layer_collection.collection
        target = bpy.data.objects.new('target_mesh_{}'.format(pcv.uuid), target_mesh)
        collection.objects.link(target)
        # still no idea, have to read about it..
        depsgraph.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        
        # use object bounding box for fast check if point can even be inside/outside of mesh and then use ray casting etc..
        bounds = [bv[:] for bv in target.bound_box]
        xmin = min([v[0] for v in bounds])
        xmax = max([v[0] for v in bounds])
        ymin = min([v[1] for v in bounds])
        ymax = max([v[1] for v in bounds])
        zmin = min([v[2] for v in bounds])
        zmax = max([v[2] for v in bounds])
        
        def is_in_bound_box(v):
            x = False
            if(xmin < v[0] < xmax):
                x = True
            y = False
            if(ymin < v[1] < ymax):
                y = True
            z = False
            if(zmin < v[2] < zmax):
                z = True
            if(x and y and z):
                return True
            return False
        
        # v1 raycasting in three axes and counting hits
        def is_point_inside_mesh_v1(p, o, ):
            axes = [Vector((1.0, 0.0, 0.0)), Vector((0.0, 1.0, 0.0)), Vector((0.0, 0.0, 1.0)), ]
            r = False
            for a in axes:
                v = p
                c = 0
                while True:
                    _, l, n, i = o.ray_cast(v, v + a * 10000.0)
                    if(i == -1):
                        break
                    c += 1
                    v = l + a * 0.00001
                if(c % 2 == 0):
                    r = True
                    break
            return not r
        
        # v2 raycasting and counting hits
        def is_point_inside_mesh_v2(p, o, shift=0.000001, search_distance=10000.0, ):
            p = Vector(p)
            path = []
            hits = 0
            direction = Vector((0.0, 0.0, 1.0))
            opposite_direction = direction.copy()
            opposite_direction.negate()
            path.append(p)
            loc = p
            ok = True
            
            def shift_vector(co, no, v):
                return co + (no.normalized() * v)
            
            while(ok):
                end = shift_vector(loc, direction, search_distance)
                loc = shift_vector(loc, direction, shift)
                _, loc, nor, ind = o.ray_cast(loc, end)
                if(ind != -1):
                    a = shift_vector(loc, opposite_direction, shift)
                    path.append(a)
                    hits += 1
                else:
                    ok = False
            if(hits % 2 == 1):
                return True
            return False
        
        # v3 closest point on mesh normal checking
        def is_point_inside_mesh_v3(p, o, ):
            _, loc, nor, ind = o.closest_point_on_mesh(p)
            if(ind != -1):
                v = loc - p
                d = v.dot(nor)
                if(d >= 0):
                    return True
            return False
        
        # if(debug_mode()):
        #     import cProfile
        #     import pstats
        #     import io
        #     pr = cProfile.Profile()
        #     pr.enable()
        
        indexes = []
        prgs = debug.Progress(len(vs), indent=1, prefix="> ")
        for i, v in enumerate(vs):
            prgs.step()
            vv = Vector(v)
            '''
            inside1 = is_point_inside_mesh_v1(vv, target, )
            inside2 = is_point_inside_mesh_v2(vv, target, )
            inside3 = is_point_inside_mesh_v3(vv, target, )
            # intersect
            if(not inside1 and not inside2 and not inside3):
                indexes.append(i)
            # # exclude
            # if(inside1 and inside2 and inside3):
            #     indexes.append(i)
            '''
            
            in_bb = is_in_bound_box(v)
            if(not in_bb):
                # is not in bounds i can skip completely
                indexes.append(i)
                continue
            
            inside3 = is_point_inside_mesh_v3(vv, target, )
            if(not inside3):
                indexes.append(i)
        
        # if(debug_mode()):
        #     pr.disable()
        #     s = io.StringIO()
        #     sortby = 'cumulative'
        #     ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        #     ps.print_stats()
        #     print(s.getvalue())
        
        c = PCVManager.cache[pcv.uuid]
        vs = c['vertices']
        ns = c['normals']
        cs = c['colors']
        vs = np.delete(vs, indexes, axis=0, )
        ns = np.delete(ns, indexes, axis=0, )
        cs = np.delete(cs, indexes, axis=0, )
        
        log("removed: {} points".format(len(indexes)), 1)
        
        # cleanup
        collection.objects.unlink(target)
        bpy.data.objects.remove(target)
        bpy.data.meshes.remove(target_mesh)
        o.to_mesh_clear()
        
        # put to cache..
        pcv = context.object.point_cloud_visualizer
        PCVManager.update(pcv.uuid, vs, ns, cs, )
        
        _d = datetime.timedelta(seconds=time.time() - _t)
        log("completed in {}.".format(_d), 1)
        
        return {'FINISHED'}


class PCV_OT_filter_boolean_exclude(Operator):
    bl_idname = "point_cloud_visualizer.filter_boolean_exclude"
    bl_label = "Exclude"
    bl_description = ""
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        
        pcv = context.object.point_cloud_visualizer
        ok = False
        for k, v in PCVManager.cache.items():
            if(v['uuid'] == pcv.uuid):
                if(v['ready']):
                    if(v['draw']):
                        o = pcv.filter_boolean_object
                        if(o is not None):
                            if(o.type in ('MESH', 'CURVE', 'SURFACE', 'FONT', )):
                                ok = True
        return ok
    
    def execute(self, context):
        log("Exclude:", 0)
        _t = time.time()
        
        pcv = context.object.point_cloud_visualizer
        o = pcv.filter_boolean_object
        if(o is None):
            raise Exception()
        
        def apply_matrix(vs, ns, matrix, ):
            matrot = matrix.decompose()[1].to_matrix().to_4x4()
            dtv = vs.dtype
            dtn = ns.dtype
            rvs = np.zeros(vs.shape, dtv)
            rns = np.zeros(ns.shape, dtn)
            for i in range(len(vs)):
                co = matrix @ Vector(vs[i])
                no = matrot @ Vector(ns[i])
                rvs[i] = np.array(co.to_tuple(), dtv)
                rns[i] = np.array(no.to_tuple(), dtn)
            return rvs, rns
        
        c = PCVManager.cache[pcv.uuid]
        vs = c['vertices']
        ns = c['normals']
        cs = c['colors']
        
        # apply parent matrix to points
        m = c['object'].matrix_world.copy()
        # vs, ns = apply_matrix(vs, ns, m)
        
        sc = context.scene
        depsgraph = bpy.context.evaluated_depsgraph_get()
        # make target
        tmp_mesh = o.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph, )
        target_mesh = tmp_mesh.copy()
        target_mesh.name = 'target_mesh_{}'.format(pcv.uuid)
        target_mesh.transform(o.matrix_world.copy())
        target_mesh.transform(m.inverted())
        view_layer = context.view_layer
        collection = view_layer.active_layer_collection.collection
        target = bpy.data.objects.new('target_mesh_{}'.format(pcv.uuid), target_mesh)
        collection.objects.link(target)
        # still no idea, have to read about it..
        depsgraph.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        
        # use object bounding box for fast check if point can even be inside/outside of mesh and then use ray casting etc..
        bounds = [bv[:] for bv in target.bound_box]
        xmin = min([v[0] for v in bounds])
        xmax = max([v[0] for v in bounds])
        ymin = min([v[1] for v in bounds])
        ymax = max([v[1] for v in bounds])
        zmin = min([v[2] for v in bounds])
        zmax = max([v[2] for v in bounds])
        
        def is_in_bound_box(v):
            x = False
            if(xmin < v[0] < xmax):
                x = True
            y = False
            if(ymin < v[1] < ymax):
                y = True
            z = False
            if(zmin < v[2] < zmax):
                z = True
            if(x and y and z):
                return True
            return False
        
        # v1 raycasting in three axes and counting hits
        def is_point_inside_mesh_v1(p, o, ):
            axes = [Vector((1.0, 0.0, 0.0)), Vector((0.0, 1.0, 0.0)), Vector((0.0, 0.0, 1.0)), ]
            r = False
            for a in axes:
                v = p
                c = 0
                while True:
                    _, l, n, i = o.ray_cast(v, v + a * 10000.0)
                    if(i == -1):
                        break
                    c += 1
                    v = l + a * 0.00001
                if(c % 2 == 0):
                    r = True
                    break
            return not r
        
        # v2 raycasting and counting hits
        def is_point_inside_mesh_v2(p, o, shift=0.000001, search_distance=10000.0, ):
            p = Vector(p)
            path = []
            hits = 0
            direction = Vector((0.0, 0.0, 1.0))
            opposite_direction = direction.copy()
            opposite_direction.negate()
            path.append(p)
            loc = p
            ok = True
            
            def shift_vector(co, no, v):
                return co + (no.normalized() * v)
            
            while(ok):
                end = shift_vector(loc, direction, search_distance)
                loc = shift_vector(loc, direction, shift)
                _, loc, nor, ind = o.ray_cast(loc, end)
                if(ind != -1):
                    a = shift_vector(loc, opposite_direction, shift)
                    path.append(a)
                    hits += 1
                else:
                    ok = False
            if(hits % 2 == 1):
                return True
            return False
        
        # v3 closest point on mesh normal checking
        def is_point_inside_mesh_v3(p, o, ):
            _, loc, nor, ind = o.closest_point_on_mesh(p)
            if(ind != -1):
                v = loc - p
                d = v.dot(nor)
                if(d >= 0):
                    return True
            return False
        
        indexes = []
        prgs = debug.Progress(len(vs), indent=1, prefix="> ")
        for i, v in enumerate(vs):
            prgs.step()
            vv = Vector(v)
            '''
            inside1 = is_point_inside_mesh_v1(vv, target, )
            inside2 = is_point_inside_mesh_v2(vv, target, )
            inside3 = is_point_inside_mesh_v3(vv, target, )
            # # intersect
            # if(not inside1 and not inside2 and not inside3):
            #     indexes.append(i)
            # exclude
            if(inside1 and inside2 and inside3):
                indexes.append(i)
            '''
            
            in_bb = is_in_bound_box(v)
            if(in_bb):
                # indexes.append(i)
                # continue
                
                # is in bound so i can check further
                inside3 = is_point_inside_mesh_v3(vv, target, )
                if(inside3):
                    indexes.append(i)
            
            # inside3 = is_point_inside_mesh_v3(vv, target, )
            # if(inside3):
            #     indexes.append(i)
        
        c = PCVManager.cache[pcv.uuid]
        vs = c['vertices']
        ns = c['normals']
        cs = c['colors']
        vs = np.delete(vs, indexes, axis=0, )
        ns = np.delete(ns, indexes, axis=0, )
        cs = np.delete(cs, indexes, axis=0, )
        
        log("removed: {} points".format(len(indexes)), 1)
        
        # cleanup
        collection.objects.unlink(target)
        bpy.data.objects.remove(target)
        bpy.data.meshes.remove(target_mesh)
        o.to_mesh_clear()
        
        # put to cache..
        pcv = context.object.point_cloud_visualizer
        PCVManager.update(pcv.uuid, vs, ns, cs, )
        
        _d = datetime.timedelta(seconds=time.time() - _t)
        log("completed in {}.".format(_d), 1)
        
        return {'FINISHED'}


class PCV_OT_reload(Operator):
    bl_idname = "point_cloud_visualizer.reload"
    bl_label = "Reload"
    bl_description = "Reload points from file"
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        
        pcv = context.object.point_cloud_visualizer
        # if(pcv.filepath != '' and pcv.uuid != '' and not pcv.runtime):
        if(pcv.filepath != '' and pcv.uuid != ''):
            return True
        return False
    
    def execute(self, context):
        pcv = context.object.point_cloud_visualizer
        
        draw = False
        for k, v in PCVManager.cache.items():
            if(v['uuid'] == pcv.uuid):
                if(v['ready']):
                    if(v['draw']):
                        draw = True
                        bpy.ops.point_cloud_visualizer.erase()
        
        if(pcv.runtime):
            c = PCVManager.cache[pcv.uuid]
            points = c['points']
            vs = np.column_stack((points['x'], points['y'], points['z'], ))
            ns = np.column_stack((points['nx'], points['ny'], points['nz'], ))
            cs = c['colors_original']
            PCVManager.update(pcv.uuid, vs, ns, cs, )
        else:
            bpy.ops.point_cloud_visualizer.load_ply_to_cache(filepath=pcv.filepath)
        
        if(draw):
            bpy.ops.point_cloud_visualizer.draw()
        
        return {'FINISHED'}


class PCV_OT_sequence_preload(Operator):
    bl_idname = "point_cloud_visualizer.sequence_preload"
    bl_label = "Preload Sequence"
    bl_description = "Preload sequence of PLY files. Files should be numbered starting at 1. Missing files in sequence will be skipped."
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        
        pcv = context.object.point_cloud_visualizer
        if(pcv.uuid in PCVSequence.cache.keys()):
            return False
        ok = False
        for k, v in PCVManager.cache.items():
            if(v['uuid'] == pcv.uuid):
                if(v['ready']):
                    if(v['draw']):
                        ok = True
        return ok
    
    def execute(self, context):
        log('Preload Sequence..')
        pcv = context.object.point_cloud_visualizer
        
        # pcv.sequence_enabled = True
        
        dirpath = os.path.dirname(pcv.filepath)
        files = []
        for (p, ds, fs) in os.walk(dirpath):
            files.extend(fs)
            break
        
        fs = [f for f in files if f.lower().endswith('.ply')]
        f = os.path.split(pcv.filepath)[1]
        
        pattern = re.compile(r'(\d+)(?!.*(\d+))')
        
        m = re.search(pattern, f, )
        if(m is not None):
            prefix = f[:m.start()]
            suffix = f[m.end():]
        else:
            self.report({'ERROR'}, 'Filename does not contain any sequence number')
            return {'CANCELLED'}
        
        sel = []
        
        for n in fs:
            m = re.search(pattern, n, )
            if(m is not None):
                # some numbers present, lets compare with selected file prefix/suffix
                pre = n[:m.start()]
                if(pre == prefix):
                    # prefixes match
                    suf = n[m.end():]
                    if(suf == suffix):
                        # suffixes match, extract number
                        si = n[m.start():m.end()]
                        try:
                            # try convert it to integer
                            i = int(si)
                            # and store as selected file
                            sel.append((i, n))
                        except ValueError:
                            pass
        
        # sort by sequence number
        sel.sort()
        # fabricate list with missing sequence numbers as None
        sequence = [[None] for i in range(sel[-1][0])]
        for i, n in sel:
            sequence[i - 1] = (i, n)
        for i in range(len(sequence)):
            if(sequence[i][0] is None):
                sequence[i] = [i, None]
        
        log('found files:', 1)
        for i, n in sequence:
            log('{}: {}'.format(i, n), 2)
        
        log('preloading..', 1)
        # this is our sequence with matching filenames, sorted by numbers with missing as None, now load it all..
        cache = []
        for i, n in sequence:
            if(n is not None):
                p = os.path.join(dirpath, n)
                points = []
                try:
                    points = io.PlyPointCloudReader(p).points
                except Exception as e:
                    self.report({'ERROR'}, str(e))
                if(len(points) == 0):
                    self.report({'ERROR'}, "No vertices loaded from file at {}".format(p))
                else:
                    if(not set(('x', 'y', 'z')).issubset(points.dtype.names)):
                        self.report({'ERROR'}, "Loaded data seems to miss vertex locations.")
                        return {'CANCELLED'}
                    
                    vs = np.column_stack((points['x'], points['y'], points['z'], ))
                    vs = vs.astype(np.float32)
                    
                    if(not set(('nx', 'ny', 'nz')).issubset(points.dtype.names)):
                        ns = None
                    else:
                        ns = np.column_stack((points['nx'], points['ny'], points['nz'], ))
                        ns = ns.astype(np.float32)
                    if(not set(('red', 'green', 'blue')).issubset(points.dtype.names)):
                        cs = None
                    else:
                        cs = np.column_stack((points['red'] / 255, points['green'] / 255, points['blue'] / 255, np.ones(len(points), dtype=float, ), ))
                        cs = cs.astype(np.float32)
                    
                    cache.append({'index': i,
                                  'name': n,
                                  'path': p,
                                  'vs': vs,
                                  'ns': ns,
                                  'cs': cs,
                                  'points': points, })
        
        log('...', 1)
        log('loaded {} item(s)'.format(len(cache)), 1)
        log('initializing..', 1)
        
        PCVSequence.init()
        
        ci = {'data': cache,
              'uuid': pcv.uuid,
              'pcv': pcv, }
        PCVSequence.cache[pcv.uuid] = ci
        
        log('force frame update..', 1)
        sc = bpy.context.scene
        cf = sc.frame_current
        sc.frame_current = cf
        log('done.', 1)
        
        return {'FINISHED'}


class PCV_OT_sequence_clear(Operator):
    bl_idname = "point_cloud_visualizer.sequence_clear"
    bl_label = "Clear Sequence"
    bl_description = "Clear preloaded sequence cache and reset all"
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        
        pcv = context.object.point_cloud_visualizer
        if(pcv.uuid in PCVSequence.cache.keys()):
            return True
        return False
        # ok = False
        # for k, v in PCVManager.cache.items():
        #     if(v['uuid'] == pcv.uuid):
        #         if(v['ready']):
        #             if(v['draw']):
        #                 ok = True
        # return ok
    
    def execute(self, context):
        pcv = context.object.point_cloud_visualizer
        
        del PCVSequence.cache[pcv.uuid]
        if(len(PCVSequence.cache.items()) == 0):
            PCVSequence.deinit()
        
        # c = PCVManager.cache[pcv.uuid]
        # vs = c['vertices']
        # ns = c['normals']
        # cs = c['colors']
        # PCVManager.update(pcv.uuid, vs, ns, cs, )
        
        bpy.ops.point_cloud_visualizer.reload()
        
        return {'FINISHED'}


class PCV_OT_seq_init(Operator):
    bl_idname = "point_cloud_visualizer.seq_init"
    bl_label = "seq_init"
    
    def execute(self, context):
        PCVSequence.init()
        context.area.tag_redraw()
        return {'FINISHED'}


class PCV_OT_seq_deinit(Operator):
    bl_idname = "point_cloud_visualizer.seq_deinit"
    bl_label = "seq_deinit"
    
    def execute(self, context):
        PCVSequence.deinit()
        context.area.tag_redraw()
        return {'FINISHED'}


class PCV_OT_generate_point_cloud(Operator):
    bl_idname = "point_cloud_visualizer.generate_from_mesh"
    bl_label = "Generate"
    bl_description = "Generate colored point cloud from mesh (or object convertible to mesh)"
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        
        # pcv = context.object.point_cloud_visualizer
        # if(pcv.uuid in PCVSequence.cache.keys()):
        #     return False
        # ok = False
        # for k, v in PCVManager.cache.items():
        #     if(v['uuid'] == pcv.uuid):
        #         if(v['ready']):
        #             if(v['draw']):
        #                 ok = True
        # return ok
        
        return True
    
    def execute(self, context):
        log("Generate From Mesh:", 0)
        _t = time.time()
        
        o = context.object
        
        if(o.type not in ('MESH', 'CURVE', 'SURFACE', 'FONT', )):
            self.report({'ERROR'}, "Object does not have geometry data.")
            return {'CANCELLED'}
        
        pcv = o.point_cloud_visualizer
        
        if(pcv.generate_source not in ('SURFACE', 'VERTICES', 'PARTICLES', )):
            self.report({'ERROR'}, "Source not implemented.")
            return {'CANCELLED'}
        
        n = pcv.generate_number_of_points
        r = random.Random(pcv.generate_seed)
        
        if(pcv.generate_colors in ('CONSTANT', 'UVTEX', 'VCOLS', 'GROUP_MONO', 'GROUP_COLOR', )):
            if(o.type in ('CURVE', 'SURFACE', 'FONT', ) and pcv.generate_colors != 'CONSTANT'):
                self.report({'ERROR'}, "Object type does not support UV textures, vertex colors or vertex groups.")
                return {'CANCELLED'}
        else:
            self.report({'ERROR'}, "Color generation not implemented.")
            return {'CANCELLED'}
        
        if(o.type != 'MESH'):
            vcols = None
            uvtex = None
            vgroup = None
        else:
            # all of following should return None is not available, at least for mesh object
            vcols = o.data.vertex_colors.active
            uvtex = o.data.uv_layers.active
            vgroup = o.vertex_groups.active
        
        generate_constant_color = tuple([c ** (1 / 2.2) for c in pcv.generate_constant_color]) + (1.0, )
        
        if(pcv.generate_source == 'VERTICES'):
            try:
                sampler = sample.PCVVertexSampler(context, o,
                                                  colorize=pcv.generate_colors,
                                                  constant_color=generate_constant_color,
                                                  vcols=vcols, uvtex=uvtex, vgroup=vgroup, )
            except Exception as e:
                self.report({'ERROR'}, str(e), )
                return {'CANCELLED'}
        elif(pcv.generate_source == 'SURFACE'):
            if(pcv.generate_algorithm == 'WEIGHTED_RANDOM_IN_TRIANGLE'):
                try:
                    sampler = sample.PCVTriangleSurfaceSampler(context, o, n, r,
                                                               colorize=pcv.generate_colors,
                                                               constant_color=generate_constant_color,
                                                               vcols=vcols, uvtex=uvtex, vgroup=vgroup,
                                                               exact_number_of_points=pcv.generate_exact_number_of_points, )
                except Exception as e:
                    self.report({'ERROR'}, str(e), )
                    return {'CANCELLED'}
            elif(pcv.generate_algorithm == 'POISSON_DISK_SAMPLING'):
                try:
                    sampler = sample.PCVPoissonDiskSurfaceSampler(context, o, r, minimal_distance=pcv.generate_minimal_distance,
                                                                  sampling_exponent=pcv.generate_sampling_exponent,
                                                                  colorize=pcv.generate_colors,
                                                                  constant_color=generate_constant_color,
                                                                  vcols=vcols, uvtex=uvtex, vgroup=vgroup, )
                except Exception as e:
                    self.report({'ERROR'}, str(e), )
                    return {'CANCELLED'}
            else:
                self.report({'ERROR'}, "Algorithm not implemented.")
                return {'CANCELLED'}
            
        elif(pcv.generate_source == 'PARTICLES'):
            try:
                alive_only = True
                if(pcv.generate_source_psys == 'ALL'):
                    alive_only = False
                sampler = sample.PCVParticleSystemSampler(context, o, alive_only=alive_only,
                                                          colorize=pcv.generate_colors,
                                                          constant_color=generate_constant_color,
                                                          vcols=vcols, uvtex=uvtex, vgroup=vgroup, )
            except Exception as e:
                self.report({'ERROR'}, str(e), )
                return {'CANCELLED'}
        else:
            self.report({'ERROR'}, "Source type not implemented.")
            return {'CANCELLED'}
        
        vs = sampler.vs
        ns = sampler.ns
        cs = sampler.cs
        
        log("generated {} points.".format(len(vs)), 1)
        
        ok = False
        for k, v in PCVManager.cache.items():
            if(v['uuid'] == pcv.uuid):
                if(v['ready']):
                    if(v['draw']):
                        ok = True
        if(ok):
            bpy.ops.point_cloud_visualizer.erase()
        
        c = PCVControl(o)
        c.draw(vs, ns, cs)
        
        if(debug_mode()):
            o.display_type = 'BOUNDS'
        
        _d = datetime.timedelta(seconds=time.time() - _t)
        log("completed in {}.".format(_d), 1)
        
        return {'FINISHED'}


class PCV_OT_reset_runtime(Operator):
    bl_idname = "point_cloud_visualizer.reset_runtime"
    bl_label = "Reset Runtime"
    bl_description = "Reset PCV to its default state if in runtime mode (displayed data is set with python and not with ui)"
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        pcv = context.object.point_cloud_visualizer
        if(pcv.runtime):
            return True
        return False
    
    def execute(self, context):
        o = context.object
        c = PCVControl(o)
        c.erase()
        c.reset()
        return {'FINISHED'}


class PCV_OT_generate_volume_point_cloud(Operator):
    bl_idname = "point_cloud_visualizer.generate_volume_from_mesh"
    bl_label = "Generate Volume"
    bl_description = "Generate colored point cloud in mesh (or object convertible to mesh) volume"
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        
        # pcv = context.object.point_cloud_visualizer
        # if(pcv.uuid in PCVSequence.cache.keys()):
        #     return False
        # ok = False
        # for k, v in PCVManager.cache.items():
        #     if(v['uuid'] == pcv.uuid):
        #         if(v['ready']):
        #             if(v['draw']):
        #                 ok = True
        # return ok
        
        return True
    
    def execute(self, context):
        log("Generate From Mesh:", 0)
        _t = time.time()
        
        o = context.object
        pcv = o.point_cloud_visualizer
        n = pcv.generate_number_of_points
        r = random.Random(pcv.generate_seed)
        g = sample.PCVRandomVolumeSampler(o, n, r, )
        vs = g.vs
        ns = g.ns
        cs = g.cs
        
        log("generated {} points.".format(len(vs)), 1)
        
        ok = False
        for k, v in PCVManager.cache.items():
            if(v['uuid'] == pcv.uuid):
                if(v['ready']):
                    if(v['draw']):
                        ok = True
        if(ok):
            bpy.ops.point_cloud_visualizer.erase()
        
        c = PCVControl(o)
        c.draw(vs, ns, cs)
        
        if(debug_mode()):
            o.display_type = 'BOUNDS'
        
        _d = datetime.timedelta(seconds=time.time() - _t)
        log("completed in {}.".format(_d), 1)
        
        return {'FINISHED'}


class PCV_OT_color_adjustment_shader_reset(Operator):
    bl_idname = "point_cloud_visualizer.color_adjustment_shader_reset"
    bl_label = "Reset"
    bl_description = "Reset color adjustment values"
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        
        pcv = context.object.point_cloud_visualizer
        ok = False
        for k, v in PCVManager.cache.items():
            if(v['uuid'] == pcv.uuid):
                if(v['ready']):
                    if(v['draw']):
                        if(pcv.color_adjustment_shader_enabled):
                            ok = True
        return ok
    
    def execute(self, context):
        pcv = context.object.point_cloud_visualizer
        
        pcv.color_adjustment_shader_exposure = 0.0
        pcv.color_adjustment_shader_gamma = 1.0
        pcv.color_adjustment_shader_brightness = 0.0
        pcv.color_adjustment_shader_contrast = 1.0
        pcv.color_adjustment_shader_hue = 0.0
        pcv.color_adjustment_shader_saturation = 0.0
        pcv.color_adjustment_shader_value = 0.0
        pcv.color_adjustment_shader_invert = False
        
        for area in bpy.context.screen.areas:
            if(area.type == 'VIEW_3D'):
                area.tag_redraw()
        
        return {'FINISHED'}


class PCV_OT_color_adjustment_shader_apply(Operator):
    bl_idname = "point_cloud_visualizer.color_adjustment_shader_apply"
    bl_label = "Apply"
    bl_description = "Apply color adjustments to points, reset and exit"
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        
        pcv = context.object.point_cloud_visualizer
        ok = False
        for k, v in PCVManager.cache.items():
            if(v['uuid'] == pcv.uuid):
                if(v['ready']):
                    if(v['draw']):
                        if(pcv.color_adjustment_shader_enabled):
                            ok = True
        return ok
    
    def execute(self, context):
        pcv = context.object.point_cloud_visualizer
        
        c = PCVManager.cache[pcv.uuid]
        vs = c['vertices']
        ns = c['normals']
        cs = c['colors']
        
        cs = cs * (2 ** pcv.color_adjustment_shader_exposure)
        cs = np.clip(cs, 0.0, 1.0, )
        cs = cs ** (1 / pcv.color_adjustment_shader_gamma)
        cs = np.clip(cs, 0.0, 1.0, )
        cs = (cs - 0.5) * pcv.color_adjustment_shader_contrast + 0.5 + pcv.color_adjustment_shader_brightness
        cs = np.clip(cs, 0.0, 1.0, )
        
        h = pcv.color_adjustment_shader_hue
        s = pcv.color_adjustment_shader_saturation
        v = pcv.color_adjustment_shader_value
        if(h > 1.0):
            h = h % 1.0
        for _i, ca in enumerate(cs):
            col = Color(ca[:3])
            _h, _s, _v = col.hsv
            _h = (_h + h) % 1.0
            _s += s
            _v += v
            col.hsv = (_h, _s, _v)
            cs[_i][0] = col.r
            cs[_i][1] = col.g
            cs[_i][2] = col.b
        cs = np.clip(cs, 0.0, 1.0, )
        
        if(pcv.color_adjustment_shader_invert):
            cs = 1.0 - cs
        cs = np.clip(cs, 0.0, 1.0, )
        
        bpy.ops.point_cloud_visualizer.color_adjustment_shader_reset()
        pcv.color_adjustment_shader_enabled = False
        
        if('extra' in c.keys()):
            del c['extra']
        
        PCVManager.update(pcv.uuid, vs, ns, cs, )
        
        return {'FINISHED'}


class PCV_OT_clip_planes_from_bbox(Operator):
    bl_idname = "point_cloud_visualizer.clip_planes_from_bbox"
    bl_label = "Set Clip Planes From Object Bounding Box"
    bl_description = "Set clip planes from object bounding box"
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        
        pcv = context.object.point_cloud_visualizer
        ok = False
        
        bbo = pcv.clip_planes_from_bbox_object
        if(bbo is not None):
            ok = True
        
        return ok
    
    def execute(self, context):
        pcv = context.object.point_cloud_visualizer
        bbo = pcv.clip_planes_from_bbox_object
        
        mw = bbo.matrix_world
        vs = [mw @ Vector(v) for v in bbo.bound_box]
        
        # 0 front left down
        # 1 front left up
        # 2 back left up
        # 3 back left down
        # 4 front right down
        # 5 front right up
        # 6 back right up
        # 7 back right down
        #          2-----------6
        #         /           /|
        #       1-----------5  |
        #       |           |  |
        #       |           |  |
        #       |  3        |  7
        #       |           | /
        #       0-----------4
        
        fs = (
            (0, 4, 5, 1),  # front
            (3, 2, 6, 7),  # back
            (1, 5, 6, 2),  # top
            (0, 3, 7, 4),  # bottom
            (0, 1, 2, 3),  # left
            (4, 7, 6, 5),  # right
        )
        
        quads = [[vs[fs[i][j]] for j in range(4)] for i in range(6)]
        ns = [mathutils.geometry.normal(quads[i]) for i in range(6)]
        for i in range(6):
            # FIXME: if i need to do this, it is highly probable i have something wrong.. somewhere..
            ns[i].negate()
        
        ds = []
        for i in range(6):
            v = quads[i][0]
            n = ns[i]
            d = mathutils.geometry.distance_point_to_plane(Vector(), v, n)
            ds.append(d)
        
        a = [ns[i].to_tuple() + (ds[i], ) for i in range(6)]
        pcv.clip_plane0 = a[0]
        pcv.clip_plane1 = a[1]
        pcv.clip_plane2 = a[2]
        pcv.clip_plane3 = a[3]
        pcv.clip_plane4 = a[4]
        pcv.clip_plane5 = a[5]
        
        pcv.clip_shader_enabled = True
        pcv.clip_plane0_enabled = True
        pcv.clip_plane1_enabled = True
        pcv.clip_plane2_enabled = True
        pcv.clip_plane3_enabled = True
        pcv.clip_plane4_enabled = True
        pcv.clip_plane5_enabled = True
        
        return {'FINISHED'}


class PCV_OT_clip_planes_reset(Operator):
    bl_idname = "point_cloud_visualizer.clip_planes_reset"
    bl_label = "Reset Clip Planes"
    bl_description = "Reset all clip planes"
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        
        pcv = context.object.point_cloud_visualizer
        ok = True
        
        return ok
    
    def execute(self, context):
        pcv = context.object.point_cloud_visualizer
        
        pcv.clip_planes_from_bbox_object = None
        
        z = (0.0, 0.0, 0.0, 0.0, )
        pcv.clip_plane0 = z
        pcv.clip_plane1 = z
        pcv.clip_plane2 = z
        pcv.clip_plane3 = z
        pcv.clip_plane4 = z
        pcv.clip_plane5 = z
        
        pcv.clip_shader_enabled = False
        pcv.clip_plane0_enabled = False
        pcv.clip_plane1_enabled = False
        pcv.clip_plane2_enabled = False
        pcv.clip_plane3_enabled = False
        pcv.clip_plane4_enabled = False
        pcv.clip_plane5_enabled = False
        
        return {'FINISHED'}


class PCV_OT_clip_planes_from_camera_view(Operator):
    bl_idname = "point_cloud_visualizer.clip_planes_from_camera_view"
    bl_label = "Set Clip Planes From Camera View"
    bl_description = "Set clip planes from active camera view"
    
    @classmethod
    def poll(cls, context):
        if(context.object is None):
            return False
        
        pcv = context.object.point_cloud_visualizer
        ok = False
        
        s = bpy.context.scene
        o = s.camera
        if(o):
            ok = True
        
        return ok
    
    def execute(self, context):
        pcv = context.object.point_cloud_visualizer
        
        o = bpy.context.object
        
        s = bpy.context.scene
        c = s.camera
        cd = c.data
        rx = s.render.resolution_x
        ry = s.render.resolution_y
        w = 0.5 * cd.sensor_width / cd.lens
        if(rx > ry):
            x = w
            y = w * ry / rx
        else:
            x = w * rx / ry
            y = w
        
        lr = Vector((x, -y, -1.0, ))
        ur = Vector((x, y, -1.0, ))
        ll = Vector((-x, -y, -1.0, ))
        ul = Vector((-x, y, -1.0, ))
        
        z = Vector()
        n0 = mathutils.geometry.normal((z, lr, ur))
        n1 = mathutils.geometry.normal((z, ur, ul))
        n2 = mathutils.geometry.normal((z, ul, ll))
        n3 = mathutils.geometry.normal((z, ll, lr))
        
        n0.negate()
        n1.negate()
        n2.negate()
        n3.negate()
        
        m = c.matrix_world
        l, r, _ = m.decompose()
        rm = r.to_matrix().to_4x4()
        lrm = Matrix.Translation(l).to_4x4() @ rm
        
        omi = o.matrix_world.inverted()
        _, r, _ = omi.decompose()
        orm = r.to_matrix().to_4x4()
        
        n0 = orm @ rm @ n0
        n1 = orm @ rm @ n1
        n2 = orm @ rm @ n2
        n3 = orm @ rm @ n3
        
        v0 = omi @ lrm @ lr
        v1 = omi @ lrm @ ur
        v2 = omi @ lrm @ ul
        v3 = omi @ lrm @ ll
        
        d0 = mathutils.geometry.distance_point_to_plane(Vector(), v0, n0)
        d1 = mathutils.geometry.distance_point_to_plane(Vector(), v1, n1)
        d2 = mathutils.geometry.distance_point_to_plane(Vector(), v2, n2)
        d3 = mathutils.geometry.distance_point_to_plane(Vector(), v3, n3)
        
        # TODO: add plane behind camera (not much needed anyway, but for consistency), but more important, add plane in clipping distance set on camera
        # TODO: ORTHO camera does not work
        
        pcv.clip_plane0 = n0.to_tuple() + (d0, )
        pcv.clip_plane1 = n1.to_tuple() + (d1, )
        pcv.clip_plane2 = n2.to_tuple() + (d2, )
        pcv.clip_plane3 = n3.to_tuple() + (d3, )
        pcv.clip_plane4 = (0.0, 0.0, 0.0, 0.0, )
        pcv.clip_plane5 = (0.0, 0.0, 0.0, 0.0, )
        
        pcv.clip_shader_enabled = True
        pcv.clip_plane0_enabled = True
        pcv.clip_plane1_enabled = True
        pcv.clip_plane2_enabled = True
        pcv.clip_plane3_enabled = True
        pcv.clip_plane4_enabled = False
        pcv.clip_plane5_enabled = False
        
        return {'FINISHED'}
