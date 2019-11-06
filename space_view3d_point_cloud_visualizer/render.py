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

import bpy
from bpy.types import Operator
from mathutils import Matrix, Vector, Quaternion, Color
from bpy_extras.object_utils import world_to_camera_view
import gpu
from gpu.types import GPUOffScreen, GPUShader, GPUBatch, GPUVertBuf, GPUVertFormat
from gpu_extras.batch import batch_for_shader
import bgl

from .debug import log, debug_mode
from .machine import PCVManager, load_shader_code


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
                    shader_data_vert, shader_data_frag, shader_data_geom = load_shader_code('depth_illumination')
                    shader = GPUShader(shader_data_vert, shader_data_frag)
                    batch = batch_for_shader(shader, 'POINTS', {"position": vs, "normal": ns, })
                elif(pcv.dev_depth_false_colors):
                    shader_data_vert, shader_data_frag, shader_data_geom = load_shader_code('depth_false_colors')
                    shader = GPUShader(shader_data_vert, shader_data_frag)
                    batch = batch_for_shader(shader, 'POINTS', {"position": vs, })
                else:
                    shader_data_vert, shader_data_frag, shader_data_geom = load_shader_code('depth_simple')
                    shader = GPUShader(shader_data_vert, shader_data_frag)
                    batch = batch_for_shader(shader, 'POINTS', {"position": vs, })
            elif(pcv.dev_normal_colors_enabled):
                shader_data_vert, shader_data_frag, shader_data_geom = load_shader_code('normal_colors')
                shader = GPUShader(shader_data_vert, shader_data_frag)
                batch = batch_for_shader(shader, 'POINTS', {"position": vs, "normal": ns, })
            elif(pcv.dev_position_colors_enabled):
                shader_data_vert, shader_data_frag, shader_data_geom = load_shader_code('position_colors')
                shader = GPUShader(shader_data_vert, shader_data_frag)
                batch = batch_for_shader(shader, 'POINTS', {"position": vs, })
            elif(pcv.illumination):
                if(use_smoothstep):
                    shader_data_vert, shader_data_frag, shader_data_geom = load_shader_code('render_illumination_smooth')
                    shader = GPUShader(shader_data_vert, shader_data_frag)
                    batch = batch_for_shader(shader, 'POINTS', {"position": vs, "color": cs, "normal": ns, })
                else:
                    shader_data_vert, shader_data_frag, shader_data_geom = load_shader_code('illumination')
                    shader = GPUShader(shader_data_vert, shader_data_frag)
                    batch = batch_for_shader(shader, 'POINTS', {"position": vs, "color": cs, "normal": ns, })
            else:
                if(use_smoothstep):
                    shader_data_vert, shader_data_frag, shader_data_geom = load_shader_code('render_simple_smooth')
                    shader = GPUShader(shader_data_vert, shader_data_frag)
                    batch = batch_for_shader(shader, 'POINTS', {"position": vs, "color": cs, })
                else:
                    shader_data_vert, shader_data_frag, shader_data_geom = load_shader_code('simple')
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
