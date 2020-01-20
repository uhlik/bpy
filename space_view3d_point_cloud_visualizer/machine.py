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
import uuid
import time
import datetime
import numpy as np

import bpy
from gpu.types import GPUShader, GPUBatch, GPUVertBuf, GPUVertFormat
from gpu_extras.batch import batch_for_shader
import bgl
from mathutils import Matrix, Vector, Quaternion, Color
from bpy.app.handlers import persistent

from .debug import log, debug_mode
from . import io_ply


def preferences():
    a = os.path.split(os.path.split(os.path.realpath(__file__))[0])[1]
    p = bpy.context.preferences.addons[a].preferences
    return p


class PCVManager():
    cache = {}
    handle = None
    initialized = False
    
    '''
    @classmethod
    def points_batch_for_shader(cls, shader, content, ):
        for k, v in content.items():
            vbo_len = len(v)
            break
        vbo_format = shader.format_calc()
        vbo = GPUVertBuf(vbo_format, vbo_len, )
        for k, v in content.items():
            vbo.attr_fill(k, v, )
        batch = GPUBatch(type='POINTS', buf=vbo, )
        return vbo, batch
    '''
    
    @classmethod
    def load_ply_to_cache(cls, operator, context, ):
        # TODO: split this, it is used on a few places almost in identical form
        
        pcv = context.object.point_cloud_visualizer
        filepath = pcv.filepath
        
        __t = time.time()
        
        log('load data..')
        _t = time.time()
        
        # FIXME ply loading might not work with all ply files, for example, file spec seems does not forbid having two or more blocks of vertices with different props, currently i load only first block of vertices. maybe construct some messed up ply and test how for example meshlab behaves
        points = []
        try:
            # points = io_ply.BinPlyPointCloudReader(filepath).points
            points = io_ply.PlyPointCloudReader(filepath).points
        except Exception as e:
            if(operator is not None):
                operator.report({'ERROR'}, str(e))
            else:
                raise e
        if(len(points) == 0):
            operator.report({'ERROR'}, "No vertices loaded from file at {}".format(filepath))
            return False
        
        _d = datetime.timedelta(seconds=time.time() - _t)
        log("completed in {}.".format(_d))
        
        log('shuffle data..')
        _t = time.time()
        
        # preferences = bpy.context.preferences
        # addon_prefs = preferences.addons[__name__].preferences
        addon_prefs = preferences()
        if(addon_prefs.shuffle_points):
            np.random.shuffle(points)
        
        _d = datetime.timedelta(seconds=time.time() - _t)
        log("completed in {}.".format(_d))
        
        log('process data..')
        _t = time.time()
        
        if(not set(('x', 'y', 'z')).issubset(points.dtype.names)):
            # this is very unlikely..
            operator.report({'ERROR'}, "Loaded data seems to miss vertex locations.")
            return False
        
        # FIXME checking for normals/colors in points is kinda scattered all over.. chceck should be upon loading / setting from external script
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
        if(vs.dtype != np.float32):
            # convert to float32 for display if needed..
            vs = vs.astype(np.float32)
        
        if(normals):
            ns = np.column_stack((points['nx'], points['ny'], points['nz'], ))
            if(ns.dtype != np.float32):
                # convert to float32 for display if needed..
                ns = ns.astype(np.float32)
            
        else:
            n = len(points)
            ns = np.column_stack((np.full(n, 0.0, dtype=np.float32, ),
                                  np.full(n, 0.0, dtype=np.float32, ),
                                  np.full(n, 1.0, dtype=np.float32, ), ))
        
        if(vcols):
            # convert to float32 for display is not needed because colors are always processed to float32 array..
            
            # preferences = bpy.context.preferences
            # addon_prefs = preferences.addons[__name__].preferences
            if(addon_prefs.convert_16bit_colors and points['red'].dtype == 'uint16'):
                # r8 = (points['red'] / 256).astype('uint8')
                # g8 = (points['green'] / 256).astype('uint8')
                # b8 = (points['blue'] / 256).astype('uint8')
                # if(addon_prefs.gamma_correct_16bit_colors):
                #     cs = np.column_stack(((r8 / 255) ** (1 / 2.2),
                #                           (g8 / 255) ** (1 / 2.2),
                #                           (b8 / 255) ** (1 / 2.2),
                #                           np.ones(len(points), dtype=float, ), ))
                # else:
                #     cs = np.column_stack((r8 / 255, g8 / 255, b8 / 255, np.ones(len(points), dtype=float, ), ))
                # cs = cs.astype(np.float32)
                
                r_f32 = points['red'].astype(np.float32)
                g_f32 = points['green'].astype(np.float32)
                b_f32 = points['blue'].astype(np.float32)
                r_f32 /= 256.0
                g_f32 /= 256.0
                b_f32 /= 256.0
                if(addon_prefs.gamma_correct_16bit_colors):
                    r_f32 /= 255.0
                    g_f32 /= 255.0
                    b_f32 /= 255.0
                    r_f32 = r_f32 ** (1 / 2.2)
                    g_f32 = g_f32 ** (1 / 2.2)
                    b_f32 = b_f32 ** (1 / 2.2)
                    cs = np.column_stack((r_f32, g_f32, b_f32, np.ones(len(points), dtype=np.float32, ), ))
                else:
                    r_f32 /= 255.0
                    g_f32 /= 255.0
                    b_f32 /= 255.0
                    cs = np.column_stack((r_f32, g_f32, b_f32, np.ones(len(points), dtype=np.float32, ), ))
                
            else:
                # # 'uint8'
                # cs = np.column_stack((points['red'] / 255, points['green'] / 255, points['blue'] / 255, np.ones(len(points), dtype=float, ), ))
                # cs = cs.astype(np.float32)
                
                r_f32 = points['red'].astype(np.float32)
                g_f32 = points['green'].astype(np.float32)
                b_f32 = points['blue'].astype(np.float32)
                r_f32 /= 255.0
                g_f32 /= 255.0
                b_f32 /= 255.0
                cs = np.column_stack((r_f32, g_f32, b_f32, np.ones(len(points), dtype=np.float32, ), ))
                
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
        
        u = str(uuid.uuid1())
        o = context.object
        
        pcv.uuid = u
        
        d = PCVManager.new()
        d['filepath'] = filepath
        
        d['points'] = points
        
        d['uuid'] = u
        d['stats'] = len(vs)
        d['vertices'] = vs
        d['colors'] = cs
        d['normals'] = ns
        
        d['length'] = len(vs)
        dp = pcv.display_percent
        l = int((len(vs) / 100) * dp)
        if(dp >= 99):
            l = len(vs)
        d['display_length'] = l
        d['current_display_length'] = l
        
        # FIXME: put this to draw button, now user can't ser display percentage and with very large clouds may run out of memory
        ienabled = pcv.illumination
        d['illumination'] = ienabled
        if(ienabled):
            shader_data_vert, shader_data_frag, shader_data_geom = load_shader_code('illumination')
            shader = GPUShader(shader_data_vert, shader_data_frag)
            batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "color": cs[:l], "normal": ns[:l], })
        else:
            shader_data_vert, shader_data_frag, shader_data_geom = load_shader_code('simple')
            shader = GPUShader(shader_data_vert, shader_data_frag)
            batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "color": cs[:l], })
        
        d['shader'] = shader
        d['batch'] = batch
        d['ready'] = True
        d['object'] = o
        d['name'] = o.name
        
        PCVManager.add(d)
        
        _d = datetime.timedelta(seconds=time.time() - _t)
        log("completed in {}.".format(_d))
        
        log("-" * 50)
        __d = datetime.timedelta(seconds=time.time() - __t)
        log("load and process completed in {}.".format(__d))
        log("-" * 50)
        
        # with new file browser in 2.81, screen is not redrawn, so i have to do it manually..
        cls._redraw()
        
        return True
    
    @classmethod
    def render(cls, uuid, ):
        # TODO: split this, to bloated and messy, and also shader choosing logic is bad
        
        bgl.glEnable(bgl.GL_PROGRAM_POINT_SIZE)
        bgl.glEnable(bgl.GL_DEPTH_TEST)
        bgl.glEnable(bgl.GL_BLEND)
        
        # TODO: replace all 'batch_for_shader' (2.80/scripts/modules/gpu_extras/batch.py) calls with something custom made and keep buffer cached. faster shader switching, less memory used, etc..
        
        ci = PCVManager.cache[uuid]
        
        shader = ci['shader']
        batch = ci['batch']
        
        if(ci['current_display_length'] != ci['display_length']):
            l = ci['display_length']
            ci['current_display_length'] = l
            vs = ci['vertices']
            cs = ci['colors']
            ns = ci['normals']
            if(ci['illumination']):
                batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "color": cs[:l], "normal": ns[:l], })
            else:
                batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "color": cs[:l], })
            ci['batch'] = batch
        
        o = ci['object']
        try:
            pcv = o.point_cloud_visualizer
        except ReferenceError:
            # FIXME undo still doesn't work in some cases, from what i've seen, only when i am undoing operations on parent object, especially when you undo/redo e.g. transforms around load/draw operators, filepath property gets reset and the whole thing is drawn, but ui looks like loding never happened, i've added a quick fix storing path in cache, but it all depends on object name and this is bad.
            # NOTE parent object reference check should be before drawing, not in the middle, it's not that bad, it's pretty early, but it's still messy, this will require rewrite of handler and render functions in manager.. so don't touch until broken
            log("PCVManager.render: ReferenceError (possibly after undo/redo?)")
            # blender on undo/redo swaps whole scene to different one stored in memory and therefore stored object references are no longer valid
            # so find object with the same name, not the best solution, but lets see how it goes..
            o = bpy.data.objects[ci['name']]
            # update stored reference
            ci['object'] = o
            pcv = o.point_cloud_visualizer
            # push back correct uuid, since undo changed it, why? WHY? why do i even bother?
            pcv.uuid = uuid
            # push back filepath, it might get lost during undo/redo
            pcv.filepath = ci['filepath']
        
        if(not o.visible_get()):
            # if parent object is not visible, skip drawing
            # this should checked earlier, but until now i can't be sure i have correct object reference
            
            # NOTE: use bpy.context.view_layer.objects.active instead of context.active_object and add option to not hide cloud when parent object is hidden? seems like this is set when object is clicked in outliner even when hidden, at least properties buttons are changed.. if i unhide and delete the object, props buttons are not drawn, if i click on another already hidden object, correct buttons are back, so i need to check if there is something active.. also this would require rewriting all panels polls, now they check for context.active_object and if None, which is when object is hidden, panel is not drawn..
            
            bgl.glDisable(bgl.GL_PROGRAM_POINT_SIZE)
            bgl.glDisable(bgl.GL_DEPTH_TEST)
            bgl.glDisable(bgl.GL_BLEND)
            return
        
        if(ci['illumination'] != pcv.illumination):
            vs = ci['vertices']
            cs = ci['colors']
            ns = ci['normals']
            l = ci['current_display_length']
            if(pcv.illumination):
                shader_data_vert, shader_data_frag, shader_data_geom = load_shader_code('illumination')
                shader = GPUShader(shader_data_vert, shader_data_frag)
                batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "color": cs[:l], "normal": ns[:l], })
                ci['illumination'] = True
            else:
                shader_data_vert, shader_data_frag, shader_data_geom = load_shader_code('simple')
                shader = GPUShader(shader_data_vert, shader_data_frag)
                batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "color": cs[:l], })
                ci['illumination'] = False
            ci['shader'] = shader
            ci['batch'] = batch
        
        shader.bind()
        pm = bpy.context.region_data.perspective_matrix
        shader.uniform_float("perspective_matrix", pm)
        shader.uniform_float("object_matrix", o.matrix_world)
        shader.uniform_float("point_size", pcv.point_size)
        shader.uniform_float("alpha_radius", pcv.alpha_radius)
        shader.uniform_float("global_alpha", pcv.global_alpha)
        
        if(pcv.illumination and pcv.has_normals and ci['illumination']):
            cm = Matrix(((-1.0, 0.0, 0.0, 0.0, ), (0.0, -0.0, 1.0, 0.0, ), (0.0, -1.0, -0.0, 0.0, ), (0.0, 0.0, 0.0, 1.0, ), ))
            _, obrot, _ = o.matrix_world.decompose()
            mr = obrot.to_matrix().to_4x4()
            mr.invert()
            direction = cm @ pcv.light_direction
            direction = mr @ direction
            shader.uniform_float("light_direction", direction)
            
            # def get_space3dview():
            #     for a in bpy.context.screen.areas:
            #         if(a.type == "VIEW_3D"):
            #             return a.spaces[0]
            #     return None
            #
            # s3dv = get_space3dview()
            # region3d = s3dv.region_3d
            # eye = region3d.view_matrix[2][:3]
            #
            # # shader.uniform_float("light_direction", Vector(eye) * -1)
            # shader.uniform_float("light_direction", Vector(eye))
            
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
        
        if(not pcv.override_default_shader):
            # NOTE: just don't draw default shader, quick and easy solution, other shader will be drawn instead, would better to not create it..
            batch.draw(shader)
            
            # # remove extra if present, will be recreated if needed and if left stored it might cause problems
            # if('extra' in ci.keys()):
            #     del ci['extra']
        
        if(pcv.vertex_normals and pcv.has_normals):
            def make(ci):
                l = ci['current_display_length']
                vs = ci['vertices'][:l]
                ns = ci['normals'][:l]
                
                shader_data_vert, shader_data_frag, shader_data_geom = load_shader_code('normals')
                shader = GPUShader(shader_data_vert, shader_data_frag, geocode=shader_data_geom, )
                batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "normal": ns[:l], }, )
                
                d = {'shader': shader,
                     'batch': batch,
                     'position': vs,
                     'normal': ns,
                     'current_display_length': l, }
                ci['vertex_normals'] = d
                
                return shader, batch
            
            if("vertex_normals" not in ci.keys()):
                shader, batch = make(ci)
            else:
                d = ci['vertex_normals']
                shader = d['shader']
                batch = d['batch']
                ok = True
                if(ci['current_display_length'] != d['current_display_length']):
                    ok = False
                if(not ok):
                    shader, batch = make(ci)
            
            shader.bind()
            pm = bpy.context.region_data.perspective_matrix
            shader.uniform_float("perspective_matrix", pm)
            shader.uniform_float("object_matrix", o.matrix_world)
            # col = bpy.context.preferences.addons[__name__].preferences.normal_color[:]
            col = preferences().normal_color[:]
            col = tuple([c ** (1 / 2.2) for c in col]) + (pcv.vertex_normals_alpha, )
            shader.uniform_float("color", col, )
            shader.uniform_float("length", pcv.vertex_normals_size, )
            shader.uniform_float("global_alpha", pcv.global_alpha)
            batch.draw(shader)
        
        if(pcv.dev_depth_enabled):
            
            # if(debug_mode()):
            #     import cProfile
            #     import pstats
            #     import io
            #     pr = cProfile.Profile()
            #     pr.enable()
            
            vs = ci['vertices']
            ns = ci['normals']
            l = ci['current_display_length']
            
            use_stored = False
            if('extra' in ci.keys()):
                t = 'DEPTH'
                for k, v in ci['extra'].items():
                    if(k == t):
                        if(v['length'] == l):
                            if(v['illumination'] == pcv.illumination and v['false_colors'] == pcv.dev_depth_false_colors):
                                use_stored = True
                                batch = v['batch']
                                shader = v['shader']
                                break
            
            if(not use_stored):
                if(pcv.illumination):
                    shader_data_vert, shader_data_frag, shader_data_geom = load_shader_code('depth_illumination')
                    shader = GPUShader(shader_data_vert, shader_data_frag)
                    batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "normal": ns[:l], })
                elif(pcv.dev_depth_false_colors):
                    shader_data_vert, shader_data_frag, shader_data_geom = load_shader_code('depth_false_colors')
                    shader = GPUShader(shader_data_vert, shader_data_frag)
                    batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], })
                else:
                    shader_data_vert, shader_data_frag, shader_data_geom = load_shader_code('depth_simple')
                    shader = GPUShader(shader_data_vert, shader_data_frag)
                    batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], })
                
                if('extra' not in ci.keys()):
                    ci['extra'] = {}
                d = {'shader': shader,
                     'batch': batch,
                     'illumination': pcv.illumination,
                     'false_colors': pcv.dev_depth_false_colors,
                     'length': l, }
                ci['extra']['DEPTH'] = d
            
            shader.bind()
            pm = bpy.context.region_data.perspective_matrix
            shader.uniform_float("perspective_matrix", pm)
            shader.uniform_float("object_matrix", o.matrix_world)
            
            # NOTE: precalculating and storing following should speed up things a bit, but then it won't reflect edits..
            # TODO: calculating center of all points is not quite correct, visually it works, but (as i've seen in bounding box shader) it's not working when distribution of points is uneven, so have a check if it might be a bit better..
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
            
            shader.uniform_float("point_size", pcv.point_size)
            shader.uniform_float("alpha_radius", pcv.alpha_radius)
            shader.uniform_float("global_alpha", pcv.global_alpha)
            
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
            
            batch.draw(shader)
            
            # if(debug_mode()):
            #     pr.disable()
            #     s = io.StringIO()
            #     sortby = 'cumulative'
            #     ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            #     ps.print_stats()
            #     print(s.getvalue())
        
        if(pcv.dev_normal_colors_enabled):
            
            vs = ci['vertices']
            ns = ci['normals']
            l = ci['current_display_length']
            
            use_stored = False
            if('extra' in ci.keys()):
                t = 'NORMAL'
                for k, v in ci['extra'].items():
                    if(k == t):
                        if(v['length'] == l):
                            use_stored = True
                            batch = v['batch']
                            shader = v['shader']
                            break
            
            if(not use_stored):
                shader_data_vert, shader_data_frag, shader_data_geom = load_shader_code('normal_colors')
                shader = GPUShader(shader_data_vert, shader_data_frag)
                batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "normal": ns[:l], })
                
                if('extra' not in ci.keys()):
                    ci['extra'] = {}
                d = {'shader': shader,
                     'batch': batch,
                     'length': l, }
                ci['extra']['NORMAL'] = d
            
            # shader = GPUShader(PCVShaders.normal_colors_vertex_shader, PCVShaders.normal_colors_fragment_shader, )
            # batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "normal": ns[:l], })
            shader.bind()
            pm = bpy.context.region_data.perspective_matrix
            shader.uniform_float("perspective_matrix", pm)
            shader.uniform_float("object_matrix", o.matrix_world)
            shader.uniform_float("point_size", pcv.point_size)
            shader.uniform_float("alpha_radius", pcv.alpha_radius)
            shader.uniform_float("global_alpha", pcv.global_alpha)
            batch.draw(shader)
        
        if(pcv.dev_position_colors_enabled):
            
            vs = ci['vertices']
            l = ci['current_display_length']
            
            use_stored = False
            if('extra' in ci.keys()):
                t = 'POSITION'
                for k, v in ci['extra'].items():
                    if(k == t):
                        if(v['length'] == l):
                            use_stored = True
                            batch = v['batch']
                            shader = v['shader']
                            break
            
            if(not use_stored):
                
                shader_data_vert, shader_data_frag, shader_data_geom = load_shader_code('position_colors')
                shader = GPUShader(shader_data_vert, shader_data_frag)
                batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], })
                
                if('extra' not in ci.keys()):
                    ci['extra'] = {}
                d = {'shader': shader,
                     'batch': batch,
                     'length': l, }
                ci['extra']['POSITION'] = d
            
            # shader = GPUShader(PCVShaders.normal_colors_vertex_shader, PCVShaders.normal_colors_fragment_shader, )
            # batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "normal": ns[:l], })
            shader.bind()
            pm = bpy.context.region_data.perspective_matrix
            shader.uniform_float("perspective_matrix", pm)
            shader.uniform_float("object_matrix", o.matrix_world)
            shader.uniform_float("point_size", pcv.point_size)
            shader.uniform_float("alpha_radius", pcv.alpha_radius)
            shader.uniform_float("global_alpha", pcv.global_alpha)
            batch.draw(shader)
        
        if(pcv.dev_selection_shader_display):
            vs = ci['vertices']
            l = ci['current_display_length']
            
            shader_data_vert, shader_data_frag, shader_data_geom = load_shader_code('selection')
            shader = GPUShader(shader_data_vert, shader_data_frag)
            
            batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], })
            shader.bind()
            pm = bpy.context.region_data.perspective_matrix
            shader.uniform_float("perspective_matrix", pm)
            shader.uniform_float("object_matrix", o.matrix_world)
            shader.uniform_float("color", pcv.dev_selection_shader_color)
            shader.uniform_float("point_size", pcv.point_size)
            shader.uniform_float("alpha_radius", pcv.alpha_radius)
            bgl.glClear(bgl.GL_DEPTH_BUFFER_BIT)
            batch.draw(shader)
        
        if(pcv.color_adjustment_shader_enabled):
            vs = ci['vertices']
            cs = ci['colors']
            l = ci['current_display_length']
            
            use_stored = False
            if('extra' in ci.keys()):
                t = 'COLOR_ADJUSTMENT'
                for k, v in ci['extra'].items():
                    if(k == t):
                        if(v['length'] == l):
                            use_stored = True
                            batch = v['batch']
                            shader = v['shader']
                            break
            
            if(not use_stored):
                
                shader_data_vert, shader_data_frag, shader_data_geom = load_shader_code('color_adjustment')
                shader = GPUShader(shader_data_vert, shader_data_frag)
                
                batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "color": cs[:l], })
                
                if('extra' not in ci.keys()):
                    ci['extra'] = {}
                d = {'shader': shader,
                     'batch': batch,
                     'length': l, }
                ci['extra']['COLOR_ADJUSTMENT'] = d
            
            shader.bind()
            pm = bpy.context.region_data.perspective_matrix
            shader.uniform_float("perspective_matrix", pm)
            shader.uniform_float("object_matrix", o.matrix_world)
            shader.uniform_float("point_size", pcv.point_size)
            shader.uniform_float("alpha_radius", pcv.alpha_radius)
            shader.uniform_float("global_alpha", pcv.global_alpha)
            
            shader.uniform_float("exposure", pcv.color_adjustment_shader_exposure)
            shader.uniform_float("gamma", pcv.color_adjustment_shader_gamma)
            shader.uniform_float("brightness", pcv.color_adjustment_shader_brightness)
            shader.uniform_float("contrast", pcv.color_adjustment_shader_contrast)
            shader.uniform_float("hue", pcv.color_adjustment_shader_hue)
            shader.uniform_float("saturation", pcv.color_adjustment_shader_saturation)
            shader.uniform_float("value", pcv.color_adjustment_shader_value)
            shader.uniform_float("invert", pcv.color_adjustment_shader_invert)
            
            batch.draw(shader)
        
        # dev
        if(pcv.dev_bbox_enabled):
            vs = ci['vertices']
            
            use_stored = False
            if('extra' in ci.keys()):
                t = 'BOUNDING_BOX'
                for k, v in ci['extra'].items():
                    if(k == t):
                        use_stored = True
                        batch = v['batch']
                        shader = v['shader']
                        break
            
            if(not use_stored):
                
                shader_data_vert, shader_data_frag, shader_data_geom = load_shader_code('bbox')
                shader = GPUShader(shader_data_vert, shader_data_frag, geocode=shader_data_geom, )
                
                batch = batch_for_shader(shader, 'POINTS', {"position": [(0.0, 0.0, 0.0, )], }, )
                
                if('extra' not in ci.keys()):
                    ci['extra'] = {}
                d = {'shader': shader,
                     'batch': batch, }
                ci['extra']['BOUNDING_BOX'] = d
            
            shader.bind()
            pm = bpy.context.region_data.perspective_matrix
            shader.uniform_float("perspective_matrix", pm)
            shader.uniform_float("object_matrix", o.matrix_world)
            # col = bpy.context.preferences.addons[__name__].preferences.normal_color[:]
            # col = tuple([c ** (1 / 2.2) for c in col]) + (pcv.vertex_normals_alpha, )
            
            # col = pcv.dev_bbox_color
            # col = tuple([c ** (1 / 2.2) for c in col]) + (1.0, )
            col = tuple(pcv.dev_bbox_color) + (pcv.dev_bbox_alpha, )
            
            shader.uniform_float("color", col, )
            # shader.uniform_float("length", pcv.vertex_normals_size, )
            shader.uniform_float("global_alpha", pcv.global_alpha)
            
            # # cx = np.sum(vs[:, 0]) / len(vs)
            # # cy = np.sum(vs[:, 1]) / len(vs)
            # # cz = np.sum(vs[:, 2]) / len(vs)
            # cx = np.median(vs[:, 0])
            # cy = np.median(vs[:, 1])
            # cz = np.median(vs[:, 2])
            # center = [cx, cy, cz]
            # # center = [0.0, 0.0, 0.0]
            # # print(center)
            # shader.uniform_float("center", center)
            
            # TODO: store values somewhere, might be slow if calculated every frame
            
            minx = np.min(vs[:, 0])
            miny = np.min(vs[:, 1])
            minz = np.min(vs[:, 2])
            maxx = np.max(vs[:, 0])
            maxy = np.max(vs[:, 1])
            maxz = np.max(vs[:, 2])
            
            def calc(mini, maxi):
                if(mini <= 0.0 and maxi <= 0.0):
                    return abs(mini) - abs(maxi)
                elif(mini <= 0.0 and maxi >= 0.0):
                    return abs(mini) + maxi
                else:
                    return maxi - mini
            
            dimensions = [calc(minx, maxx), calc(miny, maxy), calc(minz, maxz)]
            shader.uniform_float("dimensions", dimensions)
            
            center = [(minx + maxx) / 2, (miny + maxy) / 2, (minz + maxz) / 2]
            shader.uniform_float("center", center)
            
            mindim = abs(min(dimensions)) / 2 * pcv.dev_bbox_size
            shader.uniform_float("length", mindim)
            
            batch.draw(shader)
        
        # dev
        if(pcv.dev_minimal_shader_enabled):
            vs = ci['vertices']
            cs = ci['colors']
            l = ci['current_display_length']
            
            use_stored = False
            if('extra' in ci.keys()):
                t = 'MINIMAL'
                for k, v in ci['extra'].items():
                    if(k == t):
                        if(v['length'] == l):
                            use_stored = True
                            batch = v['batch']
                            shader = v['shader']
                            break
            
            if(not use_stored):
                
                shader_data_vert, shader_data_frag, shader_data_geom = load_shader_code('minimal')
                shader = GPUShader(shader_data_vert, shader_data_frag)
                
                batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "color": cs[:l], })
                
                if('extra' not in ci.keys()):
                    ci['extra'] = {}
                d = {'shader': shader,
                     'batch': batch,
                     'length': l, }
                ci['extra']['MINIMAL'] = d
            
            shader.bind()
            pm = bpy.context.region_data.perspective_matrix
            shader.uniform_float("perspective_matrix", pm)
            shader.uniform_float("object_matrix", o.matrix_world)
            shader.uniform_float("point_size", pcv.point_size)
            shader.uniform_float("global_alpha", pcv.global_alpha)
            batch.draw(shader)
        
        # dev
        if(pcv.dev_minimal_shader_variable_size_enabled):
            vs = ci['vertices']
            cs = ci['colors']
            l = ci['current_display_length']
            
            use_stored = False
            if('extra' in ci.keys()):
                t = 'MINIMAL_VARIABLE_SIZE'
                for k, v in ci['extra'].items():
                    if(k == t):
                        if(v['length'] == l):
                            use_stored = True
                            batch = v['batch']
                            shader = v['shader']
                            sizes = v['sizes']
                            break
            
            if(not use_stored):
                # # generate something to test it, later implement how to set it
                # sizes = np.random.randint(low=1, high=10, size=len(vs), )
                
                if('extra' in ci.keys()):
                    if('MINIMAL_VARIABLE_SIZE' in ci['extra'].keys()):
                        sizes = ci['extra']['MINIMAL_VARIABLE_SIZE']['sizes']
                    else:
                        sizes = np.random.randint(low=1, high=10, size=len(vs), )
                else:
                    sizes = np.random.randint(low=1, high=10, size=len(vs), )
                
                if('extra' in ci.keys()):
                    for k, v in ci['extra'].items():
                        if(k in ('MINIMAL_VARIABLE_SIZE', 'MINIMAL_VARIABLE_SIZE_AND_DEPTH', 'RICH_BILLBOARD', 'RICH_BILLBOARD_NO_DEPTH', )):
                            # FIXME: it was recently switched, try to recover already generated data, both arrays are the same, so thay are in memory just once, no problem here, the problem is with storing reference to it, this should be fixed with something, for example, shader and batch should be generated on one spot, etc.. but have to be done together with all new PCVManager.render and unified shader management. this is just mediocre workaround..
                            sizes = ci['extra'][k]['sizes']
                            sizesf = ci['extra'][k]['sizesf']
                            break
                
                shader_data_vert, shader_data_frag, shader_data_geom = load_shader_code('minimal_variable_size')
                shader = GPUShader(shader_data_vert, shader_data_frag)
                
                batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "color": cs[:l], "size": sizes[:l], })
                # batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "color": cs[:l], })
                
                if('extra' not in ci.keys()):
                    ci['extra'] = {}
                
                d = {'shader': shader,
                     'batch': batch,
                     'sizes': sizes,
                     'length': l, }
                ci['extra']['MINIMAL_VARIABLE_SIZE'] = d
            
            shader.bind()
            pm = bpy.context.region_data.perspective_matrix
            shader.uniform_float("perspective_matrix", pm)
            shader.uniform_float("object_matrix", o.matrix_world)
            # shader.uniform_float("point_size", pcv.point_size)
            shader.uniform_float("global_alpha", pcv.global_alpha)
            batch.draw(shader)
        
        # dev
        if(pcv.dev_minimal_shader_variable_size_and_depth_enabled):
            vs = ci['vertices']
            cs = ci['colors']
            l = ci['current_display_length']
            
            use_stored = False
            if('extra' in ci.keys()):
                t = 'MINIMAL_VARIABLE_SIZE_AND_DEPTH'
                for k, v in ci['extra'].items():
                    if(k == t):
                        if(v['length'] == l):
                            use_stored = True
                            batch = v['batch']
                            shader = v['shader']
                            sizes = v['sizes']
                            break
            
            if(not use_stored):
                # # generate something to test it, later implement how to set it
                # sizes = np.random.randint(low=1, high=10, size=len(vs), )
                
                if('extra' in ci.keys()):
                    if('MINIMAL_VARIABLE_SIZE_AND_DEPTH' in ci['extra'].keys()):
                        sizes = ci['extra']['MINIMAL_VARIABLE_SIZE_AND_DEPTH']['sizes']
                    else:
                        sizes = np.random.randint(low=1, high=10, size=len(vs), )
                else:
                    sizes = np.random.randint(low=1, high=10, size=len(vs), )
                
                if('extra' in ci.keys()):
                    for k, v in ci['extra'].items():
                        if(k in ('MINIMAL_VARIABLE_SIZE', 'MINIMAL_VARIABLE_SIZE_AND_DEPTH', 'RICH_BILLBOARD', 'RICH_BILLBOARD_NO_DEPTH', )):
                            # FIXME: it was recently switched, try to recover already generated data, both arrays are the same, so thay are in memory just once, no problem here, the problem is with storing reference to it, this should be fixed with something, for example, shader and batch should be generated on one spot, etc.. but have to be done together with all new PCVManager.render and unified shader management. this is just mediocre workaround..
                            sizes = ci['extra'][k]['sizes']
                            sizesf = ci['extra'][k]['sizesf']
                            break
                
                shader_data_vert, shader_data_frag, shader_data_geom = load_shader_code('minimal_variable_size_and_depth')
                shader = GPUShader(shader_data_vert, shader_data_frag)
                
                batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "color": cs[:l], "size": sizes[:l], })
                
                if('extra' not in ci.keys()):
                    ci['extra'] = {}
                
                d = {'shader': shader,
                     'batch': batch,
                     'sizes': sizes,
                     'length': l, }
                ci['extra']['MINIMAL_VARIABLE_SIZE_AND_DEPTH'] = d
            
            shader.bind()
            pm = bpy.context.region_data.perspective_matrix
            shader.uniform_float("perspective_matrix", pm)
            shader.uniform_float("object_matrix", o.matrix_world)
            shader.uniform_float("global_alpha", pcv.global_alpha)
            
            if(len(vs) == 0):
                maxdist = 1.0
                cx = 0.0
                cy = 0.0
                cz = 0.0
            else:
                # NOTE: precalculating and storing following should speed up things a bit, but then it won't reflect edits..
                # TODO: calculating center of all points is not quite correct, visually it works, but (as i've seen in bounding box shader) it's not working when distribution of points is uneven, so have a check if it might be a bit better..
                cx = np.sum(vs[:, 0]) / len(vs)
                cy = np.sum(vs[:, 1]) / len(vs)
                cz = np.sum(vs[:, 2]) / len(vs)
                _, _, s = o.matrix_world.decompose()
                l = s.length
                # FIXME: here is error in max with zero length arrays, why are they zero length anyway, putting this single fix for now
                maxd = abs(np.max(vs))
                mind = abs(np.min(vs))
                maxdist = maxd
                if(mind > maxd):
                    maxdist = mind
            shader.uniform_float("maxdist", float(maxdist) * l)
            shader.uniform_float("center", (cx, cy, cz, ))
            shader.uniform_float("brightness", pcv.dev_minimal_shader_variable_size_and_depth_brightness)
            shader.uniform_float("contrast", pcv.dev_minimal_shader_variable_size_and_depth_contrast)
            shader.uniform_float("blend", 1.0 - pcv.dev_minimal_shader_variable_size_and_depth_blend)
            
            batch.draw(shader)
        
        # dev
        if(pcv.dev_billboard_point_cloud_enabled):
            vs = ci['vertices']
            cs = ci['colors']
            l = ci['current_display_length']
            
            use_stored = False
            if('extra' in ci.keys()):
                t = 'BILLBOARD'
                for k, v in ci['extra'].items():
                    if(k == t):
                        if(v['length'] == l):
                            use_stored = True
                            batch = v['batch']
                            shader = v['shader']
                            break
            
            if(not use_stored):
                
                shader_data_vert, shader_data_frag, shader_data_geom = load_shader_code('simple_billboard')
                shader = GPUShader(shader_data_vert, shader_data_frag, geocode=shader_data_geom, )
                
                # shader = GPUShader(PCVShaders.billboard_vertex, PCVShaders.billboard_fragment, geocode=PCVShaders.billboard_geometry_disc, )
                batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "color": cs[:l], })
                
                if('extra' not in ci.keys()):
                    ci['extra'] = {}
                d = {'shader': shader,
                     'batch': batch,
                     'length': l, }
                ci['extra']['BILLBOARD'] = d
            
            shader.bind()
            shader.uniform_float("view_matrix", bpy.context.region_data.view_matrix)
            shader.uniform_float("window_matrix", bpy.context.region_data.window_matrix)
            shader.uniform_float("object_matrix", o.matrix_world)
            shader.uniform_float("size", pcv.dev_billboard_point_cloud_size)
            shader.uniform_float("alpha", pcv.global_alpha)
            batch.draw(shader)
        
        # dev
        if(pcv.dev_rich_billboard_point_cloud_enabled):
            vs = ci['vertices']
            cs = ci['colors']
            l = ci['current_display_length']
            
            use_stored = False
            if('extra' in ci.keys()):
                t = 'RICH_BILLBOARD'
                for k, v in ci['extra'].items():
                    if(k == t):
                        if(v['length'] == l):
                            use_stored = True
                            batch = v['batch']
                            shader = v['shader']
                            sizesf = v['sizesf']
                            break
            
            if(not use_stored):
                
                if('extra' in ci.keys()):
                    if('RICH_BILLBOARD' in ci['extra'].keys()):
                        sizesf = ci['extra']['RICH_BILLBOARD']['sizesf']
                    else:
                        sizesf = np.random.uniform(low=0.5, high=1.5, size=len(vs), )
                        sizesf = sizesf.astype(np.float32)
                else:
                    sizesf = np.random.uniform(low=0.5, high=1.5, size=len(vs), )
                    sizesf = sizesf.astype(np.float32)
                
                if('extra' in ci.keys()):
                    for k, v in ci['extra'].items():
                        if(k in ('MINIMAL_VARIABLE_SIZE', 'MINIMAL_VARIABLE_SIZE_AND_DEPTH', 'RICH_BILLBOARD', 'RICH_BILLBOARD_NO_DEPTH', )):
                            # FIXME: it was recently switched, try to recover already generated data, both arrays are the same, so thay are in memory just once, no problem here, the problem is with storing reference to it, this should be fixed with something, for example, shader and batch should be generated on one spot, etc.. but have to be done together with all new PCVManager.render and unified shader management. this is just mediocre workaround..
                            sizes = ci['extra'][k]['sizes']
                            sizesf = ci['extra'][k]['sizesf']
                            break
                
                shader_data_vert, shader_data_frag, shader_data_geom = load_shader_code('billboard_with_depth_and_size')
                shader = GPUShader(shader_data_vert, shader_data_frag, geocode=shader_data_geom, )
                
                batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "color": cs[:l], "sizef": sizesf[:l], })
                
                if('extra' not in ci.keys()):
                    ci['extra'] = {}
                d = {'shader': shader,
                     'batch': batch,
                     'sizesf': sizesf,
                     'length': l, }
                ci['extra']['RICH_BILLBOARD'] = d
            
            shader.bind()
            shader.uniform_float("perspective_matrix", bpy.context.region_data.perspective_matrix)
            shader.uniform_float("view_matrix", bpy.context.region_data.view_matrix)
            shader.uniform_float("window_matrix", bpy.context.region_data.window_matrix)
            shader.uniform_float("object_matrix", o.matrix_world)
            shader.uniform_float("size", pcv.dev_rich_billboard_point_cloud_size)
            shader.uniform_float("alpha", pcv.global_alpha)
            
            if(len(vs) == 0):
                maxdist = 1.0
                cx = 0.0
                cy = 0.0
                cz = 0.0
            else:
                # NOTE: precalculating and storing following should speed up things a bit, but then it won't reflect edits..
                # TODO: calculating center of all points is not quite correct, visually it works, but (as i've seen in bounding box shader) it's not working when distribution of points is uneven, so have a check if it might be a bit better..
                cx = np.sum(vs[:, 0]) / len(vs)
                cy = np.sum(vs[:, 1]) / len(vs)
                cz = np.sum(vs[:, 2]) / len(vs)
                _, _, s = o.matrix_world.decompose()
                l = s.length
                # FIXME: here is error in max with zero length arrays, why are they zero length anyway, putting this single fix for now
                maxd = abs(np.max(vs))
                mind = abs(np.min(vs))
                maxdist = maxd
                if(mind > maxd):
                    maxdist = mind
            shader.uniform_float("maxdist", float(maxdist) * l)
            shader.uniform_float("center", (cx, cy, cz, ), )
            
            shader.uniform_float("brightness", pcv.dev_rich_billboard_depth_brightness)
            shader.uniform_float("contrast", pcv.dev_rich_billboard_depth_contrast)
            shader.uniform_float("blend", 1.0 - pcv.dev_rich_billboard_depth_blend)
            
            batch.draw(shader)
        
        # dev
        if(pcv.dev_rich_billboard_point_cloud_no_depth_enabled):
            vs = ci['vertices']
            cs = ci['colors']
            l = ci['current_display_length']
            
            use_stored = False
            if('extra' in ci.keys()):
                t = 'RICH_BILLBOARD_NO_DEPTH'
                for k, v in ci['extra'].items():
                    if(k == t):
                        if(v['length'] == l):
                            use_stored = True
                            batch = v['batch']
                            shader = v['shader']
                            sizesf = v['sizesf']
                            break
            
            if(not use_stored):
                
                if('extra' in ci.keys()):
                    if('RICH_BILLBOARD_NO_DEPTH' in ci['extra'].keys()):
                        sizesf = ci['extra']['RICH_BILLBOARD_NO_DEPTH']['sizesf']
                    else:
                        sizesf = np.random.uniform(low=0.5, high=1.5, size=len(vs), )
                        sizesf = sizesf.astype(np.float32)
                else:
                    sizesf = np.random.uniform(low=0.5, high=1.5, size=len(vs), )
                    sizesf = sizesf.astype(np.float32)
                
                if('extra' in ci.keys()):
                    for k, v in ci['extra'].items():
                        if(k in ('MINIMAL_VARIABLE_SIZE', 'MINIMAL_VARIABLE_SIZE_AND_DEPTH', 'RICH_BILLBOARD', 'RICH_BILLBOARD_NO_DEPTH', )):
                            # FIXME: it was recently switched, try to recover already generated data, both arrays are the same, so thay are in memory just once, no problem here, the problem is with storing reference to it, this should be fixed with something, for example, shader and batch should be generated on one spot, etc.. but have to be done together with all new PCVManager.render and unified shader management. this is just mediocre workaround..
                            sizes = ci['extra'][k]['sizes']
                            sizesf = ci['extra'][k]['sizesf']
                            break
                
                shader_data_vert, shader_data_frag, shader_data_geom = load_shader_code('billboard_with_no_depth_and_size')
                shader = GPUShader(shader_data_vert, shader_data_frag, geocode=shader_data_geom, )
                
                batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "color": cs[:l], "sizef": sizesf[:l], })
                
                if('extra' not in ci.keys()):
                    ci['extra'] = {}
                d = {'shader': shader,
                     'batch': batch,
                     'sizesf': sizesf,
                     'length': l, }
                ci['extra']['RICH_BILLBOARD_NO_DEPTH'] = d
            
            shader.bind()
            # shader.uniform_float("perspective_matrix", bpy.context.region_data.perspective_matrix)
            shader.uniform_float("view_matrix", bpy.context.region_data.view_matrix)
            shader.uniform_float("window_matrix", bpy.context.region_data.window_matrix)
            shader.uniform_float("object_matrix", o.matrix_world)
            shader.uniform_float("size", pcv.dev_rich_billboard_point_cloud_size)
            shader.uniform_float("alpha", pcv.global_alpha)
            
            batch.draw(shader)
        
        # dev
        if(pcv.dev_phong_shader_enabled):
            vs = ci['vertices']
            ns = ci['normals']
            cs = ci['colors']
            l = ci['current_display_length']
            
            use_stored = False
            if('extra' in ci.keys()):
                t = 'PHONG'
                for k, v in ci['extra'].items():
                    if(k == t):
                        if(v['length'] == l):
                            use_stored = True
                            batch = v['batch']
                            shader = v['shader']
                            break
            
            if(not use_stored):
                shader_data_vert, shader_data_frag, shader_data_geom = load_shader_code('phong')
                shader = GPUShader(shader_data_vert, shader_data_frag)
                
                batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "normal": ns[:l], "color": cs[:l], })
                
                if('extra' not in ci.keys()):
                    ci['extra'] = {}
                d = {'shader': shader,
                     'batch': batch,
                     'length': l, }
                ci['extra']['PHONG'] = d
            
            shader.bind()
            
            # shader.uniform_float("perspective_matrix", bpy.context.region_data.perspective_matrix)
            shader.uniform_float("view", bpy.context.region_data.view_matrix)
            shader.uniform_float("projection", bpy.context.region_data.window_matrix)
            shader.uniform_float("model", o.matrix_world)
            
            shader.uniform_float("light_position", bpy.context.region_data.view_matrix.inverted().translation)
            # shader.uniform_float("light_color", (1.0, 1.0, 1.0))
            shader.uniform_float("light_color", (0.8, 0.8, 0.8, ))
            shader.uniform_float("view_position", bpy.context.region_data.view_matrix.inverted().translation)
            
            shader.uniform_float("ambient_strength", pcv.dev_phong_shader_ambient_strength)
            shader.uniform_float("specular_strength", pcv.dev_phong_shader_specular_strength)
            shader.uniform_float("specular_exponent", pcv.dev_phong_shader_specular_exponent)
            
            shader.uniform_float("alpha", pcv.global_alpha)
            shader.uniform_float("point_size", pcv.point_size)
            shader.uniform_float("alpha_radius", pcv.alpha_radius)
            
            # pm = bpy.context.region_data.perspective_matrix
            # shader.uniform_float("perspective_matrix", pm)
            # shader.uniform_float("object_matrix", o.matrix_world)
            # shader.uniform_float("point_size", pcv.point_size)
            # shader.uniform_float("global_alpha", pcv.global_alpha)
            batch.draw(shader)
        
        # dev
        if(pcv.clip_shader_enabled):
            vs = ci['vertices']
            cs = ci['colors']
            l = ci['current_display_length']
            
            use_stored = False
            if('extra' in ci.keys()):
                t = 'CLIP'
                for k, v in ci['extra'].items():
                    if(k == t):
                        if(v['length'] == l):
                            use_stored = True
                            batch = v['batch']
                            shader = v['shader']
                            break
            
            if(not use_stored):
                shader_data_vert, shader_data_frag, shader_data_geom = load_shader_code('simple_clip')
                shader = GPUShader(shader_data_vert, shader_data_frag)
                
                batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "color": cs[:l], })
                
                if('extra' not in ci.keys()):
                    ci['extra'] = {}
                d = {'shader': shader,
                     'batch': batch,
                     'length': l, }
                ci['extra']['CLIP'] = d
            
            if(pcv.clip_plane0_enabled):
                bgl.glEnable(bgl.GL_CLIP_DISTANCE0)
            if(pcv.clip_plane1_enabled):
                bgl.glEnable(bgl.GL_CLIP_DISTANCE1)
            if(pcv.clip_plane2_enabled):
                bgl.glEnable(bgl.GL_CLIP_DISTANCE2)
            if(pcv.clip_plane3_enabled):
                bgl.glEnable(bgl.GL_CLIP_DISTANCE3)
            if(pcv.clip_plane4_enabled):
                bgl.glEnable(bgl.GL_CLIP_DISTANCE4)
            if(pcv.clip_plane5_enabled):
                bgl.glEnable(bgl.GL_CLIP_DISTANCE5)
            
            shader.bind()
            
            shader.uniform_float("perspective_matrix", bpy.context.region_data.perspective_matrix)
            shader.uniform_float("object_matrix", o.matrix_world)
            shader.uniform_float("point_size", pcv.point_size)
            shader.uniform_float("alpha_radius", pcv.alpha_radius)
            shader.uniform_float("global_alpha", pcv.global_alpha)
            
            shader.uniform_float("clip_plane0", pcv.clip_plane0)
            shader.uniform_float("clip_plane1", pcv.clip_plane1)
            shader.uniform_float("clip_plane2", pcv.clip_plane2)
            shader.uniform_float("clip_plane3", pcv.clip_plane3)
            shader.uniform_float("clip_plane4", pcv.clip_plane4)
            shader.uniform_float("clip_plane5", pcv.clip_plane5)
            
            batch.draw(shader)
            
            if(pcv.clip_plane0_enabled):
                bgl.glDisable(bgl.GL_CLIP_DISTANCE0)
            if(pcv.clip_plane1_enabled):
                bgl.glDisable(bgl.GL_CLIP_DISTANCE1)
            if(pcv.clip_plane2_enabled):
                bgl.glDisable(bgl.GL_CLIP_DISTANCE2)
            if(pcv.clip_plane3_enabled):
                bgl.glDisable(bgl.GL_CLIP_DISTANCE3)
            if(pcv.clip_plane4_enabled):
                bgl.glDisable(bgl.GL_CLIP_DISTANCE4)
            if(pcv.clip_plane5_enabled):
                bgl.glDisable(bgl.GL_CLIP_DISTANCE5)
        
        # dev
        if(pcv.billboard_phong_enabled):
            vs = ci['vertices']
            ns = ci['normals']
            cs = ci['colors']
            l = ci['current_display_length']
            
            use_stored = False
            if('extra' in ci.keys()):
                t = 'BILLBOARD_PHONG'
                for k, v in ci['extra'].items():
                    if(k == t):
                        if(v['circles'] != pcv.billboard_phong_circles):
                            use_stored = False
                            break
                        if(v['length'] == l):
                            use_stored = True
                            batch = v['batch']
                            shader = v['shader']
                            break
            
            if(not use_stored):
                # use_geocode = PCVShaders.billboard_phong_fast_gs
                # if(pcv.billboard_phong_circles):
                #     use_geocode = PCVShaders.billboard_phong_circles_gs
                # shader = GPUShader(PCVShaders.billboard_phong_vs, PCVShaders.billboard_phong_fs, geocode=use_geocode, )
                
                if(pcv.billboard_phong_circles):
                    shader_data_vert, shader_data_frag, shader_data_geom = load_shader_code('phong_billboard_circles')
                    shader = GPUShader(shader_data_vert, shader_data_frag, geocode=shader_data_geom, )
                else:
                    shader_data_vert, shader_data_frag, shader_data_geom = load_shader_code('phong_billboard')
                    shader = GPUShader(shader_data_vert, shader_data_frag, geocode=shader_data_geom, )
                
                batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "normal": ns[:l], "color": cs[:l], })
                
                if('extra' not in ci.keys()):
                    ci['extra'] = {}
                d = {'shader': shader,
                     'batch': batch,
                     'circles': pcv.billboard_phong_circles,
                     'length': l, }
                ci['extra']['BILLBOARD_PHONG'] = d
            
            shader.bind()
            
            shader.uniform_float("model", o.matrix_world)
            # shader.uniform_float("view", bpy.context.region_data.view_matrix)
            # shader.uniform_float("projection", bpy.context.region_data.window_matrix)
            
            shader.uniform_float("view_matrix", bpy.context.region_data.view_matrix)
            shader.uniform_float("window_matrix", bpy.context.region_data.window_matrix)
            
            shader.uniform_float("size", pcv.billboard_phong_size)
            
            shader.uniform_float("alpha", pcv.global_alpha)
            
            shader.uniform_float("light_position", bpy.context.region_data.view_matrix.inverted().translation)
            shader.uniform_float("light_color", (0.8, 0.8, 0.8, ))
            shader.uniform_float("view_position", bpy.context.region_data.view_matrix.inverted().translation)
            
            shader.uniform_float("ambient_strength", pcv.billboard_phong_ambient_strength)
            shader.uniform_float("specular_strength", pcv.billboard_phong_specular_strength)
            shader.uniform_float("specular_exponent", pcv.billboard_phong_specular_exponent)
            
            batch.draw(shader)
        
        # dev
        if(pcv.skip_point_shader_enabled):
            vs = ci['vertices']
            cs = ci['colors']
            
            use_stored = False
            if('extra' in ci.keys()):
                t = 'SKIP'
                for k, v in ci['extra'].items():
                    if(k == t):
                        use_stored = True
                        batch = v['batch']
                        shader = v['shader']
                        break
            
            if(not use_stored):
                indices = np.indices((len(vs), ), dtype=np.int, )
                indices.shape = (-1, )
                
                shader_data_vert, shader_data_frag, shader_data_geom = load_shader_code('simple_skip_point')
                shader = GPUShader(shader_data_vert, shader_data_frag)
                
                batch = batch_for_shader(shader, 'POINTS', {"position": vs[:], "color": cs[:], "index": indices[:], })
                
                if('extra' not in ci.keys()):
                    ci['extra'] = {}
                d = {'shader': shader,
                     'batch': batch, }
                ci['extra']['SKIP'] = d
            
            shader.bind()
            
            shader.uniform_float("perspective_matrix", bpy.context.region_data.perspective_matrix)
            shader.uniform_float("object_matrix", o.matrix_world)
            shader.uniform_float("point_size", pcv.point_size)
            shader.uniform_float("alpha_radius", pcv.alpha_radius)
            shader.uniform_float("global_alpha", pcv.global_alpha)
            
            sp = pcv.skip_point_percentage
            l = int((len(vs) / 100) * sp)
            if(sp >= 99):
                l = len(vs)
            shader.uniform_float("skip_index", l)
            
            batch.draw(shader)
        
        # dev
        if(pcv.fresnel_shader_enabled):
            vs = ci['vertices']
            ns = ci['normals']
            cs = ci['colors']
            l = ci['current_display_length']
            
            use_stored = False
            if('extra' in ci.keys()):
                t = 'FRESNEL'
                for k, v in ci['extra'].items():
                    if(k == t):
                        if(v['length'] == l):
                            use_stored = True
                            batch = v['batch']
                            shader = v['shader']
                            break
            
            if(not use_stored):
                shader_data_vert, shader_data_frag, shader_data_geom = load_shader_code('fresnel')
                shader = GPUShader(shader_data_vert, shader_data_frag)
                batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "normal": ns[:l], "color": cs[:l], })
                
                if('extra' not in ci.keys()):
                    ci['extra'] = {}
                d = {'shader': shader,
                     'batch': batch,
                     'length': l, }
                ci['extra']['FRESNEL'] = d
            
            shader.bind()
            # shader.uniform_float("perspective_matrix", bpy.context.region_data.perspective_matrix)
            # shader.uniform_float("object_matrix", o.matrix_world)
            
            shader.uniform_float("view", bpy.context.region_data.view_matrix)
            shader.uniform_float("projection", bpy.context.region_data.window_matrix)
            shader.uniform_float("model", o.matrix_world)
            
            shader.uniform_float("point_size", pcv.point_size)
            shader.uniform_float("alpha_radius", pcv.alpha_radius)
            shader.uniform_float("global_alpha", pcv.global_alpha)
            shader.uniform_float("view_position", bpy.context.region_data.view_matrix.inverted().translation)
            shader.uniform_float("fresnel_sharpness", pcv.fresnel_shader_sharpness)
            shader.uniform_float("use_colors", float(pcv.fresnel_shader_colors))
            shader.uniform_float("use_invert", float(pcv.fresnel_shader_invert))
            batch.draw(shader)
        
        # and now back to some production stuff..
        
        # draw selection as a last step bucause i clear depth buffer for it
        if(pcv.filter_remove_color_selection):
            if('selection_indexes' not in ci):
                return
            vs = ci['vertices']
            indexes = ci['selection_indexes']
            try:
                # if it works, leave it..
                vs = np.take(vs, indexes, axis=0, )
            except IndexError:
                # something has changed.. some other edit hapended, selection is invalid, reset it all..
                pcv.filter_remove_color_selection = False
                del ci['selection_indexes']
            
            shader_data_vert, shader_data_frag, shader_data_geom = load_shader_code('selection')
            shader = GPUShader(shader_data_vert, shader_data_frag)
            
            batch = batch_for_shader(shader, 'POINTS', {"position": vs[:], })
            shader.bind()
            pm = bpy.context.region_data.perspective_matrix
            shader.uniform_float("perspective_matrix", pm)
            shader.uniform_float("object_matrix", o.matrix_world)
            # sc = bpy.context.preferences.addons[__name__].preferences.selection_color[:]
            sc = preferences().selection_color[:]
            shader.uniform_float("color", sc)
            shader.uniform_float("point_size", pcv.point_size)
            shader.uniform_float("alpha_radius", pcv.alpha_radius)
            bgl.glClear(bgl.GL_DEPTH_BUFFER_BIT)
            batch.draw(shader)
        
        bgl.glDisable(bgl.GL_PROGRAM_POINT_SIZE)
        bgl.glDisable(bgl.GL_DEPTH_TEST)
        bgl.glDisable(bgl.GL_BLEND)
    
    @classmethod
    def handler(cls):
        bobjects = bpy.data.objects
        
        run_gc = False
        for k, v in cls.cache.items():
            if(not bobjects.get(v['name'])):
                v['kill'] = True
                run_gc = True
            if(v['ready'] and v['draw'] and not v['kill']):
                cls.render(v['uuid'])
        if(run_gc):
            cls.gc()
    
    @classmethod
    def update(cls, uuid, vs, ns=None, cs=None, ):
        if(uuid not in PCVManager.cache):
            raise KeyError("uuid '{}' not in cache".format(uuid))
        # if(len(vs) == 0):
        #     raise ValueError("zero length")
        
        # get cache item
        c = PCVManager.cache[uuid]
        l = len(vs)
        
        if(ns is None):
            ns = np.column_stack((np.full(l, 0.0, dtype=np.float32, ),
                                  np.full(l, 0.0, dtype=np.float32, ),
                                  np.full(l, 1.0, dtype=np.float32, ), ))
        
        if(cs is None):
            # col = bpy.context.preferences.addons[__name__].preferences.default_vertex_color[:]
            col = preferences().default_vertex_color[:]
            col = tuple([c ** (1 / 2.2) for c in col]) + (1.0, )
            cs = np.column_stack((np.full(l, col[0], dtype=np.float32, ),
                                  np.full(l, col[1], dtype=np.float32, ),
                                  np.full(l, col[2], dtype=np.float32, ),
                                  np.ones(l, dtype=np.float32, ), ))
        
        # store data
        c['vertices'] = vs
        c['normals'] = ns
        c['colors'] = cs
        c['length'] = l
        c['stats'] = l
        
        o = c['object']
        pcv = o.point_cloud_visualizer
        dp = pcv.display_percent
        nl = int((l / 100) * dp)
        if(dp >= 99):
            nl = l
        c['display_length'] = nl
        c['current_display_length'] = nl
        
        # setup new shaders
        ienabled = c['illumination']
        if(ienabled):
            shader_data_vert, shader_data_frag, shader_data_geom = load_shader_code('illumination')
            shader = GPUShader(shader_data_vert, shader_data_frag)
            batch = batch_for_shader(shader, 'POINTS', {"position": vs[:nl], "color": cs[:nl], "normal": ns[:nl], })
        else:
            shader_data_vert, shader_data_frag, shader_data_geom = load_shader_code('simple')
            shader = GPUShader(shader_data_vert, shader_data_frag)
            batch = batch_for_shader(shader, 'POINTS', {"position": vs[:nl], "color": cs[:nl], })
        c['shader'] = shader
        c['batch'] = batch
        
        # redraw all viewports
        for area in bpy.context.screen.areas:
            if(area.type == 'VIEW_3D'):
                area.tag_redraw()
    
    @classmethod
    def gc(cls):
        l = []
        for k, v in cls.cache.items():
            if(v['kill']):
                l.append(k)
        for i in l:
            del cls.cache[i]
    
    @classmethod
    def init(cls):
        if(cls.initialized):
            return
        cls.handle = bpy.types.SpaceView3D.draw_handler_add(cls.handler, (), 'WINDOW', 'POST_VIEW')
        bpy.app.handlers.load_pre.append(watcher)
        cls.initialized = True
    
    @classmethod
    def deinit(cls):
        if(not cls.initialized):
            return
        for k, v in cls.cache.items():
            v['kill'] = True
        cls.gc()
        
        bpy.types.SpaceView3D.draw_handler_remove(cls.handle, 'WINDOW')
        cls.handle = None
        bpy.app.handlers.load_pre.remove(watcher)
        cls.initialized = False
    
    @classmethod
    def add(cls, data, ):
        cls.cache[data['uuid']] = data
    
    @classmethod
    def new(cls):
        # NOTE: this is redundant.. is it?
        return {'uuid': None,
                'filepath': None,
                'vertices': None,
                'normals': None,
                'colors': None,
                'display_length': None,
                'current_display_length': None,
                'illumination': False,
                'shader': False,
                'batch': False,
                'ready': False,
                'draw': False,
                'kill': False,
                'stats': None,
                'length': None,
                'name': None,
                'object': None, }
    
    @classmethod
    def _redraw(cls):
        # force redraw
        
        # for area in bpy.context.screen.areas:
        #     if(area.type == 'VIEW_3D'):
        #         area.tag_redraw()
        
        # seems like sometimes context is different, this should work..
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if(area.type == 'VIEW_3D'):
                    area.tag_redraw()


class PCVSequence():
    cache = {}
    initialized = False
    
    @classmethod
    def handler(cls, scene, depsgraph, ):
        cf = scene.frame_current
        for k, v in cls.cache.items():
            pcv = v['pcv']
            if(pcv.uuid != k):
                del cls.cache[k]
                if(len(cls.cache.items()) == 0):
                    cls.deinit()
                return
            # if(pcv.sequence_enabled):
            #     PCVManager.init()
            #     ld = len(v['data'])
            #     if(pcv.sequence_use_cyclic):
            #         cf = cf % ld
            #     if(cf > ld):
            #         PCVManager.update(k, [], None, None, )
            #     else:
            #         data = v['data'][cf - 1]
            #         PCVManager.update(k, data['vs'], data['ns'], data['cs'], )
            PCVManager.init()
            ld = len(v['data'])
            if(pcv.sequence_use_cyclic):
                cf = cf % ld
            if(cf > ld):
                PCVManager.update(k, [], None, None, )
            else:
                data = v['data'][cf - 1]
                PCVManager.update(k, data['vs'], data['ns'], data['cs'], )
    
    @classmethod
    def init(cls):
        if(cls.initialized):
            return
        bpy.app.handlers.frame_change_post.append(PCVSequence.handler)
        cls.initialized = True
    
    @classmethod
    def deinit(cls):
        if(not cls.initialized):
            return
        bpy.app.handlers.frame_change_post.remove(PCVSequence.handler)
        cls.initialized = False
        cls.cache = {}


class PCVControl():
    def __init__(self, o, ):
        self.o = o
        PCVManager.init()
    
    def _prepare(self, vs, ns, cs, ):
        if(vs is not None):
            if(len(vs) == 0):
                vs = None
        if(ns is not None):
            if(len(ns) == 0):
                ns = None
        if(cs is not None):
            if(len(cs) == 0):
                cs = None
        
        if(vs is None):
            vs = np.zeros((0, 3), dtype=np.float32,)
        else:
            # make numpy array if not already
            if(type(vs) != np.ndarray):
                vs = np.array(vs)
            # and ensure data type
            vs = vs.astype(np.float32)
        
        n = len(vs)
        
        # process normals if present, otherwise set to default (0.0, 0.0, 1.0)
        if(ns is None):
            has_normals = False
            ns = np.column_stack((np.full(n, 0.0, dtype=np.float32, ),
                                  np.full(n, 0.0, dtype=np.float32, ),
                                  np.full(n, 1.0, dtype=np.float32, ), ))
        else:
            has_normals = True
            if(type(ns) != np.ndarray):
                ns = np.array(ns)
            ns = ns.astype(np.float32)
        
        # process colors if present, otherwise set to default from preferences, append alpha 1.0
        if(cs is None):
            has_colors = False
            # col = bpy.context.preferences.addons[__name__].preferences.default_vertex_color[:]
            col = preferences().default_vertex_color[:]
            col = tuple([c ** (1 / 2.2) for c in col]) + (1.0, )
            cs = np.column_stack((np.full(n, col[0], dtype=np.float32, ),
                                  np.full(n, col[1], dtype=np.float32, ),
                                  np.full(n, col[2], dtype=np.float32, ),
                                  np.ones(n, dtype=np.float32, ), ))
        else:
            has_colors = True
            if(type(cs) != np.ndarray):
                cs = np.array(cs)
            cs = np.column_stack((cs[:, 0], cs[:, 1], cs[:, 2], np.ones(n), ))
            cs = cs.astype(np.float32)
        
        # store points to enable some other functions
        cs8 = cs * 255
        cs8 = cs8.astype(np.uint8)
        dt = [('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('nx', '<f4'), ('ny', '<f4'), ('nz', '<f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        points = np.empty(n, dtype=dt, )
        points['x'] = vs[:, 0]
        points['y'] = vs[:, 1]
        points['z'] = vs[:, 2]
        points['nx'] = ns[:, 0]
        points['ny'] = ns[:, 1]
        points['nz'] = ns[:, 2]
        points['red'] = cs8[:, 0]
        points['green'] = cs8[:, 1]
        points['blue'] = cs8[:, 2]
        
        return vs, ns, cs, points, has_normals, has_colors
    
    def _redraw(self):
        # force redraw
        
        # for area in bpy.context.screen.areas:
        #     if(area.type == 'VIEW_3D'):
        #         area.tag_redraw()
        
        # seems like sometimes context is different, this should work..
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if(area.type == 'VIEW_3D'):
                    area.tag_redraw()
    
    def draw(self, vs=None, ns=None, cs=None, ):
        o = self.o
        pcv = o.point_cloud_visualizer
        
        # check if object has been used before, i.e. has uuid and uuid item is in cache
        if(pcv.uuid != "" and pcv.runtime):
            # was used or blend was saved after it was used and uuid is saved from last time, check cache
            if(pcv.uuid in PCVManager.cache):
                # cache item is found, object has been used before
                self._update(vs, ns, cs, )
                return
        # otherwise setup as new
        
        u = str(uuid.uuid1())
        # use that as path, some checks wants this not empty
        filepath = u
        
        # validate/prepare input data
        vs, ns, cs, points, has_normals, has_colors = self._prepare(vs, ns, cs)
        n = len(vs)
        
        # build cache dict
        d = {}
        d['uuid'] = u
        d['filepath'] = filepath
        d['points'] = points
        
        # but because colors i just stored in uint8, store them also as provided to enable reload operator
        cs_orig = np.column_stack((cs[:, 0], cs[:, 1], cs[:, 2], np.ones(n), ))
        cs_orig = cs_orig.astype(np.float32)
        d['colors_original'] = cs_orig
        
        d['stats'] = n
        d['vertices'] = vs
        d['colors'] = cs
        d['normals'] = ns
        d['length'] = n
        dp = pcv.display_percent
        l = int((n / 100) * dp)
        if(dp >= 99):
            l = n
        d['display_length'] = l
        d['current_display_length'] = l
        d['illumination'] = pcv.illumination
        if(pcv.illumination):
            shader_data_vert, shader_data_frag, shader_data_geom = load_shader_code('illumination')
            shader = GPUShader(shader_data_vert, shader_data_frag)
            batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "color": cs[:l], "normal": ns[:l], })
        else:
            shader_data_vert, shader_data_frag, shader_data_geom = load_shader_code('simple')
            shader = GPUShader(shader_data_vert, shader_data_frag)
            batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "color": cs[:l], })
        d['shader'] = shader
        d['batch'] = batch
        d['ready'] = True
        d['draw'] = False
        d['kill'] = False
        d['object'] = o
        d['name'] = o.name
        
        # set properties
        pcv.uuid = u
        pcv.filepath = filepath
        pcv.has_normals = has_normals
        pcv.has_vcols = has_colors
        pcv.runtime = True
        
        PCVManager.add(d)
        
        # mark to draw
        c = PCVManager.cache[pcv.uuid]
        c['draw'] = True
        
        self._redraw()
    
    def _update(self, vs, ns, cs, ):
        o = self.o
        pcv = o.point_cloud_visualizer
        
        # validate/prepare input data
        vs, ns, cs, points, has_normals, has_colors = self._prepare(vs, ns, cs)
        n = len(vs)
        
        d = PCVManager.cache[pcv.uuid]
        d['points'] = points
        
        # kill normals, might not be no longer valid, it will be recreated later
        if('vertex_normals' in d.keys()):
            del d['vertex_normals']
        
        # but because colors i just stored in uint8, store them also as provided to enable reload operator
        cs_orig = np.column_stack((cs[:, 0], cs[:, 1], cs[:, 2], np.ones(n), ))
        cs_orig = cs_orig.astype(np.float32)
        d['colors_original'] = cs_orig
        
        d['stats'] = n
        d['vertices'] = vs
        d['colors'] = cs
        d['normals'] = ns
        d['length'] = n
        dp = pcv.display_percent
        l = int((n / 100) * dp)
        if(dp >= 99):
            l = n
        d['display_length'] = l
        d['current_display_length'] = l
        d['illumination'] = pcv.illumination
        if(pcv.illumination):
            shader_data_vert, shader_data_frag, shader_data_geom = load_shader_code('illumination')
            shader = GPUShader(shader_data_vert, shader_data_frag)
            batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "color": cs[:l], "normal": ns[:l], })
        else:
            shader_data_vert, shader_data_frag, shader_data_geom = load_shader_code('simple')
            shader = GPUShader(shader_data_vert, shader_data_frag)
            batch = batch_for_shader(shader, 'POINTS', {"position": vs[:l], "color": cs[:l], })
        d['shader'] = shader
        d['batch'] = batch
        
        pcv.has_normals = has_normals
        pcv.has_vcols = has_colors
        
        c = PCVManager.cache[pcv.uuid]
        c['draw'] = True
        
        self._redraw()
    
    def erase(self):
        o = self.o
        pcv = o.point_cloud_visualizer
        
        if(pcv.uuid == ""):
            return
        if(not pcv.runtime):
            return
        if(pcv.uuid not in PCVManager.cache.keys()):
            return
        
        # get cache item and set draw to False
        c = PCVManager.cache[pcv.uuid]
        c['draw'] = False
        
        # # force redraw
        # for area in bpy.context.screen.areas:
        #     if(area.type == 'VIEW_3D'):
        #         area.tag_redraw()
        self._redraw()
    
    def reset(self):
        o = self.o
        pcv = o.point_cloud_visualizer
        
        if(pcv.uuid == ""):
            return
        if(not pcv.runtime):
            return
        if(pcv.uuid not in PCVManager.cache.keys()):
            return
        
        # mark for deletion cache
        c = PCVManager.cache[pcv.uuid]
        c['kill'] = True
        PCVManager.gc()
        
        # reset properties
        pcv.uuid = ""
        pcv.filepath = ""
        pcv.has_normals = False
        pcv.has_vcols = False
        pcv.runtime = False
        
        self._redraw()


@persistent
def watcher(scene):
    PCVSequence.deinit()
    PCVManager.deinit()


registry = {
    'bbox': {'v': "bbox.vert", 'f': "bbox.frag", 'g': "bbox.geom", },
    'billboard_with_depth_and_size': {'v': "billboard_with_depth_and_size.vert", 'f': "billboard_with_depth_and_size.frag", 'g': "billboard_with_depth_and_size.geom", },
    'billboard_with_no_depth_and_size': {'v': "billboard_with_no_depth_and_size.vert", 'f': "billboard_with_no_depth_and_size.frag", 'g': "billboard_with_no_depth_and_size.geom", },
    'color_adjustment': {'v': "color_adjustment.vert", 'f': "color_adjustment.frag", },
    'depth_false_colors': {'v': "depth_false_colors.vert", 'f': "depth_false_colors.frag", },
    'depth_illumination': {'v': "depth_illumination.vert", 'f': "depth_illumination.frag", },
    'depth_simple': {'v': "depth_simple.vert", 'f': "depth_simple.frag", },
    'fresnel': {'v': "fresnel.vert", 'f': "fresnel.frag", },
    'illumination': {'v': "illumination.vert", 'f': "illumination.frag", },
    'minimal_variable_size_and_depth': {'v': "minimal_variable_size_and_depth.vert", 'f': "minimal_variable_size_and_depth.frag", },
    'minimal_variable_size': {'v': "minimal_variable_size.vert", 'f': "minimal_variable_size.frag", },
    'minimal': {'v': "minimal.vert", 'f': "minimal.frag", },
    'normal_colors': {'v': "normal_colors.vert", 'f': "normal_colors.frag", },
    'normals': {'v': "normals.vert", 'f': "normals.frag", 'g': "normals.geom", },
    'phong_billboard': {'v': "phong_billboard.vert", 'f': "phong_billboard.frag", 'g': "phong_billboard.geom", },
    'phong_billboard_circles': {'v': "phong_billboard.vert", 'f': "phong_billboard.frag", 'g': "phong_billboard_circles.geom", },
    'phong': {'v': "phong.vert", 'f': "phong.frag", },
    'position_colors': {'v': "position_colors.vert", 'f': "position_colors.frag", },
    'render_illumination_smooth': {'v': "render_illumination_smooth.vert", 'f': "render_illumination_smooth.frag", },
    'render_simple_smooth': {'v': "render_simple_smooth.vert", 'f': "render_simple_smooth.frag", },
    'selection': {'v': "selection.vert", 'f': "selection.frag", },
    'simple_billboard': {'v': "simple_billboard.vert", 'f': "simple_billboard.frag", 'g': "simple_billboard.geom", },
    'simple_billboard_disc': {'v': "simple_billboard.vert", 'f': "simple_billboard.frag", 'g': "simple_billboard_disc.geom", },
    'simple_clip': {'v': "simple_clip.vert", 'f': "simple_clip.frag", },
    'simple_skip_point': {'v': "simple_skip_point.vert", 'f': "simple_skip_point.frag", },
    'simple': {'v': "simple.vert", 'f': "simple.frag", },
}


def load_shader_code(name):
    if(name not in registry.keys()):
        raise TypeError("Unknown shader requested..")
    d = registry[name]
    vf = d['v']
    ff = d['f']
    gf = None
    if('g' in d.keys()):
        gf = d['g']
    
    with open(os.path.join(os.path.dirname(__file__), 'shaders', vf), mode='r', encoding='utf-8') as f:
        vs = f.read()
    with open(os.path.join(os.path.dirname(__file__), 'shaders', ff), mode='r', encoding='utf-8') as f:
        fs = f.read()
    
    gs = None
    if(gf is not None):
        with open(os.path.join(os.path.dirname(__file__), 'shaders', gf), mode='r', encoding='utf-8') as f:
            gs = f.read()
    
    return vs, fs, gs
