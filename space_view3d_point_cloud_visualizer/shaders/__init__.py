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

registry = {
    'bbox': {'v': "bbox.vert",
             'f': "bbox.frag",
             'g': "bbox.geom", },
    'billboard_with_depth_and_size': {'v': "billboard_with_depth_and_size.vert",
                                      'f': "billboard_with_depth_and_size.frag",
                                      'g': "billboard_with_depth_and_size.geom", },
    'billboard_with_no_depth_and_size': {'v': "billboard_with_no_depth_and_size.vert",
                                         'f': "billboard_with_no_depth_and_size.frag",
                                         'g': "billboard_with_no_depth_and_size.geom", },
    'color_adjustment': {'v': "color_adjustment.vert",
                         'f': "color_adjustment.frag", },
    'depth_false_colors': {'v': "depth_false_colors.vert",
                           'f': "depth_false_colors.frag", },
    'depth_illumination': {'v': "depth_illumination.vert",
                           'f': "depth_illumination.frag", },
    'depth_simple': {'v': "depth_simple.vert",
                     'f': "depth_simple.frag", },
    'illumination': {'v': "illumination.vert",
                     'f': "illumination.frag", },
    'minimal_variable_size_and_depth': {'v': "minimal_variable_size_and_depth.vert",
                                        'f': "minimal_variable_size_and_depth.frag", },
    'minimal_variable_size': {'v': "minimal_variable_size.vert",
                              'f': "minimal_variable_size.frag", },
    'minimal': {'v': "minimal.vert",
                'f': "minimal.frag", },
    'normal_colors': {'v': "normal_colors.vert",
                      'f': "normal_colors.frag", },
    'normals': {'v': "normals.vert",
                'f': "normals.frag",
                'g': "normals.geom", },
    'phong_billboard': {'v': "phong_billboard.vert",
                        'f': "phong_billboard.frag",
                        'g': "phong_billboard.geom", },
    'phong_billboard_circles': {'v': "phong_billboard.vert",
                                'f': "phong_billboard.frag",
                                'g': "phong_billboard_circles.geom", },
    'phong': {'v': "phong.vert",
              'f': "phong.frag", },
    'position_colors': {'v': "position_colors.vert",
                        'f': "position_colors.frag", },
    'render_illumination_smooth': {'v': "render_illumination_smooth.vert",
                                   'f': "render_illumination_smooth.frag", },
    'render_simple_smooth': {'v': "render_simple_smooth.vert",
                             'f': "render_simple_smooth.frag", },
    'selection': {'v': "selection.vert",
                  'f': "selection.frag", },
    'simple_billboard': {'v': "simple_billboard.vert",
                         'f': "simple_billboard.frag",
                         'g': "simple_billboard.geom", },
    'simple_billboard_disc': {'v': "simple_billboard.vert",
                              'f': "simple_billboard.frag",
                              'g': "simple_billboard_disc.geom", },
    'simple_clip': {'v': "simple_clip.vert",
                    'f': "simple_clip.frag", },
    'simple_skip_point': {'v': "simple_skip_point.vert",
                          'f': "simple_skip_point.frag", },
    'simple': {'v': "simple.vert",
               'f': "simple.frag", },
}


def load(name):
    if(name not in registry.keys()):
        raise TypeError("Unknown shader requested..")
    d = registry[name]
    vf = d['v']
    ff = d['f']
    gf = None
    if('g' in d.keys()):
        gf = d['g']
    
    with open(os.path.join(os.path.dirname(__file__), vf), mode='r', encoding='utf-8') as f:
        vs = f.read()
    with open(os.path.join(os.path.dirname(__file__), ff), mode='r', encoding='utf-8') as f:
        fs = f.read()
    
    gs = None
    if(gf is not None):
        with open(os.path.join(os.path.dirname(__file__), gf), mode='r', encoding='utf-8') as f:
            gs = f.read()
    
    return vs, fs, gs
