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


def draw_instavis(self, context, ):
    l = self.layout
    l.label(text='hello')
