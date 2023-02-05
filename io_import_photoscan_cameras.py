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

# Copyright (c) 2023 Jakub Uhlik

bl_info = {"name": "Import Agisoft PhotoScan Cameras (.xml)",
           "description": "",
           "author": "Jakub Uhlik",
           "version": (0, 1, 3),
           "blender": (2, 80, 0),
           "location": "Properties > Scene > Import Agisoft PhotoScan Cameras",
           "warning": "work in progress",
           "wiki_url": "",
           "tracker_url": "",
           "category": "Import-Export", }

import os
import sys
import math
import xml.etree.ElementTree as etree

import bpy
import bmesh
from mathutils import Matrix, Vector
from bpy.props import PointerProperty, BoolProperty, StringProperty, FloatProperty, EnumProperty
from bpy.types import Operator, Panel, PropertyGroup
from bpy_extras.io_utils import axis_conversion
from bl_ui.properties_scene import SceneButtonsPanel


PHOTOSCAN_VERSION_COMPATIBILITY = [(1, 4, 0), (1, 5, 0)]
CHECK_VERSION = True
DEBUG = False


def log(msg, indent=0, ):
    m = "{0}> {1}".format("    " * indent, msg)
    if(DEBUG):
        print(m)


'''
class Progress():
    def __init__(self, total, indent=0, prefix="> ", ):
        self.current = 0
        self.percent = -1
        self.last = -1
        self.total = total
        self.prefix = prefix
        self.indent = indent
        self.t = "    "
        self.r = "\r"
        self.n = "\n"
    
    def step(self, numdone=1):
        self.current += numdone
        self.percent = int(self.current / (self.total / 100))
        if(self.percent > self.last):
            sys.stdout.write(self.r)
            sys.stdout.write("{0}{1}{2}%".format(self.t * self.indent, self.prefix, self.percent))
            self.last = self.percent
        if(self.percent >= 100 or self.total == self.current):
            sys.stdout.write(self.r)
            sys.stdout.write("{0}{1}{2}%{3}".format(self.t * self.indent, self.prefix, 100, self.n))
'''


def add_object(name, data, ):
    so = bpy.context.scene.objects
    for i in so:
        i.select_set(False)
    o = bpy.data.objects.new(name, data)
    context = bpy.context
    view_layer = context.view_layer
    collection = view_layer.active_layer_collection.collection
    collection.objects.link(o)
    o.select_set(True)
    view_layer.objects.active = o
    return o


def get_space3dview():
    for a in bpy.context.screen.areas:
        if(a.type == "VIEW_3D"):
            return a.spaces[0]
    return None


def activate_object(obj):
    bpy.ops.object.select_all(action='DESELECT')
    context = bpy.context
    view_layer = context.view_layer
    obj.select_set(True)
    view_layer.objects.active = obj


def switch_view_to_camera():
    s3dv = get_space3dview()
    if(s3dv is not None):
        if(s3dv.region_3d.view_perspective != 'CAMERA'):
            s3dv.region_3d.view_perspective = 'CAMERA'


def camera_list(scene):
    r = []
    for o in scene.objects:
        if(o.type == 'CAMERA'):
            r.append(o)
    r.sort(key=lambda c: c.name)
    return r


'''
def show_background_image(name):
    s3dv = get_space3dview()
    if(s3dv is not None):
        s3dv.show_background_images = True
        for im in s3dv.background_images:
            imnm = im.image.name.split(".")[0]
            if(imnm == name):
                im.show_background_image = True
            else:
                im.show_background_image = False
'''


def switch_orientation(camera):
    render = bpy.context.scene.render
    x = render.resolution_x
    y = render.resolution_y
    if(x > y and camera.data.sensor_fit == 'VERTICAL'):
        render.resolution_x = y
        render.resolution_y = x
    if(x < y and camera.data.sensor_fit == 'HORIZONTAL'):
        render.resolution_x = y
        render.resolution_y = x


class PSCVersionException(Exception):
    pass


class PSCSensor():
    def __init__(self, xml):
        self.id = int(xml.attrib["id"])
        self.label = xml.attrib["label"]
        self.type = xml.attrib["type"]
        
        res = xml.find("resolution")
        self.resolution = {'width': int(res.attrib["width"]), 'height': int(res.attrib["height"]), }
        
        props = xml.findall("property")
        self.props = {}
        for p in props:
            n = p.attrib["name"]
            v = p.attrib["value"]
            if(n == "pixel_width"):
                self.props[n] = float(v)
            elif(n == "pixel_height"):
                self.props[n] = float(v)
            elif(n == "focal_length"):
                self.props[n] = float(v)
            elif(n == "fixed"):
                self.props[n] = v
            elif(n == "layer_index"):
                self.props[n] = v
            else:
                log("PSCSensorData: unknown property name: {0} with value: {1}".format(n, v), 0, )
        
        self.calibration = {}
        cals = xml.findall("calibration")
        cal = None
        for c in cals:
            # there can be also calibration element with class="initial" attribute
            # this is when sensor is precalibrated.. if it is so, then use adjusted values
            if(c.attrib["class"] == "adjusted"):
                cal = c
        
        if(cal is None):
            # skip it and hope for the best
            self.calibration['type'] = None
            self.calibration['class'] = None
            self.calibration['resolution'] = {'width': self.props['pixel_width'], 'height': self.props['pixel_height'], }
            self.calibration['f'] = self.props['focal_length']
            self.calibration['cx'] = 0.0
            self.calibration['cy'] = 0.0
        else:
            self.calibration['type'] = cal.attrib["type"]
            self.calibration['class'] = cal.attrib["class"]
            cres = cal.find("resolution")
            self.calibration['resolution'] = {'width': int(cres.attrib["width"]), 'height': int(cres.attrib["height"]), }
            # self.calibration['fx'] = float(cal.find("fx").text)
            # self.calibration['fy'] = float(cal.find("fy").text)
            self.calibration['f'] = float(cal.find("f").text)
            try:
                self.calibration['cx'] = float(cal.find("cx").text)
                self.calibration['cy'] = float(cal.find("cy").text)
            except:
                self.calibration['cx'] = 0.0
                self.calibration['cy'] = 0.0
            # self.calibration['k1'] = float(cal.find("k1").text)
            # self.calibration['k2'] = float(cal.find("k2").text)
            # self.calibration['k3'] = float(cal.find("k3").text)
        
        if('pixel_width' not in self.props):
            self.props['pixel_width'] = self.calibration['resolution']['width']
            self.props['pixel_height'] = self.calibration['resolution']['height']
            self.props['focal_length'] = self.calibration['f']
        
        sw = 0
        sh = 0
        ver = 0
        if(ver == 0):
            # sensor w/h: resolution * single pixel size
            sw = self.props['pixel_width'] * self.calibration['resolution']['width']
            sh = self.props['pixel_height'] * self.calibration['resolution']['height']
            self.orientation = "HORIZONTAL"
            if(sw < sh):
                self.orientation = "VERTICAL"
            self.calibrated_sensor_width = sw
            self.calibrated_sensor_height = sh
            
        elif(ver == 1):
            # # principal point (image center) position * pixel size * 2
            # cx = self.calibration['cx'] * self.props['pixel_width'] * 2
            # cy = self.calibration['cy'] * self.props['pixel_height'] * 2
            # self.orientation = "HORIZONTAL"
            # if(cx < cy):
            #     self.orientation = "VERTICAL"
            # self.calibrated_sensor_width = cx
            # self.calibrated_sensor_height = cy
            
            pass
            
        else:
            class SillyException(Exception):
                pass
            raise SillyException("choose one")
        
        # average fx and fy, average pw and ph
        # fx = self.calibration['fx']
        # fy = self.calibration['fy']
        f = self.calibration['f']
        pw = self.props['pixel_width']
        ph = self.props['pixel_height']
        self.calibrated_focal_length = f * ((pw + ph) / 2)
        
        self.principal_point_x = self.calibration['cx'] * self.props['pixel_width']
        self.principal_point_y = self.calibration['cy'] * self.props['pixel_height']
        
        shift_x = ((self.calibrated_sensor_width / 2) - self.principal_point_x)
        shift_y = ((self.calibrated_sensor_height / 2) - self.principal_point_y)
        
        self.shift_x_mm = shift_x
        self.shift_y_mm = shift_y
        
        # make it normalized, shift_x: 1 in blender camera is positive horizontal offset: 1 * sensor_width, negative would be shift_x: -1
        # formula from teoplib.maths.map: vmin2 + (vmax2 - vmin2) * ((v - vmin1) / (vmax1 - vmin1))
        shift_x = shift_x / self.calibrated_sensor_width
        shift_y = shift_y / self.calibrated_sensor_height
        
        # |                   <-|         |->         <-<-| |->->
        # * -x -y             * -x +y     * +x +y     * +x +y
        # . x y               . x -y      . -x y      . -x -y
        # +-------o-------+   +-------+   +-------+   +---------------+
        # |     *         |   |       |   |       |   |     .         |
        # |       +       |   |     . |   |     * |   |       +       |
        # |         .     |   o   +   |   |   +   o   |         *     |
        # +---------------+   | *     |   | .     |   +-------o-------+
        #                     |       |   |       |
        #                     +-------+   +-------+
        
        # i think there is no way how to correctly determine this. i just assume that when shooting horizontally i am holding camera in viewer up position,
        # when vertically i rotate camera to left, viewer is on left. it would be possible to get in from image metadata but, when i rotate image
        # and develop like that, it will have orientation value: "Horizontal (normal)", that's Fun! how cool is that! but i have to check it with raw files
        # maybe it is camera 'feature', not having orientation sensor..
        # so, the result is:
        #   width > height = horizontal
        #   height > width = vertical
        #   horizontal cx, cy = +,+ = shift_x, shift_y = -,-
        #   vertical cx, cy = +,+ = shift_x, shift_y = +,-
        # we'll see if this is going to work..
        
        sign_x = 1
        if(self.principal_point_x < self.calibrated_sensor_width / 2):
            sign_x = -1
            if(self.orientation == 'VERTICAL'):
                # see above..
                sign_x = 1
        sign_y = 1
        if(self.principal_point_y < self.calibrated_sensor_height / 2):
            sign_y = -1
        
        self.shift_x = shift_x * sign_x
        self.shift_y = shift_y * sign_y
        
        self.real_megapixels = self.calibration['resolution']['width'] * self.calibration['resolution']['height'] / 1e6
        
        self._xml = xml


class PSCCamera():
    def __init__(self, xml, sensor):
        self.id = int(xml.attrib["id"])
        self.label = xml.attrib["label"]
        self.sensor_id = int(xml.attrib["sensor_id"])
        
        # if(xml.attrib["enabled"] == "true"):
        #     self.enabled = True
        # else:
        #     self.enabled = False
        self.enabled = True
        
        self.resolution = {"width": sensor.resolution['width'], "height": sensor.resolution['height'], }
        
        try:
            t = xml.find("transform").text
            l = t.split(" ")
            v = []
            for i in range(len(l)):
                v.append(float(l[i]))
            m = []
            i = 0
            while(i < 16):
                m.append(tuple([v[i], v[i + 1], v[i + 2], v[i + 3]]))
                i += 4
            matrix = Matrix((m[0], m[1], m[2], m[3]))
            self.transform = t
        except Exception as e:
            matrix = Matrix()
            self.transform = None
            self.enabled = False
        
        conversion_matrix = axis_conversion(from_forward='Z', from_up='-Y', to_forward='-Z', to_up='Y').to_4x4()
        # self.matrix = matrix * conversion_matrix
        self.matrix = matrix @ conversion_matrix
        
        self._xml = xml


class PSCChunk():
    def __init__(self, xml, id):
        self.xml = xml
        self.id = id
        
        self.sensors = []
        self.cameras = []
        
        sens = self.xml.findall(".//sensor")
        for s in sens:
            try:
                sd = PSCSensor(s)
            except:
                sd = None
            self.sensors.append(sd)
        
        self.cameras = []
        cams = self.xml.findall(".//camera")
        for c in cams:
            sensor = self.sensors[int(c.attrib["sensor_id"])]
            if(sensor is None):
                # TODO: unparsed sensor, skip that, deal with that later somehow.. at least notify user
                continue
            cam = PSCCamera(c, sensor)
            self.cameras.append(cam)
        
        self.region = {}
        reg = self.xml.find(".//region")
        
        c = reg.find("center").text.split(" ")
        cf = [float(v) for v in c]
        self.region['center'] = cf
        
        s = reg.find("size").text.split(" ")
        sf = [float(v) for v in s]
        self.region['size'] = sf
        
        r = reg.find("R").text.split(" ")
        rf = [float(v) for v in r]
        m = []
        i = 0
        while(i < 9):
            m.append(tuple([rf[i], rf[i + 1], rf[i + 2]]))
            i += 3
        matrix = Matrix((m[0], m[1], m[2]))
        
        conversion_matrix = axis_conversion(from_forward='Z', from_up='-Y', to_forward='-Z', to_up='Y').to_3x3()
        matrix = matrix @ conversion_matrix
        self.region['R'] = matrix


class PSCMakeCameras():
    def __init__(self, xml_path, matrix, camera_draw_size=0.5, planes=False, chunk_regions=True, correct_principal_point=False, ):
        self.xml_path = os.path.realpath(xml_path)
        self.matrix = matrix
        self.planes = planes
        self.camera_draw_size = camera_draw_size
        self.chunk_regions = chunk_regions
        self.correct_principal_point = correct_principal_point
        self._make()
    
    def _make(self):
        log("PSCMakeCameras:", 0)
        log("loading and parsing xml from: {0}".format(self.xml_path), 1)
        self._load_parse_xml()
        
        self.cameras = []
        
        for i, ch in enumerate(self.chunks):
            log("creating chunk '{0}' ({1}/{2})".format(ch.id, i + 1, len(self.chunks)), 1)
            empty = add_object(ch.id, None)
            
            if(self.planes):
                log("creating cameras: (planes: {0})".format(self.planes), 2)
            else:
                log("creating cameras:", 2)
            camera_objects = self._create_cameras(ch.cameras, ch.sensors, empty)
            self.cameras.extend(camera_objects)
            
            if(self.chunk_regions):
                log("creating region..", 2)
                region = self._create_region(ch.id, ch.region, empty)
            
            log("setting parameters..", 2)
            ch.empty = empty
            ch.camera_objects = camera_objects
            if(self.chunk_regions):
                ch.region_object = region
            
            empty.matrix_world = self.matrix
    
    def _load_parse_xml(self):
        self._tree = etree.parse(self.xml_path)
        
        self.chunks = []
        chunks = self._tree.findall(".//chunk")
        for i, chunk in enumerate(chunks):
            ch = PSCChunk(chunk, "chunk-{0}".format(i))
            self.chunks.append(ch)
    
    def _create_cameras(self, cameras, sensors, empty):
        # calculate camera draw size from matrix to get real size in viewport
        l, r, s = self.matrix.decompose()
        ms = Matrix.Scale(s.x, 4)
        l = Vector((self.camera_draw_size, 0, 0))
        v = ms.inverted() @ l
        ds = v.x
        self.camera_draw_size_relative = ds
        
        # prgr = Progress(len(cameras), 3)
        
        camera_objects = []
        
        for c in cameras:
            cnm = str(c.label).split(".")[0]
            bc = bpy.data.cameras.new(cnm)
            
            bc.import_photoscan_cameras.include = True
            # bc.import_photoscan_cameras.image = c.label
            
            bco = add_object(cnm, bc)
            bco.show_name = True
            
            bc.lens_unit = 'MILLIMETERS'
            current_sensor = sensors[c.sensor_id]
            bc.lens = current_sensor.calibrated_focal_length
            bc.sensor_fit = current_sensor.orientation
            
            bc.show_name = True
            
            bc.sensor_width = current_sensor.calibrated_sensor_width
            bc.sensor_height = current_sensor.calibrated_sensor_height
            
            if(self.correct_principal_point):
                bc.shift_x = current_sensor.shift_x
                bc.shift_y = current_sensor.shift_y
            
            bc.display_size = self.camera_draw_size_relative
            
            if(self.planes):
                self._set_render_params(sensors[c.sensor_id])
                
                # how blender camera is drawn: when draw_size is 1.0, longer side of frame is 1 unit, how far from origin depends on focal length
                # the distance is then calculated as: (draw_size / 2) / math.tan(camera.angle / 2)
                camera_frame_distance = ((bc.display_size / 2) / math.tan(bc.angle / 2)) * 2
                # lets add a bit to it
                plane_distance = camera_frame_distance * 1.25
                
                ch = plane_distance * math.tan(bc.angle_y / 2)
                cv = plane_distance * math.tan(bc.angle_x / 2)
                
                # 3-------------2
                # |             |
                # |      *      |
                # |             |
                # 0-------------1
                
                va = Vector((-cv, -ch, -plane_distance))
                vb = Vector((cv, -ch, -plane_distance))
                vc = Vector((cv, ch, -plane_distance))
                vd = Vector((-cv, ch, -plane_distance))
                verts = [va, vb, vc, vd]
                
                pnm = "{0}_image_plane".format(cnm)
                me = bpy.data.meshes.new(pnm)
                me.from_pydata(verts, [], [(0, 1, 2, 3)])
                
                uvnm = "UVMap"
                # me.uv_textures.new(uvnm)
                me.uv_layers.new(name=uvnm)
                loops = me.uv_layers[0].data
                loops[0].uv = Vector((0.0, 0.0))
                loops[1].uv = Vector((1.0, 0.0))
                loops[2].uv = Vector((1.0, 1.0))
                loops[3].uv = Vector((0.0, 1.0))
                
                meo = add_object(pnm, me)
                meo.parent = bco
            else:
                self._set_render_params(sensors[cameras[0].sensor_id])
                
            bco.matrix_world = c.matrix
            bco.parent = empty
            
            camera_objects.append(bco)
            
            # prgr.step()
        
        return camera_objects
    
    def _set_render_params(self, sensor):
        render = bpy.context.scene.render
        render.resolution_x = sensor.calibration['resolution']['width']
        render.resolution_y = sensor.calibration['resolution']['height']
        render.resolution_percentage = 100
    
    def _create_region(self, id, region, empty):
        center = region["center"]
        size = region["size"]
        r = region["R"]
        
        # 1 unit cube
        l = 1.0 / 2
        dv = [(+l, +l, -l),
              (+l, -l, -l),
              (-l, -l, -l),
              (-l, +l, -l),
              (+l, +l, +l),
              (+l, -l, +l),
              (-l, -l, +l),
              (-l, +l, +l), ]
        df = [(0, 1, 2, 3),
              (4, 7, 6, 5),
              (0, 4, 5, 1),
              (1, 5, 6, 2),
              (2, 6, 7, 3),
              (4, 0, 3, 7), ]
        
        me = bpy.data.meshes.new("{0}_region".format(id))
        me.from_pydata(dv, [], df)
        
        conversion_matrix = axis_conversion(from_forward='Z', from_up='-Y', to_forward='-Z', to_up='Y').to_4x4()
        
        o = add_object("{0}_region".format(id), me)
        
        mt = Matrix.Translation(Vector(center)).to_4x4()
        mr = Matrix(r.to_4x4() @ conversion_matrix).inverted()
        ms = Matrix(((size[0], 0, 0), (0, size[1], 0), (0, 0, size[2]), )).to_4x4()
        m = mt @ mr @ ms
        
        o.matrix_world = m
        
        o.parent = empty
        o.display_type = 'WIRE'
        
        # delete faces, why are they even created? because i am lazy
        me = o.data
        bm = bmesh.new()
        bm.from_mesh(me)
        for f in bm.faces:
            f.select_set(True)
        # DEL_ONLYFACES > context=3
        # bmesh.ops.delete(bm, geom=bm.faces, context='FACES', )
        bmesh.ops.delete(bm, geom=bm.faces, context='FACES_ONLY', )
        o.data = bm.to_mesh(me)
        
        return o


class PSCXMLImport():
    def __init__(self, xml_path, matrix, camera_draw_size=0.5, load_images=True, images_directory=None, image_extension=".jpg", background_images=True, image_planes=False, chunk_regions=True, correct_principal_point=False, version_check=True, ):
        if(xml_path is not None):
            if(type(xml_path) is str):
                if(xml_path != ""):
                    if(os.path.exists(xml_path)):
                        self.xml_path = os.path.realpath(xml_path)
                        
                        def check_extension(check_path, check_ext, ):
                            if(check_ext.startswith(".") is False):
                                check_ext = ".{0}".format(check_ext)
                            head, tail = os.path.split(check_path)
                            name, ext = os.path.splitext(tail)
                            if(ext == check_ext):
                                return True
                            return False
                        
                        if(check_extension(self.xml_path, ".xml") is False):
                            log("{}: WARNING: does not seem to be a .xml file ({})".format(self.__class__.__name__, self.xml_path), 1, )
                    else:
                        raise ValueError("{}: file at xml_path does not exist".format(self.__class__.__name__))
                else:
                    raise ValueError("{}: xml_path is an empty string".format(self.__class__.__name__))
            else:
                raise TypeError("{}: xml_path is not a string".format(self.__class__.__name__))
        else:
            raise TypeError("{}: xml_path is None".format(self.__class__.__name__))
        
        if(matrix is not None):
            if(type(matrix) is Matrix):
                self.matrix = matrix
            else:
                raise TypeError("{}: matrix is not a Matrix".format(self.__class__.__name__))
        else:
            raise TypeError("{}: matrix is None".format(self.__class__.__name__))
        
        if(camera_draw_size is not None):
            if(type(camera_draw_size) is float):
                if(camera_draw_size > 0):
                    self.camera_draw_size = camera_draw_size
                else:
                    raise ValueError("{}: camera_draw_size is zero or negative".format(self.__class__.__name__))
            else:
                raise TypeError("{}: camera_draw_size is not a float".format(self.__class__.__name__))
        else:
            raise TypeError("{}: camera_draw_size is None".format(self.__class__.__name__))
        
        if(type(load_images) is bool):
            self.load_images = load_images
        else:
            raise TypeError("{}: load_images is not a bool".format(self.__class__.__name__))
        
        if(self.load_images):
            if(images_directory is not None):
                if(type(images_directory) is str):
                    if(images_directory != ""):
                        if(os.path.exists(images_directory)):
                            if(os.path.isdir(images_directory)):
                                self.images_directory = images_directory
                            else:
                                raise ValueError("{}: images_directory is not a directory".format(self.__class__.__name__))
                        else:
                            raise ValueError("{}: images_directory does not exist".format(self.__class__.__name__))
                    else:
                        raise ValueError("{}: images_directory is an empty string".format(self.__class__.__name__))
                else:
                    raise TypeError("{}: images_directory is not a string".format(self.__class__.__name__))
            else:
                raise TypeError("{}: images_directory is None".format(self.__class__.__name__))
            
            if(image_extension is not None):
                if(type(image_extension) is str):
                    if(image_extension != ""):
                        if(not image_extension.startswith(".")):
                            image_extension = ".{0}".format(image_extension)
                        self.image_extension = image_extension.lower()
                    else:
                        raise ValueError("{}: image_extension is an empty string".format(self.__class__.__name__))
                else:
                    raise TypeError("{}: image_extension is not a string".format(self.__class__.__name__))
            else:
                raise TypeError("{}: image_extension is None".format(self.__class__.__name__))
            
            if(type(background_images) is bool):
                self.background_images = background_images
            else:
                raise TypeError("{}: background_images is not a bool".format(self.__class__.__name__))
            
            if(type(image_planes) is bool):
                self.image_planes = image_planes
            else:
                raise TypeError("{}: image_planes is not a bool".format(self.__class__.__name__))
        else:
            self.images_directory = None
            self.background_images = False
            self.image_planes = False
        
        if(type(chunk_regions) is bool):
            self.chunk_regions = chunk_regions
        else:
            raise TypeError("{}: chunk_regions is not a bool".format(self.__class__.__name__))
        
        if(type(correct_principal_point) is bool):
            self.correct_principal_point = correct_principal_point
        else:
            raise TypeError("{}: correct_principal_point is not a bool".format(self.__class__.__name__))
        
        if(type(version_check) is bool):
            self.version_check = version_check
        else:
            raise TypeError("{}: version_check is not a bool".format(self.__class__.__name__))
        
        self._make()
    
    def _make(self):
        log("{}:".format(self.__class__.__name__), 0)
        
        # version check
        self._tree = etree.parse(self.xml_path)
        
        if(self.version_check):
            root = self._tree.getroot()
            ver_str = root.attrib["version"]
            ver_split = ver_str.split(".")
            ver = tuple(int(i) for i in ver_split)
            if(ver not in PHOTOSCAN_VERSION_COMPATIBILITY):
                raise PSCVersionException("{}: incompatible xml version: ({})".format(self.__class__.__name__, ver))
            else:
                log("checking file version: {0}: {1}".format(ver, "ok"))
        
        # make cameras
        do_planes = False
        if(self.load_images and self.image_planes):
            do_planes = True
        
        self.psc = PSCMakeCameras(self.xml_path, self.matrix, self.camera_draw_size, do_planes, self.chunk_regions, self.correct_principal_point, )
        
        # images
        if(self.load_images):
            def walk_dir(p):
                r = {"files": [],
                     "dirs": [], }
                for (root, dirs, files) in os.walk(p):
                    r["files"].extend(files)
                    r["dirs"].extend(dirs)
                    break
                return r
            
            files = walk_dir(self.images_directory)['files']
            
            fls = []
            for f in files:
                if(os.path.splitext(f)[1].lower() == self.image_extension):
                    fls.append(f)
            files = fls
            
            self.images = []
            cam_names = [c.name.split(".")[0] for c in bpy.data.cameras]
            redundant_images_log = []
            for f in files:
                fnm = os.path.splitext(f)[0]
                if(fnm in cam_names):
                    fi = cam_names.index(fnm)
                    cam_names = cam_names[:fi] + cam_names[fi + 1:]
                else:
                    redundant_images_log.append(f)
                    continue
                
                # load image
                p = os.path.join(self.images_directory, f)
                im = bpy.data.images.load(p)
                self.images.append(im)
            
            # any camera without an image?
            missing_images_log = []
            if(len(cam_names) != 0):
                for cnm in cam_names:
                    missing_images_log.append(cnm)
            
            # set background images
            if(self.background_images):
                
                # cam_names = [c.name.split(".")[0] for c in bpy.data.cameras]
                cam_names = [c.name.split(".")[0] for c in bpy.data.cameras]
                
                # # set images as viewport background images, but turn off visibility
                # s3dv = get_space3dview()
                # s3dv.show_background_images = True
                
                # for cn in cam_names:
                #     cd = bpy.data.cameras[cn]
                #     # if(cn in [i.name for i in self.images]):
                #     cd.show_background_images = True
                #     bi = cd.background_images.new()
                #     bi.image = self.images[cn]
                #     bi.draw_depth = 'FRONT'
                
                for im in self.images:
                    nm = im.name.split(".")[0]
                    cam = bpy.data.cameras.get(nm)
                    if(cam is not None):
                        cam.show_background_images = True
                        bi = cam.background_images.new()
                        bi.image = im
                        bi.display_depth = 'FRONT'
                
                # for im in self.images:
                #     bim = s3dv.background_images.new()
                #     bim.view_axis = 'CAMERA'
                #     bim.show_background_image = False
                #     bim.show_expanded = False
                #     # image not shown from the start
                #     bim.show_background_image = False
                #     bim.draw_depth = 'FRONT'
                #     bim.image = im
            
            # make image planes
            if(self.image_planes):
                # planes are already created, just assign images to uvs
                for o in bpy.data.objects:
                    if(o.type == 'CAMERA'):
                        if(len(o.children) == 1):
                            if(o.children[0].name.endswith("_image_plane")):
                                me = o.children[0].data
                                uvf = me.uv_textures["UVMap"].data[0]
                                im = None
                                for img in self.images:
                                    if(img.name.split(".")[0] == o.name):
                                        im = img
                                        break
                                uvf.image = im
        
        # activate last chunk empty
        activate_object(self.psc.chunks[-1].empty)


class PSC_import_properties(PropertyGroup):
    xml_path: StringProperty(name="Cameras XML", default="", subtype='FILE_PATH', description="Path to Agisoft PhotoScan cameras xml file", )
    camera_draw_size: FloatProperty(name="Camera Display Size", description="Size of imported cameras in viewport", default=0.25, precision=3, )
    
    load_images: BoolProperty(name="Load Camera Images", default=True, description="Load camera images", )
    images_directory: StringProperty(name="Images Directory", default="", subtype='DIR_PATH', description="Path to images directory", )
    image_extension: EnumProperty(name="Image Extension", items=[('TIFF', "TIFF (*.tif)", ""),
                                                                 ('TIFF2', "TIFF (*.tiff)", ""),
                                                                 ('JPEG', "JPEG (*.jpg)", ""),
                                                                 ('JPEG2', "JPEG (*.jpeg)", ""),
                                                                 ('PNG', "PNG (*.png)", ""),
                                                                 ('BMP', "BMP (*.bmp)", ""),
                                                                 ('OPENEXR', "OpenEXR (*.exr)", ""),
                                                                 ('TARGA', "TARGA (*.tga)", "")], default='TIFF', description="", )
    background_images: BoolProperty(name="Assign Images to Cameras", default=True, description="Assign images to cameras", )
    background_image_alpha: FloatProperty(name="Alpha", description="", default=0.5, precision=3, min=0.0, max=1.0, subtype='PERCENTAGE', )
    background_image_depth: EnumProperty(name="Depth", items=[('BACK', "Back", ""), ('FRONT', "Front", ""), ], default='FRONT', description="", )
    
    image_planes: BoolProperty(name="Create Image Planes", default=False, description="Create mesh planes in front of cameras and assign images to them", )
    
    chunk_regions: BoolProperty(name="Create Chunk Region Borders", default=True, description="Create mesh at borders of chunk region", )
    align_to_active: BoolProperty(name="Align to Active Object", default=False, description="Copy transformation from active object", )
    
    @classmethod
    def register(cls):
        bpy.types.Scene.import_photoscan_cameras = PointerProperty(type=cls)
    
    @classmethod
    def unregister(cls):
        del bpy.types.Scene.import_photoscan_cameras


class PSC_camera_properties(PropertyGroup):
    include: BoolProperty(name="Include", default=False, options={'HIDDEN'}, )
    image: StringProperty(name="Image", default="", options={'HIDDEN'}, )
    
    @classmethod
    def register(cls):
        bpy.types.Camera.import_photoscan_cameras = PointerProperty(type=cls)
    
    @classmethod
    def unregister(cls):
        del bpy.types.Camera.import_photoscan_cameras


class PSC_OT_import(Operator):
    bl_idname = "import_scene.photoscan_cameras"
    bl_label = "Agisoft PhotoScan Cameras (.xml)"
    bl_description = ""
    bl_options = {'UNDO', }
    
    @classmethod
    def poll(cls, context):
        ps = bpy.context.scene.import_photoscan_cameras
        return (ps and ps.xml_path != '')
    
    def execute(self, context):
        m = Matrix()
        ps = bpy.context.scene.import_photoscan_cameras
        if(ps.align_to_active):
            ao = context.view_layer.objects.active
            if(ao is not None):
                m = ao.matrix_world.copy()
        ed = {'TIFF': ".tiff", 'TIFF2': ".tif", 'JPEG': ".jpg", 'JPEG2': ".jpeg", 'PNG': ".png", 'BMP': ".bmp", 'OPENEXR': ".exr", 'TARGA': ".tga", }
        e = ed[ps.image_extension]
        d = {'xml_path': os.path.realpath(bpy.path.abspath(ps.xml_path)),
             'matrix': m,
             'camera_draw_size': ps.camera_draw_size,
             'load_images': ps.load_images,
             'images_directory': os.path.realpath(bpy.path.abspath(ps.images_directory)),
             'image_extension': e,
             'background_images': ps.background_images,
             # 'image_planes': ps.image_planes,
             'image_planes': False,
             'chunk_regions': ps.chunk_regions,
             'correct_principal_point': False,
             'version_check': CHECK_VERSION, }
        o = PSCXMLImport(**d)
        return {'FINISHED'}


'''
class PSC_OT_align_cameras_to_mesh(Operator):
    bl_idname = "import_scene.photoscan_cameras_align"
    bl_label = "Align Cameras to Mesh"
    bl_description = "Align cameras to active mesh. First select chunk cameras empty object holder, then mesh. World matrix will be copied to cameras empty with corrected axes."
    bl_options = {'UNDO', }
    
    @classmethod
    def poll(cls, context):
        obs = [o for o in context.scene.objects if o.select_get()]
        if(len(obs) != 2):
            return False
        o = context.active_object
        c = [i for i in obs if i is not o][0]
        return (o and o.type == 'MESH' and o in obs and c and c.type == 'EMPTY')
    
    def execute(self, context):
        obs = [o for o in context.scene.objects if o.select_get()]
        o = context.active_object
        c = [i for i in obs if i is not o][0]
        m = o.matrix_world.copy()
        c.matrix_world = m
        return {'FINISHED'}
'''


class PSC_OT_next_camera(Operator):
    bl_idname = "import_scene.photoscan_cameras_next"
    bl_label = "Next Camera"
    
    @classmethod
    def poll(cls, context):
        cams = camera_list(context.scene)
        if(len(cams) > 1):
            return True
        return False
    
    def execute(self, context):
        ao = context.active_object
        mode = ao.mode
        bpy.ops.object.mode_set(mode='OBJECT')
        
        cams = camera_list(context.scene)
        current = context.scene.camera
        
        found = False
        for i, c in enumerate(cams):
            if(c == current):
                found = True
                break
        if(found):
            if(i == len(cams) - 1):
                i = 0
            else:
                i += 1
        else:
            i = 0
        
        # print(cams[i].data.import_photoscan_cameras.include)
        # print(cams[i].data.import_photoscan_cameras.image)
        
        activate_object(cams[i])
        context.scene.camera = cams[i]
        switch_orientation(cams[i])
        # show_background_image(cams[i].name)
        switch_view_to_camera()
        
        activate_object(ao)
        bpy.ops.object.mode_set(mode=mode)
        
        return {'FINISHED'}


class PSC_OT_prev_camera(Operator):
    bl_idname = "import_scene.photoscan_cameras_prev"
    bl_label = "Previous Camera"
    
    @classmethod
    def poll(cls, context):
        cams = camera_list(context.scene)
        if(len(cams) > 1):
            return True
        return False
    
    def execute(self, context):
        ao = context.active_object
        mode = ao.mode
        bpy.ops.object.mode_set(mode='OBJECT')
        
        cams = camera_list(context.scene)
        current = context.scene.camera
        
        found = False
        for i, c in enumerate(cams):
            if(c == current):
                found = True
                break
        if(found):
            if(i == 0):
                i = len(cams) - 1
            else:
                i -= 1
        else:
            i = 0
        
        activate_object(cams[i])
        context.scene.camera = cams[i]
        switch_orientation(cams[i])
        # show_background_image(cams[i].name)
        switch_view_to_camera()
        
        activate_object(ao)
        bpy.ops.object.mode_set(mode=mode)
        
        return {'FINISHED'}


class PSC_PT_import_panel(SceneButtonsPanel, Panel):
    bl_label = "Import PhotoScan Cameras"
    bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        return True
    
    def draw(self, context):
        l = self.layout
        ps = bpy.context.scene.import_photoscan_cameras
        sub = l.column()
        
        def prop_name(cls, prop, colon=False, ):
            for p in cls.bl_rna.properties:
                if(p.identifier == prop):
                    if(colon):
                        return "{}:".format(p.name)
                    return p.name
            return ''
        
        f = 0.33
        r = sub.row(align=True, )
        s = r.split(factor=f)
        s.label(text=prop_name(ps, 'xml_path', True, ))
        s = s.split(factor=1.0)
        r = s.row(align=True, )
        # c = r.column(align=True)
        r.prop(ps, 'xml_path', text='', )
        
        # sub.prop(ps, 'xml_path')
        sub.prop(ps, 'camera_draw_size')
        
        b = sub.box()
        
        b.prop(ps, 'load_images')
        
        c = b.column()
        # c.prop(ps, 'images_directory')
        
        # f = 0.33
        r = c.row(align=True, )
        s = r.split(factor=f)
        s.label(text=prop_name(ps, 'images_directory', True, ))
        s = s.split(factor=1.0)
        r = s.row(align=True, )
        # c = r.column(align=True)
        r.prop(ps, 'images_directory', text='', )
        
        # c.prop(ps, 'image_extension')
        r = c.row(align=True, )
        s = r.split(factor=f)
        s.label(text=prop_name(ps, 'image_extension', True, ))
        s = s.split(factor=1.0)
        r = s.row(align=True, )
        r.prop(ps, 'image_extension', text='', )
        
        # c.prop(ps, 'background_images')
        # c.prop(ps, 'image_planes')
        c.prop(ps, 'background_image_alpha')
        # c.prop(ps, 'background_image_depth')
        r = c.row(align=True, )
        s = r.split(factor=f)
        s.label(text=prop_name(ps, 'background_image_depth', True, ))
        s = s.split(factor=1.0)
        r = s.row(align=True, )
        r.prop(ps, 'background_image_depth', text='', )
        
        c.enabled = False
        if(ps.load_images):
            c.enabled = True
        
        sub.prop(ps, 'chunk_regions')
        sub.prop(ps, 'align_to_active')
        sub.operator('import_scene.photoscan_cameras', text="Import", )


class PSC_PT_utilities_panel(SceneButtonsPanel, Panel):
    bl_label = "PhotoScan Cameras Utilities"
    bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        return True
    
    def draw(self, context):
        l = self.layout
        sub = l.column()
        # sub.operator('import_scene.photoscan_cameras_align')
        r = sub.row(align=True)
        r.operator('import_scene.photoscan_cameras_prev')
        r.operator('import_scene.photoscan_cameras_next')


classes = (
    PSC_import_properties,
    PSC_camera_properties,
    PSC_OT_import,
    # PSC_OT_align_cameras_to_mesh,
    PSC_OT_next_camera,
    PSC_OT_prev_camera,
    PSC_PT_import_panel,
    PSC_PT_utilities_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
