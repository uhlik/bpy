### Contents:

### addons for blender 2.80

* [Point Cloud Visualizer](#point-cloud-visualizer-for-blender-280)
* [Tube UV Unwrap](#tube-uv-unwrap-for-blender-280)
* [Fast Wavefront^2](#fast-wavefront2-for-blender-280)
* [Import Agisoft PhotoScan Cameras](#import-agisoft-photoscan-cameras)
* [Carbon Tools](#carbon-tools)
* [Time Tracker](#time-tracker-for-blender-280)

### addons for blender 2.7x

* [Point Cloud Visualizer](#point-cloud-visualizer-for-blender-27x)
* [OpenGL Lights](#opengl-lights)
* [Fast Wavefront (.obj)](#fast-wavefront-for-blender-27x)
* [UV Equalize](#uv-equalize)
* [Tube UV Unwrap](#tube-uv-unwrap-for-blender-27x)
* [Time Tracker](#time-tracker)

***
***

### addons for blender 2.80

***
***

## [Point Cloud Visualizer](https://raw.githubusercontent.com/uhlik/bpy/master/view3d_point_cloud_visualizer.py) (for blender 2.80)

**Display, render and convert to mesh colored point cloud PLY files.**

Display colored point cloud PLY in Blender's 3d viewport. Optionally render point cloud to png sequence or convert to various mesh types with vertex colors for regular rendering. 

Works with any PLY file with 'x, y, z, nx, ny, nz, red, green, blue' vertex values. Vertex normals and colors are optional. Color values must be in 0-255 range.

![Point Cloud Visualizer](https://raw.githubusercontent.com/uhlik/bpy/master/x/pcv-0.8.9.gif)

![Point Cloud Visualizer](https://raw.githubusercontent.com/uhlik/bpy/master/x/pcv-0.8.9-full.png)

### General info

##### Usage:

* Install and activate addon in a usual way.
* Add any object type to scene.
* Go to 3d View Sidebar (N) > `Point Cloud Visualizer` panel, click file browser icon, select ply file, click `Load PLY`.
* Click `Draw` button to display point cloud, `Erase` to hide point cloud. Adjust percentage of displayed points with `Display` and point size with `Size`.
* Display point normals as lines - click `Normal` icon, adjust line length with `Length` next to it. *Pro tip: for large clouds, set Display to some small percentage, adjust Length to appropriate value and then set Display back.*
* Transforming parent object transforms point cloud as well.
* `Illumination` works only when vertex normals are present
* When vertex colors are missing, cloud will be displayed in uniform gray, in this case you can enable `Illumination` to have better cloud view
* Single point cloud can be rendered on transparent and composed over regular render
* For rendering in regular render engine you can convert cloud to colored mesh

##### Display Options:

* `Display` - percentage of displayed points
* `Size` - point size in pixels
* `Normals` - display point normals as lines, adjust line length with `Length` next to it
* `Illumination` - enable extra illumination, works only when vertex normals can be loaded
* `Light Direction` - light direction
* `Light Intensity` - light intensity
* `Shadow Intensity` - shadow intensity

### Point cloud rendering

Currently only sigle point cloud per render/frame is supported. If you need more clouds at once, select another cloud parent and re-render with different suffix in `Render` subpanel. Output image is RGBA 8bit PNG - transparent background with colored point cloud, which can be composed over something else later.

##### Usage:

* Load and display ply first.
* Make a camera and adjust as needed.
* Set render image size in `Properties > Output > Dimensions`. Resolution X, Y and % are used.
* Set render path in `Properties > Output > Output`. Just path is used.
* Select cloud parent object, set point size with `Size` or percentage of rendered points with `Count`
* If `Illumination` is enabled it will be rendered as well
* Hit `Render` or `Animation`

##### Render options:

* `Size` - point render size in pixels
* `Count` - percentage of rendered points
* `Suffix` - rendered image filename suffix. If filename in `Output` path is defined result filename will be `NAME_SUFFIX_######.png`, if only path is given, result is `SUFFIX_######.png`
* `Leading Zeros` - image filename frame number leading zeros count

### Point cloud to mesh conversion:

Convert point cloud to mesh. May result in very large meshes, e.g. 1m point cloud to cubes = 8m poly mesh. Depending on what point cloud data is available and desired mesh type, some options may not be enabled.

* `Type` - Instance mesh type, Vertex, Equilateral Triangle, Tetrahedron, Cube or Ico Sphere
* `All`, `Subset` - Use all points or random subset of by given percentage
* `Size` - Mesh instance size, internal instanced mesh has size 1.0 so if you set size to 0.01, resulting instances will have actual size of 0.01 event when cloud is scaled
* `Align To Normal` - Align instance to point normal, e.g. tetrahedron point will align to normal, triangle plane will align to normal etc.
* `Colors` - Assign point color to instance vertex colors, each instance will be colored by point color (except vertices)

### Addon Preferences:

* `Default Color` - Default color to be used upon loading PLY to cache when vertex colors are missing
* `Normal Color` - Display color for vertex normals

### Changelog:

* 0.8.10 fixes
* 0.8.9 ui tweaks, code cleanup
* 0.8.8 refactored convert to mesh
* 0.8.7 fixed vcols bug in convert
* 0.8.6 ui tweaks, a few minor optimizations
* 0.8.5 convert to mesh all or subset
* 0.8.4 preferences, ui tweaks
* 0.8.3 display normals
* 0.8.2 fixed shader unknown attribute name
* 0.8.1 fixed ply with alpha, fixed convert to mesh when normals or colors are missing
* 0.8.0 convert to mesh
* 0.7.2 ui tweaks
* 0.7.1 viewport performance fixes
* 0.7.0 ascii ply support
* 0.6.6 fixed drawing after undo/redo
* 0.6.5 point cloud illumination
* 0.6.4 refactored draw handlers, fixed occasional crash on erase
* 0.6.3 added percentage of rendered points, fixed render colors to look the same as in viewport
* 0.6.2 fixed point size display in viewport, separated view and render point size
* 0.6.1 single cloud rendering almost completely rewritten to be better and faster
* 0.6.0 single cloud rendering
* 0.5.2 refactored some logic, removed icons from buttons
* 0.5.1 load ply without vertex colors, uniform grey will be used
* 0.5.0 performance improvements using numpy for loading and processing data
* 0.4.6 fixed crash when parent object is deleted while drawing, fixed removal of loaded data when parent is deleted
* 0.4.5 added 'Display' percentage, better error handling during .ply loading
* 0.4.0 almost complete rewrite for blender 2.80, performance improvements using shaders, simplified ui
* 0.3.0 new ply loader, can be used with any binary ply file with vertex coordinates and colors
* 0.2.0 display percentage
* 0.1.0 first release

##### Known bugs:

* If you duplicate object with cloud, duplicate will still control the original one until you load a different one. Currently there is no reliable way (as far as i know) to get unique id of an object and therefore no way to tell to which object stored properties (e.g. path to ply) belong.

[BlenderArtist.org thread](https://blenderartists.org/forum/showthread.php?416158-Addon-Point-Cloud-Visualizer)

***

## [Tube UV Unwrap](https://raw.githubusercontent.com/uhlik/bpy/master/uv_tube_unwrap.py) (for blender 2.80)

UV unwrap tube-like meshes (all quads, no caps, fixed number of vertices in each ring)

![Tube UV Unwrap](https://raw.githubusercontent.com/uhlik/bpy/master/x/tuv280.gif)

notes:

* Works only on tube-like parts of mesh defined by selection and active vertex (therefore you must be in vertex selection mode) and the selection must have a start and an end ring. Tube-like mesh is: all quads, no caps, fixed number of vertices in each ring. (Best example of such mesh is mesh circle extruded several times or beveled curve (not cyclic) converted to mesh.) There must be an active vertex on one of the boundary loops in selection. This active vertex define place where mesh will be 'cut' - where seam will be placed.
* Result is rectangular UV for easy texturing, scaled to fit square, horizontal and vertical distances between vertices are averaged and proportional to each other. 

usage:

1. tab to Edit mode
2. select part of mesh you want to unwrap, tube type explained above
3. make sure your selection has boundaries and there is an active vertex on one border of selection
4. hit "U" and select "Tube UV Unwrap"
5. optionally check/uncheck 'Mark Seams' or 'Flip' in operator properties

changelog:

* 0.3.0 blender 2.8 update

[BlenderArtist.org thread](http://blenderartists.org/forum/showthread.php?339782-UV-Equalize-and-Tube-Unwrap-addons)

***

## [Fast Wavefront^2](https://github.com/uhlik/bpy/tree/master/io_mesh_fast_obj) (for blender 2.80)

**Import/Export single mesh as Wavefront OBJ. Fast. Now with Cython. Binaries not included.**

Only active mesh is exported. Only single mesh is expected on import. Supported obj features: UVs, normals, vertex colors using MRGB format (ZBrush) or 'Extended' format (import only) where vertex is defined as (x,y,z,r,g,b).

![Fast Wavefront^2](https://raw.githubusercontent.com/uhlik/bpy/master/x/obj2.png)

changelog:

* 0.3.3 fallback python export implementation in case cython module is not available
* 0.3.2 import 'extended' vertex colors (x,y,z,r,g,b), optionally apply gamma correction
* 0.3.1 import obj (python only)
* 0.3.0 export implemented in cython
* 0.2.0 ported to blender 2.80

**requirements:**

* python 3.7.0 (the same as shipped with blender 2.8)
* Cython (easiest is to install with pip)

**installation on mac (win/linux should be very similar):**

1. download repository and copy whole directory `io_mesh_fast_obj` to `/Users/*USERNAME*/Library/Application Support/Blender/2.80/scripts/addons/`
2. in terminal cd to `/Users/*USERNAME*/Library/Application Support/Blender/2.80/scripts/addons/io_mesh_fast_obj/`
    1. `$ git clone http://git.blender.org/blender.git`
    2. `$ python3 setup.py build_ext --inplace`
3. now delete `blender` directory, it is no longer needed until blender is updated, then you (might) need to repeat the process

***

## [Import Agisoft PhotoScan Cameras](https://github.com/uhlik/bpy/tree/master/io_import_photoscan_cameras.py)

Import cameras from Agisoft PhotoScan xml. Works with xml version 1.4.0 which is exported from PhotoScan 1.4.x versions and xml versions 1.5.0 from Agisoft Metashape 1.5.x versions. If you want to have images actually aligned with model, undistort images first. This is done in PhotoScan by `Export > Undistort Photos..`. Because you can't in Blender set resolution for cameras independently, xml with different cameras or image resolutions might not work well.

![Import Agisoft PhotoScan Cameras](https://raw.githubusercontent.com/uhlik/bpy/master/x/pscamerasui.png)

usage:

1. go to `Properties > Scene > Import Agisoft PhotoScan Cameras` panel
2. **Cameras XML**: set path to xml
3. set import options:
    * **Camera Display Size**: size of imported cameras in viewport
    * **Load Camera Images**: load images or not
    * **Images Directory**: path to directory with undistorted images
    * **Image Extension**: images extension, they all should be the same (currently)
    * **Alpha**: camera image alpha, 0.0 - 1.0
    * **Depth**: camera display depth, front / back
4. there are some more optional properties:
    * **Create Chunk Region Borders**
    * **Align to Active Object**: if you import mesh from PhotoScan first, the transform it to correct size and orientation, this option will copy transformation from that mesh if it is active
5. hit **Import**
6. import done..
7. now you can quickly swap cameras in alphabetical order in `PhotoScan Cameras Utilities` panel

changelog:

* 0.1.2 compatibility with Agisoft Metashape XML (1.5.x)
* 0.1.1 first release

[BlenderArtist.org thread](https://blenderartists.org/t/addon-import-agisoft-photoscan-cameras/1140610)

***

## [Carbon Tools](https://github.com/uhlik/bpy/tree/master/carbon_tools.py)

Ever-evolving set of small tools, workflows and shortcuts focused mainly on processing photogrammetry scans.

![Carbon Tools](https://raw.githubusercontent.com/uhlik/bpy/master/x/carbon_tools.png)

#### Subtools

* **Extract** selected part of mesh to a new object. If edges of extracted mesh are changed, it won't be able to merge back seamlessly - use option to hide edges (lock button) to protect them.
* **Insert** it back when finished editing.
* **Extract Non-Manifold** elements with part of mesh around them (10x expanded selection) as subtool

#### Dyntopo

* **Dyntopo Setup** - Quick setup for optimizing mesh resolution. Set desired **Constant Resolution** and **Method** and hit **Dyntopo Setup**
Mode is switched to Sculpt with Dyntopo, brush is set to strength 0 - affecting only mesh resolution.
* **Dyntopo Live Settings** - current settings which can be changed during sculpting

#### Texture Paint

* **Texture Paint Setup** - Quick setup for retouching texture in Photoshop, set **Resolution** of exported images and hit **TP Setup** (Photoshop must be set in preferences as Image Editor)
* **External TP Live Commands**: **Quick Edit** - export image and open in PS, **Apply** - project image back to model, **Save All Images** - save all edited textures

#### IO

* **Import from ZBrush** (depends on other addon Fast Wavefront^2)
* **Export to ZBrush** (depends on other addon Fast Wavefront^2)
* **Transformation: Selected > Active** - Copy transformation from selected to active. Useful for setting correct scale and orientation after initial import from PhotoScan.
* **Matrix: Selected > Active** - Select matrix source object first and then target object. Matrix will be copied while keeping visual transformation intact.
* **Export to PhotoScan** (depends on other addon Fast Wavefront^2)

#### End
* **End** ends current procedure: Dyntopo and Texture Paint back to Object mode and reset all settings

#### Utilities

* **Smooth** / **Flat** shading, just shortcuts
* **UV Coverage** calculate and print how much percent covers active uv layout
* **Seams From Islands** mark seams from UV islands
* **Select Seams** select seam edges
* **Seams > Wireframe** copy seams edges to a new mesh object
* **Export UV Layout**
* **Wireframe** set display to shaded + wire + all edges and deselect object
* **Select Non-Manifold** select non-manifold elements and optionally focus camera on them (eye icon)

#### Conversions

* **UVTex > VCols** - Copy image colors from active image texture node in active material using active UV layout to new vertex colors
* **Group > VCols** - Active vertex group to new vertex colors, vertex weight to rgba(weight, weight, weight, 1.0)
* **VCols > Group** - Active vertex colors to new vertex group, vertex weight by color perceived luminance
* **Difference > Group** - Calculate difference between two selected meshes and write as vertex group to active mesh. Selected is consedered to be original, active to be modified. Objects should have the same transformation.

changelog:

* 2.2.2 ui changes, mesh difference to group, calculate uv coverege
* 0.2.1 uvtex / vcols / group conversions
* 0.2.0 first release

***

## [Time Tracker](https://raw.githubusercontent.com/uhlik/bpy/master/system_time_tracker.py) (for blender 2.80)

Simple time tracker inside blender. After you install and enable it, it will log loaded and saved files and time spent of file until it is saved. All ui is inside addon preferences.

![Time Tracker](https://raw.githubusercontent.com/uhlik/bpy/master/x/tt2.png)

Here you can enable / disable logging, clear data collected so far, set custom data path (.csv) and see short summary of collected data and open individual project directories in file browser. The project name is determined by directory name where the blend is. For example if you have project in directory named "MyNewOutstandingProject" and all blends are inside subdirectory "models", set level number to 1 and you will see project name in results. 0 is directory directly above blend, 1 is one directory above blend, and so on. If you are like me and all your projects have the same subdirectory structure, sent directory level and you are good to go.

changelog:

* 0.2.0 updated for 2.80
* 0.1.0 added simple ui
* 0.0.8 ui tweaks, more advanced options, minor bugfixes
* 0.0.7 fixed performance and sorting, added tracking of files which were closed without saving once a minute (can be enabled in preferences: check Track Scene Update)
* 0.0.6 first release

[BlenderArtist.org thread](http://blenderartists.org/forum/showthread.php?345129-Time-Tracker-addon)

***
***

### addons for blender 2.7x

***
***

## [Point Cloud Visualizer](https://raw.githubusercontent.com/uhlik/bpy/master/2.7x/view3d_point_cloud_visualizer.py) (for blender 2.7x)

Display colored point cloud PLY in Blender's 3d viewport. Works with binary point cloud PLY files with 'x, y, z, red, green, blue' vertex values. All other values are ignored. Color values must be in 0-255 range.

* Add an Empty object
* Load and display point cloud at Properties (N) panel with Empty selected
* Available controls: auto loading of selected ply file, pixel point drawing or OpenGL smooth point, drawing enable/disable, percentage of displayed points and reset button which resets all (except autoload option)
* Transforming Empty transforms point cloud as well
* Works reasonably fast with 4m points on my machine, 5 years old, not top of the line at its time

![Point Cloud Visualizer](https://raw.githubusercontent.com/uhlik/bpy/master/x/pc.gif)

changelog:

* 0.3.0 new ply loader, can be used with any binary ply file with vertex coordinates and colors
* 0.2.0 display percentage
* 0.1.0 first release

[BlenderArtist.org thread](https://blenderartists.org/forum/showthread.php?416158-Addon-Point-Cloud-Visualizer)

***

## [OpenGL Lights](https://raw.githubusercontent.com/uhlik/bpy/master/2.7x/space_view3d_opengl_lights.py)

Simple and effective access to Solid OpenGL Lights for your viewport. Comes with clean, simple interface and endless possibilities. No more fiddling with preferences and other similar confusing, complex and view obscuring addons!

![OpenGL Lights](https://raw.githubusercontent.com/uhlik/bpy/master/x/gll2.gif)

changelog:

* 2018.04.07 new ui

![OpenGL Lights](https://raw.githubusercontent.com/uhlik/bpy/master/x/gllui.png)

* 2014.08.25 more presets
* 2014.08.24 added defaults, created when no presets are available
* 2014.08.19 first release

[BlenderArtist.org thread](http://blenderartists.org/forum/showthread.php?346612-The-most-efficient-OpenGL-Lights-panel-%28with-presets-system%29)

***

## [Fast Wavefront](https://raw.githubusercontent.com/uhlik/bpy/master/2.7x/io_mesh_fast_obj.py) (for blender 2.7x)

Import/Export single mesh as Wavefront OBJ.

Only active mesh is exported. Only single mesh is expected on import. Supported obj features: UVs, normals, shading, vertex colors using MRGB format (ZBrush) or so called 'extended' format when each vertex is defined by 6 values (x, y, z, r, g, b). Export is ~3x faster than built-in obj exporter and import ~2x. It lacks some features, but saves quite a bit time when you need to move high resolution mesh from blender to ZBrush and back a few times per hour while cleaning up scans.

Comes with a few presets (created upon activation) for following workflow: import raw mesh obj from Agisoft PhotoScan, export raw mesh obj to ZBrush, import cleaned/uv unwrapped mesh obj from ZBrush, export cleaned mesh to Agisoft PhotoScan for texture generation.

changelog:

* 0.1.2 import zbrush mask as vertex group
* 0.1.1 first release

***

## [UV Equalize](https://raw.githubusercontent.com/uhlik/bpy/master/2.7x/uv_equalize.py)

Equalizes scale of UVs of selected objects to active object.

* Use when tileable texture needs to be applied on all objects and its scale should be the same across them.
* Available in Object menu of 3d view while in object mode.
* To enable, more than two mesh objects must be selected, one must be active.
* Default behavior is active object determines scale and all other objects will be adjusted. This can be overrided unchecking 'Use Active', then all objects will be averaged.
* Island scale averaging and repacking is optional and will yeld better result.

![UV Equalize](https://raw.githubusercontent.com/uhlik/bpy/master/x/eq2.gif)

changelog:

* 0.2.3 better fix for bug fixed in previous version..
* 0.2.2 fixed bug which prevented script to work, operators are used for transforming uvs, but when in image editor is loaded 'Render Result', UV will not be displayed and therefore operators will not work.. it's one line fix, just set displayed image to None..
* 0.2.1 auto deselect non mesh objects
* 0.2.0 complete rewrite, now it is pure math
* 0.1.2 fixed different uv names bug
* 0.1.1 uuid windows workaround
* 0.1.0 first release

[BlenderArtist.org thread](http://blenderartists.org/forum/showthread.php?339782-UV-Equalize-and-Tube-Unwrap-addons)

***

## [Tube UV Unwrap](https://raw.githubusercontent.com/uhlik/bpy/master/2.7x/uv_tube_unwrap.py) (for blender 2.7x)

UV unwrap tube-like meshes (all quads, no caps, fixed number of vertices in each ring)

notes:

* Works only on tube-like parts of mesh defined by selection and active vertex (therefore you must be in vertex selection mode) and the selection must have a start and an end ring. Tube-like mesh is: all quads, no caps, fixed number of vertices in each ring. (Best example of such mesh is mesh circle extruded several times or beveled curve (not cyclic) converted to mesh.) There must be an active vertex on one of the boundary loops in selection. This active vertex define place where mesh will be 'cut' - where seam will be placed.
* Result is rectangular UV for easy texturing, scaled to fit square, horizontal and vertical distances between vertices are averaged and proportional to each other. 

usage:

1. tab to Edit mode
2. select part of mesh you want to unwrap, tube type explained above
3. make sure your selection has boundaries and there is an active vertex on one border of selection
4. hit "U" and select "Tube UV Unwrap"
5. optionally check/uncheck 'Mark Seams' or 'Flip' in operator properties

![Tube UV Unwrap](https://raw.githubusercontent.com/uhlik/bpy/master/x/tube2.gif)

changelog:

* 0.2.4 removed redundant and buggy parts
* 0.2.0 almost full rewrite, now it works on selection only, any mesh will work, if selection comply to requirements
* 0.1.3 fail nicely when encountered 2 ring cylinder
* 0.1.2 got rid of changing edit/object mode
* 0.1.1 fixed accidental freeze on messy geometry, fixed first loop vertex order (also on messy geometry), uv creation part completely rewritten from scratch
* 0.1.0 first release

[BlenderArtist.org thread](http://blenderartists.org/forum/showthread.php?339782-UV-Equalize-and-Tube-Unwrap-addons)

***

## [Time Tracker](https://raw.githubusercontent.com/uhlik/bpy/master/2.7x/system_time_tracker.py)

Simple time tracker inside blender. After you install and enable it, it will log loaded and saved files and time spent of file until it is saved. All ui is inside addon preferences.

![Time Tracker](https://raw.githubusercontent.com/uhlik/bpy/master/x/tt.jpg)

Here you can enable / disable logging, clear data collected so far, set custom data path (.csv) and see short summary of collected data and open individual project directories in file browser. The project name is determined by directory name where the blend is. For example if you have project in directory named "MyNewOutstandingProject" and all blends are inside subdirectory "models", set level number to 1 and you will see project name in results. 0 is directory directly above blend, 1 is one directory above blend, and so on. If you are like me and all your projects have the same subdirectory structure, sent directory level and you are good to go.

changelog:

* 0.1.0 added simple ui
* 0.0.8 ui tweaks, more advanced options, minor bugfixes
* 0.0.7 fixed performance and sorting, added tracking of files which were closed without saving once a minute (can be enabled in preferences: check Track Scene Update)
* 0.0.6 first release

[BlenderArtist.org thread](http://blenderartists.org/forum/showthread.php?345129-Time-Tracker-addon)
