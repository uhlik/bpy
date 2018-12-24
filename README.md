### Contents:


### for blender 2.80

* [Point Cloud Visualizer](#point-cloud-visualizer-for-blender-280)
* [Tube UV Unwrap](#tube-uv-unwrap-for-blender-280)

### for blender 2.7x

* [Point Cloud Visualizer](#point-cloud-visualizer-for-blender-27x)
* [OpenGL Lights](#opengl-lights)
* [Fast Wavefront (.obj)](#fast-wavefront)
* [UV Equalize](#uv-equalize)
* [Tube UV Unwrap](#tube-uv-unwrap-for-blender-27x)
* [Time Tracker](#time-tracker)

***

## [Point Cloud Visualizer](https://raw.githubusercontent.com/uhlik/bpy/master/view3d_point_cloud_visualizer.py) (for blender 2.80)

Display colored point cloud PLY in Blender's 3d viewport. Works with binary point cloud PLY files with 'x, y, z, red, green, blue' vertex values. All other values are ignored. Color values must be in 0-255 range.

![Point Cloud Visualizer](https://raw.githubusercontent.com/uhlik/bpy/master/x/pcv280.gif)

![Point Cloud Visualizer](https://raw.githubusercontent.com/uhlik/bpy/master/x/pcv280ui0.4.5.png)

* Usage: Install and activate addon in a usual way. Add any object to scene. Go to 3d View Sidebar (N) > 'Point Cloud Visualizer' panel, click file browser icon, select ply file, click 'Load PLY'. Click 'Draw' button to display point cloud, 'Erase' to hide point cloud. Adjust percentage of displayed points with 'Display' and point size with 'Radius'.
* Transforming parent object transforms point cloud as well.
* Huge speed improvements compared to 2.7x version.

[BlenderArtist.org thread](https://blenderartists.org/forum/showthread.php?416158-Addon-Point-Cloud-Visualizer)

changelog:

* 0.5.1 load ply without vertex colors, uniform grey will be used
* 0.5.0 performance improvements using numpy for loading and processing data
* 0.4.6 fixed crash when parent object is deleted while drawing, fixed removal of loaded data when parent is deleted
* 0.4.5 added 'Display' percentage, better error handling during .ply loading
* 0.4.0 almost complete rewrite for blender 2.80, performance improvements using shaders, simplified ui
* 0.3.0 new ply loader, can be used with any binary ply file with vertex coordinates and colors
* 0.2.0 display percentage
* 0.1.0 first release

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

[BlenderArtist.org thread](http://blenderartists.org/forum/showthread.php?339782-UV-Equalize-and-Tube-Unwrap-addons)

changelog:

* 0.3.0 blender 2.8 update

***

## [Point Cloud Visualizer](https://raw.githubusercontent.com/uhlik/bpy/master/2.7x/view3d_point_cloud_visualizer.py) (for blender 2.7x)

Display colored point cloud PLY in Blender's 3d viewport. Works with binary point cloud PLY files with 'x, y, z, red, green, blue' vertex values. All other values are ignored. Color values must be in 0-255 range.

* Add an Empty object
* Load and display point cloud at Properties (N) panel with Empty selected
* Available controls: auto loading of selected ply file, pixel point drawing or OpenGL smooth point, drawing enable/disable, percentage of displayed points and reset button which resets all (except autoload option)
* Transforming Empty transforms point cloud as well
* Works reasonably fast with 4m points on my machine, 5 years old, not top of the line at its time

![Point Cloud Visualizer](https://raw.githubusercontent.com/uhlik/bpy/master/x/pc.gif)

[BlenderArtist.org thread](https://blenderartists.org/forum/showthread.php?416158-Addon-Point-Cloud-Visualizer)

changelog:

* 0.3.0 new ply loader, can be used with any binary ply file with vertex coordinates and colors
* 0.2.0 display percentage
* 0.1.0 first release

***

## [OpenGL Lights](https://raw.githubusercontent.com/uhlik/bpy/master/2.7x/space_view3d_opengl_lights.py)

Simple and effective access to Solid OpenGL Lights for your viewport. Comes with clean, simple interface and endless possibilities. No more fiddling with preferences and other similar confusing, complex and view obscuring addons!

![OpenGL Lights](https://raw.githubusercontent.com/uhlik/bpy/master/x/gll2.gif)

[BlenderArtist.org thread](http://blenderartists.org/forum/showthread.php?346612-The-most-efficient-OpenGL-Lights-panel-%28with-presets-system%29)

changelog:

* 2018.04.07 new ui

![OpenGL Lights](https://raw.githubusercontent.com/uhlik/bpy/master/x/gllui.png)

* 2014.08.25 more presets
* 2014.08.24 added defaults, created when no presets are available
* 2014.08.19 first release

***

## [Fast Wavefront](https://raw.githubusercontent.com/uhlik/bpy/master/2.7x/io_mesh_fast_obj.py)

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

[BlenderArtist.org thread](http://blenderartists.org/forum/showthread.php?339782-UV-Equalize-and-Tube-Unwrap-addons)

changelog:

* 0.2.3 better fix for bug fixed in previous version..
* 0.2.2 fixed bug which prevented script to work, operators are used for transforming uvs, but when in image editor is loaded 'Render Result', UV will not be displayed and therefore operators will not work.. it's one line fix, just set displayed image to None..
* 0.2.1 auto deselect non mesh objects
* 0.2.0 complete rewrite, now it is pure math
* 0.1.2 fixed different uv names bug
* 0.1.1 uuid windows workaround
* 0.1.0 first release

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

[BlenderArtist.org thread](http://blenderartists.org/forum/showthread.php?339782-UV-Equalize-and-Tube-Unwrap-addons)

changelog:

* 0.2.4 removed redundant and buggy parts
* 0.2.0 almost full rewrite, now it works on selection only, any mesh will work, if selection comply to requirements
* 0.1.3 fail nicely when encountered 2 ring cylinder
* 0.1.2 got rid of changing edit/object mode
* 0.1.1 fixed accidental freeze on messy geometry, fixed first loop vertex order (also on messy geometry), uv creation part completely rewritten from scratch
* 0.1.0 first release

***

## [Time Tracker](https://raw.githubusercontent.com/uhlik/bpy/master/2.7x/system_time_tracker.py)

Simple time tracker inside blender. After you install and enable it, it will log loaded and saved files and time spent of file until it is saved. All ui is inside addon preferences.

![Time Tracker](https://raw.githubusercontent.com/uhlik/bpy/master/x/tt.jpg)

Here you can enable / disable logging, clear data collected so far, set custom data path (.csv) and see short summary of collected data and open individual project directories in file browser. The project name is determined by directory name where the blend is. For example if you have project in directory named "MyNewOutstandingProject" and all blends are inside subdirectory "models", set level number to 1 and you will see project name in results. 0 is directory directly above blend, 1 is one directory above blend, and so on. If you are like me and all your projects have the same subdirectory structure, sent directory level and you are good to go.

[BlenderArtist.org thread](http://blenderartists.org/forum/showthread.php?345129-Time-Tracker-addon)

changelog:

* 0.1.0 added simple ui
* 0.0.8 ui tweaks, more advanced options, minor bugfixes
* 0.0.7 fixed performance and sorting, added tracking of files which were closed without saving once a minute (can be enabled in preferences: check Track Scene Update)
* 0.0.6 first release
