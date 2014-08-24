## [UV Equalize](https://raw.githubusercontent.com/uhlik/bpy/master/uv_equalize.py)

Equalizes scale of UVs of selected objects to active object.

* Use when tileable texture needs to be applied on all objects and its scale should be the same across them.
* Beware, active UV on each object will be repacked, in active object as well.
* Available in Object menu of 3d view while in object mode.
* To enable, more than two mesh objects must be selected, one must be active.

![UV Equalize](https://raw.githubusercontent.com/uhlik/bpy/master/x/eq.gif)

[BlenderArtist.org thread](http://blenderartists.org/forum/showthread.php?339782-UV-Equalize-and-Tube-Unwrap-addons)

changelog:

* 0.1.1 uuid windows workaround
* 0.1.0 first release

***

## [Tube UV Unwrap](https://raw.githubusercontent.com/uhlik/bpy/master/uv_tube_unwrap.py)

UV unwrap tube like meshes (all quads, no caps, fixed number of vertices in each ring)

notes:

* Works only on tube like meshes, all quads, no caps, fixed number of vertices in each ring. Best example of such mesh is mesh circle extruded several times or beveled curve converted to mesh.
* Result is right-angled UV for easy texturing
* Single selected vertex on boundary ring is required before running operator. This vertex marks loop, along which tube will be cut.
* Distances of vertices in next tube ring are averaged.
* UV is scaled to fit area.
* Seam will be marked on mesh automatically.
* Mesh must have at least 3 rings to be unwrapped. Simple cylinder with two boundary rings will not work. Add a loop cut between them.

usage:

1. tab to Edit mode
2. select single vertex on boundary ring
3. hit "U" and select "Tube UV Unwrap"

![Tube UV Unwrap](https://raw.githubusercontent.com/uhlik/bpy/master/x/tube.gif)

[BlenderArtist.org thread](http://blenderartists.org/forum/showthread.php?339782-UV-Equalize-and-Tube-Unwrap-addons)

changelog:

* 0.1.3 fail nicely when encountered 2 ring cylinder
* 0.1.2 got rid of changing edit/object mode
* 0.1.1 fixed accidental freeze on messy geometry, fixed first loop vertex order (also on messy geometry), uv creation part completely rewritten from scratch
* 0.1.0 first release

***

## [Time Tracker](https://raw.githubusercontent.com/uhlik/bpy/master/system_time_tracker.py)

Simple time tracker inside blender. After you install and enable it, it will log loaded and saved files and time spent of file until it is saved. All ui is inside addon preferences.

![Time Tracker](https://raw.githubusercontent.com/uhlik/bpy/master/x/tt.jpg)

Here you can enable / disable logging, clear data collected so far, set custom data path (.csv) and see short summary of collected data and open individual project directories in file browser. The project name is determined by directory name where the blend is. For example if you have project in directory named "MyNewOutstandingProject" and all blends are inside subdirectory "models", set level number to 1 and you will see project name in results. 0 is directory directly above blend, 1 is one directory above blend, and so on. If you are like me and all your projects have the same subdirectory structure, sent directory level and you are good to go.

[BlenderArtist.org thread](http://blenderartists.org/forum/showthread.php?345129-Time-Tracker-addon)

changelog:

* 0.0.8 ui tweaks, more advanced options, minor bugfixes
* 0.0.7 fixed performance and sorting, added tracking of files which were closed without saving once a minute (can be enabled in preferences: check Track Scene Update)
* 0.0.6 first release

***

## [OpenGL Lights](https://raw.githubusercontent.com/uhlik/bpy/master/space_view3d_opengl_lights.py)

Simple and effective access to Solid OpenGL Lights for your viewport. Comes with clean in simple interface and endless possibilities. No more fiddling with preferences and other similar confusing, complex and view obscuring addons!

![OpenGL Lights](https://raw.githubusercontent.com/uhlik/bpy/master/x/gl-lights.gif)

[BlenderArtist.org thread](http://blenderartists.org/forum/showthread.php?346612-The-most-efficient-OpenGL-Lights-panel-%28with-presets-system%29)

changelog:

* 2014.08.24 added defaults, created when no presets are available
* 2014.08.19 first release
