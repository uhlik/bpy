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

bl_info = {"name": "Time Tracker",
           "description": "Track time spent in blender. Writes data to .csv and provides summary sorted by project (directory name).",
           "author": "Jakub Uhlik",
           "version": (0, 2, 0),
           "blender": (2, 80, 0),
           "location": "main: Preferences > Addons > System > Time Tracker, quick enable/disable: View3d > Properties > Time Tracker",
           "warning": "Beta",
           "wiki_url": "",
           "tracker_url": "",
           "category": "System", }


import os
import datetime
import csv
import platform
import subprocess
import time
import shutil

import bpy
from bpy.types import Panel, Operator, AddonPreferences
from bpy.props import BoolProperty, StringProperty, IntProperty
from bpy.app.handlers import persistent


# TODO try to use timers for something? https://docs.blender.org/api/blender2.8/bpy.app.timers.html


DEBUG = False


def log(msg, indent=0, ):
    m = "{0}> {1}".format("    " * indent, msg)
    if(DEBUG):
        print(m)


class Runtime():
    start = datetime.datetime.now()
    path_message = ""
    summary_message = ""
    modified = -1
    summary = None
    level = -1
    update_last = -1
    update_step = 60


class Utils():
    @staticmethod
    def format_stamp(t):
        return t.strftime('%Y.%m.%d-%H.%M.%S')
    
    @staticmethod
    def format_time(d):
        return '{:02}:{:02}:{:02}'.format(d // 3600, d % 3600 // 60, d % 60)
    
    @staticmethod
    def format_time_summary(d):
        return '{:02}h {:02}m'.format(d // 3600, d % 3600 // 60)
    
    @staticmethod
    def format_time_summary_seconds(d):
        return '{:02}h {:02}m {:02}s'.format(d // 3600, d % 3600 // 60, d % 60)
    
    @staticmethod
    def get_default_csv_path():
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), "{0}.csv".format(os.path.splitext(os.path.split(os.path.realpath(__file__))[1])[0]))
    
    @staticmethod
    def get_preferences():
        a = os.path.splitext(os.path.split(os.path.realpath(__file__))[1])[0]
        p = bpy.context.preferences.addons[a].preferences
        return p
    
    @staticmethod
    def find_handlers():
        l = -1
        s = -1
        u = -1
        h = bpy.app.handlers
        for i in range(len(h.load_post)):
            if(h.load_post[i].__name__ == "TIME_TRACKER_load_handler"):
                l = i
        for i in range(len(h.save_post)):
            if(h.save_post[i].__name__ == "TIME_TRACKER_save_handler"):
                s = i
        for i in range(len(h.depsgraph_update_post)):
            if(h.depsgraph_update_post[i].__name__ == "TIME_TRACKER_update_handler"):
                u = i
        return l, s, u


def summary():
    prefs = Utils.get_preferences()
    p = prefs.csv_path
    
    if(not os.path.exists(p)):
        Runtime.summary_message = "File {} does not exist.".format(p)
        return []
    else:
        Runtime.summary_message = ""
    
    # get modified time of csv
    tm = os.path.getmtime(p)
    
    if(Runtime.summary is not None):
        if(tm == Runtime.modified and prefs.level == Runtime.level):
            # return already parsed results
            return Runtime.summary
    
    Runtime.modified = tm
    Runtime.level = prefs.level
    
    db = []
    with open(p, encoding='utf-8', ) as f:
        reader = csv.reader(f)
        for i, r in enumerate(reader):
            if(i == 0):
                continue
            t = int(r[2])
            if(r[4] != "" and r[5] != "" and t != 0):
                a = [r[0], r[1], t, r[3], r[4], r[5]]
                db.append(a)
    
    def proj(path, level, ):
        # split path until level is reached
        def slice_last_dir(p):
            return os.path.split(p)[0]
        
        h, t = os.path.split(path)
        pp = "{0}".format(h)
        for i in range(level):
            pp = slice_last_dir(pp)
        proj = os.path.split(pp)[1]
        return proj
    
    # split to projects
    dbp = {}
    for r in db:
        p = proj(r[5], int(prefs.level))
        try:
            dbp[p].append(r)
        except Exception:
            dbp[p] = []
            dbp[p].append(r)
    
    # sum projects
    a = []
    for proj, ls in dbp.items():
        s = 0
        for r in ls:
            s += r[2]
        d = os.path.split(ls[0][5])[0]
        n = len(ls)
        if(proj == ''):
            # substitute no directory by '/'
            proj = '/'
        if(prefs.debug_show_seconds_in_summary):
            a.append([proj, Utils.format_time_summary_seconds(s), d])
        else:
            a.append([proj, Utils.format_time_summary(s), d])
    
    # sort by project name
    a.sort(key=lambda v: v[0])
    
    # and make strings
    r = []
    for i, l in enumerate(a):
        r.append(["project '{0}': {1}".format(l[0], l[1]), l[2]])
    
    # store results, so it will not be calculated on each ui redraw
    Runtime.summary = r
    
    return r


def update(self, context):
    prefs = Utils.get_preferences()
    
    current = prefs.csv_path
    previous = prefs.previous_csv_path
    
    if(current == previous):
        # no change
        return
    
    # remove message at this point. earlier will not be visible
    Runtime.path_message = ""
    
    if(current == ""):
        # no path, use default
        current = Utils.get_default_csv_path()
    
    if(os.path.isdir(current)):
        # is directory, add default file name
        current = os.path.join(current, os.path.split(Utils.get_default_csv_path())[1])
    
    # add correct extension
    current = bpy.path.ensure_ext(current, ".csv", case_sensitive=True, )
    
    # check if directory is writeable
    d = os.path.split(current)[0]
    if(not os.access(d, os.W_OK)):
        # if not, put default path and write message
        current = Utils.get_default_csv_path()
        Runtime.path_message = "Location '{0}' is not writable.".format(d)
    
    # check if path has changed, and if so, write previous .csv to it if there is none (if it exists too)
    if(current != previous and not os.path.exists(current) and os.path.exists(previous)):
        with open(current, mode='w', encoding='utf-8') as f:
            with open(previous, encoding='utf-8') as o:
                c = "".join(o.readlines())
            f.write(c)
    
    prefs.previous_csv_path = current
    # update after previous path to prevent recursion
    prefs.csv_path = current


def scene_update_update(self, context):
    # great function name, isn't it?
    prefs = Utils.get_preferences()
    Runtime.update_step = prefs.update_interval
    # force summary update
    Runtime.summary = None
    if(prefs.scene_update):
        _, _, u = Utils.find_handlers()
        h = bpy.app.handlers
        if(u == -1):
            h.depsgraph_update_post.append(TIME_TRACKER_update_handler)
    else:
        _, _, u = Utils.find_handlers()
        h = bpy.app.handlers
        if(u != -1):
            del h.depsgraph_update_post[u]


class TIME_TRACKER_preferences(AddonPreferences):
    bl_idname = __name__
    
    csv_first_line: bpy.props.StringProperty(name=".csv First Line", description=".csv first line to be written, contains field names.", default="timestamp (YYYY.MM.DD-HH.MM.SS),event,time (seconds),time (formatted),file name,file path", )
    previous_csv_path: bpy.props.StringProperty(name="Previous CSV Path", description="Used to detect path change and to copy old csv from on change.", default=Utils.get_default_csv_path(), maxlen=1024, subtype='FILE_PATH', )
    enabled: bpy.props.BoolProperty(name="Tracking Enabled", description="When enabled, loading and saving of files is logged.", default=True, )
    level: bpy.props.IntProperty(name="Project Directory Level", description="Which level is considered as project directory. 0 is current directory, 1 is directory enclosing current directory, etc.", default=0, min=0, max=100, )
    csv_path: bpy.props.StringProperty(name="CSV Path", description="Location of .csv with tracking data.", default=Utils.get_default_csv_path(), update=update, maxlen=1024, subtype='FILE_PATH', )
    summary: bpy.props.BoolProperty(name="Show Summary", description="When enabled, shows tracked data bellow in simple format (project name - total time spent).", default=False, )
    show_advanced: bpy.props.BoolProperty(name="Show Advanced Options", description="", default=False, )
    scene_update: bpy.props.BoolProperty(name="Track Scene Update", description="Track time spent on files closed without saving using depsgraph_update_post handler.", update=scene_update_update, default=True, )
    update_interval: bpy.props.IntProperty(name="Update Interval In Seconds", description="Interval at which depsgraph_update_post handler is processed.", update=scene_update_update, default=60, min=1, max=60 * 10, )
    debug_show_seconds_in_summary: bpy.props.BoolProperty(name="Show Seconds In Summary", description="", update=scene_update_update, default=False, )
    
    def draw(self, context):
        l = self.layout
        r = l.row()
        s = r.split(factor=0.5)
        c = s.column()
        c.prop(self, "enabled")
        c = s.column()
        rr = c.row()
        rr.operator("time_tracker.open_csv")
        rr.operator("time_tracker.clear_data", )
        r = l.row()
        r.prop(self, "csv_path")
        if(Runtime.path_message is not ""):
            r = l.row()
            r.label(text=Runtime.path_message, icon='ERROR', )
        r = l.row()
        s = r.split(factor=0.75)
        c = s.column()
        c.prop(self, "summary", )
        if(self.summary is True):
            c = s.column()
            c.prop(self, "level")
            if(Runtime.summary_message is not ""):
                r = l.row()
                r.label(text=Runtime.summary_message, icon='ERROR', )
            a = summary()
            if(len(a) == 0):
                r = l.row()
                r.label(text="No data tracked yet.", icon='ERROR', )
            else:
                r = l.row()
                b = r.box()
                for i in a:
                    r = b.row()
                    s = r.split(factor=0.75)
                    c = s.column()
                    c.label(text=i[0], icon='TIME', )
                    c = s.column()
                    c.operator("time_tracker.show_project_directory", ).directory = i[1]
        r = l.row()
        r.prop(self, "show_advanced")
        if(self.show_advanced):
            r = l.row()
            r.prop(self, "debug_show_seconds_in_summary")
            r.prop(self, "scene_update")
            r.prop(self, "update_interval")


class TIME_TRACKER_OT_show_project_directory(Operator):
    bl_idname = "time_tracker.show_project_directory"
    bl_label = "Show Project Directory"
    bl_description = "Show project directory."
    
    directory: StringProperty()
    
    def execute(self, context):
        d = self.directory
        if(not os.path.exists(d)):
            self.report({'ERROR'}, "The directory {0} does not exist.".format(d))
            return {'FINISHED'}
        
        p = platform.system()
        if(p == 'Windows'):
            os.startfile(os.path.normpath(d))
        elif(p == 'Darwin'):
            subprocess.Popen(['open', d], )
        elif(p == 'Linux'):
            subprocess.Popen(['xdg-open', d], )
        else:
            raise OSError("Unknown platform: {}.".format(p))
        
        return {'FINISHED'}


class TIME_TRACKER_OT_clear_data(Operator):
    bl_idname = "time_tracker.clear_data"
    bl_label = "Clear Data"
    bl_description = "Removes all tracked data."
    
    def execute(self, context):
        prefs = Utils.get_preferences()
        p = prefs.csv_path
        with open(p, mode='w', encoding='utf-8') as f:
            f.write("{0}\n".format(prefs.csv_first_line))
        
        return {'FINISHED'}


class TIME_TRACKER_OT_open_csv(Operator):
    bl_idname = "time_tracker.open_csv"
    bl_label = "Open CSV"
    bl_description = "Open CSV with tracking data."
    
    def execute(self, context):
        prefs = Utils.get_preferences()
        csv = prefs.csv_path
        
        if(not os.path.exists(csv)):
            self.report({'ERROR'}, "No such file: {}".format(csv))
            return {'FINISHED'}
        
        p = platform.system()
        if(p == 'Windows'):
            os.startfile(os.path.normpath(csv))
        elif(p == 'Darwin'):
            subprocess.Popen(['open', csv], )
        elif(p == 'Linux'):
            subprocess.Popen(['xdg-open', csv], )
        else:
            raise OSError("Unknown platform: {}.".format(csv))
        
        return {'FINISHED'}


@persistent
def TIME_TRACKER_load_handler(null):
    track("load")


@persistent
def TIME_TRACKER_save_handler(null):
    track("save")


@persistent
def TIME_TRACKER_update_handler(null):
    t = time.time()
    if(t < Runtime.update_last + Runtime.update_step):
        return
    Runtime.update_last = t
    track("update")


def start():
    log("start")
    
    prefs = Utils.get_preferences()
    p = prefs.csv_path
    
    # write starting csv if there is none
    if(not os.path.exists(p)):
        with open(p, mode='w', encoding='utf-8') as f:
            f.write("{0}\n".format(prefs.csv_first_line))
    
    # set handlers
    l, s, u = Utils.find_handlers()
    h = bpy.app.handlers
    if(l == -1):
        h.load_post.append(TIME_TRACKER_load_handler)
    if(s == -1):
        h.save_post.append(TIME_TRACKER_save_handler)
    if(prefs.scene_update):
        if(u == -1):
            h.depsgraph_update_post.append(TIME_TRACKER_update_handler)


def track(e):
    log(e)
    
    prefs = Utils.get_preferences()
    if(not prefs.enabled):
        return
    
    p = bpy.data.filepath
    n = datetime.datetime.now()
    d = n - Runtime.start
    h, t = os.path.split(p)
    
    l = "{0},{1},{2},{3},{4},{5}\n".format(Utils.format_stamp(n), e, d.seconds, Utils.format_time(d.seconds), t, p, )
    with open(prefs.csv_path, mode='a', encoding='utf-8') as f:
        f.write(l)
    Runtime.start = n


def stop():
    log("stop")
    
    l, s, u = Utils.find_handlers()
    h = bpy.app.handlers
    if(l != -1):
        del h.load_post[l]
    if(s != -1):
        del h.save_post[s]
    if(u != -1):
        del h.depsgraph_update_post[u]


class TIME_TRACKER_PT_panel(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "View"
    bl_label = "Time Tracker"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        prefs = Utils.get_preferences()
        l = self.layout
        l.prop(prefs, 'enabled', toggle=True, text='Enabled', )
        
        if(DEBUG):
            l.separator()
            c = l.column()
            c.label(text="start: {}".format(Runtime.start))
            c.label(text="path_message: {}".format(Runtime.path_message))
            c.label(text="summary_message: {}".format(Runtime.summary_message))
            c.label(text="modified: {}".format(Runtime.modified))
            c.label(text="summary: {}".format(Runtime.summary))
            c.label(text="level: {}".format(Runtime.level))
            c.label(text="update_last: {}".format(Runtime.update_last))
            c.label(text="update_step: {}".format(Runtime.update_step))
            c.scale_y = 0.5


classes = (
    TIME_TRACKER_preferences,
    TIME_TRACKER_OT_show_project_directory,
    TIME_TRACKER_OT_clear_data,
    TIME_TRACKER_OT_open_csv,
    TIME_TRACKER_PT_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    Runtime.start = datetime.datetime.now()
    start()


def unregister():
    stop()
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == '__main__':
    register()
