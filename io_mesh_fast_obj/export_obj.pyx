# cython: language_level=3

import io
import os
import shutil
import numpy as np
from libcpp cimport bool


cdef extern from "BLI_math_vector.h":
    void normal_short_to_float_v3(float r[3], const short n[3])


cdef extern from "DNA_meshdata_types.h":
    ctypedef struct MVert:
        float co[3]
        short no[3]
        char flag
        char bweight

    ctypedef struct MPoly:
        int loopstart
        int totloop
        short mat_nr
        char flag
        char pad

    ctypedef struct MLoop:
        unsigned int v
        unsigned int e

    ctypedef struct MLoopUV:
        float uv[2]
        int flag
    
    ctypedef struct MLoopCol:
        unsigned char r, g, b, a


cdef extern from "DNA_mesh_types.h":
    ctypedef struct Mesh:
        int totvert
        int totpoly
        int totloop
        MVert *mvert
        MPoly *mpoly
        MLoop *mloop
        MLoopUV *mloopuv
        MLoopCol *mloopcol


DEBUG = False


def log(msg="", indent=0, prefix="> "):
    m = "{}{}{}".format("    " * indent, prefix, msg, )
    if(DEBUG):
        print(m)


def export(long long int pointer, bool use_normals, bool use_uv, bool use_vcols, int precision, ):
    cdef Mesh *me = <Mesh*>pointer
    
    f = io.StringIO()
    fw = f.write
    
    cdef int i
    cdef int vi
    cdef int li
    cdef int pi
    cdef float no[3]
    cdef MLoopCol col
    
    log('writing vertices..', 1)
    for i in range(me.totvert):
        fw('v {:.{p}f} {:.{p}f} {:.{p}f}\n'.format(*me.mvert[i].co, p=precision))
    
    if(use_normals):
        log('writing normals..', 1)
        for i in range(me.totvert):
            normal_short_to_float_v3(no, me.mvert[i].no)
            fw('vn {:.{p}f} {:.{p}f} {:.{p}f}\n'.format(*no, p=precision))
    
    if(use_vcols):
        log('writing vertex colors..', 1)
        vcols = np.zeros([me.totvert, 4], dtype=np.uint8, )
        for i in range(me.totloop):
            col = me.mloopcol[i]
            vi = me.mloop[i].v
            vcols[vi][0] = 255
            vcols[vi][1] = col.r
            vcols[vi][2] = col.g
            vcols[vi][3] = col.b
        for i in range(0, me.totvert, 64):
            ch = vcols[i:i + 64]
            ch = ch.flatten()
            fw(("#MRGB " + ("{:02x}" * len(ch)) + "\n").format(*ch))
    
    if(use_uv):
        log('writing uv..', 1)
        for li in range(me.totloop):
            fw('vt {:.{p}f} {:.{p}f}\n'.format(*me.mloopuv[li].uv, p=precision))
    
    log('writing polygons..', 1)
    for pi in range(me.totpoly):
        p = me.mpoly[pi]
        fw("f")
        for li in range(p.loopstart, p.loopstart + p.totloop, 1):
            vi = me.mloop[li].v
            if(use_normals and use_uv):
                fw(' {}/{}/{}'.format(vi + 1, li + 1, vi + 1))
            elif(not use_normals and use_uv):
                fw(' {}/{}'.format(vi + 1, li + 1))
            elif(use_normals and not use_uv):
                fw(' {}//{}'.format(vi + 1, vi + 1))
            else:
                fw(' {}'.format(vi + 1))
        fw("\n")
    
    return f


def export_obj(long long int pointer, str path, str obname, bool use_normals, bool use_uv, bool use_vcols, int precision=6, debug=False, ):
    global DEBUG
    DEBUG = debug
    
    f = export(pointer, use_normals, use_uv, use_vcols, precision, )
    
    log('writing obj file..', 1)
    tp = "{}.tmp".format(path)
    with open(tp, mode='w', encoding='utf-8', newline="\n", ) as of:
        of.write("# Fast Wavefront^2 (.obj) (Cython)\n")
        of.write("o {}\n".format(obname))
        of.write(f.getvalue())
    f.close()
    if(os.path.exists(path)):
        os.remove(path)
    shutil.move(tp, path)

