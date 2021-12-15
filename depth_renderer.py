#!/usr/bin/env python3
import os
import cv2
import trimesh
import numpy as np
import ctypes as ct
from glob import glob
import matplotlib.pyplot as plt


mdl_ptn = '/data/6D_pose_data/BOP/lm/models/*.ply'
mdl_pth_lst = glob(mdl_ptn)
cur_dir = os.path.dirname(os.path.realpath(__file__))
so_p = os.path.join(cur_dir, 'rastertriangle_so.so')
print("raster triangle so path:", so_p)
dll = np.ctypeslib.load_library(so_p, '.')


def load_trimesh(pth, scale2m=1.):
    mesh = trimesh.load(pth, force='mesh')
    mesh.vertices = mesh.vertices / scale2m
    return mesh


def depth_renderer(mesh, K, RT=np.eye(4), h=480, w=640, scale2m=1.):
    vtxs = np.array(mesh.vertices) / scale2m
    faces = np.array(mesh.faces)
    n_face = faces.shape[0]
    face = faces.flatten().copy()

    R, T = RT[:3, :3], RT[:3, 3]

    new_xyz = vtxs.copy()
    new_xyz = np.dot(new_xyz, R.T) + T[None, :]
    p2ds = np.dot(new_xyz.copy(), K.T)
    p2ds = p2ds[:, :2] / p2ds[:, 2:]
    p2ds = np.require(p2ds.flatten(), 'float32', 'C')

    face = np.require(face, 'int32', 'C')
    new_xyz = np.require(new_xyz, 'float32', 'C')
    zs = np.require(new_xyz[:, 2].copy(), 'float32', 'C')
    zbuf = np.require(np.zeros(h*w), 'float32', 'C')

    dll.zbuffer(
        ct.c_int(h),
        ct.c_int(w),
        p2ds.ctypes.data_as(ct.c_void_p),
        # new_xyz.ctypes.data_as(ct.c_void_p),
        zs.ctypes.data_as(ct.c_void_p),
        ct.c_int(n_face),
        face.ctypes.data_as(ct.c_void_p),
        zbuf.ctypes.data_as(ct.c_void_p),
    )

    zbuf.resize((h, w))
    msk = (zbuf > 1e-8).astype('uint8')
    zbuf *= msk.astype(zbuf.dtype)  # * 1000.0
    return zbuf, msk


def dpt2cld(dpt, cam_scale, K):
    """
    dpt: h x w depth image
    cam_scale: scale depth to (m). e.g: dpt in mm, cam_scale should be 1000
    K: camera intrinsic
    """
    h, w = dpt.shape[0], dpt.shape[1]
    xmap = np.array([[j for i in range(w)] for j in range(h)])
    ymap = np.array([[i for i in range(w)] for j in range(h)])

    if len(dpt.shape) > 2:
        dpt = dpt[:, :, 0]
    msk_dp = dpt > 1e-6
    choose = msk_dp.flatten()
    if choose.sum() < 1:
        return None, None

    dpt_mskd = dpt.flatten()[choose][:, np.newaxis].astype(np.float32)
    xmap_mskd = xmap.flatten()[choose][:, np.newaxis].astype(np.float32)
    ymap_mskd = ymap.flatten()[choose][:, np.newaxis].astype(np.float32)

    pt2 = dpt_mskd / cam_scale
    cam_cx, cam_cy = K[0][2], K[1][2]
    cam_fx, cam_fy = K[0][0], K[1][1]
    pt0 = (ymap_mskd - cam_cx) * pt2 / cam_fx
    pt1 = (xmap_mskd - cam_cy) * pt2 / cam_fy
    cld = np.concatenate((pt0, pt1, pt2), axis=1)
    return cld


def dpt2heat(dpt):
    vd = dpt[dpt > 0]
    dpt[dpt > 0] = (dpt[dpt > 0] - vd.min() + 0.4) / (vd.max() - vd.min())
    dpt = (dpt * 255).astype(np.uint8)
    colormap = plt.get_cmap('inferno')
    heatmap = (colormap(dpt.copy()) * 2**16).astype(np.uint16)[:, :, :3]
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
    heatmap[dpt == 0, :] = np.array([255, 255, 255])
    return heatmap


def main():
    for mdl_pth in mdl_pth_lst:
        mesh = load_trimesh(mdl_pth, scale2m=1000.)
        # mesh = load_trimesh(mdl_pth, scale2m=1.)
        K = np.array([[700., 0., 320.], [0., 700., 240.], [0., 0., 1.]])
        RT = np.eye(4)
        RT[:3, 3] = np.array([0., 0., 1.])
        depth, msk = depth_renderer(mesh, K, RT=RT)

        try:
            from neupeak.utils.webcv2 import imshow, waitKey
        except ImportError:
            from cv2 import imshow, waitKey
        show_dpt = dpt2heat(depth.copy())
        imshow("render_depth:", show_dpt)
        waitKey(0)


if __name__ == "__main__":
    main()
# vim: ts=4 sw=4 sts=4 expandtab
