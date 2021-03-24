#!/usr/bin/env python3
import os
import pickle as pkl
import numpy as np
from plyfile import PlyData
import ctypes as ct
import cv2
import random
from random import randint
from random import shuffle
from tqdm import tqdm
from scipy import stats
from glob import glob
from argparse import ArgumentParser
import concurrent.futures
import time
try:
    from neupeak.utils.webcv2 import imshow, waitKey
except:
    from cv2 import imshow, waitKey


parser = ArgumentParser()
parser.add_argument(
    "--cls", type=str, default="ape",
    help="Target object from {ape, benchvise, cam, can, cat, driller, duck, \
    eggbox, glue, holepuncher, iron, lamp, phone} (default ape)"
)
parser.add_argument(
    '--render_num', type=int, default=70000,
    help="Number of images you want to generate."
)
parser.add_argument(
    '--DEBUG', action="store_true",
    help="To show the generated images or not."
)
parser.add_argument(
    '--vis', action="store_true",
    help="visulaize generated images."
)
args = parser.parse_args()
DEBUG = args.DEBUG


def check_dir(pth):
    if not os.path.exists(pth):
        os.system("mkdir -p {}".format(pth))

OBJ_ID_DICT = {
    'ape':1,
    'cam':2,
    'cat':3,
    'duck':4,
    'glue':5,
    'iron':6,
    'phone':7,
    'benchvise':8,
    'can':9,
    'driller':10,
    'eggbox':11,
    'holepuncher':12,
    'lamp':13,
}

class LineModRenderDB():
    def __init__(self, cls_type, render_num=10, rewrite=False):
        self.h, self.w = 480, 640
        self.K = np.array([[700., 0., 320.],
                           [0., 700., 240.],
                           [0., 0., 1.]])

        self.cls_type = cls_type
        self.cls_id = OBJ_ID_DICT[cls_type]

        self.linemod_dir = './Linemod_preprocessed'
        self.render_dir = os.path.join(self.linemod_dir, 'renders', cls_type)
        check_dir(self.render_dir)
        self.render_num = render_num
        RT_pth = os.path.join('sampled_poses', '{}_sampled_RTs.pkl'.format(cls_type))
        self.RT_lst = pkl.load(open(RT_pth, 'rb'))

        so_p = './rastertriangle_so.so'
        self.dll = np.ctypeslib.load_library(so_p, '.')

        self.bg_img_pth_lst = glob("SUN2012pascalformat/JPEGImages/*.jpg")

        random.seed(19763)
        if render_num < len(self.RT_lst):
            random.shuffle(self.RT_lst)
            self.RT_lst = self.RT_lst[:render_num]

        print("begin loading {} render set:".format(cls_type))

        b_mdl_p = os.path.join(
            self.linemod_dir, 'models', 'obj_%02d.ply' % self.cls_id
        )
        self.npts, self.xyz, self.r, self.g, self.b, self.n_face, self.face = self.load_ply_model(b_mdl_p)
        self.face = np.require(self.face, 'int32', 'C')
        self.r = np.require(np.array(self.r), 'float32', 'C')
        self.g = np.require(np.array(self.g), 'float32', 'C')
        self.b = np.require(np.array(self.b), 'float32', 'C')

    def load_ply_model(self, model_path):
        ply = PlyData.read(model_path)
        data = ply.elements[0].data
        x = data['x']
        y = data['y']
        z = data['z']
        r = data['red']
        g = data['green']
        b = data['blue']
        face_raw = ply.elements[1].data
        face = []
        for item in face_raw:
            face.append(item[0])

        n_face = len(face)
        face = np.array(face).flatten()

        n_pts = len(x)
        xyz = np.stack([x, y, z], axis=-1) / 1000.0

        return n_pts, xyz, r, g, b, n_face, face

    def gen_pack_zbuf_render(self):
        pth_lst = []

        for idx, RT in tqdm(enumerate(self.RT_lst)):
            h, w = self.h, self.w
            R, T = RT[:, :3], RT[:, 3]
            K = self.K

            new_xyz = self.xyz.copy()
            new_xyz = np.dot(new_xyz, R.T) + T
            p2ds = np.dot(new_xyz.copy(), K.T)
            p2ds = p2ds[:, :2] / p2ds[:, 2:]
            p2ds = np.require(p2ds.flatten(), 'float32', 'C')

            zs = np.require(new_xyz[:,2].copy(), 'float32', 'C')
            zbuf = np.require(np.zeros(h*w), 'float32', 'C')
            rbuf = np.require(np.zeros(h*w), 'int32', 'C')
            gbuf = np.require(np.zeros(h*w), 'int32', 'C')
            bbuf = np.require(np.zeros(h*w), 'int32', 'C')
            xyzs = np.require(new_xyz.flatten(), 'float32', 'C')

            self.dll.rgbzbuffer(
                ct.c_int(h),
                ct.c_int(w),
                p2ds.ctypes.data_as(ct.c_void_p),
                new_xyz.ctypes.data_as(ct.c_void_p),
                zs.ctypes.data_as(ct.c_void_p),
                self.r.ctypes.data_as(ct.c_void_p),
                self.g.ctypes.data_as(ct.c_void_p),
                self.b.ctypes.data_as(ct.c_void_p),
                ct.c_int(self.n_face),
                self.face.ctypes.data_as(ct.c_void_p),
                zbuf.ctypes.data_as(ct.c_void_p),
                rbuf.ctypes.data_as(ct.c_void_p),
                gbuf.ctypes.data_as(ct.c_void_p),
                bbuf.ctypes.data_as(ct.c_void_p),
            )

            zbuf.resize((h,w))
            msk = (zbuf>1e-8).astype('uint8')
            if len( np.where(msk.flatten() > 0)[0] ) < 500:
                continue
            zbuf *= msk.astype(zbuf.dtype) # * 1000.0

            bbuf.resize((h,w)), rbuf.resize((h,w)), gbuf.resize((h,w))
            bgr = np.concatenate((bbuf[:,:,None], gbuf[:, :, None], rbuf[:, :, None]), axis=2)
            bgr = bgr.astype('uint8')

            bg = None
            len_bg_lst = len(self.bg_img_pth_lst)
            while bg is None or len(bg.shape) < 3:
                bg_pth = self.bg_img_pth_lst[randint(0, len_bg_lst-1)]
                bg = cv2.imread(bg_pth)
                if len(bg.shape) < 3:
                    continue
                bg_h, bg_w, _ = bg.shape
                if bg_h < h or bg_w < w:
                    bg = None
                    continue
                else:
                    sh = sw = 0
                    sh, sw = randint(0, bg_h-h), randint(0, bg_w-w)
                    if sh+h > bg_h:
                        sh -= 1
                    if sw+w > bg_w:
                        sw -= 1
                    bg = bg[sh:sh+h, sw:sw+w, :]
            msk_3c = np.repeat(msk[:, :, None], 3, axis=2)
            bgr = bg * (msk_3c <= 0).astype(bg.dtype) + bgr * (msk_3c).astype(bg.dtype)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            if args.vis:
                try:
                    from neupeak.utils.webcv2 import imshow, waitKey
                except:
                    from cv2 import imshow, waitKey
                imshow("bgr", bgr.astype("uint8"))
                show_zbuf = zbuf.copy()
                min_d, max_d = show_zbuf[show_zbuf > 0].min(), show_zbuf.max()
                show_zbuf[show_zbuf>0] = (show_zbuf[show_zbuf>0] - min_d) / (max_d - min_d) * 255
                show_zbuf = show_zbuf.astype(np.uint8)
                imshow("dpt", show_zbuf)
                show_msk = (msk / msk.max() * 255).astype("uint8")
                imshow("msk", show_msk)
                waitKey(0)

            data = {}
            data['depth'] = zbuf
            data['rgb'] = rgb
            data['mask'] = msk
            data['K'] = self.K
            data['RT'] = RT
            data['cls_typ'] = self.cls_type
            data['rnd_typ'] = 'render'
            data_str = pkl.dumps(data)
            sv_pth = os.path.join(self.render_dir, "{}.pkl".format(idx))
            if DEBUG:
                imshow("rgb", rgb[:, :, ::-1].astype("uint8"))
                imshow("depth", (zbuf / zbuf.max() * 255).astype("uint8"))
                imshow("mask", (msk/ msk.max() * 255).astype("uint8"))
                waitKey(0)
            pkl.dump(data, open(sv_pth, "wb"))
            pth_lst.append(os.path.abspath(sv_pth))

        plst_pth = os.path.join(self.render_dir, "file_list.txt")
        with open(plst_pth, 'w') as of:
            for pth in pth_lst:
                print(pth, file=of)


def main():
    cls_type = 'cam'
    render_num = 70000
    if len(args.cls) > 0:
        cls_type = args.cls
    if args.render_num > 0:
        render_num = args.render_num
    print("cls: ", cls_type)
    gen = LineModRenderDB(cls_type, render_num, True)
    gen.gen_pack_zbuf_render()

    # for cls_type in OBJ_ID_DICT.keys():
    #     pth_lst = []
    #     render_num = 70000
    #     gen = LineModRenderDB(cls_type, render_num, True)
    #     gen.gen_pack_zbuf_render()



if __name__ == "__main__":
    main()
# vim: ts=4 sw=4 sts=4 expandtab
