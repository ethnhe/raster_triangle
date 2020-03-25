import os
import time
import cv2
import pickle
import yaml
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import ImageFile, Image
from plyfile import PlyData
from concurrent.futures import ProcessPoolExecutor
from argparse import ArgumentParser
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
    '--fuse_num', type=int, default=10000,
    help="Number of images you want to generate."
)
parser.add_argument(
    '--DEBUG', action="store_true",
    help="To show the generated images or not."
)
args = parser.parse_args()
DEBUG = args.DEBUG

Intrinsic_matrix = {
    'linemod': np.array([[572.4114, 0., 325.2611],
                          [0., 573.57043, 242.04899],
                          [0., 0., 1.]]),
    'blender': np.array([[700.,    0.,  320.],
                         [0.,  700.,  240.],
                         [0.,    0.,    1.]])
}
lm_obj_dict={
    'ape':1,
    'benchvise':2,
    'cam':4,
    'can':5,
    'cat':6,
    'driller':8,
    'duck':9,
    'eggbox':10,
    'glue':11,
    'holepuncher':12,
    'iron':13,
    'lamp':14,
    'phone':15,
}
root = './Linemod_preprocessed'
cls_root_ptn = os.path.join(root, "data/%02d/")


def ensure_dir(pth):
    if not os.path.exists(pth):
        os.system("mkdir -p {}".format(pth))


def read_lines(pth):
    with open(pth, 'r') as f:
        return [
            line.strip() for line in f.readlines()
        ]


def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def save_pickle(data, pkl_path):
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)


def collect_train_info(cls_name):
    cls_id = lm_obj_dict[cls_name]
    cls_root = cls_root_ptn % cls_id
    tr_pth = os.path.join(
        cls_root, "train.txt"
    )
    train_fns = read_lines(tr_pth)

    return train_fns


def collect_linemod_set_info(
        linemod_dir, linemod_cls_name, cache_dir='./data/LINEMOD/cache'
):
    database = []
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)
    if os.path.exists(
            os.path.join(cache_dir,'{}_info.pkl').format(linemod_cls_name)
    ):
        return read_pickle(
            os.path.join(cache_dir,'{}_info.pkl').format(linemod_cls_name)
        )

    train_fns = collect_train_info(linemod_cls_name)
    cls_id = lm_obj_dict[linemod_cls_name]
    cls_root = cls_root_ptn % cls_id
    meta_file = open(os.path.join(cls_root, 'gt.yml'), "r")
    meta_lst = yaml.load(meta_file)

    print('begin generate database {}'.format(linemod_cls_name))
    rgb_ptn = os.path.join(cls_root, "rgb/{}.png")
    msk_ptn = os.path.join(cls_root, "mask/{}.png")
    dpt_ptn = os.path.join(cls_root, "depth/{}.png")
    for item in train_fns:
        data={}
        data['rgb_pth'] = rgb_ptn.format(item)
        data['dpt_pth'] = dpt_ptn.format(item)
        data['msk_pth'] = msk_ptn.format(item)

        meta = meta_lst[int(item)]
        if cls_id == 2:
            for i in range(0, len(meta)):
                if meta[i]['obj_id'] == 2:
                    meta = meta[i]
                    break
        else:
            meta = meta[0]
        R = np.resize(np.array(meta['cam_R_m2c']), (3, 3))
        T = np.array(meta['cam_t_m2c']) / 1000.0
        RT = np.concatenate((R, T[:, None]), axis=1)
        data['RT'] = RT
        database.append(data)

    print(
        'successfully generate database {} len {}'.format(
            linemod_cls_name, len(database)
        )
    )
    save_pickle(
        database, os.path.join(cache_dir,'{}_info.pkl').format(linemod_cls_name)
    )
    return database


def randomly_read_background(background_dir,cache_dir):
    if os.path.exists(os.path.join(cache_dir,'background_info.pkl')):
        fns = read_pickle(os.path.join(cache_dir,'background_info.pkl'))
    else:
        fns = glob(os.path.join(background_dir,'*.jpg')) + \
            glob(os.path.join(background_dir,'*.png'))
        save_pickle(fns, os.path.join(cache_dir,'background_info.pkl'))

    return cv2.imread(fns[np.random.randint(0,len(fns))])[:, :, ::-1]


def fuse_regions(rgbs, masks, depths, begins, cls_ids, background, th, tw, cls):
    fuse_order = np.arange(len(rgbs))
    np.random.shuffle(fuse_order)
    fuse_img = background
    fuse_img = cv2.resize(fuse_img,(tw,th),interpolation=cv2.INTER_LINEAR)
    fuse_mask = np.zeros([fuse_img.shape[0],fuse_img.shape[1]],np.int32)
    INF = pow(2,15)
    fuse_depth = np.ones([fuse_img.shape[0], fuse_img.shape[1]], np.uint16) * INF
    t_cls_id = lm_obj_dict[cls]
    if len(background.shape) < 3:
        return None, None, None, None
    for idx in fuse_order:
        if len(rgbs[idx].shape) < 3:
            continue
        cls_id = cls_ids[idx]
        rh,rw = masks[idx].shape
        if cls_id == t_cls_id:
            bh, bw = begins[idx][0], begins[idx][1]
        else:
            bh = np.random.randint(0,fuse_img.shape[0]-rh)
            bw = np.random.randint(0,fuse_img.shape[1]-rw)

        silhouette = masks[idx]>0
        out_silhouette = np.logical_not(silhouette)
        fuse_depth_patch = fuse_depth[bh:bh+rh, bw:bw+rw].copy()
        cover = (depths[idx] < fuse_depth_patch) * silhouette
        not_cover = np.logical_not(cover)

        fuse_mask[bh:bh+rh,bw:bw+rw] *= not_cover.astype(fuse_mask.dtype)
        cover_msk = masks[idx] * cover.astype(masks[idx].dtype)
        fuse_mask[bh:bh+rh,bw:bw+rw] += cover_msk

        fuse_img[bh:bh+rh,bw:bw+rw] *= not_cover.astype(fuse_img.dtype)[:,:,None]
        cover_rgb = rgbs[idx] * cover.astype(rgbs[idx].dtype)[:,:,None]
        fuse_img[bh:bh+rh,bw:bw+rw] += cover_rgb

        fuse_depth[bh:bh+rh, bw:bw+rw] *= not_cover.astype(fuse_depth.dtype)
        cover_dpt = depths[idx] * cover.astype(depths[idx].dtype)
        fuse_depth[bh:bh+rh, bw:bw+rw] += cover_dpt.astype(fuse_depth.dtype)

        begins[idx][0] = -begins[idx][0]+bh
        begins[idx][1] = -begins[idx][1]+bw

    dp_bg = (fuse_depth == INF)
    dp_bg_filter = np.logical_not(dp_bg)
    fuse_depth *= dp_bg_filter.astype(fuse_depth.dtype)

    return fuse_img, fuse_mask, fuse_depth, begins


def randomly_sample_foreground(image_db, linemod_dir):
    idx = np.random.randint(0,len(image_db))
    rgb_pth = image_db[idx]['rgb_pth']
    dpt_pth = image_db[idx]['dpt_pth']
    msk_pth = image_db[idx]['msk_pth']
    with Image.open(dpt_pth) as di:
        depth = np.array(di).astype(np.int16)
    with Image.open(msk_pth) as li:
        mask = np.array(li).astype(np.int16)
    with Image.open(rgb_pth) as ri:
        rgb = np.array(ri)[:, :, :3].astype(np.uint8)

    mask = np.sum(mask,2)>0
    mask = np.asarray(mask,np.int32)

    hs, ws = np.nonzero(mask)
    hmin, hmax = np.min(hs),np.max(hs)
    wmin, wmax = np.min(ws),np.max(ws)

    mask = mask[hmin:hmax,wmin:wmax]
    rgb = rgb[hmin:hmax,wmin:wmax]
    depth = depth[hmin:hmax, wmin:wmax]

    rgb *= mask.astype(np.uint8)[:,:,None]
    depth *= mask.astype(np.uint16)[:,:]
    begin = [hmin,wmin]
    pose = image_db[idx]['RT']

    return rgb, mask, depth, begin, pose


def save_fuse_data(
    output_dir, idx, fuse_img, fuse_mask, fuse_depth, fuse_begins, t_pose, cls
):
    cls_id = lm_obj_dict[cls]
    if (fuse_mask == cls_id).sum() < 20:
        return None
    os.makedirs(output_dir, exist_ok=True)
    fuse_mask = fuse_mask.astype(np.uint8)
    data = {}
    data['rgb'] = fuse_img
    data['mask'] = fuse_mask
    data['depth'] = fuse_depth.astype(np.float32) / 1000.0
    data['K'] = Intrinsic_matrix['linemod']
    data['RT'] = t_pose
    data['cls_typ'] = cls
    data['rnd_typ'] = 'fuse'
    data['begins'] = fuse_begins
    if DEBUG:
        imshow("rgb", fuse_img[:, :, ::-1])
        imshow("depth", (fuse_depth / fuse_depth.max() * 255).astype('uint8'))
        imshow("label", (fuse_mask / fuse_mask.max() * 255).astype("uint8"))
        waitKey(0)
    sv_pth = os.path.join(output_dir, "{}.pkl".format(idx))
    pickle.dump(data, open(sv_pth, 'wb'))
    sv_pth = os.path.abspath(sv_pth)
    return sv_pth


def prepare_dataset_single(
    output_dir, idx, linemod_dir, background_dir, cache_dir, seed, cls
):
    time_begin = time.time()
    np.random.seed(seed)
    rgbs, masks, depths, begins, poses, cls_ids = [], [], [], [], [], []
    image_dbs={}
    for cls_name in lm_obj_dict.keys():
        cls_id = lm_obj_dict[cls_name]
        image_dbs[cls_id] = collect_linemod_set_info(
            linemod_dir, cls_name, cache_dir
        )

    for cls_name in lm_obj_dict.keys():
        cls_id = lm_obj_dict[cls_name]
        rgb, mask, depth, begin, pose = randomly_sample_foreground(
            image_dbs[cls_id], linemod_dir
        )
        if cls_name == cls:
            t_pose = pose
        mask *= cls_id
        rgbs.append(rgb)
        masks.append(mask)
        depths.append(depth)
        begins.append(begin)
        poses.append(pose)
        cls_ids.append(cls_id)

    background = randomly_read_background(background_dir, cache_dir)

    fuse_img, fuse_mask, fuse_depth, fuse_begins= fuse_regions(
        rgbs, masks, depths, begins, cls_ids, background, 480, 640, cls
    )

    if fuse_img is not None:
        sv_pth = save_fuse_data(
            output_dir, idx, fuse_img, fuse_mask, fuse_depth, fuse_begins,
            t_pose, cls
        )
        return sv_pth


def prepare_dataset_parallel(
        output_dir, linemod_dir, fuse_num, background_dir, cache_dir,
        worker_num=8, cls="ape"
):
    exector = ProcessPoolExecutor(max_workers=worker_num)
    futures = []

    for cls_name in lm_obj_dict.keys():
        collect_linemod_set_info(linemod_dir, cls_name, cache_dir)
    randomly_read_background(background_dir, cache_dir)

    for idx in np.arange(fuse_num):
        seed = np.random.randint(500000)
        futures.append(exector.submit(
            prepare_dataset_single, output_dir, idx, linemod_dir,
            background_dir, cache_dir, seed, cls
        ))

    pth_lst = []
    for f in tqdm(futures):
        res = f.result()
        if res is not None:
            pth_lst.append(res)
    f_lst_pth = os.path.join(output_dir, "file_list.txt")
    with open(f_lst_pth, "w") as f:
        for item in pth_lst:
            print(item, file=f)


if __name__=="__main__":
    cls = args.cls
    linemod_dir = './Linemod_preprocessed'
    output_dir = os.path.join(linemod_dir, "fuse",  cls)
    ensure_dir(output_dir)
    background_dir = './SUN2012pascalformat/JPEGImages'
    cache_dir = './'
    fuse_num = args.fuse_num
    worker_num = 20
    prepare_dataset_parallel(
        output_dir, linemod_dir, fuse_num, background_dir, cache_dir,
        worker_num, cls=cls
    )
