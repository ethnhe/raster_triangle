cls_lst=('ape' 'benchvise' 'cam' 'can' 'cat' 'driller' 'duck' 'eggbox' 'glue' 'holepuncher' 'iron' 'lamp' 'phone')
cls=${cls_lst[12]}
python3 rgbd_renderer.py --cls_type ${cls}
