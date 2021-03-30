import argparse
import os, glob

import sys
sys.path.insert(0, os.path.join(os.environ['XIVO_ROOT'], 'lib'))
import pyxivo
import savers

sys.path.append(os.path.join(os.getcwd(), "pyutils"))
from utils import get_stock_parser
from constants import COSYVIO_COLLECTION_TIME


parser = get_stock_parser("Main Python interface for running XIVO.")
parser.add_argument(
    '-cfg', default='cfg/tumvi_cam0.json', help='path to the estimator configuration')
parser.add_argument(
    '-use_viewer', default=False, action='store_true',
    help='visualize trajectory and feature tracks if set')
parser.add_argument(
    '-mode', default='eval', help='[eval|dump|dumpCov|runOnly] mode to handle the state estimates. eval: save states for evaluation; dump: save to json file for further processing')
parser.add_argument(
    '-save_full_cov', default=False, action='store_true',
    help='save the entire covariance matrix, not just that of the motion state, if set'
)
parser.add_argument('-collection_time', default=COSYVIO_COLLECTION_TIME)




def main(args):
    if not os.path.exists(args.dump):
        os.makedirs(args.dump)

    ########################################
    # CHOOSE SAVERS
    ########################################
    if args.mode == 'eval':
        if args.dataset == 'tumvi':
            saver = savers.TUMVIEvalModeSaver(args)
        elif args.dataset == 'xivo':
            saver = savers.XIVOEvalModeSaver(args)
    elif args.mode == 'dump':
        if args.dataset == 'tumvi':
            saver = savers.TUMVIDumpModeSaver(args)
        elif args.dataset == 'xivo':
            saver = savers.XIVODumpModeSaver(args)
    elif args.mode == 'dumpCov':
        if args.dataset == 'tumvi':
            saver = savers.TUMVICovDumpModeSaver(args)
        elif args.dataset == 'xivo':
            saver = savers.XIVOCovDumpModeSaver(args)
    elif args.mode == 'runOnly':
        pass
    else:
        raise ValueError('mode=[eval|dump|dumpCov|runOnly]')

    ########################################
    # LOAD DATA
    ########################################
    if args.dataset == 'tumvi':
        img_dir = os.path.join(args.root, 'dataset-{}_512_16'.format(args.seq),
                               'mav0', 'cam{}'.format(args.cam_id), 'data')

        imu_path = os.path.join(args.root, 'dataset-{}_512_16'.format(args.seq),
                                'mav0', 'imu0', 'data.csv')
    elif args.dataset == 'xivo':
        img_dir = os.path.join(args.root, args.seq, 'cam0', 'data')
        imu_path = os.path.join(args.root, args.seq, 'imu0', 'data.csv')
    else:
        raise ValueError('unknown dataset argument; choose from tumvi, xivo, cosyvio, carla, alphred, sabr')

    data = []

    if args.dataset in ['tumvi', 'xivo']:
        for p in glob.glob(os.path.join(img_dir, '*.png')):
            ts = int(os.path.basename(p)[:-4])
            data.append((ts, p))


    with open(imu_path, 'r') as fid:
        for l in fid.readlines():
            if l[0].isdigit():
                v = l.strip().split(',')
                ts = int(v[0])
                w = [float(x) for x in v[1:4]]
                t = [float(x) for x in v[4:]]
                data.append((ts, (w, t)))

    data.sort(key=lambda tup: tup[0])

    ########################################
    # INITIALIZE ESTIMATOR
    ########################################
    viewer_cfg = ''
    if args.use_viewer:
        if args.dataset == 'tumvi':
            viewer_cfg = os.path.join('cfg', 'viewer.json')
        elif args.dataset == 'xivo':
            viewer_cfg = os.path.join('cfg', 'phab_viewer.json')

    #########################################
    # RUN ESTIMATOR AND SAVE DATA
    #########################################
    # this is wrapped in a try/finally block so that data will save even when
    # we hit an exception (namely, KeyboardInterrupt)
    try:
        estimator = pyxivo.Estimator(args.cfg, viewer_cfg, args.seq)
        for i, (ts, content) in enumerate(data):
            if i > 0 and i % 500 == 0:
                print('{:6}/{:6}'.format(i, len(data)))
            if isinstance(content, tuple):
                gyro, accel = content
                estimator.InertialMeas(ts, gyro[0], gyro[1], gyro[2], accel[0],
                                    accel[1], accel[2])
            else:
                estimator.VisualMeas(ts, content)
                estimator.Visualize()
                if args.mode != 'runOnly':
                    saver.onVisionUpdate(estimator, datum=(ts, content))

    finally:
        if args.mode != 'runOnly':
            saver.onResultsReady()


if __name__ == '__main__':
    main(args=parser.parse_args())
