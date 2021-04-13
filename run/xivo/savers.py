import numpy as np
import os, sys
import json
from transforms3d.quaternions import mat2quat

from cosyvio_prep import cosyvio_extrinsics_data as cd


sys.path.append(os.path.join(os.getcwd(), "pyutils"))
from to_json import to_json
from utils import upper_triangular_list, get_xivo_gt_filename, \
    get_xivo_output_filename


class BaseSaver:
    """Abstract class that outlines the functions that the other savers need
    to have."""
    def __init__(self, args):
        self.results = []
        self.resultsPath = get_xivo_output_filename(args.dump, args.dataset,
            args.seq, cam_id=args.cam_id, sen=args.sen)
    def onVisionUpdate(self, estimator, datum):
        pass
    def onResultsReady(self):
        pass


class TUMVISaver:

    def __init__(self, args):
        # parse mocap and save gt in desired format
        mocapPath = os.path.join(args.root, 'dataset-{}_512_16'.format(args.seq),
                                 'mav0', 'mocap0', 'data.csv')
        groundtruthPath = get_xivo_gt_filename(args.dump, "tumvi", args.seq)
        self.saveMocapAs(mocapPath, groundtruthPath)

    def saveMocapAs(self, mocapPath, groundtruthPath):
        gt = []
        with open(mocapPath, 'r') as fid:
            for l in fid.readlines():
                if l[0] != '#':
                    v = l.strip().split(',')
                    if (len(v) >= 8):
                        ts = int(v[0])
                        t = [float(x) for x in v[1:4]]
                        q = [float(x) for x in v[4:]]  # [w, x, y, z]
                        gt.append(
                            [ts * 1e-9, t[0], t[1], t[2], q[1], q[2], q[3], q[0]])

        np.savetxt(
            groundtruthPath,
            gt,
            fmt='%f %f %f %f %f %f %f %f')



class EvalModeSaver(BaseSaver):
    """ Callback functions used in eval mode of pyxivo.
    """
    def __init__(self, args):
        BaseSaver.__init__(self, args)

    def onVisionUpdate(self, estimator, datum):
        now = estimator.now()
        gsb = np.array(estimator.gsb())
        Tsb = gsb[:, 3]

        # print gsb[:3, :3]
        try:
            q = mat2quat(gsb[:3, :3])  # [w, x, y, z]
            # format compatible with tumvi rgbd benchmark scripts
            self.results.append(
                [now * 1e-9, Tsb[0], Tsb[1], Tsb[2], q[1], q[2], q[3], q[0]])
        except np.linalg.linalg.LinAlgError:
            pass

    def onResultsReady(self):
        np.savetxt(
            self.resultsPath,
            self.results,
            fmt='%f %f %f %f %f %f %f %f')



class DumpModeSaver(BaseSaver):
    """ Callback functions used by dump mode of pyxivo.
    """
    def __init__(self, args):
        BaseSaver.__init__(self, args)

    def onVisionUpdate(self, estimator, datum):
        ts, content = datum
        #now = estimator.now()
        g = np.array(estimator.gsc())
        T = g[:, 3]

        if np.linalg.norm(T) > 0:
            try:
                q = mat2quat(g[:3, :3])  # [w, x, y, z]
                # format compatible with tumvi rgbd benchmark scripts
                entry = dict()
                entry['ImagePath'] = str(content)
                entry['Timestamp'] = ts
                entry['TranslationXYZ'] = [T[0], T[1], T[2]]
                entry['QuaternionWXYZ'] = [q[0], q[1], q[2], q[3]]
                self.results.append(entry)

                with open(self.resultsPath, 'w') as fid:
                    json.dump(self.results, fid, indent=2)
            except np.linalg.linalg.LinAlgError:
                pass

    def onResultsReady(self):
        with open(self.resultsPath, 'w') as fid:
            json.dump(self.results, fid, indent=2)


class CovDumpModeSaver(BaseSaver):
    def __init__(self, args):
        BaseSaver.__init__(self, args)
        self.save_full_cov = args.save_full_cov

    def onVisionUpdate(self, estimator, datum):
        ts, content = datum

        # Get camera pose
        gsb = np.array(estimator.gsb())
        gbc = np.array(estimator.gbc())
        gsc = np.array(estimator.gsc())
        qsb = mat2quat(gsb[:3, :3])
        Tsb = gsb[:, 3]
        qbc = mat2quat(gbc[:3, :3])
        Tbc = gbc[:, 3]
        qsc = mat2quat(gsc[:3, :3])
        Tsc = gsc[:, 3]
        Vsb = np.array(estimator.Vsb())

        # Get calibration states
        bg = estimator.bg()
        ba = estimator.ba()
        qg = mat2quat(estimator.Rg())
        td = estimator.td()
        Ca = estimator.Ca()
        Cg = estimator.Cg()

        # Get filter covariance
        if self.save_full_cov:
            Pstate = np.array(estimator.P())
        else:
            Pstate = np.array(estimator.Pstate())

        # Group ID
        GaugeGroup = estimator.gauge_group()

        # Get filter innovations
        if not estimator.MeasurementUpdateInitialized():
            inn_Tsb = np.zeros((3,))
            inn_Wsb = np.zeros((3,))
            inn_Vsb = np.zeros((3,))
        else:
            inn_Tsb = np.array(estimator.inn_Tsb())
            inn_Wsb = np.array(estimator.inn_Wsb())
            inn_Vsb = np.array(estimator.inn_Vsb())

        # Get features
        num_instate_features = estimator.num_instate_features()
        if num_instate_features > 0:
            feature_positions = estimator.InstateFeaturePositions()
            feature_covs = estimator.InstateFeatureCovs()
            feature_ids = estimator.InstateFeatureIDs()
            feature_sinds = estimator.InstateFeatureSinds()
        else:
            feature_positions = []
            feature_covs = []
            feature_ids = []
            feature_sinds = []

        # Get groups
        num_instate_groups = estimator.num_instate_groups()
        if num_instate_groups > 0:
            group_covs = estimator.InstateGroupCovs()
            group_sinds = estimator.InstateGroupSinds()
            group_ids = estimator.InstateGroupIDs()
            group_poses = estimator.InstateGroupPoses()
        else:
            group_poses = []
            group_covs = []
            group_ids = []
            group_sinds = []

        # Save results
        entry = dict()

        entry['group'] = GaugeGroup

        entry['ImagePath'] = str(content)
        entry['Timestamp'] = ts

        entry['Tsb_XYZ'] = Tsb.tolist()
        entry['qsb_WXYZ'] = qsb.tolist()
        entry['Tbc_XYZ'] = Tbc.tolist()
        entry['qbc_WXYZ'] = qbc.tolist()
        entry['Tsc_XYZ'] = Tsc.tolist()
        entry['qsc_WXYZ'] = qsc.tolist()
        entry['Vsb_XYZ'] = Vsb.tolist()

        entry['Pstate'] = upper_triangular_list(Pstate)

        entry['MeasurementUpdateInitialized'] = estimator.MeasurementUpdateInitialized()
        entry['inn_Tsb'] = inn_Tsb.tolist()
        entry['inn_Wsb'] = inn_Wsb.tolist()
        entry['inn_Vsb'] = inn_Vsb.tolist()

        entry['bg'] = bg.tolist()
        entry['ba'] = ba.tolist()
        entry['qg_WXYZ'] = qg.tolist()
        entry['td'] = td
        entry['Ca'] = Ca
        entry['Cg'] = Cg

        entry['num_instate_features'] = num_instate_features
        entry['feature_positions'] = feature_positions
        entry['feature_covs'] = feature_covs
        entry['feature_ids'] = feature_ids
        entry['feature_sinds'] = feature_sinds

        entry['num_instate_groups'] = num_instate_groups
        entry['group_poses'] = group_poses
        entry['group_covs'] = group_covs
        entry['group_ids'] = group_ids
        entry['group_sinds'] = group_sinds

        self.results.append(entry)


    def onResultsReady(self):
        with open(self.resultsPath, 'w') as fid:
            output = { "data": self.results }
            fid.write(to_json(output))


class TUMVIEvalModeSaver(EvalModeSaver, TUMVISaver):
    def __init__(self, args):
        EvalModeSaver.__init__(self, args)
        TUMVISaver.__init__(self, args)


class TUMVIDumpModeSaver(DumpModeSaver, TUMVISaver):
    def __init__(self, args):
        DumpModeSaver.__init__(self, args)
        TUMVISaver.__init__(self, args)


class TUMVICovDumpModeSaver(CovDumpModeSaver, TUMVISaver):
    def __init__(self, args):
        CovDumpModeSaver.__init__(self, args)
        TUMVISaver.__init__(self, args)


class XIVOEvalModeSaver(EvalModeSaver, BaseSaver):
    def __init__(self, args):
        EvalModeSaver.__init__(self, args)
        BaseSaver.__init__(self, args)


class XIVODumpModeSaver(DumpModeSaver, BaseSaver):
    def __init__(self, args):
        DumpModeSaver.__init__(self, args)
        BaseSaver.__init__(self, args)


class XIVOCovDumpModeSaver(CovDumpModeSaver, BaseSaver):
    def __init__(self, args):
        CovDumpModeSaver.__init__(self, args)
        BaseSaver.__init__(self, args)
