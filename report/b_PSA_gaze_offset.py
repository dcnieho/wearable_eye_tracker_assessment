import pandas as pd
import numpy as np
from collections import defaultdict

import analysis_setup, naming
from gazeMapper import config, episode, plane, session
from glassesTools import data_types, gaze_headref, gaze_worldref, naming as gt_naming, ocv, pose, process_pool
from glassesTools.validation.config import get_validation_setup
from glassesTools.validation.dynamic import _get_position


# get project config
gazeMapper_project_path = analysis_setup.gazeMapper_projects_path / analysis_setup.gazeMapper_project_names[1]
config_dir = config.guess_config_dir(gazeMapper_project_path)
study_config = config.Study.load_from_json(config_dir)
et_recs      = [r for r in study_config.session_def.recordings if r.type==session.RecordingType.Eye_Tracker]
if len(et_recs)!=1:
    raise RuntimeError('A project with one eye tracker recording per session is expected')
et_rec = et_recs[0]

# get validation setup
val_planes = [p for p in study_config.planes if p.type==plane.Type.GlassesValidator]
if len(val_planes)!=1:
    raise RuntimeError('A project with one GlassesValidator plane is expected')
val_plane = val_planes[0]
validation_setup = get_validation_setup(config_dir/val_plane.name)
viewing_distance = validation_setup['distance']

# get sessions
sessions = session.get_sessions_from_project_directory(gazeMapper_project_path, study_config.session_def)

# process each session
for s in sessions:
    # check if recording is imported, if it has been coded and if its not already processed
    # per session, see if already processed
    rec_dir = s.working_directory / et_rec.name

    if et_rec.name not in s.recordings:
        print(f'No eye tracker recording in "{rec_dir}", skipping...')
        continue
    pose_file = rec_dir/naming.pose
    if not pose_file.is_file():
        print(f'The recording has not yet been processed with a_PSA_plane_detection.py, run that first. This is indicated by the "{naming.pose}" file existing in the recording directory "{rec_dir}", skipping...')
        continue
    target_file = rec_dir/naming.target
    if not target_file.is_file():
        print(f'The recording has not yet been processed with a_PSA_plane_detection.py, run that first. This is indicated by the "{naming.target}" file existing in the recording directory "{rec_dir}", skipping...')
        continue
    PSA_offset_file = rec_dir/naming.PSA_offsets
    if PSA_offset_file.is_file():
        print(f'The recording has already been processed, and will thus be skipped. This is indicated by the "{naming.PSA_offsets}" file existing in the recording directory "{rec_dir}", skipping...')
        continue

    print(f'Processing session "{s.name}", recording directory "{rec_dir}"...')

    # load files
    poses   = pose.read_dict_from_file(pose_file)
    targets = pd.read_csv(target_file, delimiter='\t', index_col='frame_idx')
    episodes = episode.load_episodes_from_all_recordings(study_config, rec_dir)[0]
    camera_params = ocv.CameraParams.read_from_file(rec_dir/gt_naming.scene_camera_calibration_fname)
    head_gazes = pd.read_csv(rec_dir / gt_naming.gaze_data_fname, sep='\t', dtype=defaultdict(lambda: float, **gaze_headref.Gaze._non_float))
    # also get target positions on the plane, in mm
    targets_plane = {t_id:[x*10. for x in _get_position(analysis_setup.PSA_target_locations[t_id], viewing_distance, 'deg')] for t_id in analysis_setup.PSA_target_locations}
    targets_plane = pd.DataFrame({'t_plane_x': [targets_plane[t_id][0] for t_id in targets_plane], 't_plane_y': [targets_plane[t_id][1] for t_id in targets_plane], 't_id': targets_plane.keys()})
    # and add to dataframe
    targets = targets.reset_index().merge(targets_plane, on='t_id', how='outer').set_index('frame_idx')

    # compute offsets to target
    # 1. get gazes
    frame_idx = 'frame_idx_VOR' if 'frame_idx_VOR' in head_gazes.columns else 'frame_idx'
    ts = 'timestamp_VOR' if 'timestamp_VOR' in head_gazes.columns else 'timestamp'
    sel = (head_gazes[frame_idx] >= episodes['PSA'][1][0][0]) & (head_gazes[frame_idx] <= episodes['PSA'][1][0][1])
    head_gazes = head_gazes[sel]
    columns = [frame_idx,ts,'gaze_pos_vid_x','gaze_pos_vid_y'] + [p for p in ('pup_diam_l', 'pup_diam_r') if p in head_gazes.columns]
    gaze_points = head_gazes[columns].rename(columns={'frame_idx_VOR':'frame_idx','timestamp_VOR':'timestamp'}).set_index('frame_idx')
    # 2. to most accurately compute 2D offsets, we need gaze positions projected to the plane
    head_gazes_dict = gaze_headref.read_dict_from_file(rec_dir / gt_naming.gaze_data_fname, episodes['PSA'][1], ts_column_suffixes=['VOR', ''])[0]
    progress_indicator = process_pool.JobProgress(printer=lambda x: print(x))
    progress_indicator.set_unit('samples')
    total = sum(len(head_gazes_dict[f]) for f in poses if f in head_gazes_dict)
    progress_indicator.set_total(total)
    progress_indicator.set_intervals(step:=min(50,int(total/200)), step)
    plane_gazes = gaze_worldref.from_head(poses, head_gazes_dict, camera_params, progress_indicator.update)
    # store to file
    gaze_worldref.write_dict_to_file(plane_gazes, rec_dir/naming.mapped_gaze, skip_missing=True)

    # 3. get offsets to target positions
    # add targets to data frame
    gaze_points     = gaze_points.join(targets[['t_id','t_img_x','t_img_y','t_plane_x','t_plane_y','is_bright_frame']],how='outer')
    gaze_points     = gaze_points.dropna(axis=0,subset=['timestamp']) # the below can merge in scene camera frames for which we have no gaze but only target data. Remove those again as they mess up down-stream analysis

    # check what gaze-on-plane data we have
    dq_have = data_types.get_available_data_types(plane_gazes)
    d_types = data_types.select_data_types_to_use([data_types.DataType.pose_vidpos_ray, data_types.DataType.pose_vidpos_homography, data_types.DataType.viewpos_vidpos_homography],
                                                  dq_have, True)
    d_type = d_types[0]  # pick first available data type

    # compute offsets per target
    targets_on_plane       = {int(r['t_id']): np.array([r['t_plane_x'], r['t_plane_y'], 0.]) for _,r in targets_plane.iterrows()}
    targets_for_homography = {int(r['t_id']): np.array([r['t_plane_x'], r['t_plane_y'], viewing_distance*10.]) for _,r in targets_plane.iterrows()}
    gaze_points = gaze_points.reset_index().set_index('timestamp')  # output of below is indexed by timestamp (multiple gaze samples per frame possible)
    for t in targets_for_homography:
        # find frames when target is shown
        q_target = targets[targets.t_id==t]
        this_gaze = {k:v for (k,v) in plane_gazes.items() if k in q_target.index}
        frame_idxs, timestamps, offsets = data_types.calculate_gaze_angles_to_point(
                    this_gaze,
                    poses,
                    {t: targets_on_plane[t]},
                    [d_type],
                    {t: targets_for_homography[t]},
                    viewing_distance*10.
                    )
        # add to output
        gaze_points.loc[timestamps, ('offset', 'offset_x', 'offset_y')] = offsets[t][d_type]
    # back to frame_idx as index
    gaze_points = gaze_points.reset_index().set_index('frame_idx')

    # 4. store to file
    gaze_points.to_csv(PSA_offset_file, sep='\t', na_rep='nan', float_format='%.8f')