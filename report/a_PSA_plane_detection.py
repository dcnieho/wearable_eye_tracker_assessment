import pandas as pd

import analysis_setup, naming, point_fiducial_detector
from gazeMapper import config, episode, naming as gm_naming, plane, session
from glassesTools import annotation, eyetracker, naming as gt_naming, pose, process_pool, propagating_thread, timestamps
from glassesTools.validation.config import get_validation_setup
from glassesTools.gui.video_player import GUI

# config
show_GUI = False

def do_the_work(gui: GUI | None):
    # get project config
    gazeMapper_project_path = analysis_setup.gazeMapper_projects_path / analysis_setup.gazeMapper_project_names[1]
    config_dir   = config.guess_config_dir(gazeMapper_project_path)
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
    has_started = False
    for s in sessions:
        # check if recording is imported, if it has been coded and if its not already processed
        # per session, see if already processed
        rec_dir = s.working_directory / et_rec.name
        if not rec_dir.is_dir():
            print(f'The eye tracker recording working directory "{rec_dir}" does not exist, skipping...')
            continue
        pose_file = rec_dir/naming.pose
        if pose_file.is_file():
            print(f'The recording has already been processed, and will thus be skipped. This is indicated by the "{naming.pose}" file existing in the recording directory "{rec_dir}", skipping...')
            continue
        # get interval to analyze
        if not (rec_dir/gm_naming.coding_file).is_file():
            print(f'The episode coding file "{gm_naming.coding_file}" does not exist for the recording in "{rec_dir}", skipping...')
            continue
        episodes = episode.load_episodes_from_all_recordings(study_config, rec_dir)[0]
        if 'PSA' not in episodes or len(episodes['PSA'][1])!=1:
            print(f'The episode coding file "{gm_naming.coding_file}" should contain a single PSA episode, but it has none or more for recording in "{rec_dir}", skipping...')
            continue

        print(f'Processing session "{s.name}", recording directory "{rec_dir}"...')

        # ok, we have a recording to process. set up the estimator
        in_video = session.get_video_path(s.recordings[et_rec.name].info)
        estimator= pose.Estimator(in_video, rec_dir/gt_naming.frame_timestamps_fname, rec_dir/gt_naming.scene_camera_calibration_fname)
        extra = {}
        if s.recordings[et_rec.name].info.eye_tracker==eyetracker.EyeTracker.SeeTrue_STONE:
            extra['min_radius_threshold'] = 3
        if s.recordings[et_rec.name].info.eye_tracker==eyetracker.EyeTracker.VPS_19:
            extra['max_radius'] = 20000
        if s.recordings[et_rec.name].info.eye_tracker in analysis_setup.blackout_dict:
            extra['blackout_rect'] = analysis_setup.blackout_dict[s.recordings[et_rec.name].info.eye_tracker]
        detector = point_fiducial_detector.Detector(viewing_distance, analysis_setup.PSA_phi, analysis_setup.PSA_rho, analysis_setup.PSA_target_locations, edge_cut_fac=0, **extra)
        # add plane and targets
        estimator.add_plane('fiducials', detector.detect_plane, episodes['PSA'], detector.visualize_plane)
        estimator.register_extra_processing_fun('target'      , episodes['PSA'], detector.detect_target      , {}, detector.visualize_target)
        estimator.register_extra_processing_fun('bright_frame', episodes['PSA'], detector.detect_bright_frame, {}, detector.visualize_bright_frame)
        # prep gui, if any
        estimator.attach_gui(gui)
        if gui is not None:
            gui.set_window_title(s.name)
            gui.set_show_timeline(True, timestamps.VideoTimestamps(rec_dir/gt_naming.frame_timestamps_fname), annotation.flatten_annotation_dict(episodes))
        # prep progress indicator
        progress_indicator = process_pool.JobProgress(printer=lambda x: print(x))
        progress_indicator.set_unit('frames')
        progress_indicator.set_total(total:=estimator.video_ts.get_last()[0])
        progress_indicator.set_intervals(step:=min(20,int(total/200)), step)
        estimator.set_progress_updater(progress_indicator.update)

        # run
        has_started = True
        poses, _, extra = estimator.process_video()

        # store pose
        pose.write_list_to_file(poses['fiducials'], pose_file, skip_failed=True)
        # handle targets
        targets = [[t[0], t_id]+t[1][t_id] for t in extra['target'] for t_id in t[1]]
        targets = pd.DataFrame(targets,columns=['frame_idx','t_id','t_img_x','t_img_y','t_screen_x','t_screen_y'])
        # add bright frame info
        targets = targets.merge(pd.DataFrame(extra['bright_frame'],columns=['frame_idx','is_bright_frame']), how='outer', on='frame_idx')
        targets = targets.set_index('frame_idx')
        targets['is_bright_frame'] *= 1 # cast bool to int
        # store to file
        targets.to_csv(rec_dir/naming.target, sep='\t', na_rep='nan', float_format='%.8f')

    if not has_started:
        raise RuntimeError('Nothing to analyze')

if show_GUI:
    gui = GUI(use_thread = False)
    gui.add_window('placeholder name')
    gui.set_interruptible(False)
    gui.set_detachable(True)
    gui.set_show_controls(True)
    gui.set_show_play_percentage(True)
    gui.set_show_action_tooltip(True)

    proc_thread = propagating_thread.PropagatingThread(target=do_the_work, args=(gui,), cleanup_fun=gui.stop)
    proc_thread.start()
    gui.start()
    proc_thread.join()
else:
    do_the_work(None)