import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import analysis_setup, utils, naming
from gazeMapper import config, episode, naming as gm_naming, session
from glassesTools import naming as gt_naming


data_dir = pathlib.Path('data')
data_dir.mkdir(exist_ok=True)
plot_dir = pathlib.Path('figures')
plot_dir.mkdir(exist_ok=True)


# get project config
gazeMapper_project_path = analysis_setup.gazeMapper_projects_path / analysis_setup.gazeMapper_project_names[2]
if not gazeMapper_project_path.is_dir():
    print(f'The gazeMapper project path "{gazeMapper_project_path}" does not exist, not processing station 2 slippage analysis.')
    exit(0)

config_dir = config.guess_config_dir(gazeMapper_project_path)
study_config = config.Study.load_from_json(config_dir)
et_recs      = [r for r in study_config.session_def.recordings if r.type==session.RecordingType.Eye_Tracker]
if len(et_recs)!=1:
    raise RuntimeError('A project with one eye tracker recording per session is expected')
et_rec = et_recs[0]

# get sessions
sessions = session.get_sessions_from_project_directory(gazeMapper_project_path, study_config.session_def)

# process each session
data = []
et_infos = {}
for s in sessions:
    rec_dir = s.working_directory / et_rec.name
    if et_rec.name not in s.recordings:
        print(f'No eye tracker recording in "{rec_dir}", skipping...')
        continue
    et  = s.recordings[et_rec.name].info.eye_tracker
    pid = s.name.split('_')[0]
    et_nm = et.value
    et_infos[(pid, et_nm)] = {k: getattr(s.recordings[et_rec.name].info,k) for k in ['firmware_version', 'recording_software_version']}

    # check required files are present, and load
    if not (offsets_file:=rec_dir/f'{gm_naming.gaze_offset_prefix}{analysis_setup.slippage_offset_suffix}.tsv').is_file():
        print(f'The slippage offsets file "{offsets_file.name}" does not exist for the recording in "{rec_dir}", skipping...')
        continue
    offsets = pd.read_csv(offsets_file, sep='\t')
    if not (coding_file:=rec_dir/gm_naming.coding_file).is_file():
        print(f'The episode coding file "{coding_file.name}" does not exist for the recording in "{rec_dir}", skipping...')
        continue
    coding = episode.list_to_marker_dict(episode.read_list_from_file(coding_file))
    if not (gaze_data_file:=rec_dir/gt_naming.gaze_data_fname).is_file():
        print(f'The gaze data file "{gaze_data_file.name}" does not exist for the recording in "{rec_dir}", skipping...')
        continue
    gaze_data = pd.read_csv(gaze_data_file, sep='\t')

    print(f'Processing session "{s.name}", recording directory "{rec_dir}"...')

    # prep for plot of the trials
    fig, ax = plt.subplots(3, 1)
    fig.supylabel("Gaze offset (deg)")
    for a in ax:
        a.set_ylim(-20, +20)    # Y-axis range
        a.invert_yaxis()

    # process slippage trials in sorted value order, so they appear in the plots and tables in the same order
    slippage_trial_items = sorted(analysis_setup.slippage_trials.items(), key=lambda x: x[1])
    for i, (trial, _) in enumerate(slippage_trial_items):
        if trial not in coding:
            print(f'No coding for trial type "{trial}" in session "{s.name}", skipping...')
            continue
        t_coding = coding[trial][1][0]  # only one episode expected
        # get data for this episode
        trial_gaze = offsets.loc[(offsets.frame_idx_VOR>=t_coding[0]) & (offsets.frame_idx_VOR<=t_coding[1])]
        # skip initial n seconds
        trial_gaze = trial_gaze.loc[trial_gaze.timestamp_VOR >= (trial_gaze.timestamp_VOR.iloc[0] + analysis_setup.slippage_skip_dur*1000)]

        # plot
        t = trial_gaze.timestamp_VOR - trial_gaze.timestamp_VOR.iloc[0]
        ax[i].plot(t, trial_gaze.offset_x_target_1_pose_vidpos_ray, label='x')
        ax[i].plot(t, trial_gaze.offset_y_target_1_pose_vidpos_ray, label='y')

        if i == len(analysis_setup.slippage_trials)-1:
            ax[i].set_xlabel('Time (ms)')
            ax[i].legend(ncol=2, loc='lower left', fontsize=8)
        else:
            ax[i].get_xaxis().set_visible(False)
        if i == 0:
            ax[i].set_title(f'{pid}')

        # indicate on axis what data direction is left/up and right/down
        for anc, ha, lbl in zip((1.,0.), ('right','left'), ('left, up', 'right, down')):
            ax[i].text(
                1.0, anc, lbl,
                rotation=90, rotation_mode='anchor',
                ha=ha, va='bottom',
                transform=ax[i].transAxes,
                fontsize=6, color='0.5'
            )

        ax[i].yaxis.set_label_position("right")  # Move label to right
        ax[i].set_ylabel(analysis_setup.slippage_trials[trial], rotation=270, labelpad=15)

        # compute shift range
        rx = np.nanmean(utils.local_robust_range(trial_gaze.offset_x_target_1_pose_vidpos_ray, window_size=analysis_setup.eye_trackers[et]['sample_rate']*2))
        ry = np.nanmean(utils.local_robust_range(trial_gaze.offset_y_target_1_pose_vidpos_ray, window_size=analysis_setup.eye_trackers[et]['sample_rate']*2))

        # Compute effective frequency and data loss
        # Uses the original head-referenced gaze data, so data loss is not due to issues mapping to the stimulus plane
        trial_raw_gaze = gaze_data.loc[(gaze_data.frame_idx_VOR>=t_coding[0]) & (gaze_data.frame_idx_VOR<=t_coding[1])]
        trial_raw_gaze = trial_raw_gaze.loc[trial_raw_gaze.timestamp_VOR >= (trial_raw_gaze.timestamp_VOR.iloc[0] + analysis_setup.slippage_skip_dur*1000)]
        intersample_interval = np.median(np.diff(trial_raw_gaze.timestamp))
        recording_duration = (trial_raw_gaze.timestamp.iloc[-1]-trial_raw_gaze.timestamp.iloc[0] + intersample_interval) / 1000 # to seconds
        n_samples = len(trial_raw_gaze)
        measured_Fs = n_samples / recording_duration
        percent_Fs = measured_Fs / analysis_setup.eye_trackers[et]['sample_rate']

        # Lost data
        data_loss = np.mean([np.sum(np.isnan(trial_raw_gaze.gaze_pos_vid_x)) / n_samples,
                             np.sum(np.isnan(trial_raw_gaze.gaze_pos_vid_y)) / n_samples])

        data.append({'pid': pid, 'tracker': et_nm, 'trial': trial,
                     'shift_x': rx, 'shift_y': ry,
                     'Fs': measured_Fs, 'relative_Fs': percent_Fs * 100, 'data_loss': data_loss * 100})
    # store figure
    fig.savefig(plot_dir / f'{naming.station2_1_prefix}{et_nm}_{pid}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

# make into data frame and store
if not data:
    print('No data collected, not performing further station 2 slippage analysis.')
    exit(0)
df_res = pd.DataFrame.from_records(data)
df_res.to_csv(data_dir / naming.station2_1, index=False, sep='\t', na_rep='nan', float_format='%.8f')

# also store eye tracker info
et_info = pd.DataFrame.from_dict(et_infos, orient='index')
et_info.index = pd.MultiIndex.from_tuples(et_info.index, names=['participant', 'device'])
et_info.to_csv(data_dir / f'{naming.station2_prefix}eye_tracker_info.tsv', index=True, sep='\t')