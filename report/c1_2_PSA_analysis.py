import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import analysis_setup, naming, utils
from gazeMapper import config, session


data_dir = pathlib.Path('data')
data_dir.mkdir(exist_ok=True)
plot_dir = pathlib.Path('figures')
plot_dir.mkdir(exist_ok=True)


# get project config
gazeMapper_project_path = analysis_setup.gazeMapper_projects_path / analysis_setup.gazeMapper_project_names[1]
if not gazeMapper_project_path.is_dir():
    print(f'The gazeMapper project path "{gazeMapper_project_path}" does not exist, not processing station 1 PSA analysis.')
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
for s in sessions:
    rec_dir = s.working_directory / et_rec.name
    if et_rec.name not in s.recordings:
        print(f'No eye tracker recording in "{rec_dir}", skipping...')
        continue
    et  = s.recordings[et_rec.name].info.eye_tracker
    pid = s.name.split('_')[0]
    et_nm = et.value

    offsets_file = rec_dir/naming.PSA_offsets
    if not offsets_file.is_file():
        print(f'The recording has not yet been processed with b_PSA_gaze_offset.py, run that first. This is indicated by the "{naming.PSA_offsets}" file existing in the recording directory "{rec_dir}", skipping...')
        continue

    print(f'Processing session "{s.name}", recording directory "{rec_dir}"...')

    # load files
    offsets = pd.read_csv(offsets_file, sep='\t')

    # determine duration and effective Fs
    intersample_interval = np.median(np.diff(offsets.timestamp))
    recording_duration = (offsets.timestamp.iloc[-1]-offsets.timestamp.iloc[0] + intersample_interval) / 1000 # to seconds
    n_samples = len(offsets)
    measured_Fs = n_samples / recording_duration
    percent_Fs = measured_Fs / analysis_setup.eye_trackers[et]['sample_rate']

    # Compute offset between dark and bright stimulus
    # take median gaze location 200 ms before offset of dark and bright screen
    window = int(analysis_setup.eye_trackers[et]['sample_rate'] * 0.2) # 200 ms window
    # offset of bright period: 111111->000000
    # offset of   dark period: 000000->111111
    is_bright         = offsets.is_bright_frame.to_numpy()
    offset_bright_idx = np.where(np.diff(       is_bright                                             )==-1)[0]
    offset_dark_idx   = np.where(np.diff(np.pad(is_bright, (0, 1), mode='constant', constant_values=1))== 1)[0]

    # compute offset and some other variables for each interval
    for k in np.arange(len(offset_bright_idx)):
        if np.isnan(offsets.t_id[offset_dark_idx[k]]):
            continue

        offset_bright_x = np.nanmedian(offsets.offset_x[(offset_bright_idx[k] - window): offset_bright_idx[k]])
        offset_bright_y = np.nanmedian(offsets.offset_y[(offset_bright_idx[k] - window): offset_bright_idx[k]])

        offset_dark_x = np.nanmedian(offsets.offset_x[(offset_dark_idx[k] - window): offset_dark_idx[k]])
        offset_dark_y = np.nanmedian(offsets.offset_y[(offset_dark_idx[k] - window): offset_dark_idx[k]])

        offset_dark_bright_x = offset_dark_x - offset_bright_x
        offset_dark_bright_y = offset_dark_y - offset_bright_y
        offset_dark_bright = np.hypot(offset_dark_bright_x, offset_dark_bright_y)

        pd_bright = np.nanmedian(offsets.pup_diam_l[(offset_bright_idx[k] - window): offset_bright_idx[k]]) if 'pup_diam_l' in offsets.columns else np.nan
        pd_dark   = np.nanmedian(offsets.pup_diam_l[(offset_dark_idx[k]   - window): offset_dark_idx[k]])   if 'pup_diam_l' in offsets.columns else np.nan
        if 'pup_diam_r' in offsets.columns:
            pd_bright = np.nanmean([pd_bright, np.nanmedian(offsets.pup_diam_r[(offset_bright_idx[k] - window): offset_bright_idx[k]])])
            pd_dark   = np.nanmean([  pd_dark, np.nanmedian(offsets.pup_diam_r[(offset_dark_idx[k]   - window): offset_dark_idx[k]])])

        # compute data loss for the interval
        target_id = int(offsets.t_id[offset_dark_idx[k]])
        target_location_name = analysis_setup.PSA_target_names[target_id]
        idx = offsets.t_id == target_id
        n_samples = np.sum(idx)
        # NB: note that this uses the original head-referenced gaze data, so data loss is not due to issues mapping to the stimulus plane
        lost = np.mean([np.sum(np.isnan(offsets[idx].gaze_pos_vid_x)) / n_samples,
                        np.sum(np.isnan(offsets[idx].gaze_pos_vid_y)) / n_samples])

        data.append({'pid': pid, 'tracker': et_nm,
                     'target id': target_id, 'target location': target_location_name, 'trial': k,

                     'offset_bright_x': offset_bright_x,
                     'offset_bright_y': offset_bright_y,
                     'offset_dark_x': offset_dark_x,
                     'offset_dark_y': offset_dark_y,
                     'offset_dark_bright_x': offset_dark_bright_x,
                     'offset_dark_bright_y': offset_dark_bright_y,
                     'offset_total': offset_dark_bright,

                     'pd_bright': pd_bright,
                     'pd_dark': pd_dark,
                     'pd_diff': pd_dark - pd_bright,

                     'Fs': measured_Fs, 'relative_Fs': percent_Fs * 100, 'data_loss': lost * 100})

    # plots for individual participant
    fig, ax = plt.subplots(figsize=(12,6))

    t = np.array(offsets.timestamp) / 1000
    t = t - t[0]
    ax.plot(t, offsets.offset_x, label='x')
    ax.plot(t, offsets.offset_y, label='y')

    bright = np.array(offsets.is_bright_frame).astype(bool)
    spans = utils.spans_from_bool(bright, 0.0, 1.0)
    for x_start, x_end, bright in spans:
        ax.axvspan(t[int(x_start)], t[int(x_end)-1],
                    facecolor='gold' if bright else 'k',
                    alpha=0.5 if bright else 0.2,
                    edgecolor='none',
                    zorder=0)

    # label the target intervals
    ax_rng = analysis_setup.PSA_individual_axis_limits[1] - analysis_setup.PSA_individual_axis_limits[0]
    for i in np.unique(offsets.t_id)[:3]:
        plt.text(t[np.where(offsets.t_id==i)[0][0]], analysis_setup.PSA_individual_axis_limits[0] + ax_rng/30, analysis_setup.PSA_target_names[i])

    plt.ylim(analysis_setup.PSA_individual_axis_limits)
    ax.invert_yaxis()

    # indicate on axis what data direction is left/up and right/down
    for anc, ha, lbl in zip((1.,0.), ('right','left'), ('left, up', 'right, down')):
        ax.text(
            1.0, anc, lbl,
            rotation=90, rotation_mode='anchor',
            ha=ha, va='bottom',
            transform=ax.transAxes,
            fontsize=10, color='0.5'
        )

    plt.legend(ncol=2, loc='lower left')
    plt.xlabel('Time (s)')
    plt.ylabel('Gaze offset (deg)')
    plt.title(f'{pid}')
    fig.savefig(plot_dir / f'{naming.station1_2_prefix}{et_nm}_{pid}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

# make into data frame and store
if not data:
    print('No data collected, not performing further station 2 slippage analysis.')
    exit(0)
df_res = pd.DataFrame.from_records(data)
df_res.to_csv(data_dir / naming.station1_2, index=False, sep='\t', na_rep='nan', float_format='%.8f')


# Plot PSA vectors for all participant in one plot, per eye tracker and target
df_res_avg = df_res.groupby(['pid', 'tracker', 'target id'])[['offset_dark_bright_x','offset_dark_bright_y']].median().reset_index()
for et_nm in df_res_avg.tracker.unique():
    plt.figure()
    for location in df_res_avg['target id'].unique():
        shift_x = np.array(df_res_avg[(df_res_avg.tracker==et_nm) &\
                                      (df_res_avg['target id']==location)].offset_dark_bright_x)
        shift_y = np.array(df_res_avg[(df_res_avg.tracker==et_nm) &\
                                      (df_res_avg['target id']==location)].offset_dark_bright_y)
        plt.plot(analysis_setup.PSA_target_locations[location][0], analysis_setup.PSA_target_locations[location][1], 'bo')
        for k in np.arange(len(shift_x)):
            plt.plot([analysis_setup.PSA_target_locations[location][0], analysis_setup.PSA_target_locations[location][0]+shift_x[k]],
                     [analysis_setup.PSA_target_locations[location][1], analysis_setup.PSA_target_locations[location][1]+shift_y[k]],
                     '-', color='#FF7F0E', linewidth=1)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.ylim([-20, 20])
    plt.xlim([-20, 20])
    plt.gca().invert_yaxis()

    # plt.legend()
    plt.xlabel('Horizontal apparent shift (deg)')
    plt.ylabel('Vertical apparent shift (deg)')
    plt.savefig(plot_dir / f'{naming.station1_2_prefix}{et_nm}.png', dpi=300, bbox_inches='tight')
    plt.close()
