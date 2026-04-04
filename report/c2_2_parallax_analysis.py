import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import analysis_setup, naming
from gazeMapper import config, naming as gm_naming, process, session
from glassesTools import annotation, naming as gt_naming
from glassesTools.validation.config import get_targets, get_validation_setup


data_dir = pathlib.Path('data')
data_dir.mkdir(exist_ok=True)
plot_dir = pathlib.Path('figures')
plot_dir.mkdir(exist_ok=True)

analysis_setup.parallax_distances.sort()
diff_dists = [analysis_setup.parallax_distances[0], analysis_setup.parallax_distances[-1]]

# get project config
gazeMapper_project_path = analysis_setup.gazeMapper_projects_path / analysis_setup.gazeMapper_project_names[2]
if not gazeMapper_project_path.is_dir():
    print(f'The gazeMapper project path "{gazeMapper_project_path}" does not exist, not processing station 2 parallax analysis.')
    exit(0)

config_dir = config.guess_config_dir(gazeMapper_project_path)
study_config = config.Study.load_from_json(config_dir)
et_recs      = [r for r in study_config.session_def.recordings if r.type==session.RecordingType.Eye_Tracker]
if len(et_recs)!=1:
    raise RuntimeError('A project with one eye tracker recording per session is expected')
et_rec = et_recs[0]

val_events = process.get_specific_event_types(study_config, annotation.EventType.Validate)

# get sessions
sessions = session.get_sessions_from_project_directory(gazeMapper_project_path, study_config.session_def)

# process each session
df_res = None
et_infos = {}
for s in sessions:
    rec_dir = s.working_directory / et_rec.name
    if et_rec.name not in s.recordings:
        print(f'No eye tracker recording in "{rec_dir}", skipping...')
        continue
    et  = (s.recordings[et_rec.name].info.eye_tracker, s.recordings[et_rec.name].info.eye_tracker_name or None)
    pid = s.name.split('_')[0]
    et_lbl = et[0].value if et[1] is None else f'{et[0].value}.{et[1]}'
    et_infos[(pid, et_lbl)] = {k: getattr(s.recordings[et_rec.name].info,k) for k in ['firmware_version', 'recording_software_version']}

    if not (gaze_data_file:=rec_dir/gt_naming.gaze_data_fname).is_file():
        print(f'The gaze data file "{gaze_data_file.name}" does not exist for the recording in "{rec_dir}", skipping...')
        continue
    gaze_data = pd.read_csv(gaze_data_file, sep='\t')

    print(f'Processing session "{s.name}", recording directory "{rec_dir}"...')

    # prep figure
    fig, axs = plt.subplots(2, 2, figsize=(6, 6))
    fig.suptitle(pid)
    for ax in axs.flatten():
        ax.set_xlim(analysis_setup.parallax_individual_axis_limits)
        ax.set_ylim(analysis_setup.parallax_individual_axis_limits)
        ax.invert_yaxis()
        ax.set_aspect('equal', adjustable='box')
        ax.label_outer()

    data = []
    for distance, ax in zip(analysis_setup.parallax_distances, axs.flatten()):
        episode_name = analysis_setup.parallax_episodes[distance]
        # check required files are present, and load
        if not (dq_file:=rec_dir/f'{gm_naming.validation_prefix}{episode_name}_data_quality.tsv').is_file():
            print(f'The data quality file "{dq_file.name}" does not exist for the recording in "{rec_dir}", skipping...')
            continue
        dq = pd.read_csv(dq_file, sep='\t')
        if not (fixation_coding_file:=rec_dir/f'{gm_naming.validation_prefix}{episode_name}_fixation_assignment.tsv').is_file():
            print(f'The fixation assignment file "{fixation_coding_file.name}" does not exist for the recording in "{rec_dir}", skipping...')
            continue
        fixation_coding = pd.read_csv(fixation_coding_file, sep='\t')

        # get plane for this episode
        # first find corresponding coding config
        cs = [cs for cs in val_events if cs['name']==episode_name][0]
        if len(cs['planes'])!=1:
            raise ValueError(f'Validation event "{episode_name}" should be coded for exactly one glassesValidator plane, found {len(cs["planes"])}')
        p = list(cs['planes'])[0]
        # get validation setup
        plane_defs = [pl for pl in study_config.planes if pl.name==p]
        if len(plane_defs)!=1:
            raise RuntimeError(f'The GlassesValidator plane {p} is not found in the project')
        plane_def = plane_defs[0]
        # get validation setup
        validation_setup = get_validation_setup(config_dir/plane_def.name)
        viewing_distance = validation_setup['distance']
        # get targets on the plane, make relative to center target
        df_target = get_targets(config_dir/plane_def.name, 'targetPositions_converted.csv')
        df_target.x -= df_target.loc[validation_setup['centerTarget']].x
        df_target.y -= df_target.loc[validation_setup['centerTarget']].y
        df_target['x_deg'] = np.degrees(np.arctan(df_target['x'] / viewing_distance))
        df_target['y_deg'] = np.degrees(np.arctan(df_target['y'] / viewing_distance))

        # prep figure
        ax.set_title(f'Distance: {distance} cm')
        ax.plot(df_target['x_deg'], df_target['y_deg'], 'bo')
        for i, row in dq.iterrows():
            target_x    , target_y     = df_target.loc[row.target,'x']    , df_target.loc[row.target,'y']
            target_x_deg, target_y_deg = df_target.loc[row.target,'x_deg'], df_target.loc[row.target,'y_deg']

            # plot
            ax.plot([target_x_deg, target_x_deg + row.acc_x],
                    [target_y_deg, target_y_deg + row.acc_y],'-', color='#FF7F0E')

            # Extract gaze data for this target
            gaze_episode = fixation_coding[(fixation_coding.marker_interval == row.marker_interval) & \
                                           (fixation_coding.target          == row.target)][['start_timestamp','end_timestamp']]

            start_idx   = np.searchsorted(gaze_data.timestamp_VOR, gaze_episode.start_timestamp.iloc[0])
            end_idx     = np.searchsorted(gaze_data.timestamp_VOR, gaze_episode.end_timestamp.iloc[0])
            target_gaze = gaze_data.iloc[start_idx:end_idx+1]

            # Compute effective frequency and data loss
            intersample_interval = np.median(np.diff(target_gaze.timestamp))
            recording_duration = (target_gaze.timestamp.iloc[-1]-target_gaze.timestamp.iloc[0] + intersample_interval) / 1000 # to seconds
            n_samples = len(target_gaze)
            measured_Fs = n_samples / recording_duration
            percent_Fs = measured_Fs / analysis_setup.eye_trackers[et]['sample_rate']

            # Lost data
            data_loss = np.mean([np.sum(np.isnan(target_gaze.gaze_pos_vid_x)) / n_samples,
                                 np.sum(np.isnan(target_gaze.gaze_pos_vid_y)) / n_samples])

            # store
            data.append({'pid': pid, 'tracker': et_lbl, 'distance': distance,
                         'target_id': row.target, 'target_x': target_x, 'target_y': target_y,
                         'target_x_deg': target_x_deg, 'target_y_deg': target_y_deg,
                         'shift_x': row.acc_x, 'shift_y': row.acc_y,
                         'Fs': measured_Fs, 'relative_Fs': percent_Fs * 100, 'data_loss': data_loss * 100})

    # turn into data frame
    this_df_res = pd.DataFrame.from_records(data)
    # append to overall results
    if df_res is None:
        df_res = this_df_res
    else:
        df_res = pd.concat([df_res, this_df_res], ignore_index=True)

    # participant parallax error (median difference between nearest and furthest)
    this_df_res_avg = this_df_res.groupby(['pid', 'tracker','distance','target_id','target_x_deg','target_y_deg'])[['shift_x', 'shift_y']].median().reset_index()
    this_shift_df = this_df_res_avg[this_df_res_avg['distance']==diff_dists[0]][['shift_x', 'shift_y']].reset_index(drop=True)-this_df_res_avg[this_df_res_avg['distance']==diff_dists[1]][['shift_x', 'shift_y']].reset_index(drop=True)
    this_shift_df['target_id'] = this_df_res_avg[this_df_res_avg['distance']==diff_dists[0]]['target_id'].values
    axs[1, 1].set_title(f'Parallax error: {diff_dists[0]} cm - {diff_dists[1]} cm')
    axs[1, 1].plot(df_target['x_deg'], df_target['y_deg'], 'bo', zorder=1)
    for i, row in this_shift_df.iterrows():
        target_x_deg, target_y_deg = df_target.loc[row.target_id, ['x_deg', 'y_deg']]
        axs[1, 1].arrow(target_x_deg, target_y_deg, row.shift_x, row.shift_y, color="#FF0E0E", lw=2.5, head_width=0.5, length_includes_head=True, zorder=2)

    axs[1, 0].set_xlabel("Horizontal position (deg)")
    axs[1, 1].set_xlabel("Horizontal position (deg)")
    axs[0, 0].set_ylabel("Vertical position (deg)")
    axs[1, 0].set_ylabel("Vertical position (deg)")
    plt.tight_layout()
    fig.savefig(plot_dir / f'{naming.station2_2_prefix}{et_lbl}_{pid}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

# make into data frame and store
if df_res is None or df_res.empty:
    print('No data collected, not performing further station 2 parallax analysis.')
    exit(0)
df_res.to_csv(data_dir / naming.station2_2, index=False, sep='\t', na_rep='nan', float_format='%.8f')

# also store eye tracker info
et_info = pd.DataFrame.from_dict(et_infos, orient='index')
et_info.index = pd.MultiIndex.from_tuples(et_info.index, names=['participant', 'device'])
et_info.to_csv(data_dir / f'{naming.station2_prefix}eye_tracker_info.tsv', index=True, sep='\t')

# make plots showing parallax
# average over multiple looks at the target
df_res_avg = df_res.groupby(['pid', 'tracker','distance','target_id','target_x_deg','target_y_deg'])[['shift_x', 'shift_y']].median().reset_index()
# Compute difference between distances (e.g., 30 cm - 200 cm)
pivot_x = df_res_avg.pivot(index=['pid', 'tracker', 'target_id'], columns='distance', values='shift_x')
pivot_y = df_res_avg.pivot(index=['pid', 'tracker', 'target_id'], columns='distance', values='shift_y')
shift_df = pd.DataFrame({'shift_x': pivot_x[diff_dists[0]] - pivot_x[diff_dists[1]], 'shift_y': pivot_y[diff_dists[0]] - pivot_y[diff_dists[1]]}).reset_index()
# average over participants
shift_df_avg = shift_df.groupby(['tracker','target_id'])[['shift_x', 'shift_y']].median().reset_index()

# plot, lines per participant and average (median) across participants
for et_lbl in shift_df_avg.tracker.unique():
    fig = plt.figure()
    plt.axis((-14, 14, -14, 14))
    plt.plot(df_target['x_deg'], df_target['y_deg'], 'bo', zorder=1)  # plot target locations (NB: target positions in deg are the same for all distances)

    df_et     = shift_df    [(shift_df    .tracker==et_lbl)]
    df_et_avg = shift_df_avg[(shift_df_avg.tracker==et_lbl)]

    for i, row in df_et.iterrows():
        target_x_deg, target_y_deg = df_target.loc[row.target_id, ['x_deg', 'y_deg']]
        plt.plot([target_x_deg, target_x_deg + row.shift_x],
                 [target_y_deg, target_y_deg + row.shift_y],'-', color="#FC9236", lw=1, zorder=2)
    for i, row in df_et_avg.iterrows():
        target_x_deg, target_y_deg = df_target.loc[row.target_id, ['x_deg', 'y_deg']]
        plt.arrow(target_x_deg, target_y_deg, row.shift_x, row.shift_y, color="#FF0E0E", lw=2.5, head_width=0.5, length_includes_head=True, zorder=3)
    plt.title(f'Parallax error: {diff_dists[0]} cm - {diff_dists[1]} cm')
    plt.xlabel('Horizontal gaze position (deg)')
    plt.ylabel('Vertical gaze position (deg)')
    fig.gca().invert_yaxis()
    fig.gca().set_aspect('equal', adjustable='box')
    fig.savefig(plot_dir / f'{naming.station2_2_prefix}{et_lbl}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
