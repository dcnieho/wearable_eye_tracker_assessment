import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import analysis_setup, utils, naming
from gazeMapper import config, naming as gm_naming, plane, process, session
from glassesTools import annotation, naming as gt_naming
from glassesTools.validation.config import get_targets, get_validation_setup


data_dir = pathlib.Path('data')
data_dir.mkdir(exist_ok=True)
plot_dir = pathlib.Path('figures')
plot_dir.mkdir(exist_ok=True)


# get project config
gazeMapper_project_path = analysis_setup.gazeMapper_projects_path / analysis_setup.gazeMapper_project_names[1]
if not gazeMapper_project_path.is_dir():
    print(f'The gazeMapper project path "{gazeMapper_project_path}" does not exist, not processing station 1 fixation analysis.')
    exit(0)

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
val_events = process.get_specific_event_types(study_config, specific_event_type=annotation.EventType.Validate)
if len(val_events)!=1:
    raise RuntimeError('A project with one validation episode configured is expected')
val_event = val_events[0]
# get targets on the plane, make relative to center target
df_target = get_targets(config_dir/val_plane.name, 'targetPositions_converted.csv')
df_target.x -= df_target.loc[validation_setup['centerTarget']].x
df_target.y -= df_target.loc[validation_setup['centerTarget']].y
df_target[['rho', 'phi']] = df_target.apply(lambda r: pd.Series(utils.cart2pol(r.x, r.y)), axis=1)
df_target[['eccentricity', 'ring']] = df_target.apply(lambda r: pd.Series(utils.get_eccentricity_ring(r.rho, r.phi, viewing_distance, (analysis_setup.fixation_rings_lower, analysis_setup.fixation_rings_upper))), axis=1)
df_target['x_deg'] = np.degrees(np.arctan(df_target['x'] / viewing_distance))
df_target['y_deg'] = np.degrees(np.arctan(df_target['y'] / viewing_distance))
# ensure ring is integer
df_target['ring'] = df_target['ring'].astype(int)


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
    et_infos[(pid, et.value)] = {k: getattr(s.recordings[et_rec.name].info,k) for k in ['firmware_version', 'recording_software_version']}

    # check required files are present, and load
    if not (dq_file:=rec_dir/f'{gm_naming.validation_prefix}{val_event["name"]}_data_quality.tsv').is_file():
        print(f'The data quality file "{dq_file.name}" does not exist for the recording in "{rec_dir}", skipping...')
        continue
    dq = pd.read_csv(dq_file, sep='\t')
    if not (fixation_coding_file:=rec_dir/f'{gm_naming.validation_prefix}{val_event["name"]}_fixation_assignment.tsv').is_file():
        print(f'The fixation assignment file "{fixation_coding_file.name}" does not exist for the recording in "{rec_dir}", skipping...')
        continue
    fixation_coding = pd.read_csv(fixation_coding_file, sep='\t')
    if not (gaze_data_file:=rec_dir/gt_naming.gaze_data_fname).is_file():
        print(f'The gaze data file "{gaze_data_file.name}" does not exist for the recording in "{rec_dir}", skipping...')
        continue
    gaze_data = pd.read_csv(gaze_data_file, sep='\t')

    print(f'Processing session "{s.name}", recording directory "{rec_dir}"...')

    for k, row in dq.iterrows():
        # which validation interval?
        validation_interval = row.marker_interval

        # Which target position?
        target_x    , target_y     = df_target.loc[row.target,'x']    , df_target.loc[row.target,'y']
        target_x_deg, target_y_deg = df_target.loc[row.target,'x_deg'], df_target.loc[row.target,'y_deg']
        target_eccentricity, target_ring = df_target.loc[row.target,'eccentricity'], df_target.loc[row.target,'ring']

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

        data.append({'pid': pid, 'tracker': et.value, 'target_id': row.target,
                     'target_x': target_x, 'target_y': target_y,
                     'target_x_deg': target_x_deg, 'target_y_deg': target_y_deg,
                     'eccentricity': np.round(target_eccentricity),
                     'acc_x': row.acc_x, 'acc_y': row.acc_y, 'acc': row.acc,
                     'rms_x': row.rms_x, 'rms_y': row.rms_y, 'rms': row.rms,
                     'std_x': row.std_x, 'std_y': row.std_y, 'std': row['std'],
                     'Fs': measured_Fs, 'relative_Fs': percent_Fs * 100, 'data_loss': data_loss * 100,
                     'ring': target_ring})

# make into data frame and store
if not data:
    print('No data collected, not performing further station 2 slippage analysis.')
    exit(0)
df_res = pd.DataFrame.from_records(data)
df_res.to_csv(data_dir / naming.station1_1, index=False, sep='\t', na_rep='nan', float_format='%.8f')

# also store eye tracker info
et_info = pd.DataFrame.from_dict(et_infos, orient='index')
et_info.index = pd.MultiIndex.from_tuples(et_info.index, names=['participant', 'device'])
et_info.to_csv(data_dir / f'{naming.station1_prefix}eye_tracker_info.tsv', index=True, sep='\t')

# make plots showing values for each point, aggregated over participants, per eye tracker
metrics = {"acc": 'Accuracy (deg)',
           "rms": 'Precision (RMS-S2S, deg)',
           "std": 'Precision (STD, deg)',
           "data_loss": 'Data Loss (%)'}
metric_fields = list(metrics.keys())
for et in df_res.tracker.unique():
    df_et = (
        df_res.loc[df_res["tracker"] == et,
                   ["target_x_deg", "target_y_deg", *metric_fields]]
        .groupby(["target_x_deg", "target_y_deg"], as_index=False)[metric_fields]
        .mean()
    )

    fig, axs = plt.subplots(2, 2, figsize=(7.5, 5))
    for ax, metric in zip(axs.flat, metrics):
        sns.scatterplot(
            data=df_et, x="target_x_deg", y="target_y_deg",
            hue=metric, size=metric, palette="viridis",
            ax=ax
        )
        ax.set_title(metrics[metric])
        ax.invert_yaxis()
        ax.set_aspect('equal', adjustable='box')
        ax.label_outer()
        ax.legend(bbox_to_anchor=(1, 1.02), loc="upper left")
        utils.format_legend_numbers(ax, max_decimals=1 if metric=='data_loss' else 2)

    axs[1, 0].set_xlabel("Horizontal position (deg)")
    axs[1, 1].set_xlabel("Horizontal position (deg)")
    axs[0, 0].set_ylabel("Vertical position (deg)")
    axs[1, 0].set_ylabel("Vertical position (deg)")
    plt.tight_layout()
    fig.savefig(plot_dir / f'{naming.station1_1_prefix}{et}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

# make plots showing values for each point, per participant, per eye tracker
for et in df_res.tracker.unique():
    for pid in df_res.pid.unique():
        df_et_pid = df_res.loc[(df_res["tracker"] == et) & (df_res["pid"] == pid),
                               ["target_x_deg", "target_y_deg", "acc_x", "acc_y", *metric_fields]]
        if df_et_pid.empty:
            continue
        # aggregate over target presentations by taking mean, to get more stable estimates per target location
        df_et_pid_aggr = df_et_pid.groupby(["target_x_deg", "target_y_deg"], as_index=False)[metric_fields].mean()
        fig, axs = plt.subplots(2, 2, figsize=(7.5, 5))
        fig.suptitle(pid)
        for ax, metric in zip(axs.flat, metric_fields):
            sns.scatterplot(
                data=df_et_pid_aggr, x="target_x_deg", y="target_y_deg",
                hue=metric, size=metric, palette="viridis",
                ax=ax
            )
            if metric=='acc':
                # also plot individual measurement points (not averaged over target presentations) to get a sense of variability
                # first target matching
                for t, row in df_et_pid.iterrows():
                    ax.plot([row['target_x_deg'], row['target_x_deg'] + row['acc_x']], [row['target_y_deg'], row['target_y_deg'] + row['acc_y']],'r-', lw=1.5)
                # now also fixation locations themselves
                ax.plot(df_et_pid.target_x_deg+df_et_pid.acc_x, df_et_pid.target_y_deg+df_et_pid.acc_y, 'ro', markersize=3)
            ax.set_title(metrics[metric])
            ax.invert_yaxis()
            ax.set_aspect('equal', adjustable='box')
            ax.label_outer()
            # make legend and fix numeric precision of legend entries
            ax.legend(bbox_to_anchor=(1, 1.02), loc="upper left")
            utils.format_legend_numbers(ax, max_decimals=1 if metric=='data_loss' else 2)


        axs[1, 0].set_xlabel("Horizontal position (deg)")
        axs[1, 1].set_xlabel("Horizontal position (deg)")
        axs[0, 0].set_ylabel("Vertical position (deg)")
        axs[1, 0].set_ylabel("Vertical position (deg)")
        plt.tight_layout()
        fig.savefig(plot_dir / f'{naming.station1_1_prefix}{et}_{pid}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)