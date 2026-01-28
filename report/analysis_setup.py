import pathlib

from glassesTools import eyetracker, json

# path where the gazeMapper projects are located
gazeMapper_projects_path = pathlib.Path('.').resolve().parent
gazeMapper_project_names = {1: 'gazeMapper_station1', 2: 'gazeMapper_station2'}

# eye tracker info
eye_trackers = json.load(pathlib.Path("eye_trackers.json"))
eye_trackers = {eyetracker.EyeTracker(k): v for k, v in eye_trackers.items()}

# station 1
## 1. Head-fixed grid fixation
### target setup
# If phi  <0: upper half, ring1 = [10], ring2 = [15, 18], ring3 = [20, 25]
# If phi >=0: lower half, ring1 = [10], ring2 = [20]    , ring3 = [30]
fixation_rings_upper = [[0], [10], [15, 18], [20, 25]]
fixation_rings_lower = [[0], [10], [20]    , [30]]
## 2. PSA setup
### 1. get fiducials
PSA_phi = [30.0, 70.0, 110.0, 150.0, 210.0, 270.0, 330.0, 30.0, 110.0, 150.0, 210.0, 270.0, 330.0, 30.0, 110.0, 210.0, 270.0]
PSA_rho = [4,    4,    4,     4,     4,     4,     4,     8,    8,     8,     8,     8,     8,     12,   12,    12,    12]
### 2. targets
PSA_target_locations= {1:(-15.,0.), 2:(0.,0.) , 3:(15.,0.)}     # deg
PSA_target_names    = {1:'left'   , 2:'middle', 3:'right'}
### 3. cosmetics
PSA_individual_axis_limits = (-10, 10)


# station 2
## 1. Slippage setup
slippage_offset_suffix = 'slippage'
slippage_trials = {'Slippage horizontal': 'side-to-side', 'Slippage vertical': 'up-and-down', 'Slippage depth': 'back-and-forth'}
slippage_skip_dur = 2.0  # seconds to skip at start of each trial
## 2. Parallax setup
parallax_distances = [30, 100, 200]  # cm
parallax_episodes = {d:f'Parallax {d}' for d in parallax_distances}
parallax_individual_axis_limits = (-10, 10)



# further analysis setup
# black out (part of scene video to black out before analysis) setup per eye tracker
blackout_dict = {
    eyetracker.EyeTracker.VPS_19: [0,0,526,78]  # time code in the scene video
}

# report settings
## color coding in tables
rms_lim          = [0.1, 0.5]
std_lim          = [0.1, 0.5]
loss_lim         = [3, 10]
colors      = ((100,255,100),(255,255,100),(255,100,100))
fs_lim           = [95, 105]
colors_fs   = ((255,100,100),(255,255,100),(100,255,100),(255,255,100),(255,100,100))
### for accuracy, set per task. You can use either fixed thresholds:
fixation_acc_lim = [1, 3]
PSA_acc_lim      = [1, 2]
slippage_acc_lim = [3, 6]
parallax_acc_lim = [1, 3]
### or enable data-driven thresholds (set percentiles to use, e.g., [33, 67] for terciles) or None to use fixed thresholds
fixation_acc_percentiles = None # [20, 80]
PSA_acc_percentiles      = None
slippage_acc_percentiles = None
parallax_acc_percentiles = None


## display names for the columns
et_colnames = {
    'name': 'Name',
    'weight': 'Weight\n(gr)',
    'sample_rate': 'Sampling\nfrequency\n(Hz)',
    'firmware_version': 'Firmware\nversion',
    'recording_software_version': 'Recording\nsoftware\nversion',
    'calibration': "Calibration\nmethod",
    'n_cal_attempts': "Number of\ncalibration\nattempts"
}
colnames = {
    'pid': 'Participant',
    'tracker': 'Name',
    'ring': 'Ring',
    'target location': 'Target\nlocation',
    'distance': 'Distance\n(cm)',

    'acc': 'Acc\n(deg)',
    'rms': 'RMS\n(deg)',
    'std': 'STD\n(deg)',
    'Fs': 'Sampling\nfrequency\n(Hz)',
    'relative_Fs': 'Sampling\nfrequency\n(%)',
    'data_loss': 'Data\nloss\n(%)',

    'offset_total': 'Apparent\ngaze\nshift\n(deg)',
    'pd_diff': 'Pupil\ndiameter\nchange\n(mm)',

    'shift': 'Total\napparent\ngaze\nshift\n(deg)',
    'shift_x': 'Horizontal\napparent\ngaze\nshift\n(deg)',
    'shift_y': 'Vertical\napparent\ngaze\nshift\n(deg)',
}