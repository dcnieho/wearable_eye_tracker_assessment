# Analysis workflow
This document describes the workflow for analyzing eye tracking data collected for station 1 and station 2, culminating in a summary benchmark report card, and a detailed report card per tested eye tracker.

## General set up
1. Make a new python venv
2. Install packages: `pip install gazeMapper open3d seaborn reportlab`
3. Prepare the gazeMapper projects for station 1 and 2. This repository already comes with the empty but correctly configured gazeMapper projects [gazeMapper_station1](../gazeMapper_station1) [gazeMapper_station2](../gazeMapper_station2) in the parent directory. If you wish to use another folder, you can run gazeMapper, create the project folder that you wish, and then copy over the config folder from the relevant gazeMapper project that comes with this repository ([gazeMapper_station1](../gazeMapper_station1) or [gazeMapper_station2](../gazeMapper_station2)). Also adapt the path to the gazeMapper project(s) that is set up in [`analysis_setup.py`](analysis_setup.py).
4. Information about the eye trackers you assess is stored in the [`eye_trackers.json`](eye_trackers.json) file. Information about several eye trackers is already provided there, add any further eye trackers that you wish to include in the report card. Besides the fields shown in the file, you can also specify the fields `firmware_version` and `recording_software_version` in this json file, either because they cannot be automatically extracted by gazeMapper from the recordings you provide, or because you want to override the extracted information.
5. If you want to use devkit or prototype eye trackers, additionally register them in gazeMapper using the `Registered custom eye trackers` setting. See the note about Generic eye trackers in the [glassesTools readme](https://github.com/dcnieho/glassestools?tab=readme-ov-file#eye-tracker-support).

With the above set up done, most of the analysis is a set of actions that can be performed in gazeMapper, after which a few custom scripts are run. The various steps are detailed below, per station.

## Station 1 analysis workflow
1. Open the gazeMapper project for station 1.
2. Import the eye tracking recording you want to analyze. Ensure the session to which it is imported has a descriptive name. We recommend using the template `<participant ID>_<eye tracker name>`. Some of the analysis scripts depend on the session name starting with `<participant ID>_`. The exact eye tracker name does not matter, the scripts read it from the recording info in the gazeMapper recording folder. An example session name could be `P01_neon`.
3. Once the recording(s) are imported, run the following actions in the following order (mostly constrained by the GUI) for each recording:
    a. Detect Markers
    b. Auto code episodes. This should automatically code one `Sync` episode (for eye tracker gaze data and scene camera synchronization), two `Fixation task` episodes for the two iterations through the fixation grid and one long `PSA` episode, covering whole pupil size artefact (PSA) measurement task.
    c. Code episodes: check manually that the sync, fixation task and PSA episodes mentioned above are correctly coded. If not, manually adjust or add the coding. Code the frame _after_ the start ArUco marker disappears and the frame _before_ the end marker appears. For the fixation task, also check that the target presentations are coded correctly. If not, you will receive an error message when running the Validate action below (so you could also wait until you receive the error message and only check and correct those recordings that fail).
    d. run Detect Markers again
    e. Sync et to cam
    f. Gaze to plane
    g. Validate (this processes the fixation grid data)
4. Next, the recordings will be further analyzed with a custom script. This is needed as support for planes not using ArUco markers as fiducials is currently not built into gazeMapper. We need this for the PSA trials, where red dots are used as fiducial markers.
    a. Run the `a_PSA_plane_detection.py` script. This will run PSA plane detection, pose determination, and coding of target location for any recording for which this has not yet been done. Note that optionally a GUI can be shown to inspect the detection results on the frames as they are processed.
    c. Run the `b_PSA_gaze_offset.py` script to determine gaze offsets from the target for each frame.

## Station 2 analysis workflow
1. Open the gazeMapper project for station 2.
2. Import the eye tracking recording you want to analyze. Ensure the session to which it is imported has a descriptive name. We recommend using the template `<participant ID>_<eye tracker name>`. Some of the analysis scripts depend on the session name starting with `<participant ID>_`. The exact eye tracker name does not matter, the scripts read it from the recording info in the gazeMapper recording folder. An example session name could be `P01_neon`.
3. Once the recording(s) are imported, run the following actions in the following order (mostly constrained by the GUI) for each recording:
    a. Detect Markers
    b. Auto code episodes. This should automatically code one `Sync` episode (for eye tracker gaze data and scene camera synchronization), three slippage episodes (one `Slippage horizontal`, one `Slippage vertical` and one `Slippage depth`) and five episodes for each of the three parallax intervals (`Parallax 30`, `Parallax 100` and `Parallax 200`). It is possible that this action yields only one long episode for a given parallax interval, instead of 5 episodes (there will be a warning on the console when this occurs). This occurs for a given parallax episode when not all target presentations are correctly detected. In that case, select Code Episodes -> Code targets for Parallax 100 (or which parallax interval it is that did not process correctly) and check that all target presentations (when the ArUco markers near the center change) are correctly coded (code from the first frame where the center markers have changed until the last frame before they change back). Once you have fixed this, run Auto code episodes again.
    c. Code episodes: check manually that the sync, slippage and parallax episodes mentioned above are correctly coded. If not, manually adjust or add the coding. As described above, for the parallax episodes, also check that the target presentations are coded correctly. If not, you will receive an error message when running the Validate action below.
    d. run Detect Markers again
    e. Sync et to cam
    f. Gaze to plane
    g. Validate (this processes the data for the parallax episodes)
    h. Compute Gaze Offsets (this processed the data for the slippage episodes)

## Final steps: collating the gazeMapper output and generating the report cards
Now that all the recordings have been processed through gazeMapper (and the custom scripts for Station 1) several further scripts need to be run to collate the results for each task, and then further scripts for generating the benchmarking report cards. It is possible to collect and analyze data for only one of the stations. In that case the report card will only contain data for that station.

### Result collation, per task
Both stations have two tasks. As such, there are for scripts to run. These can be run in any order:
- `c1_1_fixation_analysis.py`
- `c1_2_PSA_analysis.py`
- `c2_1_slippage_analysis.py`
- `c2_2_parallax_analysis.py`

### Report card generation
The summary benchmark report card is generated using the `d_generate_summary_report.py` script. The per-eye tracker detailed report cards are generated using the `d_generate_per_tracker_report.py` script.