# Stimulus code
This document describes the stimulus code for presenting the visual tasks of station 1 and station 2 (see the protocol documents for details).
This is part of the code released for the paper Niehorster, D. C., Marano, G., Pettenella, A., Carminati, M., Melloni, F., Merigo, L. & Nyström, M. (in prep). A methodology for systematically assessing wearable eye tracker performance.

## Set up
Set up for the two stations is contained in the `setup_station1.json` and `setup_station2.json` JSON files. Important here is that you set up the screen geometry correctly (it does not have to be the same for the two stations, but can be).
Further configured in these json files are the tasks themselves: if you e.g. wish to use a different number of repetitions for the fixation, PSA or parallax tasks, a different number of target positions for the PSA task, or different temporal parameters (e.g., duration of fixation points, bright and dark periods or recording duration for the slippage tasks), these are set in these files.
To change the ArUco markers and target locations used in the fixation task of station 1, and the PSA and parallax tasks of station 2, one should edit the `markerPositions_` and `targetPositions_` files. See [here](https://github.com/dcnieho/gazeMapper?tab=readme-ov-file#gazemapper-planes) for more information about what these contain.

Note that if you change these parameters, the settings of the gazeMapper projects that come preconfigured with this repository (`gazeMapper_station1` and `gazeMapper_station2` in the top-level directory of this repository) will need to be adapted to correspond.

## Running the tasks
The tasks for station 1 and station 2 are displayed by running the `stim_station1.py` and `stim_station2.py` scripts, respectively. Ensure to carefully follow the protocol described in the `Protocol_station1.docx` and `Protocol_station2.docx` files in the top-level directory of this repository.