from psychopy import visual, monitors, tools, event, core, sound
import numpy as np
import pandas as pd
import random
from collections import defaultdict
import cv2


def open_window(screen_config: dict) -> visual.Window:
    mon = monitors.Monitor('temp')
    mon.setWidth(screen_config["width"])
    mon.setDistance(screen_config["viewing_distance"])
    mon.setSizePix(screen_config["resolution"])
    win = visual.Window(monitor=mon,
                        fullscr=True,
                        color=screen_config["background_color"],
                        screen=screen_config["which_monitor"] or 0,
                        units=screen_config["units"],
                        viewPos=screen_config["origin"] if "origin" in screen_config else None,
                        allowGUI=False,
                        multiSample=True,
                        numSamples=4,
                        infoMsg='')
    win.mouseVisible = False
    if not all((x==y for x,y in zip(win.size,screen_config["resolution"]))):
        raise RuntimeError(f'expected resolution of {screen_config["resolution"]}, but got {win.size}')
    tol = screen_config["refresh_rate_tolerance"]
    actual_refresh_rate = 1/win.monitorFramePeriod
    if actual_refresh_rate<screen_config["refresh_rate"]-tol or actual_refresh_rate>screen_config["refresh_rate"]+tol:  # check within tolerance (Hz)
        raise RuntimeError(f'expected framerate of {screen_config["refresh_rate"]}, but got {actual_refresh_rate}')

    return win

class ABCFixPoint:
    def __init__(self, win: visual.Window, outer_diameter=0.7, inner_diameter=0.1,
                 outer_color='black', inner_color='white', units='degFlatPos'):
        self.outer_dot      = visual.Circle(win, fillColor = outer_color,
                                            radius = outer_diameter/2,
                                            units = units)
        self.inner_dot      = visual.Circle(win, fillColor = outer_color,
                                            radius = inner_diameter/2,
                                            units = units)
        self.line_vertical  = visual.Rect(win, fillColor=inner_color,
                                          width=inner_diameter, height=outer_diameter,
                                          units = units)
        self.line_horizontal= visual.Rect(win, fillColor=inner_color,
                                          width=outer_diameter, height=inner_diameter,
                                          units = units)

    def set_size(self, diameter: float):
        self.outer_dot.radius       = diameter/2.
        self.line_vertical.size     = (self.line_vertical.width, diameter)
        self.line_horizontal.size   = (diameter, self.line_horizontal.height)

    def set_pos(self, pos):
        self.outer_dot.pos          = pos
        self.inner_dot.pos          = pos
        self.line_vertical.pos      = pos
        self.line_horizontal.pos    = pos

    def get_pos(self):
        return self.outer_dot.pos

    def get_size(self):
        return self.outer_dot.size

    def draw(self):
        self.outer_dot.draw()
        self.line_vertical.draw()
        self.line_horizontal.draw()
        self.inner_dot.draw()

class SegmentationMarker:
    def __init__(self, win: visual.Window, refresh_rate: int, duration: float,
                 marker_size: float, units: str, margin: float, background_color: str):
        self.win = win
        self.refresh_rate = refresh_rate
        self.duration = duration
        self.size = marker_size
        self.units = units
        self.margin = margin
        self.background_color = background_color

        self._marker    = visual.ImageStim(win, image=None, size=self.size, units=self.units)
        self._win       = win
        self.update()

    def update(self):
        self._marker_bg = visual.Rect(self._win,
                                      units =self.units,
                                      width =self.size*(1+self.margin*2),
                                      height=self.size*(1+self.margin*2),
                                      fillColor=self.background_color)

    def draw(self, m_id: int):
        check_escape(self.win)
        self._marker.image = get_aruco_marker(m_id, self.size, self.units, self.win)
        n_segment_marker_frames = int(self.duration*self.refresh_rate)
        for _ in range(n_segment_marker_frames):
            self._marker_bg.draw()
            self._marker.draw()
            self.win.flip()

def read_coord_file(file):
    return pd.read_csv(file, dtype=defaultdict(lambda: np.float32, ID='int32', color='str')).dropna(axis=0, how='all').set_index('ID')

_aruco_dict: cv2.aruco.Dictionary = None
_aruco_border_bits: int = None
def load_aruco_dict(aruco_dict_name: str, border_bits: int):
    global _aruco_dict
    global _aruco_border_bits
    str_to_dict: dict[str, int] = {k: getattr(cv2.aruco,k) for k in ['DICT_4X4_50', 'DICT_4X4_100', 'DICT_4X4_250', 'DICT_4X4_1000', 'DICT_5X5_50', 'DICT_5X5_100', 'DICT_5X5_250', 'DICT_5X5_1000', 'DICT_6X6_50', 'DICT_6X6_100', 'DICT_6X6_250', 'DICT_6X6_1000', 'DICT_7X7_50', 'DICT_7X7_100', 'DICT_7X7_250', 'DICT_7X7_1000', 'DICT_ARUCO_ORIGINAL', 'DICT_APRILTAG_16H5', 'DICT_APRILTAG_25H9', 'DICT_APRILTAG_36H10', 'DICT_APRILTAG_36H11', 'DICT_ARUCO_MIP_36H12']}

    if aruco_dict_name not in str_to_dict:
        possible = '"'+'", "'.join(str_to_dict.keys())+'"'
        raise ValueError(f'ArUco dictionary with name "{aruco_dict_name}" is not known. Possible values are: {possible}.')
    if border_bits<1:
        raise ValueError('The number of border bits for ArUco markers must be 1 or higher.')

    _aruco_dict = cv2.aruco.getPredefinedDictionary(str_to_dict[aruco_dict_name])
    _aruco_border_bits = border_bits

def get_aruco_marker(m_id: int, size: float, units: str, win: visual.Window):
    if _aruco_dict is None or _aruco_border_bits is None:
        raise RuntimeError('You must call load_aruco_dict() before you try to load an ArUco marker')
    size = tools.monitorunittools.convertToPix(np.array([-.5*size,.5*size]),np.array([0.,0.]),units,win)
    size = int(size[1]-size[0])
    # NB: flipud because PsychoPy draws images loaded from memory upside down
    return np.flipud(cv2.aruco.generateImageMarker(_aruco_dict, m_id, size, borderBits=_aruco_border_bits).astype(np.float32)/255.*2-1)

def check_escape(win: visual.Window, keys=None):
    if keys is None:
        keys = event.getKeys()
    if 'escape' in keys:
        win.close()
        core.quit()

def pol2cart(rho: float, phi: float) -> tuple[float,float]:
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)


def run_setup_check(win: visual.Window, config: dict, screen_config: list[float]):
    # prepare visual objects
    fixation_target  = ABCFixPoint(win,
                                   outer_diameter=config["target"]["outer_diameter"], inner_diameter=config["target"]["inner_diameter"],
                                   outer_color   =config["target"]["outer_color"]   , inner_color   =config["target"]["inner_color"],
                                   units=config["units"])
    background_patch = visual.Circle(win, radius=config["background_patch"]["radius"], units=config["units"], fillColor=config["background_patch"]["color"], edges='circle')
    # set up background and position it correctly (correcting for origin)
    window_background = visual.Rect(win, size=win.size, units='pix', fillColor=config["background_color"])
    window_background.units = config["units"]
    if 'origin' in screen_config:
        window_background.pos = [-x for x in screen_config["origin"]]

    draw_tobii = False
    while True:
        window_background.draw()
        if draw_tobii:
            visual.Rect(win,
                        width = 5.45,
                        height = 5.45,
                        units = 'cm',
                        fillColor = 'white',
                        pos = (0,0)).draw()
            visual.Circle(win,
                          radius = 2.15,
                          units = 'cm',
                          fillColor = 'black',
                          pos = (0,0)).draw()
            visual.Circle(win,
                          radius = 1.15,
                          units = 'cm',
                          fillColor = 'white',
                          pos = (0,0)).draw()
            visual.Circle(win,
                          radius = 0.15,
                          units = 'cm',
                          fillColor = 'black',
                          pos = (0,0)).draw()
        else:
            for p in config["positions"]:
                background_patch.pos = p
                fixation_target.set_pos(p)
                background_patch.draw()
                fixation_target.draw()
        win.flip()
        key = event.waitKeys(keyList=['space','t'])
        if key:
            if 'space' in key:
                break
            elif 't' in key:
                draw_tobii = not draw_tobii


def run_et_sync(win: visual.Window, textstim: visual.TextStim, segmenter:SegmentationMarker, config: dict, refresh_rate: int):
    textstim.text = 'A black-and-white fixation target will appear on top of a grey circle. Carefully look\nat the center of this fixation target. After a while, the grey background circle will turn red.\nWhen the circle is red, slowly and continuously move your head side to side (like shaking "no")\nfive times using the metronome as guidance for the tempo.\nKeep looking at the center of the fixation target the whole time.\nWhen the background circle turns grey again, stop moving your head but keep looking at the fixation target.\n\n(Press the spacebar to start)'
    textstim.draw()
    win.flip()
    event.waitKeys()
    # signal et_sync start
    for m_id in config["segment_marker"]["start_IDs"]:
        segmenter.draw(m_id)
    # run et_sync
    run_et_sync_impl(win, config, refresh_rate)
    # signal et_sync end
    for m_id in config["segment_marker"]["end_IDs"]:
        segmenter.draw(m_id)

def run_et_sync_impl(win: visual.Window, config: dict, refresh_rate: int):
    # prepare visual objects
    fixation_target  = ABCFixPoint(win,
                                   outer_diameter=config["target"]["outer_diameter"], inner_diameter=config["target"]["inner_diameter"],
                                   outer_color   =config["target"]["outer_color"]   , inner_color   =config["target"]["inner_color"],
                                   units=config["units"])
    fixation_target.set_pos((0, 0))
    background_patch = visual.Circle(win, radius=config["background_patch"]["radius"], edges='circle', units=config["units"])
    background_patch.pos = (0, 0)
    window_background = visual.Rect(win, size=win.size, units='pix', fillColor=config['background_color'])

    # prepare sound objects
    if (has_metronome:="metronome" in config):
        beep = sound.Sound(value=config["metronome"]["frequency"], secs=config["metronome"]["duration"], volume=config["metronome"]["volume"], stereo=True)
        n_frame_metronome = int(config["metronome"]["interval"]*refresh_rate)

    check_escape(win)

    # number of frames to achieve wanted duration
    n_frames_task     = int(config["duration"]         *refresh_rate)
    n_frames_pre_post = int(config["pre_post_duration"]*refresh_rate)

    # draw before sync
    background_patch.fillColor = config["background_patch"]["pre_post_color"]
    for _ in range(n_frames_pre_post):
        window_background.draw()
        background_patch.draw()
        fixation_target.draw()
        win.flip()
    check_escape(win)
    # draw during sync
    background_patch.fillColor = config["background_patch"]["color"]
    for fr_idx in range(n_frames_task):
        if has_metronome and fr_idx%n_frame_metronome==0:
            beep.play()
        window_background.draw()
        background_patch.draw()
        fixation_target.draw()
        win.flip()
    check_escape(win)
    # draw after sync
    background_patch.fillColor = config["background_patch"]["pre_post_color"]
    for _ in range(n_frames_pre_post):
        window_background.draw()
        background_patch.draw()
        fixation_target.draw()
        win.flip()


def prepare_fixation_sequence(win: visual.Window, config: dict, screen_config: dict, run_id: int = 0, placeholders: bool = True):
    # get markers and target positions
    target_positions   = read_coord_file(config["targets"]["file"])
    fiducial_positions = read_coord_file(config["markers"]["files"][run_id] if "files" in config["markers"] else config["markers"]["file"])

    # adjust markers for run
    id_off = 0
    if run_id>0 and "IDoffset" in config["markers"]:
        id_off = config["markers"]["IDoffset"] * run_id
        fiducial_positions.index += id_off

    indicator_IDs: dict[int,dict[int,int]] = {}
    # prepare indicator IDs for each fiducial marker and target position
    replace_IDs = config["markers"].get("replace_IDs", None)
    if replace_IDs is not None:
        replace_IDs = [r+id_off for r in replace_IDs]
        for ri,m_id in enumerate(replace_IDs):
            if config["markers"]["replace_method"]=="condensed":
                m_id_start = config["markers"]["replace_ID_start"]+run_id*config["markers"]["replace_ID_offset"]+ri
                indicator_IDs[m_id] = {t_id: m_id_start+t_ii*len(replace_IDs) for t_ii,t_id in enumerate(target_positions.index)}
            elif config["markers"]["replace_method"]=="validator":
                m_id_start = config["markers"]["replace_ID_start"]
                m_id_step  = config["markers"]["replace_ID_offset"]
                indicator_IDs[m_id] = {t_id: m_id_start+ri*m_id_step+t_id for t_id in target_positions.index}

    # check there are no overlapping ArUco markers
    if duplicates:=(set([indicator_IDs[y][x] for y in indicator_IDs for x in indicator_IDs[y]]) if indicator_IDs else set()) & set(fiducial_positions.index):
        raise ValueError(f"For the current setup, marker IDs specified in marker file {config['markers']['file']} would also appear as target indicator markers. Adapt the marker positions file or config->markers->replace_ID_start, config->markers->replace_ID_offset and config->markers->IDoffset. Duplicates: {sorted(list(duplicates))}.")

    # Create a background with ArUco markers and circle placeholders where target will appear
    stimList = []
    # draw all ArUco markers
    # NB ArUco markers are drawn using the 'deg' and not the 'degFlatPos' coordinate system (specify
    # this in config) so that these markers are evenly spaced and not more eccentric than necessary
    for m_id, marker in fiducial_positions.iterrows():
        aruco_im = visual.ImageStim(win, image=None, size=config["markers"]["size"], units=config["markers"]["units"])
        aruco_bg = visual.Rect(win, units=config["markers"]["units"],
                               width =config["markers"]["size"]*(1+config["markers"]["margin"]*2),
                               height=config["markers"]["size"]*(1+config["markers"]["margin"]*2),
                               fillColor=config["markers"]["background_color"])

        aruco_bg.pos = [marker.x, marker.y]
        aruco_im.image = get_aruco_marker(m_id, config["markers"]["size"], config["markers"]["units"], win)
        aruco_im.pos = [marker.x, marker.y]

        stimList.append(aruco_bg)
        stimList.append(aruco_im)

        aruco_bg.draw()
        aruco_im.draw()

    # draw placeholder circles at all target positions
    if placeholders:
        if (use_file_color:='color' in target_positions.columns) or config["targets"]["placeholder"]["color"]:
            # if color specified in the the target_positions file, use that. Else, fall back on placeholder color
            # specified in config. If neither is specified, do not draw placeholders
            for _, target in target_positions.iterrows():
                circle = visual.Circle(win, radius=config["targets"]["placeholder"]["diameter"]/2, units=config["targets"]["units"], fillColor=target.color if use_file_color else config["targets"]["placeholder"]["color"])
                circle.pos = [target.x, target.y]
                stimList.append(circle)
                circle.draw()

    # Get screenshot of background, so that we can draw unchanging things at once
    background = visual.ImageStim(win, visual.BufferImageStim(win, stim=stimList).image)    # because https://github.com/psychopy/psychopy/issues/840
    if 'origin' in screen_config:
        background.pos = [-x for x in screen_config["origin"]]
    return {'background': background, 'target_positions': target_positions, 'fiducial_positions': fiducial_positions, 'indicator_IDs': indicator_IDs}

def run_fixation_sequence(win: visual.Window, config: dict, refresh_rate: int, task_vars: dict):
    # prepare visual objects
    fixation_target = ABCFixPoint(win, outer_color=config["targets"]["look"]["outer_color"], inner_color=config["targets"]["look"]["inner_color"], units=config["targets"]["units"])
    aruco_ims = {m_id: visual.ImageStim(win, image=None, size=config["markers"]["size"], units=config["markers"]["units"]) for m_id in task_vars["indicator_IDs"]}
    if (have_cue:=config["targets"]["cue"]["color"] is not None):
        cue = visual.Circle(win, radius=config["targets"]["cue"]["diameter"]/2, units=config["targets"]["units"], fillColor=config["targets"]["cue"]["color"])

    # prepare target order
    if config["targets"]["first_ID"] not in task_vars['target_positions'].index:
        raise ValueError(f'Target ID set in setup.json->targets->first_ID ({config["targets"]["first_ID"]}) is not in the targets file {config["targets"]["file"]}')
    ts = [t for t in task_vars['target_positions'].index.to_list() if t!=config["targets"]["first_ID"]]
    random.shuffle(ts)
    targets = [config["targets"]["first_ID"]] + ts
    if not config["show_segment_between_repetitions"] and config["n_repetitions"]>1:
        all_ts = [t for t in task_vars['target_positions'].index.to_list()]
        for _ in range(config["n_repetitions"]-1):
            while True:
                # shuffle until first target is not the same as last target of previous repetition
                random.shuffle(all_ts)
                if all_ts[0]!=targets[-1]:
                    break
            targets += all_ts

    # prepare task parameters
    n_shrink_frames = int(config["targets"]["shrink"]["duration"]*refresh_rate)
    shrink_sizes    = np.linspace(config["targets"]["look"]["diameter_max"], config["targets"]["look"]["diameter_min"], n_shrink_frames)
    n_capture_frames= int(config["targets"]["duration"]*refresh_rate)

    # show fixation target sequence
    old_pos = task_vars['target_positions'].loc[targets[0],['x','y']].to_numpy()
    for t_id in targets:
        check_escape(win)

        # Move target to new position
        pos   = task_vars['target_positions'].loc[t_id,['x','y']].to_numpy()
        d     = np.hypot(old_pos[0]-pos[0], old_pos[1]-pos[1])
        old_pos_pix = tools.monitorunittools.convertToPix(np.array([0.,0.]),old_pos.astype(np.float64),config["targets"]["units"],win)
        pos_pix     = tools.monitorunittools.convertToPix(np.array([0.,0.]),    pos.astype(np.float64),config["targets"]["units"],win)
        d_pix = np.hypot(old_pos_pix[0]-pos_pix[0], old_pos_pix[1]-pos_pix[1])
        # Target should move at constant speed regardless of distance to cover, duration contains time to move
        # over width of whole screen. Adjust time to proportion of screen width covered by current move
        move_duration = max(config["targets"]["move"]["min_duration"], config["targets"]["move"]["duration"]*d_pix/win.size[0])
        n_move_frames = int(move_duration*refresh_rate)
        if config["targets"]["move"]["move_with_acceleration"]:
            accel   = 0 if not d else d/(move_duration/2)**2    # solve x=.5*a*t^2 for a, use d/2 for x
            moveVec = [0.,0.] if not d else [(x-y)/d for x,y in zip(pos,old_pos)]
            def calc_pos(frac):
                if frac<.5:
                    return [p+m*.5*accel*(frac*move_duration)**2 for p,m in zip(old_pos,moveVec)]
                else:
                    # implement deceleration by accelerating from the other side in backward time
                    return [p-m*.5*accel*((1-frac)*move_duration)**2 for p,m in zip(pos,moveVec)]
            tar_pos = [calc_pos(x) for x in np.linspace(0., 1., n_move_frames)]
        else:
            x_tar_pos = np.linspace(old_pos[0], pos[0], n_move_frames)
            y_tar_pos = np.linspace(old_pos[1], pos[1], n_move_frames)
            tar_pos = [[x,y] for x,y in zip(x_tar_pos,y_tar_pos)]

        # replace markers indicating which target is shown (multiple for redundancy)
        for m_id in task_vars['indicator_IDs']:
            m_new_id = task_vars['indicator_IDs'][m_id][t_id]
            aruco_ims[m_id].image = get_aruco_marker(m_new_id, config["markers"]["size"], config["markers"]["units"], win)
            aruco_ims[m_id].pos   = task_vars['fiducial_positions'].loc[m_id,['x','y']].tolist()

        # move to next target
        fixation_target.set_size(config["targets"]["move"]["diameter"])
        if have_cue:
            cue.pos = pos
        for p in tar_pos:
            task_vars['background'].draw()
            if have_cue:
                cue.draw()
            fixation_target.set_pos(p)
            fixation_target.draw()
            win.flip()

        # shrink
        for s in shrink_sizes:
            task_vars['background'].draw()
            fixation_target.set_size(s)
            fixation_target.draw()
            win.flip()

        # fixation period
        for _ in range(n_capture_frames):
            task_vars['background'].draw()
            [aruco_ims[m_id].draw() for m_id in aruco_ims]
            fixation_target.draw()
            win.flip()

        old_pos = pos