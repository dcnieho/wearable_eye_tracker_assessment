# -*- coding: utf-8 -*-
from psychopy import visual, event, sound
import json
import traceback
import pathlib
import os

import utils

def prepare_slippage(win: visual.Window, config: dict, screen_config: dict):
    return utils.prepare_fixation_sequence(win, config, screen_config, placeholders=False)
def prepare_parallax(win: visual.Window, config: dict, screen_config: dict, run: int):
    return utils.prepare_fixation_sequence(win, config, screen_config, run_id=run, placeholders=True)

def run_slippage(win: visual.Window, config: dict, refresh_rate: int, task_vars: dict):
    # prepare visual objects
    fixation_target = utils.ABCFixPoint(win, outer_diameter=config["targets"]["look"]["diameter"], outer_color=config["targets"]["look"]["outer_color"], inner_color=config["targets"]["look"]["inner_color"], units=config["targets"]["units"])

    # prepare sound objects
    if (has_metronome:="metronome" in config):
        beep = sound.Sound(value=config["metronome"]["frequency"], secs=config["metronome"]["duration"], volume=config["metronome"]["volume"], stereo=True)
        n_frame_metronome = int(config["metronome"]["interval"]*refresh_rate)

    # prepare task parameters
    n_capture_frames= int(config["targets"]["duration"]*refresh_rate)

    # show fixation target sequence
    pos   = task_vars['target_positions'].loc[1,['x','y']].to_numpy()
    fixation_target.set_pos(pos)
    for fr_idx in range(n_capture_frames):
        if has_metronome and fr_idx%n_frame_metronome==0:
            beep.play()
        task_vars['background'].draw()
        fixation_target.draw()
        win.flip()

def main():
    os.chdir(pathlib.Path(__file__).parent)  # set working directory to script directory to ensure relative paths work

    # read protocol setup
    with open("setup_station2.json") as fp:
        config = json.load(fp)

    try:
        # Open window, check
        win = utils.open_window(config["screen"])

        # prepare fiducials
        utils.load_aruco_dict(config["aruco"]["dict"], config["aruco"]["border_bits"])

        # prepare trial segmentation
        segmenter = utils.SegmentationMarker(
            win, config["screen"]["refresh_rate"],
            config["segment_marker"]["duration"],
            config["segment_marker"]["size"], config["segment_marker"]["units"], config["segment_marker"]["margin"],
            config["segment_marker"]["background_color"])
        # prepare instruction
        textstim = visual.TextStim(win, text="", height=config["instruction_text"]["height"], units=config["instruction_text"]["units"], color=config["instruction_text"]["color"], wrapWidth=120.)

        # show setup screen
        utils.run_setup_check(win, config['setup_check'], config["screen"])

        # run et_sync
        utils.run_et_sync(win, textstim, segmenter, config['et_sync'], config["screen"]["refresh_rate"])

        # parallax instruction
        for d_int in range(len(config['parallax']['distances'])):
            win.monitor.setDistance(config['parallax']['distances'][d_int])
            segmenter.update()
            textstim.text = f'Sit at a distance of {config["parallax"]["distances"][d_int]} cm.\n\nA black-and-white fixation target will appear in the center of the\nscreen. Carefully look at its center all the time.\nAfter a while, it will start moving. Once it starts moving,\nfollow the fixation target around, and keep carefully\nlooking at its center all the time.\n\n(Press the spacebar to start)'
            textstim.draw()
            win.flip()
            event.waitKeys()
            # prepare parallax
            task_vars = utils.prepare_fixation_sequence(win, config["parallax"], config["screen"], run_id=d_int, placeholders=True)
            # signal parallax start
            for m_id in config["parallax"]["segment_marker"]["start_IDs"][d_int]:
                segmenter.draw(m_id)
            # run parallax
            utils.run_fixation_sequence(win, config['parallax'], config["screen"]["refresh_rate"], task_vars)
            # signal parallax end
            for m_id in config["parallax"]["segment_marker"]["end_IDs"][d_int]:
                segmenter.draw(m_id)

        # slippage
        win.monitor.setDistance(config['screen']['viewing_distance'])
        segmenter.update()

        for s_int in range(3):
            match s_int:
                case 0:
                    task_str = 'move them up and down'
                    movie_filename = "station2_slippage_updown.mp4"
                case 1:
                    task_str = 'slide them along the nose bridge'
                    movie_filename = "station2_slippage_slide_nosebridge.mp4"
                case 2:
                    task_str = 'rotate them around the nose bridge (one side up, other side down)'
                    movie_filename = "station2_slippage_rotate_nosebridge.mp4"
            textstim.text = 'A black-and-white fixation target will appear in the center of the\nscreen. Carefully look at its center all the time.\nWhile you keep fixating the fixation target, hold the glasses between your fingers and\n' + task_str + '.\n\n(Press the spacebar to start)'
            textstim.pos = (0, 10)
            movie = visual.MovieStim(
                win,
                filename=movie_filename,
                units='deg',
                size=(20.0, 11.25),  # keep the aspect ratio consistent with your file (e.g., 16:9)
                pos=(0.0, -5.0),     # below the text
                loop=True,           # loop while we wait
                noAudio=False,       # <<< enable audio playback
                volume=1.0,          # set volume to max (0.0 to 1.0)
            )

            # play video and draw instruction text until spacebar is pressed
            event.clearEvents()
            movie.play()  # start audio+video
            while True:
                movie.draw()
                textstim.draw()
                win.flip()
                keys = event.getKeys()
                if 'space' in keys:
                    break
                utils.check_escape(win, keys)
            movie.stop()

            # prepare slippage
            task_vars = utils.prepare_fixation_sequence(win, config["slippage"], config["screen"], placeholders=False)
            # signal slippage start
            for m_id in config["slippage"]["segment_marker"]["start_IDs"][s_int]:
                segmenter.draw(m_id)
            # run slippage
            run_slippage(win, config["slippage"], config["screen"]["refresh_rate"], task_vars)
            # signal slippage end
            for m_id in config["slippage"]["segment_marker"]["end_IDs"][s_int]:
                segmenter.draw(m_id)

    except Exception as e:
        tb_lines = traceback.format_exception(type(e), e, e.__traceback__)
        print("".join(tb_lines))
    finally:
        if 'win' in locals():
            win.close()

if __name__=="__main__":
    main()