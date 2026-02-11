# -*- coding: utf-8 -*-
from psychopy import visual, core, event
import numpy as np
import json
import traceback
import pathlib
import os

import utils

def run_psa(win: visual.Window, config: dict, screen_config: dict):
    # prepare visual objects
    fixation_target  = utils.ABCFixPoint(
        win,
        outer_diameter=config["target"]["outer_diameter"], inner_diameter=config["target"]["inner_diameter"],
        outer_color   =config["target"]["outer_color"]   , inner_color   =config["target"]["inner_color"],
        units=config["target"]["units"])
    background_patch = visual.Circle(win, radius=config["background_patch"]["radius"], units=config["target"]["units"], fillColor=config["background_patch"]["color"], edges='circle')
    # set up background and position it correctly (correcting for origin)
    window_background = visual.Rect(win, size=win.size, units='pix', fillColor=config["background_colors"]["precycle"])
    window_background.units = screen_config["units"]
    window_background.pos = [-x for x in screen_config["origin"]]
    # fiducials
    if (l1:=len(config["fiducials"]["rho"]))!=(l2:=len(config["fiducials"]["phi"])):
        raise ValueError(f"length of config['PSA']['fiducials']['rho'] ({l1}) should be equal to length of config['PSA']['fiducials']['phi'] ({l2})")
    fiducials_array  = [visual.Circle(win, pos=utils.pol2cart(config["fiducials"]["rho"][r], phi/180*np.pi), radius=config["fiducials"]["radius"], units=config["fiducials"]["units"], edges='circle', fillColor=config["fiducials"]["color"])
                        for r,phi in enumerate(config["fiducials"]["phi"])]

    # convert durations
    n_frames_pre    = int(config["durations"]["precycle"]*screen_config["refresh_rate"])
    n_frames_bright = int(config["durations"]["bright"]*screen_config["refresh_rate"])
    n_frames_dark   = int(config["durations"]["dark"]*screen_config["refresh_rate"])

    def draw(draw_fiducials: bool):
        window_background.draw()
        if draw_fiducials:
            [f.draw() for f in fiducials_array]
        background_patch.draw()
        fixation_target.draw()

    target_positions = config["target"]["positions"]
    np.random.shuffle(target_positions)
    for _, p in enumerate(target_positions):
        background_patch.pos = (p[0], p[1])
        fixation_target.set_pos((p[0], p[1]))

        # show sequence start screen
        window_background.color = config["background_colors"]["precycle"]
        for _ in range(n_frames_pre):
            draw(draw_fiducials=False)
            win.flip()

        for _ in range(config["n_cycle"]):
            utils.check_escape(win)

            # bright background
            window_background.color = config["background_colors"]["bright"]
            for _ in range(n_frames_bright):
                draw(draw_fiducials=True)
                win.flip()

            utils.check_escape(win)

            # dark background
            window_background.color = config["background_colors"]["dark"]
            for _ in range(n_frames_dark):
                draw(draw_fiducials=True)
                win.flip()

def main():
    os.chdir(pathlib.Path(__file__).parent)  # set working directory to script directory to ensure relative paths work

    # read protocol setup
    with open("setup_station1.json") as fp:
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
        # prepare instruction (NB, doesn't respect origin, so set origin as pos)
        textstim = visual.TextStim(win, text="", height=config["instruction_text"]["height"], color=config["instruction_text"]["color"], wrapWidth=120., pos=config["screen"]["origin"])

        # show setup screen
        utils.run_setup_check(win, config['setup_check'], config["screen"])

        # run et_sync
        utils.run_et_sync(win, textstim, segmenter, config['et_sync'], config["screen"]["refresh_rate"])

        # fixation sequence instruction
        textstim.text = 'A black-and-white fixation target will appear in the center of the\nscreen. Carefully look at its center all the time.\nAfter a while, it will start moving. Once it starts moving,\nfollow the fixation target around, and keep carefully looking at its center all the time.\n\n(Press the spacebar to start)'
        textstim.draw()
        win.flip()
        event.waitKeys()
        # prepare fixation sequence
        task_vars = utils.prepare_fixation_sequence(win, config["validation"], config["screen"])
        # run fixation sequence
        for r in range(config["validation"]["n_repetitions"]):
            # signal fixation sequence start
            for m_id in config["validation"]["segment_marker"]["start_IDs"]:
                segmenter.draw(m_id)
            # run fixation sequence
            utils.run_fixation_sequence(win, config["validation"], config["screen"]["refresh_rate"], task_vars)
            # signal fixation sequence end
            for m_id in config["validation"]["segment_marker"]["end_IDs"]:
                segmenter.draw(m_id)
            # blank screen to separate multiple fixation sequences
            if r!=config["validation"]["n_repetitions"]-1:
                win.flip()
                core.wait(1.)

        # PSA instruction
        textstim.text = 'A black-and-white fixation target will appear on top of a grey circle.\nCarefully look at the center of this target at all times.\n\n(Press the spacebar to start)'
        textstim.draw()
        win.flip()
        event.waitKeys()
        # signal PSA start
        for m_id in config["PSA"]["segment_marker"]["start_IDs"]:
            segmenter.draw(m_id)
        # run PSA
        run_psa(win, config['PSA'], config["screen"])
        # signal PSA end
        for m_id in config["PSA"]["segment_marker"]["end_IDs"]:
            segmenter.draw(m_id)

    except Exception as e:
        tb_lines = traceback.format_exception(type(e), e, e.__traceback__)
        print("".join(tb_lines))
    finally:
        if 'win' in locals():
            win.close()

if __name__=="__main__":
    main()