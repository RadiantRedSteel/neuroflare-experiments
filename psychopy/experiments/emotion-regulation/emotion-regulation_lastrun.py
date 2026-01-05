#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2025.2.3),
    on January 03, 2026, at 16:58
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware, parallel
from psychopy.tools import environmenttools
from psychopy.constants import (
    NOT_STARTED, STARTED, PLAYING, PAUSED, STOPPED, STOPPING, FINISHED, PRESSED, 
    RELEASED, FOREVER, priority
)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

# Run 'Before Experiment' code from code_iasp_block_randomizer
# ------------------------------------------------------------
# code_iasp_block_randomizer
#
# Generates all trial CSV files used by the experiment:
#   - emotion_regulation_unpleasant_trials.csv
#   - emotion_regulation_neutral_trials.csv
#   - emotion_regulation_practice_trials.csv
#
# Performs the following:
#   • Checks for missing or extra IASP images
#   • Randomizes technique assignment to blocks
#   • Randomizes block order
#   • Randomizes photo order within each block
#   • Inserts placeholder images if needed
#
# This code runs once before the experiment begins and writes
# fully randomized trial files that the main loops will read.
#
# IMPORTANT:
#   This component does NOT slice or select trials during the
#   experiment. It only creates the CSVs.
# ------------------------------------------------------------

import random
import csv
import os

placeholder_filename = 'IASP/placeholder.jpg'

# Fallback: if placeholder image is absent, use empty string to avoid PsychoPy crash
if not os.path.exists(placeholder_filename):
    print("Warning: placeholder image not found; using empty string to avoid crash.")
    placeholder_filename = ''

# Techniques and blocks
techniques = ['View', 'Suppress', 'Reappraise', 'Suppress and Reappraise']
block_ids = ['block1', 'block2', 'block3', 'block4']

# Photo sets per block
photo_sets = {
    'block1': [1019, 1040, 1070, 1110, 1201, 1321, 2130, 2691, 2700, 2751, 2811, 3016, 3215, 3301, 5920, 6010, 6242, 6311, 6370, 6800, 6940, 8485, 9041, 9180, 9230, 9300, 9340, 9404, 9410, 9419, 9425, 9430, 9471, 9561, 9584, 9611, 9635, 9900, 9910, 9920],
    'block2': [1022, 1050, 1080, 1111, 1274, 1302, 2120, 2683, 2695, 2705, 2800, 3005, 3181, 3300, 4621, 5973, 6241, 6260, 6315, 6571, 6840, 8480, 9040, 9140, 9220, 9270, 9330, 9373, 9411, 9420, 9426, 9440, 9480, 9570, 9592, 9620, 9700, 9901, 9911, 9921],
    'block3': [1026, 1051, 1090, 1114, 1275, 1301, 2095, 2220, 2694, 2704, 2799, 3001, 3110, 3230, 3530, 5971, 6212, 6250, 6313, 6560, 6383, 8231, 9001, 9120, 9210, 9254, 9320, 9342, 9409, 9417, 9423, 9429, 9470, 9520, 9582, 9600, 9630, 9830, 9903, 9913],
    'block4': [1030, 1052, 1101, 1200, 1300, 1932, 2205, 2692, 2703, 2795, 2900, 3030, 3225, 3500, 5950, 6200, 6243, 6312, 6550, 6821, 8230, 9000, 9050, 9181, 9250, 9301, 9341, 9405, 9415, 9421, 9427, 9452, 9490, 9571, 9594, 9622, 9810, 9902, 9912, 9925],
}

block_neutral = [2272, 2383, 2393, 2396, 2397, 2435, 2480, 2514, 2518, 2575, 2580, 2593, 2594, 2870, 2880, 2980, 5395, 5455, 5471, 5520, 5740, 7002, 7004, 7036, 7037, 7041, 7130, 7140, 7205, 7217, 7491, 7493, 7495, 7496, 7504, 7546, 7550, 7640, 7705, 8221]
block_practice = [1205, 3103, 3180, 3213, 3220, 3280, 5970, 6190]

all_sets = list(photo_sets.values()) + [block_neutral, block_practice]

# Check to see if any photos are missing
def check_missing_photos(photo_id: str) -> bool:
    return not os.path.exists(f"IASP/{photo_id}.jpg")

missing_list = [
    photo_id
    for block in all_sets
    for photo_id in block
    if check_missing_photos(photo_id)
]

missing_set = set(missing_list)

if missing_list:
    print(f"Missing stimuli: {len(missing_set)}")
    print("Names:", sorted(missing_set))
else:
    print("All IASP files found.")

# ----------------------------------------------------------
# Check for extra files in IASP/ that are not referenced in any block
def _collect_image_ids(folder: str) -> set[int]:
    ids = set()
    try:
        for name in os.listdir(folder):
            lower = name.lower()
            if lower.endswith(('.jpg', '.jpeg')):
                stem, _ = os.path.splitext(name)
                if stem.lower() == 'placeholder':
                    continue
                if stem.isdigit():
                    ids.add(int(stem))
    except FileNotFoundError:
        pass
    return ids

iasp_folder = 'IASP'
available_ids = _collect_image_ids(iasp_folder)
expected_ids = {pid for block in all_sets for pid in block}
extra_ids = sorted(available_ids - expected_ids)

if not os.path.isdir(iasp_folder):
    print("IASP folder not found; skipping extra-file check.")
elif extra_ids:
    print(f"Extra stimuli in IASP not used: {len(extra_ids)}")
    print("Names:", extra_ids)
else:
    print("No extra stimuli in IASP source folder.")

# ----------------------------------------------------------
# Create negative blocks trials file
# Randomly assign techniques to blocks
negative_file_name = 'emotion_regulation_unpleasant_trials.csv'

random.shuffle(techniques)
block_to_technique = dict(zip(block_ids, techniques))
negative_trials_technique_order = [] # Referenced in reset routine

# Randomize block order
block_order = block_ids.copy()
random.shuffle(block_order)

# Build trial list
negative_trials = []
negative_trial_number = 1

for block in block_order:
    technique = block_to_technique[block]
    negative_trials_technique_order.append(technique)
    photos = photo_sets[block].copy()
    random.shuffle(photos)
    
    for photo_id in photos:
        if photo_id in missing_set:
            photo_path = placeholder_filename
        else:
            photo_path = f"IASP/{photo_id}.jpg"

        negative_trials.append({
            'trial_number': negative_trial_number,
            'block_id': block,
            'technique': technique,
            'photo_filename': photo_path
        })
        negative_trial_number += 1

# Save to CSV
with open(negative_file_name, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=negative_trials[0].keys())
    writer.writeheader()
    writer.writerows(negative_trials)

print(f"Saved randomized trial file: {negative_file_name}")

# ----------------------------------------------------------
# Create block_neutral trials file
neutral_file_name = 'emotion_regulation_neutral_trials.csv'
neutral_trials = []
neutral_trial_number = 1

# Randomize photo order
neutral_photos = block_neutral.copy()
random.shuffle(neutral_photos)

for photo_id in neutral_photos:
    if photo_id in missing_set:
        photo_path = placeholder_filename
    else:
        photo_path = f"IASP/{photo_id}.jpg"

    neutral_trials.append({
        'trial_number': neutral_trial_number,
        'block_id': 'block_neutral',
        'technique': 'View',
        'photo_filename': photo_path
    })
    neutral_trial_number += 1

# Save block_neutral to CSV
with open(neutral_file_name, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=neutral_trials[0].keys())
    writer.writeheader()
    writer.writerows(neutral_trials)

print(f"Saved randomized trial file: {neutral_file_name}")

# ----------------------------------------------------------
# Create block_practice trials file
practice_file_name = 'emotion_regulation_practice_trials.csv'
practice_trials = []
practice_trial_number = 1

# Randomize photo order
practice_photos = block_practice.copy()
random.shuffle(practice_photos)

for photo_id in practice_photos:
    if photo_id in missing_set:
        photo_path = placeholder_filename
    else:
        photo_path = f"IASP/{photo_id}.jpg"

    practice_trials.append({
        'trial_number': practice_trial_number,
        'block_id': 'block_practice',
        'technique': 'View',
        'photo_filename': photo_path
    })
    practice_trial_number += 1

# Save block_practice to CSV
with open(practice_file_name, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=practice_trials[0].keys())
    writer.writeheader()
    writer.writerows(practice_trials)

print(f"Saved randomized trial file: {practice_file_name}")
# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2025.2.3'
expName = 'emotion-regulation'  # from the Builder filename that created this script
expVersion = '0.5'
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'expVersion|hid': expVersion,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [2560, 1440]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # replace default participant ID
    if prefs.piloting['replaceParticipantID']:
        expInfo['participant'] = 'pilot'

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version=expVersion,
        extraInfo=expInfo, runtimeInfo=None,
        originPath='D:\\Github\\neuroflare-experiments\\psychopy\\experiments\\emotion-regulation\\emotion-regulation_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('debug')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[-0.5000, -0.5000, -0.5000], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [-0.5000, -0.5000, -0.5000]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    if PILOTING:
        # show a visual indicator if we're in piloting mode
        if prefs.piloting['showPilotingIndicator']:
            win.showPilotingIndicator()
        # always show the mouse in piloting mode
        if prefs.piloting['forceMouseVisible']:
            win.mouseVisible = True
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    ioSession = ioServer = eyetracker = None
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ptb'
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], currentRoutine=None):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    currentRoutine : psychopy.data.Routine
        Current Routine we are in at time of pausing, if any. This object tells PsychoPy what Components to pause/play/dispatch.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='PsychToolbox',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # dispatch messages on response components
        if currentRoutine is not None:
            for comp in currentRoutine.getDispatchComponents():
                comp.device.dispatchMessages()
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # update experiment info
    expInfo['date'] = data.getDateStr()
    expInfo['expName'] = expName
    expInfo['expVersion'] = expVersion
    expInfo['psychopyVersion'] = psychopyVersion
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='PsychToolbox'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "experimentSetup" ---
    # Run 'Begin Experiment' code from code_experiment_setup
    # ------------------------------------------------------------
    # code_experiment_setup
    # Global screen geometry and shared layout variables.
    #
    # Computes window aspect ratio and establishes a unified
    # coordinate system in 'height' units so all routines scale
    # consistently across monitors.
    #
    # Also defines shared wrap-width values for text components.
    # ------------------------------------------------------------
    
    # ============================================================
    # =============== GLOBAL SCREEN GEOMETRY =====================
    # ============================================================
    # Centralized screen geometry so all routines share the same coordinate system
    
    screen_width, screen_height = win.size
    aspect = screen_width / screen_height  # width / height
    
    # In 'height' units:
    # - Full screen height = 1.0
    # - Full screen width  = aspect
    leftEdge = -aspect / 2
    rightEdge = aspect / 2
    topEdge = 0.5
    bottomEdge = -0.5
    
    # Wrap width as a percentage of screen width
    aspect80 = aspect * 0.80 # 80% of screen width
    aspect20 = aspect * 0.20  # 20% of screen width
    comp_wrap_width_body = aspect80
    comp_wrap_width_header = aspect80
    comp_wrap_width_continue = aspect80 
    
    # Run 'Begin Experiment' code from code_state_measure_setup
    # ------------------------------------------------------------
    # code_state_measure_setup
    # Global setup logic for the stateMeasure routine.
    #
    # Handles scaling rules, slider configuration helpers, and
    # shared variables needed to dynamically construct rating
    # scales based on loop parameters.
    #
    # The stateMeasure routine depends on this setup code to
    # size components correctly and interpret SAM/text settings.
    # ------------------------------------------------------------
    
    # --- Helper functions for dynamic slider configuration ---
    def parse_tick_values(raw):
        # If PsychoPy already parsed it into a list, just return it
        if isinstance(raw, list):
            return raw
        # Otherwise assume it's a comma-separated string
        return [int(x.strip()) for x in raw.split(',')]
    
    # ===============================================================
    # ============== STATE-MEASURE SPECIFIC GEOMETRY ================
    # ===============================================================
    
    def sm_compute_tick_positions(tick_values, rating_type):
        """
        Compute slider size, slider position, and min/max label positions
        for both SAM and Generic rating types.
        """
    
        if rating_type == "SAM":
            # SAM uses the precomputed geometry
            slider_size = (sm_sam_comp_slider_width, sm_sam_comp_slider_size_height)
            slider_pos  = (0, sm_sam_comp_slider_pos_y)
    
            # Tick positions already computed globally
            tick_positions = sm_sam_tick_positions
    
            label_min_pos = (tick_positions[0],  sm_sam_comp_label_pos_y)
            label_max_pos = (tick_positions[-1], sm_sam_comp_label_pos_y)
    
        else:
            # Generic slider: evenly spaced ticks across the slider width
            slider_size = (sm_gen_comp_slider_size_width, sm_gen_comp_slider_size_height)
            slider_pos  = (0, sm_gen_comp_slider_pos_y)
    
            # Compute generic tick positions dynamically
            num_ticks = len(tick_values)
            half_width = sm_gen_comp_slider_size_width / 2
    
            # Even spacing across the width
            tick_positions = [
                -half_width + (i / (num_ticks - 1)) * sm_gen_comp_slider_size_width
                for i in range(num_ticks)
            ]
    
            # Labels sit below the slider
            label_min_pos = (tick_positions[0],  sm_gen_comp_label_pos_y)
            label_max_pos = (tick_positions[-1], sm_gen_comp_label_pos_y)
    
        return {
            "slider_size": slider_size,
            "slider_pos": slider_pos,
            "tick_positions": tick_positions,
            "label_min_pos": label_min_pos,
            "label_max_pos": label_max_pos
        }
    
    # ===============================================================
    # =================== GEN SPECIFIC GEOMETRY =====================
    # ===============================================================
    
    # --- Generic Slider Scaling ---
    # Full-width slider across screen
    # Generic slider (used when rating_type != "SAM")
    
    # Slider layout (height units)
    sm_gen_comp_slider_size_width  = aspect * 0.6  # 60% of screen width
    sm_gen_comp_slider_size_height = 0.1           # slider thickness/height
    sm_gen_comp_slider_pos_y       = 0             # vertical position of the slider center
    
    # Label Min/Max layout
    sm_gen_comp_label_pos_y = sm_gen_comp_slider_pos_y - 0.2
    
    # ===============================================================
    # =================== SAM SPECIFIC GEOMETRY =====================
    # ===============================================================
    
    # --- SAM image geometry (pixels) ---
    sm_sam_img_px_width  = 1168
    sm_sam_img_px_height = 231
    sm_sam_img_aspect    = sm_sam_img_px_width / sm_sam_img_px_height  # ≈ 5.056
    
    # Pictogram/gap geometry (pixels)
    # Note: there are 5 pictograms and 4 gaps spanning the full image width
    sm_sam_pictogram_px_width     = 215
    sm_sam_pictogram_gap_px_width = 23
    
    # --- Layout decisions (height units) ---
    sm_sam_comp_img_size_height = 0.25                                             # 25% of screen height
    sm_sam_comp_img_size_width  = sm_sam_comp_img_size_height * sm_sam_img_aspect  # scaled width based on aspect
    sm_sam_comp_img_pos_y       = 0.1                                              # vertical position of the image
    
    # Message Layout
    sm_sam_comp_message_pos_y = sm_sam_comp_img_pos_y + 0.2
    
    # Slider layout (height units)
    sm_sam_comp_slider_pos_y       = -0.09      # vertical position of the slider center
    sm_sam_comp_slider_size_height = 0.06       # slider thickness/height
    sm_sam_slider_px_width    = sm_sam_img_px_width - 215
    sm_sam_comp_slider_width  = sm_sam_comp_img_size_width * (sm_sam_slider_px_width / sm_sam_img_px_width)
    
    # Scale factor: px -> 'height' units within the image width
    sm_sam_scale_factor = sm_sam_comp_img_size_width / sm_sam_img_px_width
    
    sm_sam_pictogram_w  = sm_sam_pictogram_px_width * sm_sam_scale_factor
    sm_sam_gap_w        = sm_sam_pictogram_gap_px_width * sm_sam_scale_factor
    
    # --- Compute 9 tick x-positions aligned to pictograms ---
    sm_sam_tick_positions = []
    sm_sam_tick_x = -sm_sam_comp_img_size_width / 2 + sm_sam_pictogram_w / 2
    
    for i in range(9):
        sm_sam_tick_positions.append(sm_sam_tick_x)
        if i % 2 == 0:
            sm_sam_tick_x += sm_sam_pictogram_w / 2 + sm_sam_gap_w / 2
        else:
            sm_sam_tick_x += sm_sam_gap_w / 2 + sm_sam_pictogram_w / 2
    
    # Label Min/Max layout
    sm_sam_comp_label_pos_y = sm_sam_comp_slider_pos_y - 0.15
    # Run 'Begin Experiment' code from code_iasp_setup
    # ------------------------------------------------------------
    # code_iasp_setup
    # Computes on-screen size for IASP images based on the window's
    # aspect ratio. Preserves the original 4:3 image ratio and scales
    # to the largest size that fits without distortion.
    #
    # Used by iaspView to size the image consistently across monitors.
    # ------------------------------------------------------------
    
    # Image aspect ratio (width / height)
    iasp_img_aspect = 1024 / 768  # = 1.3333
    
    # If the window is wider than the image, height is the limiting factor
    if aspect >= iasp_img_aspect:
        # Height fills the screen (minus a small margin if you want)
        iasp_comp_img_size_height = 1.0
        iasp_comp_img_size_width = iasp_comp_img_size_height * aspect
    else:
        # Width fills the screen
        iasp_comp_img_size_width = aspect
        iasp_comp_img_size_height = iasp_comp_img_size_width / aspect
    
    # If you want a little breathing room around the image:
    #iasp_margin = 0.05  # 5% margin
    #iasp_comp_img_height *= (1 - iasp_margin)
    #iasp_comp_img_width  *= (1 - iasp_margin)
    # Run 'Begin Experiment' code from code_iasp_block_randomizer
    # Initialize slice indices
    start = 0
    end = 40
    
    # Store for logging
    thisExp.addData('block_order', block_order)
    thisExp.addData('block_to_technique', block_to_technique)
    
    # --- Initialize components for Routine "welcome" ---
    t_welcome_body = visual.TextStim(win=win, name='t_welcome_body',
        text='',
        font='Arial',
        units='height', pos=(0.0, 0.05), draggable=False, height=0.07, wrapWidth=comp_wrap_width_body, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    t_welcome_continue = visual.TextStim(win=win, name='t_welcome_continue',
        text='Press the SPACEBAR to continue',
        font='Arial',
        units='height', pos=(0, -0.35), draggable=False, height=0.04, wrapWidth=comp_wrap_width_continue, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_welcome = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "stateMeasure" ---
    image_SAM = visual.ImageStim(
        win=win,
        name='image_SAM', units='height', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=1.0,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    t_sm_message = visual.TextStim(win=win, name='t_sm_message',
        text='',
        font='Arial',
        pos=(0, .35), draggable=False, height=0.07, wrapWidth=comp_wrap_width_header, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    t_sm_continue = visual.TextStim(win=win, name='t_sm_continue',
        text='Press the SPACEBAR to continue',
        font='Arial',
        pos=(0, -0.35), draggable=False, height=0.04, wrapWidth=comp_wrap_width_continue, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    key_resp_sm = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "instructionNeutral" ---
    t_neutral_instruction = visual.TextStim(win=win, name='t_neutral_instruction',
        text='',
        font='Arial',
        pos=(0.0, 0.05), draggable=False, height=0.07, wrapWidth=comp_wrap_width_body, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    t_neutral_continue = visual.TextStim(win=win, name='t_neutral_continue',
        text='Press the SPACEBAR to begin',
        font='Arial',
        pos=(0, -0.35), draggable=False, height=0.04, wrapWidth=comp_wrap_width_continue, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_neutral = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "emotionRegulationCue" ---
    cross_fixation = visual.ShapeStim(
        win=win, name='cross_fixation', vertices='cross',units='height', 
        size=(0.2, 0.2),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    t_emotion_regulation_cue = visual.TextStim(win=win, name='t_emotion_regulation_cue',
        text='',
        font='Arial',
        pos=(0.0, 0.05), draggable=False, height=0.14, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    t_blank_delayer = visual.TextStim(win=win, name='t_blank_delayer',
        text=None,
        font='Arial',
        pos=(0.0, 0.05), draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    
    # --- Initialize components for Routine "iaspView" ---
    image_iasp = visual.ImageStim(
        win=win,
        name='image_iasp', units='height', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=1.0,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    p_port_iasp = parallel.ParallelPort(address='0x0378')
    
    # --- Initialize components for Routine "stateMeasure" ---
    image_SAM = visual.ImageStim(
        win=win,
        name='image_SAM', units='height', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=1.0,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    t_sm_message = visual.TextStim(win=win, name='t_sm_message',
        text='',
        font='Arial',
        pos=(0, .35), draggable=False, height=0.07, wrapWidth=comp_wrap_width_header, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    t_sm_continue = visual.TextStim(win=win, name='t_sm_continue',
        text='Press the SPACEBAR to continue',
        font='Arial',
        pos=(0, -0.35), draggable=False, height=0.04, wrapWidth=comp_wrap_width_continue, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    key_resp_sm = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "instructionPractice" ---
    t_practice_instruction = visual.TextStim(win=win, name='t_practice_instruction',
        text='',
        font='Arial',
        pos=(0.0, 0.05), draggable=False, height=0.07, wrapWidth=comp_wrap_width_body, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    t_practice_continue = visual.TextStim(win=win, name='t_practice_continue',
        text='Press the SPACEBAR to begin',
        font='Arial',
        pos=(0, -0.35), draggable=False, height=0.04, wrapWidth=comp_wrap_width_continue, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_practice = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "emotionRegulationCue" ---
    cross_fixation = visual.ShapeStim(
        win=win, name='cross_fixation', vertices='cross',units='height', 
        size=(0.2, 0.2),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    t_emotion_regulation_cue = visual.TextStim(win=win, name='t_emotion_regulation_cue',
        text='',
        font='Arial',
        pos=(0.0, 0.05), draggable=False, height=0.14, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    t_blank_delayer = visual.TextStim(win=win, name='t_blank_delayer',
        text=None,
        font='Arial',
        pos=(0.0, 0.05), draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    
    # --- Initialize components for Routine "iaspView" ---
    image_iasp = visual.ImageStim(
        win=win,
        name='image_iasp', units='height', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=1.0,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    p_port_iasp = parallel.ParallelPort(address='0x0378')
    
    # --- Initialize components for Routine "stateMeasure" ---
    image_SAM = visual.ImageStim(
        win=win,
        name='image_SAM', units='height', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=1.0,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    t_sm_message = visual.TextStim(win=win, name='t_sm_message',
        text='',
        font='Arial',
        pos=(0, .35), draggable=False, height=0.07, wrapWidth=comp_wrap_width_header, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    t_sm_continue = visual.TextStim(win=win, name='t_sm_continue',
        text='Press the SPACEBAR to continue',
        font='Arial',
        pos=(0, -0.35), draggable=False, height=0.04, wrapWidth=comp_wrap_width_continue, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    key_resp_sm = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "instructionUnpleasant" ---
    t_unpleasant_instruction = visual.TextStim(win=win, name='t_unpleasant_instruction',
        text='',
        font='Arial',
        pos=(0.0, 0.05), draggable=False, height=0.07, wrapWidth=comp_wrap_width_body, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    t_unpleasant_continue = visual.TextStim(win=win, name='t_unpleasant_continue',
        text='Otherwise, press the SPACEBAR to begin the emotion regulation tasks',
        font='Arial',
        pos=(0, -0.35), draggable=False, height=0.04, wrapWidth=comp_wrap_width_continue, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_unpleasant = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "reset" ---
    # Run 'Begin Experiment' code from code_slicer_helper
    reset_routine_counter = 0
    just_view_text = "Just View\n\nSimply look at the picture as it appears.\nDo not try to change your feelings or reactions.\nLet your natural response happen and just observe."
    suppress_text = "Suppress Emotion\n\nTry not to feel the unpleasant emotions the picture might bring up.\nYou can push the feelings down, ignore them, or keep a neutral expression.\nFocus on staying calm and steady."
    reappraise_text = "Reappraise\n\nChange the way you think about the picture so it feels less unpleasant.\nYou might imagine the situation has a neutral or positive outcome, that it is staged or not real, or that the people are safe afterward."
    suppress_and_reappraise_text = "Suppress and Reappraise\n\nUse both strategies at the same time.\nFirst reinterpret the picture to make it feel less unpleasant.\nAt the same time, try not to show or feel the unpleasant emotion."
    t_reset_instruction = visual.TextStim(win=win, name='t_reset_instruction',
        text='',
        font='Arial',
        pos=(0.0, 0.05), draggable=False, height=0.07, wrapWidth=comp_wrap_width_body, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    t_reset_continue = visual.TextStim(win=win, name='t_reset_continue',
        text='Press the SPACEBAR to begin',
        font='Arial',
        pos=(0, -0.35), draggable=False, height=0.04, wrapWidth=comp_wrap_width_continue, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    key_resp_reset = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "emotionRegulationCue" ---
    cross_fixation = visual.ShapeStim(
        win=win, name='cross_fixation', vertices='cross',units='height', 
        size=(0.2, 0.2),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    t_emotion_regulation_cue = visual.TextStim(win=win, name='t_emotion_regulation_cue',
        text='',
        font='Arial',
        pos=(0.0, 0.05), draggable=False, height=0.14, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    t_blank_delayer = visual.TextStim(win=win, name='t_blank_delayer',
        text=None,
        font='Arial',
        pos=(0.0, 0.05), draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    
    # --- Initialize components for Routine "iaspView" ---
    image_iasp = visual.ImageStim(
        win=win,
        name='image_iasp', units='height', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=1.0,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    p_port_iasp = parallel.ParallelPort(address='0x0378')
    
    # --- Initialize components for Routine "stateMeasure" ---
    image_SAM = visual.ImageStim(
        win=win,
        name='image_SAM', units='height', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=1.0,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    t_sm_message = visual.TextStim(win=win, name='t_sm_message',
        text='',
        font='Arial',
        pos=(0, .35), draggable=False, height=0.07, wrapWidth=comp_wrap_width_header, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    t_sm_continue = visual.TextStim(win=win, name='t_sm_continue',
        text='Press the SPACEBAR to continue',
        font='Arial',
        pos=(0, -0.35), draggable=False, height=0.04, wrapWidth=comp_wrap_width_continue, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    key_resp_sm = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "goodbye" ---
    t_goodbye = visual.TextStim(win=win, name='t_goodbye',
        text='You have completed the trial. Thank you!',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.07, wrapWidth=comp_wrap_width_body, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    if eyetracker is not None:
        eyetracker.enableEventReporting()
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "experimentSetup" ---
    # create an object to store info about Routine experimentSetup
    experimentSetup = data.Routine(
        name='experimentSetup',
        components=[],
    )
    experimentSetup.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for experimentSetup
    experimentSetup.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    experimentSetup.tStart = globalClock.getTime(format='float')
    experimentSetup.status = STARTED
    experimentSetup.maxDuration = None
    # keep track of which components have finished
    experimentSetupComponents = experimentSetup.components
    for thisComponent in experimentSetup.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "experimentSetup" ---
    thisExp.currentRoutine = experimentSetup
    experimentSetup.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=experimentSetup,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            experimentSetup.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if experimentSetup.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in experimentSetup.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "experimentSetup" ---
    for thisComponent in experimentSetup.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for experimentSetup
    experimentSetup.tStop = globalClock.getTime(format='float')
    experimentSetup.tStopRefresh = tThisFlipGlobal
    thisExp.nextEntry()
    # the Routine "experimentSetup" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    intro_prompts = data.TrialHandler2(
        name='intro_prompts',
        nReps=1.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('loopEmotionRegulationIntro.csv'), 
        seed=None, 
        isTrials=False, 
    )
    thisExp.addLoop(intro_prompts)  # add the loop to the experiment
    thisIntro_prompt = intro_prompts.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisIntro_prompt.rgb)
    if thisIntro_prompt != None:
        for paramName in thisIntro_prompt:
            globals()[paramName] = thisIntro_prompt[paramName]
    
    for thisIntro_prompt in intro_prompts:
        intro_prompts.status = STARTED
        if hasattr(thisIntro_prompt, 'status'):
            thisIntro_prompt.status = STARTED
        currentLoop = intro_prompts
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # abbreviate parameter names if possible (e.g. rgb = thisIntro_prompt.rgb)
        if thisIntro_prompt != None:
            for paramName in thisIntro_prompt:
                globals()[paramName] = thisIntro_prompt[paramName]
        
        # --- Prepare to start Routine "welcome" ---
        # create an object to store info about Routine welcome
        welcome = data.Routine(
            name='welcome',
            components=[t_welcome_body, t_welcome_continue, key_resp_welcome],
        )
        welcome.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        t_welcome_body.setText(Message)
        # create starting attributes for key_resp_welcome
        key_resp_welcome.keys = []
        key_resp_welcome.rt = []
        _key_resp_welcome_allKeys = []
        # store start times for welcome
        welcome.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        welcome.tStart = globalClock.getTime(format='float')
        welcome.status = STARTED
        thisExp.addData('welcome.started', welcome.tStart)
        welcome.maxDuration = None
        # keep track of which components have finished
        welcomeComponents = welcome.components
        for thisComponent in welcome.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "welcome" ---
        thisExp.currentRoutine = welcome
        welcome.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisIntro_prompt, 'status') and thisIntro_prompt.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *t_welcome_body* updates
            
            # if t_welcome_body is starting this frame...
            if t_welcome_body.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                t_welcome_body.frameNStart = frameN  # exact frame index
                t_welcome_body.tStart = t  # local t and not account for scr refresh
                t_welcome_body.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(t_welcome_body, 'tStartRefresh')  # time at next scr refresh
                # update status
                t_welcome_body.status = STARTED
                t_welcome_body.setAutoDraw(True)
            
            # if t_welcome_body is active this frame...
            if t_welcome_body.status == STARTED:
                # update params
                pass
            
            # *t_welcome_continue* updates
            
            # if t_welcome_continue is starting this frame...
            if t_welcome_continue.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                t_welcome_continue.frameNStart = frameN  # exact frame index
                t_welcome_continue.tStart = t  # local t and not account for scr refresh
                t_welcome_continue.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(t_welcome_continue, 'tStartRefresh')  # time at next scr refresh
                # update status
                t_welcome_continue.status = STARTED
                t_welcome_continue.setAutoDraw(True)
            
            # if t_welcome_continue is active this frame...
            if t_welcome_continue.status == STARTED:
                # update params
                pass
            
            # *key_resp_welcome* updates
            waitOnFlip = False
            
            # if key_resp_welcome is starting this frame...
            if key_resp_welcome.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_welcome.frameNStart = frameN  # exact frame index
                key_resp_welcome.tStart = t  # local t and not account for scr refresh
                key_resp_welcome.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_welcome, 'tStartRefresh')  # time at next scr refresh
                # update status
                key_resp_welcome.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_welcome.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_welcome.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_welcome.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_welcome.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_welcome_allKeys.extend(theseKeys)
                if len(_key_resp_welcome_allKeys):
                    key_resp_welcome.keys = _key_resp_welcome_allKeys[-1].name  # just the last key pressed
                    key_resp_welcome.rt = _key_resp_welcome_allKeys[-1].rt
                    key_resp_welcome.duration = _key_resp_welcome_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=welcome,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                welcome.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if welcome.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in welcome.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "welcome" ---
        for thisComponent in welcome.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for welcome
        welcome.tStop = globalClock.getTime(format='float')
        welcome.tStopRefresh = tThisFlipGlobal
        thisExp.addData('welcome.stopped', welcome.tStop)
        # the Routine "welcome" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisIntro_prompt as finished
        if hasattr(thisIntro_prompt, 'status'):
            thisIntro_prompt.status = FINISHED
        # if awaiting a pause, pause now
        if intro_prompts.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            intro_prompts.status = STARTED
    # completed 1.0 repeats of 'intro_prompts'
    intro_prompts.status = FINISHED
    
    
    # set up handler to look after randomisation of conditions etc
    state_measure_pretrial = data.TrialHandler2(
        name='state_measure_pretrial',
        nReps=1.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions(
        '../../shared/loop-templates/loopStateMeasure.csv', 
        selection='0:7'
    )
    , 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(state_measure_pretrial)  # add the loop to the experiment
    thisState_measure_pretrial = state_measure_pretrial.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisState_measure_pretrial.rgb)
    if thisState_measure_pretrial != None:
        for paramName in thisState_measure_pretrial:
            globals()[paramName] = thisState_measure_pretrial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisState_measure_pretrial in state_measure_pretrial:
        state_measure_pretrial.status = STARTED
        if hasattr(thisState_measure_pretrial, 'status'):
            thisState_measure_pretrial.status = STARTED
        currentLoop = state_measure_pretrial
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisState_measure_pretrial.rgb)
        if thisState_measure_pretrial != None:
            for paramName in thisState_measure_pretrial:
                globals()[paramName] = thisState_measure_pretrial[paramName]
        
        # --- Prepare to start Routine "stateMeasure" ---
        # create an object to store info about Routine stateMeasure
        stateMeasure = data.Routine(
            name='stateMeasure',
            components=[image_SAM, t_sm_message, t_sm_continue, key_resp_sm],
        )
        stateMeasure.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_sm_helper
        allowContinue = False
        
        # Parse tick values from the condition file
        tick_values_parsed = parse_tick_values(tick_values)
        
        try:
            logging.debug(f"Starting state-measure in loop: {currentLoop.name}")
            logging.debug(f"State-measure current row: {rating_category}, {rating_type}")
            logging.debug(f"State-measure tick values: {tick_values_parsed}")
        except:
            logging.error("Error printing current state measure row.")
        
        # Compute geometry based on rating type
        sm_geometry = sm_compute_tick_positions(tick_values_parsed, rating_type)
        
        sm_comp_slider_size = sm_geometry["slider_size"]
        sm_comp_slider_pos  = sm_geometry["slider_pos"]
        sm_comp_label_min_pos = sm_geometry["label_min_pos"]
        sm_comp_label_max_pos = sm_geometry["label_max_pos"]
        
        t_sm_label_min = visual.TextStim(
            win=win,
            text=rating_min_label,
            pos=(sm_comp_label_min_pos[0], sm_comp_label_min_pos[1]),
            wrapWidth=aspect20,
            anchorVert='top',
            height=0.05,
            color='white',
            font='Noto Sans'
        )
        
        t_sm_label_max = visual.TextStim(
            win=win,
            text=rating_max_label,
            pos=(sm_comp_label_max_pos[0], sm_comp_label_max_pos[1]),
            wrapWidth=aspect20,
            anchorVert='top',
            height=0.05,
            color='white',
            font='Noto Sans'
        )
        
        #t_label_min.setPos(sm_comp_label_min_pos)
        #t_label_max.setPos(sm_comp_label_max_pos)
        
        # --- Create a fresh slider for this trial ---
        sm_slider = visual.Slider(
            win=win,
            ticks=tick_values_parsed,
            labels=tick_values_parsed,
            granularity=granularity,
            style=style.lower(),
            pos=sm_comp_slider_pos,
            size=sm_comp_slider_size,
            labelHeight=0.05,
            colorSpace='rgb',
            markerColor='Red',
            lineColor='White',
            labelColor='LightGray',
            font='Noto Sans'
        )
        
        # --- SAM image logic ---
        if rating_type == 'SAM':
            image_SAM.setImage(picture_path)
            image_SAM.setAutoDraw(True)
        else:
            image_SAM.setAutoDraw(False)
        
        image_SAM.setPos((0, sm_sam_comp_img_pos_y))
        image_SAM.setSize((sm_sam_comp_img_size_width, sm_sam_comp_img_size_height))
        image_SAM.setImage(picture_path)
        t_sm_message.setText(rating_message)
        # create starting attributes for key_resp_sm
        key_resp_sm.keys = []
        key_resp_sm.rt = []
        _key_resp_sm_allKeys = []
        # store start times for stateMeasure
        stateMeasure.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        stateMeasure.tStart = globalClock.getTime(format='float')
        stateMeasure.status = STARTED
        thisExp.addData('stateMeasure.started', stateMeasure.tStart)
        stateMeasure.maxDuration = None
        # keep track of which components have finished
        stateMeasureComponents = stateMeasure.components
        for thisComponent in stateMeasure.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "stateMeasure" ---
        thisExp.currentRoutine = stateMeasure
        stateMeasure.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisState_measure_pretrial, 'status') and thisState_measure_pretrial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from code_sm_helper
            # Draw state measure components
            sm_slider.draw()
            t_sm_label_min.draw()
            t_sm_label_max.draw()
            
            # If rated, allow spacebar to end the routine
            if sm_slider.getRating() is not None:
                allowContinue = True
            
            
            # *image_SAM* updates
            
            # if image_SAM is starting this frame...
            if image_SAM.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                image_SAM.frameNStart = frameN  # exact frame index
                image_SAM.tStart = t  # local t and not account for scr refresh
                image_SAM.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_SAM, 'tStartRefresh')  # time at next scr refresh
                # update status
                image_SAM.status = STARTED
                image_SAM.setAutoDraw(True)
            
            # if image_SAM is active this frame...
            if image_SAM.status == STARTED:
                # update params
                pass
            
            # *t_sm_message* updates
            
            # if t_sm_message is starting this frame...
            if t_sm_message.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                t_sm_message.frameNStart = frameN  # exact frame index
                t_sm_message.tStart = t  # local t and not account for scr refresh
                t_sm_message.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(t_sm_message, 'tStartRefresh')  # time at next scr refresh
                # update status
                t_sm_message.status = STARTED
                t_sm_message.setAutoDraw(True)
            
            # if t_sm_message is active this frame...
            if t_sm_message.status == STARTED:
                # update params
                pass
            
            # *t_sm_continue* updates
            
            # if t_sm_continue is starting this frame...
            if t_sm_continue.status == NOT_STARTED and allowContinue == True:
                # keep track of start time/frame for later
                t_sm_continue.frameNStart = frameN  # exact frame index
                t_sm_continue.tStart = t  # local t and not account for scr refresh
                t_sm_continue.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(t_sm_continue, 'tStartRefresh')  # time at next scr refresh
                # update status
                t_sm_continue.status = STARTED
                t_sm_continue.setAutoDraw(True)
            
            # if t_sm_continue is active this frame...
            if t_sm_continue.status == STARTED:
                # update params
                pass
            
            # *key_resp_sm* updates
            waitOnFlip = False
            
            # if key_resp_sm is starting this frame...
            if key_resp_sm.status == NOT_STARTED and allowContinue == True:
                # keep track of start time/frame for later
                key_resp_sm.frameNStart = frameN  # exact frame index
                key_resp_sm.tStart = t  # local t and not account for scr refresh
                key_resp_sm.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_sm, 'tStartRefresh')  # time at next scr refresh
                # update status
                key_resp_sm.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_sm.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_sm.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_sm.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_sm.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_sm_allKeys.extend(theseKeys)
                if len(_key_resp_sm_allKeys):
                    key_resp_sm.keys = _key_resp_sm_allKeys[-1].name  # just the last key pressed
                    key_resp_sm.rt = _key_resp_sm_allKeys[-1].rt
                    key_resp_sm.duration = _key_resp_sm_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=stateMeasure,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                stateMeasure.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if stateMeasure.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in stateMeasure.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "stateMeasure" ---
        for thisComponent in stateMeasure.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for stateMeasure
        stateMeasure.tStop = globalClock.getTime(format='float')
        stateMeasure.tStopRefresh = tThisFlipGlobal
        thisExp.addData('stateMeasure.stopped', stateMeasure.tStop)
        # Run 'End Routine' code from code_sm_helper
        currentLoop.addData('rating', sm_slider.getRating())
        currentLoop.addData('rating_rt', sm_slider.getRT())
        
        try:
            logging.data(f"State-measure rating: {rating_category}, {sm_slider.getRating()}")
        except:
            logging.error("Error printing state-measure rating")
        # the Routine "stateMeasure" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisState_measure_pretrial as finished
        if hasattr(thisState_measure_pretrial, 'status'):
            thisState_measure_pretrial.status = FINISHED
        # if awaiting a pause, pause now
        if state_measure_pretrial.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            state_measure_pretrial.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'state_measure_pretrial'
    state_measure_pretrial.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    # get names of stimulus parameters
    if state_measure_pretrial.trialList in ([], [None], None):
        params = []
    else:
        params = state_measure_pretrial.trialList[0].keys()
    # save data for this loop
    state_measure_pretrial.saveAsExcel(filename + '.xlsx', sheetName='state_measure_pretrial',
        stimOut=params,
        dataOut=['n','all_mean','all_std', 'all_raw'])
    
    # --- Prepare to start Routine "instructionNeutral" ---
    # create an object to store info about Routine instructionNeutral
    instructionNeutral = data.Routine(
        name='instructionNeutral',
        components=[t_neutral_instruction, t_neutral_continue, key_resp_neutral],
    )
    instructionNeutral.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    t_neutral_instruction.setText('First, you will just view neutral pictures. \n\nLook at the picture naturally without trying to change your thoughts or feelings. Avoid thinking about unrelated topics.')
    # create starting attributes for key_resp_neutral
    key_resp_neutral.keys = []
    key_resp_neutral.rt = []
    _key_resp_neutral_allKeys = []
    # store start times for instructionNeutral
    instructionNeutral.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instructionNeutral.tStart = globalClock.getTime(format='float')
    instructionNeutral.status = STARTED
    thisExp.addData('instructionNeutral.started', instructionNeutral.tStart)
    instructionNeutral.maxDuration = None
    # keep track of which components have finished
    instructionNeutralComponents = instructionNeutral.components
    for thisComponent in instructionNeutral.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructionNeutral" ---
    thisExp.currentRoutine = instructionNeutral
    instructionNeutral.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *t_neutral_instruction* updates
        
        # if t_neutral_instruction is starting this frame...
        if t_neutral_instruction.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            t_neutral_instruction.frameNStart = frameN  # exact frame index
            t_neutral_instruction.tStart = t  # local t and not account for scr refresh
            t_neutral_instruction.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(t_neutral_instruction, 'tStartRefresh')  # time at next scr refresh
            # update status
            t_neutral_instruction.status = STARTED
            t_neutral_instruction.setAutoDraw(True)
        
        # if t_neutral_instruction is active this frame...
        if t_neutral_instruction.status == STARTED:
            # update params
            pass
        
        # *t_neutral_continue* updates
        
        # if t_neutral_continue is starting this frame...
        if t_neutral_continue.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            t_neutral_continue.frameNStart = frameN  # exact frame index
            t_neutral_continue.tStart = t  # local t and not account for scr refresh
            t_neutral_continue.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(t_neutral_continue, 'tStartRefresh')  # time at next scr refresh
            # update status
            t_neutral_continue.status = STARTED
            t_neutral_continue.setAutoDraw(True)
        
        # if t_neutral_continue is active this frame...
        if t_neutral_continue.status == STARTED:
            # update params
            pass
        
        # *key_resp_neutral* updates
        waitOnFlip = False
        
        # if key_resp_neutral is starting this frame...
        if key_resp_neutral.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_neutral.frameNStart = frameN  # exact frame index
            key_resp_neutral.tStart = t  # local t and not account for scr refresh
            key_resp_neutral.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_neutral, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_resp_neutral.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_neutral.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_neutral.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_neutral.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_neutral.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_neutral_allKeys.extend(theseKeys)
            if len(_key_resp_neutral_allKeys):
                key_resp_neutral.keys = _key_resp_neutral_allKeys[-1].name  # just the last key pressed
                key_resp_neutral.rt = _key_resp_neutral_allKeys[-1].rt
                key_resp_neutral.duration = _key_resp_neutral_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=instructionNeutral,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            instructionNeutral.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if instructionNeutral.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in instructionNeutral.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructionNeutral" ---
    for thisComponent in instructionNeutral.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instructionNeutral
    instructionNeutral.tStop = globalClock.getTime(format='float')
    instructionNeutral.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instructionNeutral.stopped', instructionNeutral.tStop)
    thisExp.nextEntry()
    # the Routine "instructionNeutral" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    iasp_neutral_trials = data.TrialHandler2(
        name='iasp_neutral_trials',
        nReps=1.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('emotion_regulation_neutral_trials.csv'), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(iasp_neutral_trials)  # add the loop to the experiment
    thisIasp_neutral_trial = iasp_neutral_trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisIasp_neutral_trial.rgb)
    if thisIasp_neutral_trial != None:
        for paramName in thisIasp_neutral_trial:
            globals()[paramName] = thisIasp_neutral_trial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisIasp_neutral_trial in iasp_neutral_trials:
        iasp_neutral_trials.status = STARTED
        if hasattr(thisIasp_neutral_trial, 'status'):
            thisIasp_neutral_trial.status = STARTED
        currentLoop = iasp_neutral_trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisIasp_neutral_trial.rgb)
        if thisIasp_neutral_trial != None:
            for paramName in thisIasp_neutral_trial:
                globals()[paramName] = thisIasp_neutral_trial[paramName]
        
        # --- Prepare to start Routine "emotionRegulationCue" ---
        # create an object to store info about Routine emotionRegulationCue
        emotionRegulationCue = data.Routine(
            name='emotionRegulationCue',
            components=[cross_fixation, t_emotion_regulation_cue, t_blank_delayer],
        )
        emotionRegulationCue.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_delay_calculator
        # Gives a random float between 0.5 and 1.5 seconds (i.e., 500–1500 ms)
        # Used for t_blank_delayer
        delay_time = random.uniform(0.5, 1.5)
        currentLoop.addData('delay_time', delay_time)
        t_emotion_regulation_cue.setText(technique)
        # store start times for emotionRegulationCue
        emotionRegulationCue.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        emotionRegulationCue.tStart = globalClock.getTime(format='float')
        emotionRegulationCue.status = STARTED
        thisExp.addData('emotionRegulationCue.started', emotionRegulationCue.tStart)
        emotionRegulationCue.maxDuration = None
        win.color = [-1.0000, -1.0000, -1.0000]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        # keep track of which components have finished
        emotionRegulationCueComponents = emotionRegulationCue.components
        for thisComponent in emotionRegulationCue.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "emotionRegulationCue" ---
        thisExp.currentRoutine = emotionRegulationCue
        emotionRegulationCue.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisIasp_neutral_trial, 'status') and thisIasp_neutral_trial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *cross_fixation* updates
            
            # if cross_fixation is starting this frame...
            if cross_fixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                cross_fixation.frameNStart = frameN  # exact frame index
                cross_fixation.tStart = t  # local t and not account for scr refresh
                cross_fixation.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cross_fixation, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cross_fixation.started')
                # update status
                cross_fixation.status = STARTED
                cross_fixation.setAutoDraw(True)
            
            # if cross_fixation is active this frame...
            if cross_fixation.status == STARTED:
                # update params
                pass
            
            # if cross_fixation is stopping this frame...
            if cross_fixation.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > cross_fixation.tStartRefresh + 0.3-frameTolerance:
                    # keep track of stop time/frame for later
                    cross_fixation.tStop = t  # not accounting for scr refresh
                    cross_fixation.tStopRefresh = tThisFlipGlobal  # on global time
                    cross_fixation.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cross_fixation.stopped')
                    # update status
                    cross_fixation.status = FINISHED
                    cross_fixation.setAutoDraw(False)
            
            # *t_emotion_regulation_cue* updates
            
            # if t_emotion_regulation_cue is starting this frame...
            if t_emotion_regulation_cue.status == NOT_STARTED and tThisFlip >= 0.3-frameTolerance:
                # keep track of start time/frame for later
                t_emotion_regulation_cue.frameNStart = frameN  # exact frame index
                t_emotion_regulation_cue.tStart = t  # local t and not account for scr refresh
                t_emotion_regulation_cue.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(t_emotion_regulation_cue, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 't_emotion_regulation_cue.started')
                # update status
                t_emotion_regulation_cue.status = STARTED
                t_emotion_regulation_cue.setAutoDraw(True)
            
            # if t_emotion_regulation_cue is active this frame...
            if t_emotion_regulation_cue.status == STARTED:
                # update params
                pass
            
            # if t_emotion_regulation_cue is stopping this frame...
            if t_emotion_regulation_cue.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > t_emotion_regulation_cue.tStartRefresh + 1-frameTolerance:
                    # keep track of stop time/frame for later
                    t_emotion_regulation_cue.tStop = t  # not accounting for scr refresh
                    t_emotion_regulation_cue.tStopRefresh = tThisFlipGlobal  # on global time
                    t_emotion_regulation_cue.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 't_emotion_regulation_cue.stopped')
                    # update status
                    t_emotion_regulation_cue.status = FINISHED
                    t_emotion_regulation_cue.setAutoDraw(False)
            
            # *t_blank_delayer* updates
            
            # if t_blank_delayer is starting this frame...
            if t_blank_delayer.status == NOT_STARTED and tThisFlip >= 1.3-frameTolerance:
                # keep track of start time/frame for later
                t_blank_delayer.frameNStart = frameN  # exact frame index
                t_blank_delayer.tStart = t  # local t and not account for scr refresh
                t_blank_delayer.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(t_blank_delayer, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 't_blank_delayer.started')
                # update status
                t_blank_delayer.status = STARTED
                t_blank_delayer.setAutoDraw(True)
            
            # if t_blank_delayer is active this frame...
            if t_blank_delayer.status == STARTED:
                # update params
                pass
            
            # if t_blank_delayer is stopping this frame...
            if t_blank_delayer.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > t_blank_delayer.tStartRefresh + delay_time-frameTolerance:
                    # keep track of stop time/frame for later
                    t_blank_delayer.tStop = t  # not accounting for scr refresh
                    t_blank_delayer.tStopRefresh = tThisFlipGlobal  # on global time
                    t_blank_delayer.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 't_blank_delayer.stopped')
                    # update status
                    t_blank_delayer.status = FINISHED
                    t_blank_delayer.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=emotionRegulationCue,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                emotionRegulationCue.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if emotionRegulationCue.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in emotionRegulationCue.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "emotionRegulationCue" ---
        for thisComponent in emotionRegulationCue.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for emotionRegulationCue
        emotionRegulationCue.tStop = globalClock.getTime(format='float')
        emotionRegulationCue.tStopRefresh = tThisFlipGlobal
        thisExp.addData('emotionRegulationCue.stopped', emotionRegulationCue.tStop)
        setupWindow(expInfo=expInfo, win=win)
        # the Routine "emotionRegulationCue" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "iaspView" ---
        # create an object to store info about Routine iaspView
        iaspView = data.Routine(
            name='iaspView',
            components=[image_iasp, p_port_iasp],
        )
        iaspView.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        image_iasp.setSize([iasp_comp_img_size_width, iasp_comp_img_size_height])
        image_iasp.setImage(photo_filename)
        # store start times for iaspView
        iaspView.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        iaspView.tStart = globalClock.getTime(format='float')
        iaspView.status = STARTED
        thisExp.addData('iaspView.started', iaspView.tStart)
        iaspView.maxDuration = 5
        win.color = [-1.0000, -1.0000, -1.0000]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        # keep track of which components have finished
        iaspViewComponents = iaspView.components
        for thisComponent in iaspView.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "iaspView" ---
        thisExp.currentRoutine = iaspView
        iaspView.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisIasp_neutral_trial, 'status') and thisIasp_neutral_trial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # is it time to end the Routine? (based on local clock)
            if tThisFlip > iaspView.maxDuration-frameTolerance:
                iaspView.maxDurationReached = True
                continueRoutine = False
            
            # *image_iasp* updates
            
            # if image_iasp is starting this frame...
            if image_iasp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                image_iasp.frameNStart = frameN  # exact frame index
                image_iasp.tStart = t  # local t and not account for scr refresh
                image_iasp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_iasp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_iasp.started')
                # update status
                image_iasp.status = STARTED
                image_iasp.setAutoDraw(True)
            
            # if image_iasp is active this frame...
            if image_iasp.status == STARTED:
                # update params
                pass
            
            # if image_iasp is stopping this frame...
            if image_iasp.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > image_iasp.tStartRefresh + 4-frameTolerance:
                    # keep track of stop time/frame for later
                    image_iasp.tStop = t  # not accounting for scr refresh
                    image_iasp.tStopRefresh = tThisFlipGlobal  # on global time
                    image_iasp.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_iasp.stopped')
                    # update status
                    image_iasp.status = FINISHED
                    image_iasp.setAutoDraw(False)
            # *p_port_iasp* updates
            
            # if p_port_iasp is starting this frame...
            if p_port_iasp.status == NOT_STARTED and image_IASP.status == STARTED:
                # keep track of start time/frame for later
                p_port_iasp.frameNStart = frameN  # exact frame index
                p_port_iasp.tStart = t  # local t and not account for scr refresh
                p_port_iasp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_port_iasp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_port_iasp.started')
                # update status
                p_port_iasp.status = STARTED
                p_port_iasp.status = STARTED
                win.callOnFlip(p_port_iasp.setData, int(1))
            
            # if p_port_iasp is stopping this frame...
            if p_port_iasp.status == STARTED:
                if bool(image_iasp.status == STOPPED):
                    # keep track of stop time/frame for later
                    p_port_iasp.tStop = t  # not accounting for scr refresh
                    p_port_iasp.tStopRefresh = tThisFlipGlobal  # on global time
                    p_port_iasp.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'p_port_iasp.stopped')
                    # update status
                    p_port_iasp.status = FINISHED
                    win.callOnFlip(p_port_iasp.setData, int(0))
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=iaspView,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                iaspView.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if iaspView.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in iaspView.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "iaspView" ---
        for thisComponent in iaspView.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for iaspView
        iaspView.tStop = globalClock.getTime(format='float')
        iaspView.tStopRefresh = tThisFlipGlobal
        thisExp.addData('iaspView.stopped', iaspView.tStop)
        setupWindow(expInfo=expInfo, win=win)
        if p_port_iasp.status == STARTED:
            win.callOnFlip(p_port_iasp.setData, int(0))
        # the Routine "iaspView" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisIasp_neutral_trial as finished
        if hasattr(thisIasp_neutral_trial, 'status'):
            thisIasp_neutral_trial.status = FINISHED
        # if awaiting a pause, pause now
        if iasp_neutral_trials.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            iasp_neutral_trials.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'iasp_neutral_trials'
    iasp_neutral_trials.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    # get names of stimulus parameters
    if iasp_neutral_trials.trialList in ([], [None], None):
        params = []
    else:
        params = iasp_neutral_trials.trialList[0].keys()
    # save data for this loop
    iasp_neutral_trials.saveAsExcel(filename + '.xlsx', sheetName='iasp_neutral_trials',
        stimOut=params,
        dataOut=['n','all_mean','all_std', 'all_raw'])
    
    # set up handler to look after randomisation of conditions etc
    state_measure_neutral = data.TrialHandler2(
        name='state_measure_neutral',
        nReps=1.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions(
        '../../shared/loop-templates/loopStateMeasure.csv', 
        selection='[2, 3, 4, 5, 6]'
    )
    , 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(state_measure_neutral)  # add the loop to the experiment
    thisState_measure_neutral = state_measure_neutral.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisState_measure_neutral.rgb)
    if thisState_measure_neutral != None:
        for paramName in thisState_measure_neutral:
            globals()[paramName] = thisState_measure_neutral[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisState_measure_neutral in state_measure_neutral:
        state_measure_neutral.status = STARTED
        if hasattr(thisState_measure_neutral, 'status'):
            thisState_measure_neutral.status = STARTED
        currentLoop = state_measure_neutral
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisState_measure_neutral.rgb)
        if thisState_measure_neutral != None:
            for paramName in thisState_measure_neutral:
                globals()[paramName] = thisState_measure_neutral[paramName]
        
        # --- Prepare to start Routine "stateMeasure" ---
        # create an object to store info about Routine stateMeasure
        stateMeasure = data.Routine(
            name='stateMeasure',
            components=[image_SAM, t_sm_message, t_sm_continue, key_resp_sm],
        )
        stateMeasure.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_sm_helper
        allowContinue = False
        
        # Parse tick values from the condition file
        tick_values_parsed = parse_tick_values(tick_values)
        
        try:
            logging.debug(f"Starting state-measure in loop: {currentLoop.name}")
            logging.debug(f"State-measure current row: {rating_category}, {rating_type}")
            logging.debug(f"State-measure tick values: {tick_values_parsed}")
        except:
            logging.error("Error printing current state measure row.")
        
        # Compute geometry based on rating type
        sm_geometry = sm_compute_tick_positions(tick_values_parsed, rating_type)
        
        sm_comp_slider_size = sm_geometry["slider_size"]
        sm_comp_slider_pos  = sm_geometry["slider_pos"]
        sm_comp_label_min_pos = sm_geometry["label_min_pos"]
        sm_comp_label_max_pos = sm_geometry["label_max_pos"]
        
        t_sm_label_min = visual.TextStim(
            win=win,
            text=rating_min_label,
            pos=(sm_comp_label_min_pos[0], sm_comp_label_min_pos[1]),
            wrapWidth=aspect20,
            anchorVert='top',
            height=0.05,
            color='white',
            font='Noto Sans'
        )
        
        t_sm_label_max = visual.TextStim(
            win=win,
            text=rating_max_label,
            pos=(sm_comp_label_max_pos[0], sm_comp_label_max_pos[1]),
            wrapWidth=aspect20,
            anchorVert='top',
            height=0.05,
            color='white',
            font='Noto Sans'
        )
        
        #t_label_min.setPos(sm_comp_label_min_pos)
        #t_label_max.setPos(sm_comp_label_max_pos)
        
        # --- Create a fresh slider for this trial ---
        sm_slider = visual.Slider(
            win=win,
            ticks=tick_values_parsed,
            labels=tick_values_parsed,
            granularity=granularity,
            style=style.lower(),
            pos=sm_comp_slider_pos,
            size=sm_comp_slider_size,
            labelHeight=0.05,
            colorSpace='rgb',
            markerColor='Red',
            lineColor='White',
            labelColor='LightGray',
            font='Noto Sans'
        )
        
        # --- SAM image logic ---
        if rating_type == 'SAM':
            image_SAM.setImage(picture_path)
            image_SAM.setAutoDraw(True)
        else:
            image_SAM.setAutoDraw(False)
        
        image_SAM.setPos((0, sm_sam_comp_img_pos_y))
        image_SAM.setSize((sm_sam_comp_img_size_width, sm_sam_comp_img_size_height))
        image_SAM.setImage(picture_path)
        t_sm_message.setText(rating_message)
        # create starting attributes for key_resp_sm
        key_resp_sm.keys = []
        key_resp_sm.rt = []
        _key_resp_sm_allKeys = []
        # store start times for stateMeasure
        stateMeasure.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        stateMeasure.tStart = globalClock.getTime(format='float')
        stateMeasure.status = STARTED
        thisExp.addData('stateMeasure.started', stateMeasure.tStart)
        stateMeasure.maxDuration = None
        # keep track of which components have finished
        stateMeasureComponents = stateMeasure.components
        for thisComponent in stateMeasure.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "stateMeasure" ---
        thisExp.currentRoutine = stateMeasure
        stateMeasure.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisState_measure_neutral, 'status') and thisState_measure_neutral.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from code_sm_helper
            # Draw state measure components
            sm_slider.draw()
            t_sm_label_min.draw()
            t_sm_label_max.draw()
            
            # If rated, allow spacebar to end the routine
            if sm_slider.getRating() is not None:
                allowContinue = True
            
            
            # *image_SAM* updates
            
            # if image_SAM is starting this frame...
            if image_SAM.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                image_SAM.frameNStart = frameN  # exact frame index
                image_SAM.tStart = t  # local t and not account for scr refresh
                image_SAM.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_SAM, 'tStartRefresh')  # time at next scr refresh
                # update status
                image_SAM.status = STARTED
                image_SAM.setAutoDraw(True)
            
            # if image_SAM is active this frame...
            if image_SAM.status == STARTED:
                # update params
                pass
            
            # *t_sm_message* updates
            
            # if t_sm_message is starting this frame...
            if t_sm_message.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                t_sm_message.frameNStart = frameN  # exact frame index
                t_sm_message.tStart = t  # local t and not account for scr refresh
                t_sm_message.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(t_sm_message, 'tStartRefresh')  # time at next scr refresh
                # update status
                t_sm_message.status = STARTED
                t_sm_message.setAutoDraw(True)
            
            # if t_sm_message is active this frame...
            if t_sm_message.status == STARTED:
                # update params
                pass
            
            # *t_sm_continue* updates
            
            # if t_sm_continue is starting this frame...
            if t_sm_continue.status == NOT_STARTED and allowContinue == True:
                # keep track of start time/frame for later
                t_sm_continue.frameNStart = frameN  # exact frame index
                t_sm_continue.tStart = t  # local t and not account for scr refresh
                t_sm_continue.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(t_sm_continue, 'tStartRefresh')  # time at next scr refresh
                # update status
                t_sm_continue.status = STARTED
                t_sm_continue.setAutoDraw(True)
            
            # if t_sm_continue is active this frame...
            if t_sm_continue.status == STARTED:
                # update params
                pass
            
            # *key_resp_sm* updates
            waitOnFlip = False
            
            # if key_resp_sm is starting this frame...
            if key_resp_sm.status == NOT_STARTED and allowContinue == True:
                # keep track of start time/frame for later
                key_resp_sm.frameNStart = frameN  # exact frame index
                key_resp_sm.tStart = t  # local t and not account for scr refresh
                key_resp_sm.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_sm, 'tStartRefresh')  # time at next scr refresh
                # update status
                key_resp_sm.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_sm.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_sm.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_sm.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_sm.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_sm_allKeys.extend(theseKeys)
                if len(_key_resp_sm_allKeys):
                    key_resp_sm.keys = _key_resp_sm_allKeys[-1].name  # just the last key pressed
                    key_resp_sm.rt = _key_resp_sm_allKeys[-1].rt
                    key_resp_sm.duration = _key_resp_sm_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=stateMeasure,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                stateMeasure.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if stateMeasure.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in stateMeasure.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "stateMeasure" ---
        for thisComponent in stateMeasure.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for stateMeasure
        stateMeasure.tStop = globalClock.getTime(format='float')
        stateMeasure.tStopRefresh = tThisFlipGlobal
        thisExp.addData('stateMeasure.stopped', stateMeasure.tStop)
        # Run 'End Routine' code from code_sm_helper
        currentLoop.addData('rating', sm_slider.getRating())
        currentLoop.addData('rating_rt', sm_slider.getRT())
        
        try:
            logging.data(f"State-measure rating: {rating_category}, {sm_slider.getRating()}")
        except:
            logging.error("Error printing state-measure rating")
        # the Routine "stateMeasure" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisState_measure_neutral as finished
        if hasattr(thisState_measure_neutral, 'status'):
            thisState_measure_neutral.status = FINISHED
        # if awaiting a pause, pause now
        if state_measure_neutral.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            state_measure_neutral.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'state_measure_neutral'
    state_measure_neutral.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    # get names of stimulus parameters
    if state_measure_neutral.trialList in ([], [None], None):
        params = []
    else:
        params = state_measure_neutral.trialList[0].keys()
    # save data for this loop
    state_measure_neutral.saveAsExcel(filename + '.xlsx', sheetName='state_measure_neutral',
        stimOut=params,
        dataOut=['n','all_mean','all_std', 'all_raw'])
    
    # --- Prepare to start Routine "instructionPractice" ---
    # create an object to store info about Routine instructionPractice
    instructionPractice = data.Routine(
        name='instructionPractice',
        components=[t_practice_instruction, t_practice_continue, key_resp_practice],
    )
    instructionPractice.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    t_practice_instruction.setText('Next, you will see examples of unpleasant pictures. \n\nLook at the picture naturally without trying to change your thoughts or feelings. Avoid thinking about unrelated topics.\n')
    # create starting attributes for key_resp_practice
    key_resp_practice.keys = []
    key_resp_practice.rt = []
    _key_resp_practice_allKeys = []
    # store start times for instructionPractice
    instructionPractice.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instructionPractice.tStart = globalClock.getTime(format='float')
    instructionPractice.status = STARTED
    thisExp.addData('instructionPractice.started', instructionPractice.tStart)
    instructionPractice.maxDuration = None
    # keep track of which components have finished
    instructionPracticeComponents = instructionPractice.components
    for thisComponent in instructionPractice.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructionPractice" ---
    thisExp.currentRoutine = instructionPractice
    instructionPractice.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *t_practice_instruction* updates
        
        # if t_practice_instruction is starting this frame...
        if t_practice_instruction.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            t_practice_instruction.frameNStart = frameN  # exact frame index
            t_practice_instruction.tStart = t  # local t and not account for scr refresh
            t_practice_instruction.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(t_practice_instruction, 'tStartRefresh')  # time at next scr refresh
            # update status
            t_practice_instruction.status = STARTED
            t_practice_instruction.setAutoDraw(True)
        
        # if t_practice_instruction is active this frame...
        if t_practice_instruction.status == STARTED:
            # update params
            pass
        
        # *t_practice_continue* updates
        
        # if t_practice_continue is starting this frame...
        if t_practice_continue.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            t_practice_continue.frameNStart = frameN  # exact frame index
            t_practice_continue.tStart = t  # local t and not account for scr refresh
            t_practice_continue.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(t_practice_continue, 'tStartRefresh')  # time at next scr refresh
            # update status
            t_practice_continue.status = STARTED
            t_practice_continue.setAutoDraw(True)
        
        # if t_practice_continue is active this frame...
        if t_practice_continue.status == STARTED:
            # update params
            pass
        
        # *key_resp_practice* updates
        waitOnFlip = False
        
        # if key_resp_practice is starting this frame...
        if key_resp_practice.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_practice.frameNStart = frameN  # exact frame index
            key_resp_practice.tStart = t  # local t and not account for scr refresh
            key_resp_practice.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_practice, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_resp_practice.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_practice.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_practice.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_practice.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_practice.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_practice_allKeys.extend(theseKeys)
            if len(_key_resp_practice_allKeys):
                key_resp_practice.keys = _key_resp_practice_allKeys[-1].name  # just the last key pressed
                key_resp_practice.rt = _key_resp_practice_allKeys[-1].rt
                key_resp_practice.duration = _key_resp_practice_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=instructionPractice,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            instructionPractice.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if instructionPractice.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in instructionPractice.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructionPractice" ---
    for thisComponent in instructionPractice.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instructionPractice
    instructionPractice.tStop = globalClock.getTime(format='float')
    instructionPractice.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instructionPractice.stopped', instructionPractice.tStop)
    thisExp.nextEntry()
    # the Routine "instructionPractice" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    iasp_practice_trials = data.TrialHandler2(
        name='iasp_practice_trials',
        nReps=1.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('emotion_regulation_practice_trials.csv'), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(iasp_practice_trials)  # add the loop to the experiment
    thisIasp_practice_trial = iasp_practice_trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisIasp_practice_trial.rgb)
    if thisIasp_practice_trial != None:
        for paramName in thisIasp_practice_trial:
            globals()[paramName] = thisIasp_practice_trial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisIasp_practice_trial in iasp_practice_trials:
        iasp_practice_trials.status = STARTED
        if hasattr(thisIasp_practice_trial, 'status'):
            thisIasp_practice_trial.status = STARTED
        currentLoop = iasp_practice_trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisIasp_practice_trial.rgb)
        if thisIasp_practice_trial != None:
            for paramName in thisIasp_practice_trial:
                globals()[paramName] = thisIasp_practice_trial[paramName]
        
        # --- Prepare to start Routine "emotionRegulationCue" ---
        # create an object to store info about Routine emotionRegulationCue
        emotionRegulationCue = data.Routine(
            name='emotionRegulationCue',
            components=[cross_fixation, t_emotion_regulation_cue, t_blank_delayer],
        )
        emotionRegulationCue.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_delay_calculator
        # Gives a random float between 0.5 and 1.5 seconds (i.e., 500–1500 ms)
        # Used for t_blank_delayer
        delay_time = random.uniform(0.5, 1.5)
        currentLoop.addData('delay_time', delay_time)
        t_emotion_regulation_cue.setText(technique)
        # store start times for emotionRegulationCue
        emotionRegulationCue.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        emotionRegulationCue.tStart = globalClock.getTime(format='float')
        emotionRegulationCue.status = STARTED
        thisExp.addData('emotionRegulationCue.started', emotionRegulationCue.tStart)
        emotionRegulationCue.maxDuration = None
        win.color = [-1.0000, -1.0000, -1.0000]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        # keep track of which components have finished
        emotionRegulationCueComponents = emotionRegulationCue.components
        for thisComponent in emotionRegulationCue.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "emotionRegulationCue" ---
        thisExp.currentRoutine = emotionRegulationCue
        emotionRegulationCue.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisIasp_practice_trial, 'status') and thisIasp_practice_trial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *cross_fixation* updates
            
            # if cross_fixation is starting this frame...
            if cross_fixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                cross_fixation.frameNStart = frameN  # exact frame index
                cross_fixation.tStart = t  # local t and not account for scr refresh
                cross_fixation.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cross_fixation, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cross_fixation.started')
                # update status
                cross_fixation.status = STARTED
                cross_fixation.setAutoDraw(True)
            
            # if cross_fixation is active this frame...
            if cross_fixation.status == STARTED:
                # update params
                pass
            
            # if cross_fixation is stopping this frame...
            if cross_fixation.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > cross_fixation.tStartRefresh + 0.3-frameTolerance:
                    # keep track of stop time/frame for later
                    cross_fixation.tStop = t  # not accounting for scr refresh
                    cross_fixation.tStopRefresh = tThisFlipGlobal  # on global time
                    cross_fixation.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cross_fixation.stopped')
                    # update status
                    cross_fixation.status = FINISHED
                    cross_fixation.setAutoDraw(False)
            
            # *t_emotion_regulation_cue* updates
            
            # if t_emotion_regulation_cue is starting this frame...
            if t_emotion_regulation_cue.status == NOT_STARTED and tThisFlip >= 0.3-frameTolerance:
                # keep track of start time/frame for later
                t_emotion_regulation_cue.frameNStart = frameN  # exact frame index
                t_emotion_regulation_cue.tStart = t  # local t and not account for scr refresh
                t_emotion_regulation_cue.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(t_emotion_regulation_cue, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 't_emotion_regulation_cue.started')
                # update status
                t_emotion_regulation_cue.status = STARTED
                t_emotion_regulation_cue.setAutoDraw(True)
            
            # if t_emotion_regulation_cue is active this frame...
            if t_emotion_regulation_cue.status == STARTED:
                # update params
                pass
            
            # if t_emotion_regulation_cue is stopping this frame...
            if t_emotion_regulation_cue.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > t_emotion_regulation_cue.tStartRefresh + 1-frameTolerance:
                    # keep track of stop time/frame for later
                    t_emotion_regulation_cue.tStop = t  # not accounting for scr refresh
                    t_emotion_regulation_cue.tStopRefresh = tThisFlipGlobal  # on global time
                    t_emotion_regulation_cue.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 't_emotion_regulation_cue.stopped')
                    # update status
                    t_emotion_regulation_cue.status = FINISHED
                    t_emotion_regulation_cue.setAutoDraw(False)
            
            # *t_blank_delayer* updates
            
            # if t_blank_delayer is starting this frame...
            if t_blank_delayer.status == NOT_STARTED and tThisFlip >= 1.3-frameTolerance:
                # keep track of start time/frame for later
                t_blank_delayer.frameNStart = frameN  # exact frame index
                t_blank_delayer.tStart = t  # local t and not account for scr refresh
                t_blank_delayer.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(t_blank_delayer, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 't_blank_delayer.started')
                # update status
                t_blank_delayer.status = STARTED
                t_blank_delayer.setAutoDraw(True)
            
            # if t_blank_delayer is active this frame...
            if t_blank_delayer.status == STARTED:
                # update params
                pass
            
            # if t_blank_delayer is stopping this frame...
            if t_blank_delayer.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > t_blank_delayer.tStartRefresh + delay_time-frameTolerance:
                    # keep track of stop time/frame for later
                    t_blank_delayer.tStop = t  # not accounting for scr refresh
                    t_blank_delayer.tStopRefresh = tThisFlipGlobal  # on global time
                    t_blank_delayer.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 't_blank_delayer.stopped')
                    # update status
                    t_blank_delayer.status = FINISHED
                    t_blank_delayer.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=emotionRegulationCue,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                emotionRegulationCue.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if emotionRegulationCue.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in emotionRegulationCue.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "emotionRegulationCue" ---
        for thisComponent in emotionRegulationCue.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for emotionRegulationCue
        emotionRegulationCue.tStop = globalClock.getTime(format='float')
        emotionRegulationCue.tStopRefresh = tThisFlipGlobal
        thisExp.addData('emotionRegulationCue.stopped', emotionRegulationCue.tStop)
        setupWindow(expInfo=expInfo, win=win)
        # the Routine "emotionRegulationCue" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "iaspView" ---
        # create an object to store info about Routine iaspView
        iaspView = data.Routine(
            name='iaspView',
            components=[image_iasp, p_port_iasp],
        )
        iaspView.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        image_iasp.setSize([iasp_comp_img_size_width, iasp_comp_img_size_height])
        image_iasp.setImage(photo_filename)
        # store start times for iaspView
        iaspView.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        iaspView.tStart = globalClock.getTime(format='float')
        iaspView.status = STARTED
        thisExp.addData('iaspView.started', iaspView.tStart)
        iaspView.maxDuration = 5
        win.color = [-1.0000, -1.0000, -1.0000]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        # keep track of which components have finished
        iaspViewComponents = iaspView.components
        for thisComponent in iaspView.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "iaspView" ---
        thisExp.currentRoutine = iaspView
        iaspView.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisIasp_practice_trial, 'status') and thisIasp_practice_trial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # is it time to end the Routine? (based on local clock)
            if tThisFlip > iaspView.maxDuration-frameTolerance:
                iaspView.maxDurationReached = True
                continueRoutine = False
            
            # *image_iasp* updates
            
            # if image_iasp is starting this frame...
            if image_iasp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                image_iasp.frameNStart = frameN  # exact frame index
                image_iasp.tStart = t  # local t and not account for scr refresh
                image_iasp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_iasp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_iasp.started')
                # update status
                image_iasp.status = STARTED
                image_iasp.setAutoDraw(True)
            
            # if image_iasp is active this frame...
            if image_iasp.status == STARTED:
                # update params
                pass
            
            # if image_iasp is stopping this frame...
            if image_iasp.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > image_iasp.tStartRefresh + 4-frameTolerance:
                    # keep track of stop time/frame for later
                    image_iasp.tStop = t  # not accounting for scr refresh
                    image_iasp.tStopRefresh = tThisFlipGlobal  # on global time
                    image_iasp.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_iasp.stopped')
                    # update status
                    image_iasp.status = FINISHED
                    image_iasp.setAutoDraw(False)
            # *p_port_iasp* updates
            
            # if p_port_iasp is starting this frame...
            if p_port_iasp.status == NOT_STARTED and image_IASP.status == STARTED:
                # keep track of start time/frame for later
                p_port_iasp.frameNStart = frameN  # exact frame index
                p_port_iasp.tStart = t  # local t and not account for scr refresh
                p_port_iasp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_port_iasp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_port_iasp.started')
                # update status
                p_port_iasp.status = STARTED
                p_port_iasp.status = STARTED
                win.callOnFlip(p_port_iasp.setData, int(1))
            
            # if p_port_iasp is stopping this frame...
            if p_port_iasp.status == STARTED:
                if bool(image_iasp.status == STOPPED):
                    # keep track of stop time/frame for later
                    p_port_iasp.tStop = t  # not accounting for scr refresh
                    p_port_iasp.tStopRefresh = tThisFlipGlobal  # on global time
                    p_port_iasp.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'p_port_iasp.stopped')
                    # update status
                    p_port_iasp.status = FINISHED
                    win.callOnFlip(p_port_iasp.setData, int(0))
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=iaspView,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                iaspView.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if iaspView.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in iaspView.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "iaspView" ---
        for thisComponent in iaspView.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for iaspView
        iaspView.tStop = globalClock.getTime(format='float')
        iaspView.tStopRefresh = tThisFlipGlobal
        thisExp.addData('iaspView.stopped', iaspView.tStop)
        setupWindow(expInfo=expInfo, win=win)
        if p_port_iasp.status == STARTED:
            win.callOnFlip(p_port_iasp.setData, int(0))
        # the Routine "iaspView" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisIasp_practice_trial as finished
        if hasattr(thisIasp_practice_trial, 'status'):
            thisIasp_practice_trial.status = FINISHED
        # if awaiting a pause, pause now
        if iasp_practice_trials.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            iasp_practice_trials.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'iasp_practice_trials'
    iasp_practice_trials.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    # get names of stimulus parameters
    if iasp_practice_trials.trialList in ([], [None], None):
        params = []
    else:
        params = iasp_practice_trials.trialList[0].keys()
    # save data for this loop
    iasp_practice_trials.saveAsExcel(filename + '.xlsx', sheetName='iasp_practice_trials',
        stimOut=params,
        dataOut=['n','all_mean','all_std', 'all_raw'])
    
    # set up handler to look after randomisation of conditions etc
    state_measure_practice = data.TrialHandler2(
        name='state_measure_practice',
        nReps=1.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions(
        '../../shared/loop-templates/loopStateMeasure.csv', 
        selection='[2, 3, 4, 5, 6]'
    )
    , 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(state_measure_practice)  # add the loop to the experiment
    thisState_measure_practice = state_measure_practice.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisState_measure_practice.rgb)
    if thisState_measure_practice != None:
        for paramName in thisState_measure_practice:
            globals()[paramName] = thisState_measure_practice[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisState_measure_practice in state_measure_practice:
        state_measure_practice.status = STARTED
        if hasattr(thisState_measure_practice, 'status'):
            thisState_measure_practice.status = STARTED
        currentLoop = state_measure_practice
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisState_measure_practice.rgb)
        if thisState_measure_practice != None:
            for paramName in thisState_measure_practice:
                globals()[paramName] = thisState_measure_practice[paramName]
        
        # --- Prepare to start Routine "stateMeasure" ---
        # create an object to store info about Routine stateMeasure
        stateMeasure = data.Routine(
            name='stateMeasure',
            components=[image_SAM, t_sm_message, t_sm_continue, key_resp_sm],
        )
        stateMeasure.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_sm_helper
        allowContinue = False
        
        # Parse tick values from the condition file
        tick_values_parsed = parse_tick_values(tick_values)
        
        try:
            logging.debug(f"Starting state-measure in loop: {currentLoop.name}")
            logging.debug(f"State-measure current row: {rating_category}, {rating_type}")
            logging.debug(f"State-measure tick values: {tick_values_parsed}")
        except:
            logging.error("Error printing current state measure row.")
        
        # Compute geometry based on rating type
        sm_geometry = sm_compute_tick_positions(tick_values_parsed, rating_type)
        
        sm_comp_slider_size = sm_geometry["slider_size"]
        sm_comp_slider_pos  = sm_geometry["slider_pos"]
        sm_comp_label_min_pos = sm_geometry["label_min_pos"]
        sm_comp_label_max_pos = sm_geometry["label_max_pos"]
        
        t_sm_label_min = visual.TextStim(
            win=win,
            text=rating_min_label,
            pos=(sm_comp_label_min_pos[0], sm_comp_label_min_pos[1]),
            wrapWidth=aspect20,
            anchorVert='top',
            height=0.05,
            color='white',
            font='Noto Sans'
        )
        
        t_sm_label_max = visual.TextStim(
            win=win,
            text=rating_max_label,
            pos=(sm_comp_label_max_pos[0], sm_comp_label_max_pos[1]),
            wrapWidth=aspect20,
            anchorVert='top',
            height=0.05,
            color='white',
            font='Noto Sans'
        )
        
        #t_label_min.setPos(sm_comp_label_min_pos)
        #t_label_max.setPos(sm_comp_label_max_pos)
        
        # --- Create a fresh slider for this trial ---
        sm_slider = visual.Slider(
            win=win,
            ticks=tick_values_parsed,
            labels=tick_values_parsed,
            granularity=granularity,
            style=style.lower(),
            pos=sm_comp_slider_pos,
            size=sm_comp_slider_size,
            labelHeight=0.05,
            colorSpace='rgb',
            markerColor='Red',
            lineColor='White',
            labelColor='LightGray',
            font='Noto Sans'
        )
        
        # --- SAM image logic ---
        if rating_type == 'SAM':
            image_SAM.setImage(picture_path)
            image_SAM.setAutoDraw(True)
        else:
            image_SAM.setAutoDraw(False)
        
        image_SAM.setPos((0, sm_sam_comp_img_pos_y))
        image_SAM.setSize((sm_sam_comp_img_size_width, sm_sam_comp_img_size_height))
        image_SAM.setImage(picture_path)
        t_sm_message.setText(rating_message)
        # create starting attributes for key_resp_sm
        key_resp_sm.keys = []
        key_resp_sm.rt = []
        _key_resp_sm_allKeys = []
        # store start times for stateMeasure
        stateMeasure.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        stateMeasure.tStart = globalClock.getTime(format='float')
        stateMeasure.status = STARTED
        thisExp.addData('stateMeasure.started', stateMeasure.tStart)
        stateMeasure.maxDuration = None
        # keep track of which components have finished
        stateMeasureComponents = stateMeasure.components
        for thisComponent in stateMeasure.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "stateMeasure" ---
        thisExp.currentRoutine = stateMeasure
        stateMeasure.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisState_measure_practice, 'status') and thisState_measure_practice.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from code_sm_helper
            # Draw state measure components
            sm_slider.draw()
            t_sm_label_min.draw()
            t_sm_label_max.draw()
            
            # If rated, allow spacebar to end the routine
            if sm_slider.getRating() is not None:
                allowContinue = True
            
            
            # *image_SAM* updates
            
            # if image_SAM is starting this frame...
            if image_SAM.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                image_SAM.frameNStart = frameN  # exact frame index
                image_SAM.tStart = t  # local t and not account for scr refresh
                image_SAM.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_SAM, 'tStartRefresh')  # time at next scr refresh
                # update status
                image_SAM.status = STARTED
                image_SAM.setAutoDraw(True)
            
            # if image_SAM is active this frame...
            if image_SAM.status == STARTED:
                # update params
                pass
            
            # *t_sm_message* updates
            
            # if t_sm_message is starting this frame...
            if t_sm_message.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                t_sm_message.frameNStart = frameN  # exact frame index
                t_sm_message.tStart = t  # local t and not account for scr refresh
                t_sm_message.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(t_sm_message, 'tStartRefresh')  # time at next scr refresh
                # update status
                t_sm_message.status = STARTED
                t_sm_message.setAutoDraw(True)
            
            # if t_sm_message is active this frame...
            if t_sm_message.status == STARTED:
                # update params
                pass
            
            # *t_sm_continue* updates
            
            # if t_sm_continue is starting this frame...
            if t_sm_continue.status == NOT_STARTED and allowContinue == True:
                # keep track of start time/frame for later
                t_sm_continue.frameNStart = frameN  # exact frame index
                t_sm_continue.tStart = t  # local t and not account for scr refresh
                t_sm_continue.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(t_sm_continue, 'tStartRefresh')  # time at next scr refresh
                # update status
                t_sm_continue.status = STARTED
                t_sm_continue.setAutoDraw(True)
            
            # if t_sm_continue is active this frame...
            if t_sm_continue.status == STARTED:
                # update params
                pass
            
            # *key_resp_sm* updates
            waitOnFlip = False
            
            # if key_resp_sm is starting this frame...
            if key_resp_sm.status == NOT_STARTED and allowContinue == True:
                # keep track of start time/frame for later
                key_resp_sm.frameNStart = frameN  # exact frame index
                key_resp_sm.tStart = t  # local t and not account for scr refresh
                key_resp_sm.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_sm, 'tStartRefresh')  # time at next scr refresh
                # update status
                key_resp_sm.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_sm.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_sm.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_sm.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_sm.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_sm_allKeys.extend(theseKeys)
                if len(_key_resp_sm_allKeys):
                    key_resp_sm.keys = _key_resp_sm_allKeys[-1].name  # just the last key pressed
                    key_resp_sm.rt = _key_resp_sm_allKeys[-1].rt
                    key_resp_sm.duration = _key_resp_sm_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=stateMeasure,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                stateMeasure.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if stateMeasure.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in stateMeasure.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "stateMeasure" ---
        for thisComponent in stateMeasure.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for stateMeasure
        stateMeasure.tStop = globalClock.getTime(format='float')
        stateMeasure.tStopRefresh = tThisFlipGlobal
        thisExp.addData('stateMeasure.stopped', stateMeasure.tStop)
        # Run 'End Routine' code from code_sm_helper
        currentLoop.addData('rating', sm_slider.getRating())
        currentLoop.addData('rating_rt', sm_slider.getRT())
        
        try:
            logging.data(f"State-measure rating: {rating_category}, {sm_slider.getRating()}")
        except:
            logging.error("Error printing state-measure rating")
        # the Routine "stateMeasure" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisState_measure_practice as finished
        if hasattr(thisState_measure_practice, 'status'):
            thisState_measure_practice.status = FINISHED
        # if awaiting a pause, pause now
        if state_measure_practice.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            state_measure_practice.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'state_measure_practice'
    state_measure_practice.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    # get names of stimulus parameters
    if state_measure_practice.trialList in ([], [None], None):
        params = []
    else:
        params = state_measure_practice.trialList[0].keys()
    # save data for this loop
    state_measure_practice.saveAsExcel(filename + '.xlsx', sheetName='state_measure_practice',
        stimOut=params,
        dataOut=['n','all_mean','all_std', 'all_raw'])
    
    # --- Prepare to start Routine "instructionUnpleasant" ---
    # create an object to store info about Routine instructionUnpleasant
    instructionUnpleasant = data.Routine(
        name='instructionUnpleasant',
        components=[t_unpleasant_instruction, t_unpleasant_continue, key_resp_unpleasant],
    )
    instructionUnpleasant.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    t_unpleasant_instruction.setText('Next, you will begin the main set of unpleasant picture trials.\n\nIf at any point you would like to stop, please let the experimenter know.')
    # create starting attributes for key_resp_unpleasant
    key_resp_unpleasant.keys = []
    key_resp_unpleasant.rt = []
    _key_resp_unpleasant_allKeys = []
    # store start times for instructionUnpleasant
    instructionUnpleasant.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instructionUnpleasant.tStart = globalClock.getTime(format='float')
    instructionUnpleasant.status = STARTED
    thisExp.addData('instructionUnpleasant.started', instructionUnpleasant.tStart)
    instructionUnpleasant.maxDuration = None
    # keep track of which components have finished
    instructionUnpleasantComponents = instructionUnpleasant.components
    for thisComponent in instructionUnpleasant.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructionUnpleasant" ---
    thisExp.currentRoutine = instructionUnpleasant
    instructionUnpleasant.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *t_unpleasant_instruction* updates
        
        # if t_unpleasant_instruction is starting this frame...
        if t_unpleasant_instruction.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            t_unpleasant_instruction.frameNStart = frameN  # exact frame index
            t_unpleasant_instruction.tStart = t  # local t and not account for scr refresh
            t_unpleasant_instruction.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(t_unpleasant_instruction, 'tStartRefresh')  # time at next scr refresh
            # update status
            t_unpleasant_instruction.status = STARTED
            t_unpleasant_instruction.setAutoDraw(True)
        
        # if t_unpleasant_instruction is active this frame...
        if t_unpleasant_instruction.status == STARTED:
            # update params
            pass
        
        # *t_unpleasant_continue* updates
        
        # if t_unpleasant_continue is starting this frame...
        if t_unpleasant_continue.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            t_unpleasant_continue.frameNStart = frameN  # exact frame index
            t_unpleasant_continue.tStart = t  # local t and not account for scr refresh
            t_unpleasant_continue.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(t_unpleasant_continue, 'tStartRefresh')  # time at next scr refresh
            # update status
            t_unpleasant_continue.status = STARTED
            t_unpleasant_continue.setAutoDraw(True)
        
        # if t_unpleasant_continue is active this frame...
        if t_unpleasant_continue.status == STARTED:
            # update params
            pass
        
        # *key_resp_unpleasant* updates
        waitOnFlip = False
        
        # if key_resp_unpleasant is starting this frame...
        if key_resp_unpleasant.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_unpleasant.frameNStart = frameN  # exact frame index
            key_resp_unpleasant.tStart = t  # local t and not account for scr refresh
            key_resp_unpleasant.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_unpleasant, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_resp_unpleasant.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_unpleasant.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_unpleasant.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_unpleasant.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_unpleasant.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_unpleasant_allKeys.extend(theseKeys)
            if len(_key_resp_unpleasant_allKeys):
                key_resp_unpleasant.keys = _key_resp_unpleasant_allKeys[-1].name  # just the last key pressed
                key_resp_unpleasant.rt = _key_resp_unpleasant_allKeys[-1].rt
                key_resp_unpleasant.duration = _key_resp_unpleasant_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=instructionUnpleasant,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            instructionUnpleasant.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if instructionUnpleasant.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in instructionUnpleasant.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructionUnpleasant" ---
    for thisComponent in instructionUnpleasant.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instructionUnpleasant
    instructionUnpleasant.tStop = globalClock.getTime(format='float')
    instructionUnpleasant.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instructionUnpleasant.stopped', instructionUnpleasant.tStop)
    thisExp.nextEntry()
    # the Routine "instructionUnpleasant" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    block_loop = data.TrialHandler2(
        name='block_loop',
        nReps=1.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('current_slice.csv'), 
        seed=None, 
        isTrials=False, 
    )
    thisExp.addLoop(block_loop)  # add the loop to the experiment
    thisBlock_loop = block_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisBlock_loop.rgb)
    if thisBlock_loop != None:
        for paramName in thisBlock_loop:
            globals()[paramName] = thisBlock_loop[paramName]
    
    for thisBlock_loop in block_loop:
        block_loop.status = STARTED
        if hasattr(thisBlock_loop, 'status'):
            thisBlock_loop.status = STARTED
        currentLoop = block_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # abbreviate parameter names if possible (e.g. rgb = thisBlock_loop.rgb)
        if thisBlock_loop != None:
            for paramName in thisBlock_loop:
                globals()[paramName] = thisBlock_loop[paramName]
        
        # --- Prepare to start Routine "reset" ---
        # create an object to store info about Routine reset
        reset = data.Routine(
            name='reset',
            components=[t_reset_instruction, t_reset_continue, key_resp_reset],
        )
        reset.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_slicer_helper
        # ------------------------------------------------------------
        # code_slicer_helper
        #
        # Selects the current 40 trial slice from the pre-randomized
        # CSV file. The surrounding loop uses this slice to determine
        # which trials to run for the current block.
        #
        # Begin Routine:
        #   • Builds a "start:end" slice string (e.g., "0:40")
        #   • Logs the slice to the data file for analysis
        #
        # End Routine:
        #   • Advances start/end by 40 to move to the next block
        #
        # This component does NOT randomize trials. It only selects
        # which portion of the CSV to use on each loop iteration.
        # ------------------------------------------------------------
        
        # Define slice for current block
        row_section = f"{start}:{end}"
        print(row_section)
        
        # Log which slice was used
        #currentLoop.addData('row_section', row_section)
        
        # Change current emotion regulation strategy text
        current_technique = negative_trials_technique_order[reset_routine_counter]
        t_reset_instruction_text = ""
        
        if current_technique == "View":
            t_reset_instruction_text = just_view_text
        elif current_technique == "Suppress":
            t_reset_instruction_text = suppress_text
        elif current_technique == "Reappraise":
            t_reset_instruction_text = reappraise_text
        elif current_technique == "Suppress and Reappraise":
            t_reset_instruction_text = suppress_and_reappraise_text
        else:
            t_reset_instruction_text = "Unknown technique. Please contact the experimenter."
        t_reset_instruction.setText(t_reset_instruction_text)
        # create starting attributes for key_resp_reset
        key_resp_reset.keys = []
        key_resp_reset.rt = []
        _key_resp_reset_allKeys = []
        # store start times for reset
        reset.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        reset.tStart = globalClock.getTime(format='float')
        reset.status = STARTED
        thisExp.addData('reset.started', reset.tStart)
        reset.maxDuration = None
        # keep track of which components have finished
        resetComponents = reset.components
        for thisComponent in reset.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "reset" ---
        thisExp.currentRoutine = reset
        reset.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisBlock_loop, 'status') and thisBlock_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *t_reset_instruction* updates
            
            # if t_reset_instruction is starting this frame...
            if t_reset_instruction.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                t_reset_instruction.frameNStart = frameN  # exact frame index
                t_reset_instruction.tStart = t  # local t and not account for scr refresh
                t_reset_instruction.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(t_reset_instruction, 'tStartRefresh')  # time at next scr refresh
                # update status
                t_reset_instruction.status = STARTED
                t_reset_instruction.setAutoDraw(True)
            
            # if t_reset_instruction is active this frame...
            if t_reset_instruction.status == STARTED:
                # update params
                pass
            
            # *t_reset_continue* updates
            
            # if t_reset_continue is starting this frame...
            if t_reset_continue.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                t_reset_continue.frameNStart = frameN  # exact frame index
                t_reset_continue.tStart = t  # local t and not account for scr refresh
                t_reset_continue.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(t_reset_continue, 'tStartRefresh')  # time at next scr refresh
                # update status
                t_reset_continue.status = STARTED
                t_reset_continue.setAutoDraw(True)
            
            # if t_reset_continue is active this frame...
            if t_reset_continue.status == STARTED:
                # update params
                pass
            
            # *key_resp_reset* updates
            waitOnFlip = False
            
            # if key_resp_reset is starting this frame...
            if key_resp_reset.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_reset.frameNStart = frameN  # exact frame index
                key_resp_reset.tStart = t  # local t and not account for scr refresh
                key_resp_reset.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_reset, 'tStartRefresh')  # time at next scr refresh
                # update status
                key_resp_reset.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_reset.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_reset.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_reset.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_reset.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_reset_allKeys.extend(theseKeys)
                if len(_key_resp_reset_allKeys):
                    key_resp_reset.keys = _key_resp_reset_allKeys[-1].name  # just the last key pressed
                    key_resp_reset.rt = _key_resp_reset_allKeys[-1].rt
                    key_resp_reset.duration = _key_resp_reset_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=reset,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                reset.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if reset.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in reset.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "reset" ---
        for thisComponent in reset.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for reset
        reset.tStop = globalClock.getTime(format='float')
        reset.tStopRefresh = tThisFlipGlobal
        thisExp.addData('reset.stopped', reset.tStop)
        # Run 'End Routine' code from code_slicer_helper
        # Advance slice for next block
        start += 40
        end += 40
        
        if reset_routine_counter < len(negative_trials_technique_order):
            reset_routine_counter += 1
        # the Routine "reset" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        iasp_unpleasant_trials = data.TrialHandler2(
            name='iasp_unpleasant_trials',
            nReps=1.0, 
            method='sequential', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions(
            'emotion_regulation_unpleasant_trials.csv', 
            selection=row_section
        )
        , 
            seed=None, 
            isTrials=True, 
        )
        thisExp.addLoop(iasp_unpleasant_trials)  # add the loop to the experiment
        thisIasp_unpleasant_trial = iasp_unpleasant_trials.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisIasp_unpleasant_trial.rgb)
        if thisIasp_unpleasant_trial != None:
            for paramName in thisIasp_unpleasant_trial:
                globals()[paramName] = thisIasp_unpleasant_trial[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisIasp_unpleasant_trial in iasp_unpleasant_trials:
            iasp_unpleasant_trials.status = STARTED
            if hasattr(thisIasp_unpleasant_trial, 'status'):
                thisIasp_unpleasant_trial.status = STARTED
            currentLoop = iasp_unpleasant_trials
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisIasp_unpleasant_trial.rgb)
            if thisIasp_unpleasant_trial != None:
                for paramName in thisIasp_unpleasant_trial:
                    globals()[paramName] = thisIasp_unpleasant_trial[paramName]
            
            # --- Prepare to start Routine "emotionRegulationCue" ---
            # create an object to store info about Routine emotionRegulationCue
            emotionRegulationCue = data.Routine(
                name='emotionRegulationCue',
                components=[cross_fixation, t_emotion_regulation_cue, t_blank_delayer],
            )
            emotionRegulationCue.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from code_delay_calculator
            # Gives a random float between 0.5 and 1.5 seconds (i.e., 500–1500 ms)
            # Used for t_blank_delayer
            delay_time = random.uniform(0.5, 1.5)
            currentLoop.addData('delay_time', delay_time)
            t_emotion_regulation_cue.setText(technique)
            # store start times for emotionRegulationCue
            emotionRegulationCue.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            emotionRegulationCue.tStart = globalClock.getTime(format='float')
            emotionRegulationCue.status = STARTED
            thisExp.addData('emotionRegulationCue.started', emotionRegulationCue.tStart)
            emotionRegulationCue.maxDuration = None
            win.color = [-1.0000, -1.0000, -1.0000]
            win.colorSpace = 'rgb'
            win.backgroundImage = ''
            win.backgroundFit = 'none'
            # keep track of which components have finished
            emotionRegulationCueComponents = emotionRegulationCue.components
            for thisComponent in emotionRegulationCue.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "emotionRegulationCue" ---
            thisExp.currentRoutine = emotionRegulationCue
            emotionRegulationCue.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisIasp_unpleasant_trial, 'status') and thisIasp_unpleasant_trial.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *cross_fixation* updates
                
                # if cross_fixation is starting this frame...
                if cross_fixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    cross_fixation.frameNStart = frameN  # exact frame index
                    cross_fixation.tStart = t  # local t and not account for scr refresh
                    cross_fixation.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(cross_fixation, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cross_fixation.started')
                    # update status
                    cross_fixation.status = STARTED
                    cross_fixation.setAutoDraw(True)
                
                # if cross_fixation is active this frame...
                if cross_fixation.status == STARTED:
                    # update params
                    pass
                
                # if cross_fixation is stopping this frame...
                if cross_fixation.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > cross_fixation.tStartRefresh + 0.3-frameTolerance:
                        # keep track of stop time/frame for later
                        cross_fixation.tStop = t  # not accounting for scr refresh
                        cross_fixation.tStopRefresh = tThisFlipGlobal  # on global time
                        cross_fixation.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'cross_fixation.stopped')
                        # update status
                        cross_fixation.status = FINISHED
                        cross_fixation.setAutoDraw(False)
                
                # *t_emotion_regulation_cue* updates
                
                # if t_emotion_regulation_cue is starting this frame...
                if t_emotion_regulation_cue.status == NOT_STARTED and tThisFlip >= 0.3-frameTolerance:
                    # keep track of start time/frame for later
                    t_emotion_regulation_cue.frameNStart = frameN  # exact frame index
                    t_emotion_regulation_cue.tStart = t  # local t and not account for scr refresh
                    t_emotion_regulation_cue.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(t_emotion_regulation_cue, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 't_emotion_regulation_cue.started')
                    # update status
                    t_emotion_regulation_cue.status = STARTED
                    t_emotion_regulation_cue.setAutoDraw(True)
                
                # if t_emotion_regulation_cue is active this frame...
                if t_emotion_regulation_cue.status == STARTED:
                    # update params
                    pass
                
                # if t_emotion_regulation_cue is stopping this frame...
                if t_emotion_regulation_cue.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > t_emotion_regulation_cue.tStartRefresh + 1-frameTolerance:
                        # keep track of stop time/frame for later
                        t_emotion_regulation_cue.tStop = t  # not accounting for scr refresh
                        t_emotion_regulation_cue.tStopRefresh = tThisFlipGlobal  # on global time
                        t_emotion_regulation_cue.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 't_emotion_regulation_cue.stopped')
                        # update status
                        t_emotion_regulation_cue.status = FINISHED
                        t_emotion_regulation_cue.setAutoDraw(False)
                
                # *t_blank_delayer* updates
                
                # if t_blank_delayer is starting this frame...
                if t_blank_delayer.status == NOT_STARTED and tThisFlip >= 1.3-frameTolerance:
                    # keep track of start time/frame for later
                    t_blank_delayer.frameNStart = frameN  # exact frame index
                    t_blank_delayer.tStart = t  # local t and not account for scr refresh
                    t_blank_delayer.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(t_blank_delayer, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 't_blank_delayer.started')
                    # update status
                    t_blank_delayer.status = STARTED
                    t_blank_delayer.setAutoDraw(True)
                
                # if t_blank_delayer is active this frame...
                if t_blank_delayer.status == STARTED:
                    # update params
                    pass
                
                # if t_blank_delayer is stopping this frame...
                if t_blank_delayer.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > t_blank_delayer.tStartRefresh + delay_time-frameTolerance:
                        # keep track of stop time/frame for later
                        t_blank_delayer.tStop = t  # not accounting for scr refresh
                        t_blank_delayer.tStopRefresh = tThisFlipGlobal  # on global time
                        t_blank_delayer.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 't_blank_delayer.stopped')
                        # update status
                        t_blank_delayer.status = FINISHED
                        t_blank_delayer.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=emotionRegulationCue,
                    )
                    # skip the frame we paused on
                    continue
                
                # has a Component requested the Routine to end?
                if not continueRoutine:
                    emotionRegulationCue.forceEnded = routineForceEnded = True
                # has the Routine been forcibly ended?
                if emotionRegulationCue.forceEnded or routineForceEnded:
                    break
                # has every Component finished?
                continueRoutine = False
                for thisComponent in emotionRegulationCue.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "emotionRegulationCue" ---
            for thisComponent in emotionRegulationCue.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for emotionRegulationCue
            emotionRegulationCue.tStop = globalClock.getTime(format='float')
            emotionRegulationCue.tStopRefresh = tThisFlipGlobal
            thisExp.addData('emotionRegulationCue.stopped', emotionRegulationCue.tStop)
            setupWindow(expInfo=expInfo, win=win)
            # the Routine "emotionRegulationCue" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "iaspView" ---
            # create an object to store info about Routine iaspView
            iaspView = data.Routine(
                name='iaspView',
                components=[image_iasp, p_port_iasp],
            )
            iaspView.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            image_iasp.setSize([iasp_comp_img_size_width, iasp_comp_img_size_height])
            image_iasp.setImage(photo_filename)
            # store start times for iaspView
            iaspView.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            iaspView.tStart = globalClock.getTime(format='float')
            iaspView.status = STARTED
            thisExp.addData('iaspView.started', iaspView.tStart)
            iaspView.maxDuration = 5
            win.color = [-1.0000, -1.0000, -1.0000]
            win.colorSpace = 'rgb'
            win.backgroundImage = ''
            win.backgroundFit = 'none'
            # keep track of which components have finished
            iaspViewComponents = iaspView.components
            for thisComponent in iaspView.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "iaspView" ---
            thisExp.currentRoutine = iaspView
            iaspView.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisIasp_unpleasant_trial, 'status') and thisIasp_unpleasant_trial.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # is it time to end the Routine? (based on local clock)
                if tThisFlip > iaspView.maxDuration-frameTolerance:
                    iaspView.maxDurationReached = True
                    continueRoutine = False
                
                # *image_iasp* updates
                
                # if image_iasp is starting this frame...
                if image_iasp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image_iasp.frameNStart = frameN  # exact frame index
                    image_iasp.tStart = t  # local t and not account for scr refresh
                    image_iasp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_iasp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_iasp.started')
                    # update status
                    image_iasp.status = STARTED
                    image_iasp.setAutoDraw(True)
                
                # if image_iasp is active this frame...
                if image_iasp.status == STARTED:
                    # update params
                    pass
                
                # if image_iasp is stopping this frame...
                if image_iasp.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > image_iasp.tStartRefresh + 4-frameTolerance:
                        # keep track of stop time/frame for later
                        image_iasp.tStop = t  # not accounting for scr refresh
                        image_iasp.tStopRefresh = tThisFlipGlobal  # on global time
                        image_iasp.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_iasp.stopped')
                        # update status
                        image_iasp.status = FINISHED
                        image_iasp.setAutoDraw(False)
                # *p_port_iasp* updates
                
                # if p_port_iasp is starting this frame...
                if p_port_iasp.status == NOT_STARTED and image_IASP.status == STARTED:
                    # keep track of start time/frame for later
                    p_port_iasp.frameNStart = frameN  # exact frame index
                    p_port_iasp.tStart = t  # local t and not account for scr refresh
                    p_port_iasp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(p_port_iasp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'p_port_iasp.started')
                    # update status
                    p_port_iasp.status = STARTED
                    p_port_iasp.status = STARTED
                    win.callOnFlip(p_port_iasp.setData, int(1))
                
                # if p_port_iasp is stopping this frame...
                if p_port_iasp.status == STARTED:
                    if bool(image_iasp.status == STOPPED):
                        # keep track of stop time/frame for later
                        p_port_iasp.tStop = t  # not accounting for scr refresh
                        p_port_iasp.tStopRefresh = tThisFlipGlobal  # on global time
                        p_port_iasp.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'p_port_iasp.stopped')
                        # update status
                        p_port_iasp.status = FINISHED
                        win.callOnFlip(p_port_iasp.setData, int(0))
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=iaspView,
                    )
                    # skip the frame we paused on
                    continue
                
                # has a Component requested the Routine to end?
                if not continueRoutine:
                    iaspView.forceEnded = routineForceEnded = True
                # has the Routine been forcibly ended?
                if iaspView.forceEnded or routineForceEnded:
                    break
                # has every Component finished?
                continueRoutine = False
                for thisComponent in iaspView.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "iaspView" ---
            for thisComponent in iaspView.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for iaspView
            iaspView.tStop = globalClock.getTime(format='float')
            iaspView.tStopRefresh = tThisFlipGlobal
            thisExp.addData('iaspView.stopped', iaspView.tStop)
            setupWindow(expInfo=expInfo, win=win)
            if p_port_iasp.status == STARTED:
                win.callOnFlip(p_port_iasp.setData, int(0))
            # the Routine "iaspView" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            # mark thisIasp_unpleasant_trial as finished
            if hasattr(thisIasp_unpleasant_trial, 'status'):
                thisIasp_unpleasant_trial.status = FINISHED
            # if awaiting a pause, pause now
            if iasp_unpleasant_trials.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                iasp_unpleasant_trials.status = STARTED
            thisExp.nextEntry()
            
        # completed 1.0 repeats of 'iasp_unpleasant_trials'
        iasp_unpleasant_trials.status = FINISHED
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # get names of stimulus parameters
        if iasp_unpleasant_trials.trialList in ([], [None], None):
            params = []
        else:
            params = iasp_unpleasant_trials.trialList[0].keys()
        # save data for this loop
        iasp_unpleasant_trials.saveAsExcel(filename + '.xlsx', sheetName='iasp_unpleasant_trials',
            stimOut=params,
            dataOut=['n','all_mean','all_std', 'all_raw'])
        
        # set up handler to look after randomisation of conditions etc
        state_measure_unpleasant = data.TrialHandler2(
            name='state_measure_unpleasant',
            nReps=1.0, 
            method='sequential', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions(
            '../../shared/loop-templates/loopStateMeasure.csv', 
            selection='[7, 2, 3, 4, 5, 6]'
        )
        , 
            seed=None, 
            isTrials=True, 
        )
        thisExp.addLoop(state_measure_unpleasant)  # add the loop to the experiment
        thisState_measure_unpleasant = state_measure_unpleasant.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisState_measure_unpleasant.rgb)
        if thisState_measure_unpleasant != None:
            for paramName in thisState_measure_unpleasant:
                globals()[paramName] = thisState_measure_unpleasant[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisState_measure_unpleasant in state_measure_unpleasant:
            state_measure_unpleasant.status = STARTED
            if hasattr(thisState_measure_unpleasant, 'status'):
                thisState_measure_unpleasant.status = STARTED
            currentLoop = state_measure_unpleasant
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisState_measure_unpleasant.rgb)
            if thisState_measure_unpleasant != None:
                for paramName in thisState_measure_unpleasant:
                    globals()[paramName] = thisState_measure_unpleasant[paramName]
            
            # --- Prepare to start Routine "stateMeasure" ---
            # create an object to store info about Routine stateMeasure
            stateMeasure = data.Routine(
                name='stateMeasure',
                components=[image_SAM, t_sm_message, t_sm_continue, key_resp_sm],
            )
            stateMeasure.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from code_sm_helper
            allowContinue = False
            
            # Parse tick values from the condition file
            tick_values_parsed = parse_tick_values(tick_values)
            
            try:
                logging.debug(f"Starting state-measure in loop: {currentLoop.name}")
                logging.debug(f"State-measure current row: {rating_category}, {rating_type}")
                logging.debug(f"State-measure tick values: {tick_values_parsed}")
            except:
                logging.error("Error printing current state measure row.")
            
            # Compute geometry based on rating type
            sm_geometry = sm_compute_tick_positions(tick_values_parsed, rating_type)
            
            sm_comp_slider_size = sm_geometry["slider_size"]
            sm_comp_slider_pos  = sm_geometry["slider_pos"]
            sm_comp_label_min_pos = sm_geometry["label_min_pos"]
            sm_comp_label_max_pos = sm_geometry["label_max_pos"]
            
            t_sm_label_min = visual.TextStim(
                win=win,
                text=rating_min_label,
                pos=(sm_comp_label_min_pos[0], sm_comp_label_min_pos[1]),
                wrapWidth=aspect20,
                anchorVert='top',
                height=0.05,
                color='white',
                font='Noto Sans'
            )
            
            t_sm_label_max = visual.TextStim(
                win=win,
                text=rating_max_label,
                pos=(sm_comp_label_max_pos[0], sm_comp_label_max_pos[1]),
                wrapWidth=aspect20,
                anchorVert='top',
                height=0.05,
                color='white',
                font='Noto Sans'
            )
            
            #t_label_min.setPos(sm_comp_label_min_pos)
            #t_label_max.setPos(sm_comp_label_max_pos)
            
            # --- Create a fresh slider for this trial ---
            sm_slider = visual.Slider(
                win=win,
                ticks=tick_values_parsed,
                labels=tick_values_parsed,
                granularity=granularity,
                style=style.lower(),
                pos=sm_comp_slider_pos,
                size=sm_comp_slider_size,
                labelHeight=0.05,
                colorSpace='rgb',
                markerColor='Red',
                lineColor='White',
                labelColor='LightGray',
                font='Noto Sans'
            )
            
            # --- SAM image logic ---
            if rating_type == 'SAM':
                image_SAM.setImage(picture_path)
                image_SAM.setAutoDraw(True)
            else:
                image_SAM.setAutoDraw(False)
            
            image_SAM.setPos((0, sm_sam_comp_img_pos_y))
            image_SAM.setSize((sm_sam_comp_img_size_width, sm_sam_comp_img_size_height))
            image_SAM.setImage(picture_path)
            t_sm_message.setText(rating_message)
            # create starting attributes for key_resp_sm
            key_resp_sm.keys = []
            key_resp_sm.rt = []
            _key_resp_sm_allKeys = []
            # store start times for stateMeasure
            stateMeasure.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            stateMeasure.tStart = globalClock.getTime(format='float')
            stateMeasure.status = STARTED
            thisExp.addData('stateMeasure.started', stateMeasure.tStart)
            stateMeasure.maxDuration = None
            # keep track of which components have finished
            stateMeasureComponents = stateMeasure.components
            for thisComponent in stateMeasure.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "stateMeasure" ---
            thisExp.currentRoutine = stateMeasure
            stateMeasure.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisState_measure_unpleasant, 'status') and thisState_measure_unpleasant.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # Run 'Each Frame' code from code_sm_helper
                # Draw state measure components
                sm_slider.draw()
                t_sm_label_min.draw()
                t_sm_label_max.draw()
                
                # If rated, allow spacebar to end the routine
                if sm_slider.getRating() is not None:
                    allowContinue = True
                
                
                # *image_SAM* updates
                
                # if image_SAM is starting this frame...
                if image_SAM.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image_SAM.frameNStart = frameN  # exact frame index
                    image_SAM.tStart = t  # local t and not account for scr refresh
                    image_SAM.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_SAM, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    image_SAM.status = STARTED
                    image_SAM.setAutoDraw(True)
                
                # if image_SAM is active this frame...
                if image_SAM.status == STARTED:
                    # update params
                    pass
                
                # *t_sm_message* updates
                
                # if t_sm_message is starting this frame...
                if t_sm_message.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    t_sm_message.frameNStart = frameN  # exact frame index
                    t_sm_message.tStart = t  # local t and not account for scr refresh
                    t_sm_message.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(t_sm_message, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    t_sm_message.status = STARTED
                    t_sm_message.setAutoDraw(True)
                
                # if t_sm_message is active this frame...
                if t_sm_message.status == STARTED:
                    # update params
                    pass
                
                # *t_sm_continue* updates
                
                # if t_sm_continue is starting this frame...
                if t_sm_continue.status == NOT_STARTED and allowContinue == True:
                    # keep track of start time/frame for later
                    t_sm_continue.frameNStart = frameN  # exact frame index
                    t_sm_continue.tStart = t  # local t and not account for scr refresh
                    t_sm_continue.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(t_sm_continue, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    t_sm_continue.status = STARTED
                    t_sm_continue.setAutoDraw(True)
                
                # if t_sm_continue is active this frame...
                if t_sm_continue.status == STARTED:
                    # update params
                    pass
                
                # *key_resp_sm* updates
                waitOnFlip = False
                
                # if key_resp_sm is starting this frame...
                if key_resp_sm.status == NOT_STARTED and allowContinue == True:
                    # keep track of start time/frame for later
                    key_resp_sm.frameNStart = frameN  # exact frame index
                    key_resp_sm.tStart = t  # local t and not account for scr refresh
                    key_resp_sm.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp_sm, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    key_resp_sm.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp_sm.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp_sm.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if key_resp_sm.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp_sm.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                    _key_resp_sm_allKeys.extend(theseKeys)
                    if len(_key_resp_sm_allKeys):
                        key_resp_sm.keys = _key_resp_sm_allKeys[-1].name  # just the last key pressed
                        key_resp_sm.rt = _key_resp_sm_allKeys[-1].rt
                        key_resp_sm.duration = _key_resp_sm_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=stateMeasure,
                    )
                    # skip the frame we paused on
                    continue
                
                # has a Component requested the Routine to end?
                if not continueRoutine:
                    stateMeasure.forceEnded = routineForceEnded = True
                # has the Routine been forcibly ended?
                if stateMeasure.forceEnded or routineForceEnded:
                    break
                # has every Component finished?
                continueRoutine = False
                for thisComponent in stateMeasure.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "stateMeasure" ---
            for thisComponent in stateMeasure.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for stateMeasure
            stateMeasure.tStop = globalClock.getTime(format='float')
            stateMeasure.tStopRefresh = tThisFlipGlobal
            thisExp.addData('stateMeasure.stopped', stateMeasure.tStop)
            # Run 'End Routine' code from code_sm_helper
            currentLoop.addData('rating', sm_slider.getRating())
            currentLoop.addData('rating_rt', sm_slider.getRT())
            
            try:
                logging.data(f"State-measure rating: {rating_category}, {sm_slider.getRating()}")
            except:
                logging.error("Error printing state-measure rating")
            # the Routine "stateMeasure" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            # mark thisState_measure_unpleasant as finished
            if hasattr(thisState_measure_unpleasant, 'status'):
                thisState_measure_unpleasant.status = FINISHED
            # if awaiting a pause, pause now
            if state_measure_unpleasant.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                state_measure_unpleasant.status = STARTED
            thisExp.nextEntry()
            
        # completed 1.0 repeats of 'state_measure_unpleasant'
        state_measure_unpleasant.status = FINISHED
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # get names of stimulus parameters
        if state_measure_unpleasant.trialList in ([], [None], None):
            params = []
        else:
            params = state_measure_unpleasant.trialList[0].keys()
        # save data for this loop
        state_measure_unpleasant.saveAsExcel(filename + '.xlsx', sheetName='state_measure_unpleasant',
            stimOut=params,
            dataOut=['n','all_mean','all_std', 'all_raw'])
        # mark thisBlock_loop as finished
        if hasattr(thisBlock_loop, 'status'):
            thisBlock_loop.status = FINISHED
        # if awaiting a pause, pause now
        if block_loop.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            block_loop.status = STARTED
    # completed 1.0 repeats of 'block_loop'
    block_loop.status = FINISHED
    
    
    # --- Prepare to start Routine "goodbye" ---
    # create an object to store info about Routine goodbye
    goodbye = data.Routine(
        name='goodbye',
        components=[t_goodbye],
    )
    goodbye.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for goodbye
    goodbye.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    goodbye.tStart = globalClock.getTime(format='float')
    goodbye.status = STARTED
    thisExp.addData('goodbye.started', goodbye.tStart)
    goodbye.maxDuration = 4
    # keep track of which components have finished
    goodbyeComponents = goodbye.components
    for thisComponent in goodbye.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "goodbye" ---
    thisExp.currentRoutine = goodbye
    goodbye.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 4.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # is it time to end the Routine? (based on local clock)
        if tThisFlip > goodbye.maxDuration-frameTolerance:
            goodbye.maxDurationReached = True
            continueRoutine = False
        
        # *t_goodbye* updates
        
        # if t_goodbye is starting this frame...
        if t_goodbye.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            t_goodbye.frameNStart = frameN  # exact frame index
            t_goodbye.tStart = t  # local t and not account for scr refresh
            t_goodbye.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(t_goodbye, 'tStartRefresh')  # time at next scr refresh
            # update status
            t_goodbye.status = STARTED
            t_goodbye.setAutoDraw(True)
        
        # if t_goodbye is active this frame...
        if t_goodbye.status == STARTED:
            # update params
            pass
        
        # if t_goodbye is stopping this frame...
        if t_goodbye.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > t_goodbye.tStartRefresh + 4-frameTolerance:
                # keep track of stop time/frame for later
                t_goodbye.tStop = t  # not accounting for scr refresh
                t_goodbye.tStopRefresh = tThisFlipGlobal  # on global time
                t_goodbye.frameNStop = frameN  # exact frame index
                # update status
                t_goodbye.status = FINISHED
                t_goodbye.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=goodbye,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            goodbye.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if goodbye.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in goodbye.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "goodbye" ---
    for thisComponent in goodbye.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for goodbye
    goodbye.tStop = globalClock.getTime(format='float')
    goodbye.tStopRefresh = tThisFlipGlobal
    thisExp.addData('goodbye.stopped', goodbye.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if goodbye.maxDurationReached:
        routineTimer.addTime(-goodbye.maxDuration)
    elif goodbye.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-4.000000)
    thisExp.nextEntry()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # run any 'at exit' functions
    for fcn in runAtExit:
        fcn()
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
