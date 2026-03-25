#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2025.2.3),
    on March 19, 2026, at 01:14
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
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
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

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2025.2.3'
expName = 'pain-appraisal'  # from the Builder filename that created this script
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
_winSize = [1536, 864]
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
        originPath='D:\\Github\\neuroflare-experiments\\neuroflare\\experiments\\pain-appraisal\\pain-appraisal_lastrun.py',
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
            winType='pyglet', allowGUI=True, allowStencil=True,
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
    _THIS_DIR = os.path.dirname(__file__)
    _REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..', '..'))
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    
    from neuroflare.shared.state_measure import StateMeasure, parse_tick_values
    
    sm = StateMeasure(win=win, aspect=aspect, screen_width=screen_width, screen_height=screen_height)
    # Run 'Begin Experiment' code from code_trigger_setup
    # ------------------------------------------------------------
    # code_trigger_setup
    
    # Manage a Brain Products TriggerBox over a virtual serial port.
    # Provides connection discovery, reconnection, and high/low 
    # trigger management without crashing when the COM port is missing.
    # ------------------------------------------------------------
    from neuroflare.shared.trigger_box import TriggerBoxManager
    tb = TriggerBoxManager()
    tb.begin_experiment()
    
    # Emotion Regulation (30-39):
    trigger_neutral_image = (30, "neutral_image")
    trigger_practice_image = (31, "practice_image")
    trigger_negative_image = (32, "negative_image")
    trigger_regulation_cue = (38, "regulation_cue")
    
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
    t_sm_continue = visual.TextStim(win=win, name='t_sm_continue',
        text='Press the SPACEBAR to continue',
        font='Arial',
        pos=(0, -0.35), draggable=False, height=0.04, wrapWidth=comp_wrap_width_continue, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_sm = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "intro" ---
    t_intro = visual.TextStim(win=win, name='t_intro',
        text="We are now going to ask you to rate different words that describe pain. Click the number that best matches your experience for each word.\u200b\n\n\nFor each word shown, rate how well it describes your current pain using the following scale: \u200b\n\n0 – none, 1 – Mild, 2 – Moderate, 3 – Severe\u200b\n\n\u200b\n\n**When you're ready, press the spacebar to begin.**",
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_intro = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "fixation" ---
    cross_fixation = visual.ShapeStim(
        win=win, name='cross_fixation', vertices='cross',units='height', 
        size=(0.2, 0.2),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "pain_adjectives01" ---
    t_pain_adj = visual.TextStim(win=win, name='t_pain_adj',
        text='',
        font='Arial',
        pos=(0, 0.25), draggable=False, height=0.15, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    slider_pain_adj = visual.Slider(win=win, name='slider_pain_adj',
        startValue=None, size=1.0, pos=[0,0], units=win.units,
        labels=["None","Mild","Moderate","Severe"],ticks=None, granularity=1,
        style='radio', styleTweaks=[], opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Noto Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-1, readOnly=False)
    # Run 'Begin Experiment' code from code_pain_helper
    # --- Slider Scaling ---
    slider_pain_width = aspect * 0.7   # 70% of screen width
    
    # Slider layout (height units)
    slider_pain_y      = -0.09                     # vertical position of the slider center
    slider_pain_height = 0.06                      # slider thickness/height
    t_pain_adj_continue = visual.TextStim(win=win, name='t_pain_adj_continue',
        text='Press the SPACEBAR to continue',
        font='Arial',
        pos=(0, -0.35), draggable=False, height=0.04, wrapWidth=comp_wrap_width_continue, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    key_resp_pain_adj = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "fixation" ---
    cross_fixation = visual.ShapeStim(
        win=win, name='cross_fixation', vertices='cross',units='height', 
        size=(0.2, 0.2),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "fixation" ---
    cross_fixation = visual.ShapeStim(
        win=win, name='cross_fixation', vertices='cross',units='height', 
        size=(0.2, 0.2),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "Pain_current" ---
    textCurrentPain = visual.TextStim(win=win, name='textCurrentPain',
        text='',
        font='Arial',
        pos=(0, 0.35), draggable=False, height=0.05, wrapWidth=1.5, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    slider_curr_pain_level = visual.Slider(win=win, name='slider_curr_pain_level',
        startValue=None, size=1.0, pos=[0,0], units=win.units,
        labels=["No Pain","Mild","Discomforting","Distressing","Horrible","Excrutiating"], ticks=(0, 1, 2, 3, 4, 5), granularity=1.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Noto Sans', labelHeight=0.033,
        flip=False, ori=0.0, depth=-1, readOnly=False)
    tb_continue_sm = visual.TextBox2(
         win, text='Please press the SPACEBAR to continue', placeholder='Type here...', font='Arial',
         ori=0.0, pos=(0, -0.4), draggable=False,      letterHeight=0.05,
         size=(0.5, 0.25), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='tb_continue_sm',
         depth=-2, autoLog=True,
    )
    key_resp_pain_curr = keyboard.Keyboard(deviceName='defaultKeyboard')
    # Run 'Begin Experiment' code from currPainHelper
    # Scale SAM properly
    # Window and coordinate setup
    screen_width, screen_height = win.size
    aspect = screen_width / screen_height  # width / height
    
    # In 'height' units:
    # - Full screen height = 1.0
    # - Full screen width  = aspect
    leftEdge = -aspect / 2
    rightEdge = aspect / 2
    topEdge = 0.5
    bottomEdge = -0.5
    
    
    # --- Generic Slider Scaling ---
    # Full-width slider across screen
    slider_gen_width = aspect * 0.7   # 70% of screen width
    slider_gen_width2 = aspect * 0.75
    
    # --- SAM image geometry (pixels) ---
    sam_px_width = 1168
    sam_px_height = 231
    sam_aspect = sam_px_width / sam_px_height  # ≈ 5.065
    
    # Pictogram/gap geometry (pixels)
    pictogram_px_width = 215
    gap_px_width = 23
    num_pictograms = 5
    num_gaps = num_pictograms - 1
    
    # Sanity check total width in px (info only)
    total_calc_px = num_pictograms * pictogram_px_width + num_gaps * gap_px_width
    # Note: total_calc_px ≈ 1175; close enough to 1170 given source image padding.
    
    # --- Layout decisions (height units) ---
    sam_height = 0.25                         # 25% of screen height
    sam_width  = sam_height * sam_aspect      # scaled width based on aspect
    sam_y      = 0.1                          # vertical position of the image
    
    # Slider layout (height units)
    slider_y      = -0.09                     # vertical position of the slider center
    slider_height = 0.06                      # slider thickness/height
    slider_px_width = sam_px_width - 215
    slider_width = sam_width * (slider_px_width / sam_px_width) # match slider width to image width
    
    # Label MinMax layout
    label_y = slider_y - 0.2 # slightly below the slider
    
    # Scale factor: px -> 'height' units within the image width
    scale_factor = sam_width / sam_px_width
    
    pictogram_w = pictogram_px_width * scale_factor
    gap_w       = gap_px_width * scale_factor
    
    # --- Compute 9 tick x-positions aligned to pictograms (1,3,5,7,9) and gaps (2,4,6,8) ---
    tick_positions = []
    # Start at the center of the first pictogram: left edge + half pictogram width
    x = -sam_width / 2 + pictogram_w / 2
    
    for i in range(9):
        tick_positions.append(x)
        # Advance: pictogram -> gap -> pictogram -> gap ...
        if i % 2 == 0:  # 0,2,4,6,8 (odd-numbered ticks under pictograms)
            x += pictogram_w / 2 + gap_w / 2
        else:           # 1,3,5,7 (even-numbered ticks under gaps)
            x += gap_w / 2 + pictogram_w / 2
    
    # Helpful mapping: tick index (0..8) -> value (1..9)
    tick_values = list(range(1, 10))
    slider_curr_pain_generic = visual.Slider(win=win, name='slider_curr_pain_generic',
        startValue=None, size=1.0, pos=[0,0], units=win.units,
        labels=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), ticks=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), granularity=1.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Noto Sans', labelHeight=0.045,
        flip=False, ori=0.0, depth=-5, readOnly=False)
    slider_curr_pain_conf = visual.Slider(win=win, name='slider_curr_pain_conf',
        startValue=None, size=1.0, pos=[0,0], units=win.units,
        labels=["Not at all \nconfident","Slightly \nconfident", "Moderately \nconfident", "Very \nconfident", "Extremely \nconfident"], ticks=(1, 2, 3, 4, 5), granularity=1.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Noto Sans', labelHeight=0.04,
        flip=False, ori=0.0, depth=-6, readOnly=False)
    textLabelMin = visual.TextStim(win=win, name='textLabelMin',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=0.5, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-7.0);
    textLabelMax = visual.TextStim(win=win, name='textLabelMax',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=0.5, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-8.0);
    
    # --- Initialize components for Routine "stateMeasure" ---
    t_sm_continue = visual.TextStim(win=win, name='t_sm_continue',
        text='Press the SPACEBAR to continue',
        font='Arial',
        pos=(0, -0.35), draggable=False, height=0.04, wrapWidth=comp_wrap_width_continue, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_sm = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "End" ---
    end_text = visual.TextStim(win=win, name='end_text',
        text='End of this secion.\n\nThank you.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
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
    # Run 'End Routine' code from code_trigger_setup
    # Clear out any residual triggers
    # Acts as both a marker and connection insurance.
    tb.send_event(value=1, name="experiment_start")
    tb.send_idle() # immediately return to baseline
    
    thisExp.nextEntry()
    # the Routine "experimentSetup" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "welcome" ---
    # create an object to store info about Routine welcome
    welcome = data.Routine(
        name='welcome',
        components=[t_welcome_body, t_welcome_continue, key_resp_welcome],
    )
    welcome.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    t_welcome_body.setText('Please relax and remain as still as possible while we record your brain activity.\n\nFollow the on-screen instructions to assess your body pain at present.')
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
    thisExp.nextEntry()
    # the Routine "welcome" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    state_measure_pretrial = data.TrialHandler2(
        name='state_measure_pretrial',
        nReps=1.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions(
        '../../shared/loop-templates/loopStateMeasure.csv', 
        selection='[0, 1, 2, 3, 4, 5, 6]'
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
            components=[t_sm_continue, key_resp_sm],
        )
        stateMeasure.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_sm_helper
        # --- Create a fresh slider for this trial ---
        allowContinue = False
        sm.begin_routine_from_category(rating_category)
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
            # If rated, allow spacebar to end the routine
            if sm.has_rating():
                allowContinue = True
            
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
        # Log rating, rt, and turn off AutoDraw
        sm.end_routine(currentLoop)
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
    
    # --- Prepare to start Routine "intro" ---
    # create an object to store info about Routine intro
    intro = data.Routine(
        name='intro',
        components=[t_intro, key_resp_intro],
    )
    intro.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_intro
    key_resp_intro.keys = []
    key_resp_intro.rt = []
    _key_resp_intro_allKeys = []
    # store start times for intro
    intro.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    intro.tStart = globalClock.getTime(format='float')
    intro.status = STARTED
    thisExp.addData('intro.started', intro.tStart)
    intro.maxDuration = None
    # keep track of which components have finished
    introComponents = intro.components
    for thisComponent in intro.components:
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
    
    # --- Run Routine "intro" ---
    thisExp.currentRoutine = intro
    intro.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *t_intro* updates
        
        # if t_intro is starting this frame...
        if t_intro.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            t_intro.frameNStart = frameN  # exact frame index
            t_intro.tStart = t  # local t and not account for scr refresh
            t_intro.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(t_intro, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 't_intro.started')
            # update status
            t_intro.status = STARTED
            t_intro.setAutoDraw(True)
        
        # if t_intro is active this frame...
        if t_intro.status == STARTED:
            # update params
            pass
        
        # *key_resp_intro* updates
        waitOnFlip = False
        
        # if key_resp_intro is starting this frame...
        if key_resp_intro.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_intro.frameNStart = frameN  # exact frame index
            key_resp_intro.tStart = t  # local t and not account for scr refresh
            key_resp_intro.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_intro, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_intro.started')
            # update status
            key_resp_intro.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_intro.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_intro.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_intro.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_intro.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_intro_allKeys.extend(theseKeys)
            if len(_key_resp_intro_allKeys):
                key_resp_intro.keys = _key_resp_intro_allKeys[-1].name  # just the last key pressed
                key_resp_intro.rt = _key_resp_intro_allKeys[-1].rt
                key_resp_intro.duration = _key_resp_intro_allKeys[-1].duration
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
                currentRoutine=intro,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            intro.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if intro.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in intro.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "intro" ---
    for thisComponent in intro.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for intro
    intro.tStop = globalClock.getTime(format='float')
    intro.tStopRefresh = tThisFlipGlobal
    thisExp.addData('intro.stopped', intro.tStop)
    # check responses
    if key_resp_intro.keys in ['', [], None]:  # No response was made
        key_resp_intro.keys = None
    thisExp.addData('key_resp_intro.keys',key_resp_intro.keys)
    if key_resp_intro.keys != None:  # we had a response
        thisExp.addData('key_resp_intro.rt', key_resp_intro.rt)
        thisExp.addData('key_resp_intro.duration', key_resp_intro.duration)
    thisExp.nextEntry()
    # the Routine "intro" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    pain_adjectives_loop = data.TrialHandler2(
        name='pain_adjectives_loop',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('pain_adjectives_01.xlsx'), 
        seed=1234, 
        isTrials=True, 
    )
    thisExp.addLoop(pain_adjectives_loop)  # add the loop to the experiment
    thisPain_adjectives_loop = pain_adjectives_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPain_adjectives_loop.rgb)
    if thisPain_adjectives_loop != None:
        for paramName in thisPain_adjectives_loop:
            globals()[paramName] = thisPain_adjectives_loop[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisPain_adjectives_loop in pain_adjectives_loop:
        pain_adjectives_loop.status = STARTED
        if hasattr(thisPain_adjectives_loop, 'status'):
            thisPain_adjectives_loop.status = STARTED
        currentLoop = pain_adjectives_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisPain_adjectives_loop.rgb)
        if thisPain_adjectives_loop != None:
            for paramName in thisPain_adjectives_loop:
                globals()[paramName] = thisPain_adjectives_loop[paramName]
        
        # --- Prepare to start Routine "fixation" ---
        # create an object to store info about Routine fixation
        fixation = data.Routine(
            name='fixation',
            components=[cross_fixation],
        )
        fixation.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for fixation
        fixation.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        fixation.tStart = globalClock.getTime(format='float')
        fixation.status = STARTED
        thisExp.addData('fixation.started', fixation.tStart)
        fixation.maxDuration = 1
        # keep track of which components have finished
        fixationComponents = fixation.components
        for thisComponent in fixation.components:
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
        
        # --- Run Routine "fixation" ---
        thisExp.currentRoutine = fixation
        fixation.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # if trial has changed, end Routine now
            if hasattr(thisPain_adjectives_loop, 'status') and thisPain_adjectives_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # is it time to end the Routine? (based on local clock)
            if tThisFlip > fixation.maxDuration-frameTolerance:
                fixation.maxDurationReached = True
                continueRoutine = False
            
            # *cross_fixation* updates
            
            # if cross_fixation is starting this frame...
            if cross_fixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                cross_fixation.frameNStart = frameN  # exact frame index
                cross_fixation.tStart = t  # local t and not account for scr refresh
                cross_fixation.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cross_fixation, 'tStartRefresh')  # time at next scr refresh
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
                if tThisFlipGlobal > cross_fixation.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    cross_fixation.tStop = t  # not accounting for scr refresh
                    cross_fixation.tStopRefresh = tThisFlipGlobal  # on global time
                    cross_fixation.frameNStop = frameN  # exact frame index
                    # update status
                    cross_fixation.status = FINISHED
                    cross_fixation.setAutoDraw(False)
            
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
                    currentRoutine=fixation,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                fixation.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if fixation.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in fixation.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "fixation" ---
        for thisComponent in fixation.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for fixation
        fixation.tStop = globalClock.getTime(format='float')
        fixation.tStopRefresh = tThisFlipGlobal
        thisExp.addData('fixation.stopped', fixation.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if fixation.maxDurationReached:
            routineTimer.addTime(-fixation.maxDuration)
        elif fixation.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        
        # --- Prepare to start Routine "pain_adjectives01" ---
        # create an object to store info about Routine pain_adjectives01
        pain_adjectives01 = data.Routine(
            name='pain_adjectives01',
            components=[t_pain_adj, slider_pain_adj, t_pain_adj_continue, key_resp_pain_adj],
        )
        pain_adjectives01.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        slider_pain_adj.reset()
        slider_pain_adj.setPos((0, slider_pain_y))
        slider_pain_adj.setSize((slider_pain_width, slider_pain_height))
        # Run 'Begin Routine' code from code_pain_helper
        allowContinue = False
        
        # create starting attributes for key_resp_pain_adj
        key_resp_pain_adj.keys = []
        key_resp_pain_adj.rt = []
        _key_resp_pain_adj_allKeys = []
        # store start times for pain_adjectives01
        pain_adjectives01.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        pain_adjectives01.tStart = globalClock.getTime(format='float')
        pain_adjectives01.status = STARTED
        thisExp.addData('pain_adjectives01.started', pain_adjectives01.tStart)
        pain_adjectives01.maxDuration = None
        # keep track of which components have finished
        pain_adjectives01Components = pain_adjectives01.components
        for thisComponent in pain_adjectives01.components:
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
        
        # --- Run Routine "pain_adjectives01" ---
        thisExp.currentRoutine = pain_adjectives01
        pain_adjectives01.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisPain_adjectives_loop, 'status') and thisPain_adjectives_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *t_pain_adj* updates
            
            # if t_pain_adj is starting this frame...
            if t_pain_adj.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                t_pain_adj.frameNStart = frameN  # exact frame index
                t_pain_adj.tStart = t  # local t and not account for scr refresh
                t_pain_adj.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(t_pain_adj, 'tStartRefresh')  # time at next scr refresh
                # update status
                t_pain_adj.status = STARTED
                t_pain_adj.setAutoDraw(True)
            
            # if t_pain_adj is active this frame...
            if t_pain_adj.status == STARTED:
                # update params
                t_pain_adj.setText(pain_adjs, log=False)
            
            # *slider_pain_adj* updates
            
            # if slider_pain_adj is starting this frame...
            if slider_pain_adj.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                slider_pain_adj.frameNStart = frameN  # exact frame index
                slider_pain_adj.tStart = t  # local t and not account for scr refresh
                slider_pain_adj.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(slider_pain_adj, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'slider_pain_adj.started')
                # update status
                slider_pain_adj.status = STARTED
                slider_pain_adj.setAutoDraw(True)
            
            # if slider_pain_adj is active this frame...
            if slider_pain_adj.status == STARTED:
                # update params
                pass
            # Run 'Each Frame' code from code_pain_helper
            # Check if either slider has been rated
            rated = (slider_pain_adj.getRating() is not None) 
            
            # If rated, allow spacebar to end the routine
            if rated:
                allowContinue = True
            else:
                allowContinue = False
            
            # *t_pain_adj_continue* updates
            
            # if t_pain_adj_continue is starting this frame...
            if t_pain_adj_continue.status == NOT_STARTED and allowContinue == True:
                # keep track of start time/frame for later
                t_pain_adj_continue.frameNStart = frameN  # exact frame index
                t_pain_adj_continue.tStart = t  # local t and not account for scr refresh
                t_pain_adj_continue.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(t_pain_adj_continue, 'tStartRefresh')  # time at next scr refresh
                # update status
                t_pain_adj_continue.status = STARTED
                t_pain_adj_continue.setAutoDraw(True)
            
            # if t_pain_adj_continue is active this frame...
            if t_pain_adj_continue.status == STARTED:
                # update params
                pass
            
            # *key_resp_pain_adj* updates
            waitOnFlip = False
            
            # if key_resp_pain_adj is starting this frame...
            if key_resp_pain_adj.status == NOT_STARTED and allowContinue == True:
                # keep track of start time/frame for later
                key_resp_pain_adj.frameNStart = frameN  # exact frame index
                key_resp_pain_adj.tStart = t  # local t and not account for scr refresh
                key_resp_pain_adj.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_pain_adj, 'tStartRefresh')  # time at next scr refresh
                # update status
                key_resp_pain_adj.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_pain_adj.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_pain_adj.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_pain_adj.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_pain_adj.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_pain_adj_allKeys.extend(theseKeys)
                if len(_key_resp_pain_adj_allKeys):
                    key_resp_pain_adj.keys = _key_resp_pain_adj_allKeys[-1].name  # just the last key pressed
                    key_resp_pain_adj.rt = _key_resp_pain_adj_allKeys[-1].rt
                    key_resp_pain_adj.duration = _key_resp_pain_adj_allKeys[-1].duration
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
                    currentRoutine=pain_adjectives01,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                pain_adjectives01.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if pain_adjectives01.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in pain_adjectives01.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "pain_adjectives01" ---
        for thisComponent in pain_adjectives01.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for pain_adjectives01
        pain_adjectives01.tStop = globalClock.getTime(format='float')
        pain_adjectives01.tStopRefresh = tThisFlipGlobal
        thisExp.addData('pain_adjectives01.stopped', pain_adjectives01.tStop)
        pain_adjectives_loop.addData('slider_pain_adj.response', slider_pain_adj.getRating())
        pain_adjectives_loop.addData('slider_pain_adj.rt', slider_pain_adj.getRT())
        # the Routine "pain_adjectives01" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisPain_adjectives_loop as finished
        if hasattr(thisPain_adjectives_loop, 'status'):
            thisPain_adjectives_loop.status = FINISHED
        # if awaiting a pause, pause now
        if pain_adjectives_loop.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            pain_adjectives_loop.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'pain_adjectives_loop'
    pain_adjectives_loop.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # set up handler to look after randomisation of conditions etc
    pain_adjectives_loop2 = data.TrialHandler2(
        name='pain_adjectives_loop2',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('pain_adjectives_01.xlsx'), 
        seed=1, 
        isTrials=True, 
    )
    thisExp.addLoop(pain_adjectives_loop2)  # add the loop to the experiment
    thisPain_adjectives_loop2 = pain_adjectives_loop2.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPain_adjectives_loop2.rgb)
    if thisPain_adjectives_loop2 != None:
        for paramName in thisPain_adjectives_loop2:
            globals()[paramName] = thisPain_adjectives_loop2[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisPain_adjectives_loop2 in pain_adjectives_loop2:
        pain_adjectives_loop2.status = STARTED
        if hasattr(thisPain_adjectives_loop2, 'status'):
            thisPain_adjectives_loop2.status = STARTED
        currentLoop = pain_adjectives_loop2
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisPain_adjectives_loop2.rgb)
        if thisPain_adjectives_loop2 != None:
            for paramName in thisPain_adjectives_loop2:
                globals()[paramName] = thisPain_adjectives_loop2[paramName]
        
        # --- Prepare to start Routine "fixation" ---
        # create an object to store info about Routine fixation
        fixation = data.Routine(
            name='fixation',
            components=[cross_fixation],
        )
        fixation.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for fixation
        fixation.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        fixation.tStart = globalClock.getTime(format='float')
        fixation.status = STARTED
        thisExp.addData('fixation.started', fixation.tStart)
        fixation.maxDuration = 1
        # keep track of which components have finished
        fixationComponents = fixation.components
        for thisComponent in fixation.components:
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
        
        # --- Run Routine "fixation" ---
        thisExp.currentRoutine = fixation
        fixation.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # if trial has changed, end Routine now
            if hasattr(thisPain_adjectives_loop2, 'status') and thisPain_adjectives_loop2.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # is it time to end the Routine? (based on local clock)
            if tThisFlip > fixation.maxDuration-frameTolerance:
                fixation.maxDurationReached = True
                continueRoutine = False
            
            # *cross_fixation* updates
            
            # if cross_fixation is starting this frame...
            if cross_fixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                cross_fixation.frameNStart = frameN  # exact frame index
                cross_fixation.tStart = t  # local t and not account for scr refresh
                cross_fixation.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cross_fixation, 'tStartRefresh')  # time at next scr refresh
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
                if tThisFlipGlobal > cross_fixation.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    cross_fixation.tStop = t  # not accounting for scr refresh
                    cross_fixation.tStopRefresh = tThisFlipGlobal  # on global time
                    cross_fixation.frameNStop = frameN  # exact frame index
                    # update status
                    cross_fixation.status = FINISHED
                    cross_fixation.setAutoDraw(False)
            
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
                    currentRoutine=fixation,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                fixation.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if fixation.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in fixation.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "fixation" ---
        for thisComponent in fixation.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for fixation
        fixation.tStop = globalClock.getTime(format='float')
        fixation.tStopRefresh = tThisFlipGlobal
        thisExp.addData('fixation.stopped', fixation.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if fixation.maxDurationReached:
            routineTimer.addTime(-fixation.maxDuration)
        elif fixation.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        # mark thisPain_adjectives_loop2 as finished
        if hasattr(thisPain_adjectives_loop2, 'status'):
            thisPain_adjectives_loop2.status = FINISHED
        # if awaiting a pause, pause now
        if pain_adjectives_loop2.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            pain_adjectives_loop2.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'pain_adjectives_loop2'
    pain_adjectives_loop2.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # set up handler to look after randomisation of conditions etc
    pain_current_loop = data.TrialHandler2(
        name='pain_current_loop',
        nReps=1.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions(
        'D:/Github/neuroflare-experiments/neuroflare/shared/loop-templates/loopSpontaneousPainExpected.xlsx', 
        selection='0:10'
    )
    , 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(pain_current_loop)  # add the loop to the experiment
    thisPain_current_loop = pain_current_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPain_current_loop.rgb)
    if thisPain_current_loop != None:
        for paramName in thisPain_current_loop:
            globals()[paramName] = thisPain_current_loop[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisPain_current_loop in pain_current_loop:
        pain_current_loop.status = STARTED
        if hasattr(thisPain_current_loop, 'status'):
            thisPain_current_loop.status = STARTED
        currentLoop = pain_current_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisPain_current_loop.rgb)
        if thisPain_current_loop != None:
            for paramName in thisPain_current_loop:
                globals()[paramName] = thisPain_current_loop[paramName]
        
        # --- Prepare to start Routine "fixation" ---
        # create an object to store info about Routine fixation
        fixation = data.Routine(
            name='fixation',
            components=[cross_fixation],
        )
        fixation.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for fixation
        fixation.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        fixation.tStart = globalClock.getTime(format='float')
        fixation.status = STARTED
        thisExp.addData('fixation.started', fixation.tStart)
        fixation.maxDuration = 1
        # keep track of which components have finished
        fixationComponents = fixation.components
        for thisComponent in fixation.components:
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
        
        # --- Run Routine "fixation" ---
        thisExp.currentRoutine = fixation
        fixation.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # if trial has changed, end Routine now
            if hasattr(thisPain_current_loop, 'status') and thisPain_current_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # is it time to end the Routine? (based on local clock)
            if tThisFlip > fixation.maxDuration-frameTolerance:
                fixation.maxDurationReached = True
                continueRoutine = False
            
            # *cross_fixation* updates
            
            # if cross_fixation is starting this frame...
            if cross_fixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                cross_fixation.frameNStart = frameN  # exact frame index
                cross_fixation.tStart = t  # local t and not account for scr refresh
                cross_fixation.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cross_fixation, 'tStartRefresh')  # time at next scr refresh
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
                if tThisFlipGlobal > cross_fixation.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    cross_fixation.tStop = t  # not accounting for scr refresh
                    cross_fixation.tStopRefresh = tThisFlipGlobal  # on global time
                    cross_fixation.frameNStop = frameN  # exact frame index
                    # update status
                    cross_fixation.status = FINISHED
                    cross_fixation.setAutoDraw(False)
            
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
                    currentRoutine=fixation,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                fixation.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if fixation.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in fixation.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "fixation" ---
        for thisComponent in fixation.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for fixation
        fixation.tStop = globalClock.getTime(format='float')
        fixation.tStopRefresh = tThisFlipGlobal
        thisExp.addData('fixation.stopped', fixation.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if fixation.maxDurationReached:
            routineTimer.addTime(-fixation.maxDuration)
        elif fixation.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        
        # --- Prepare to start Routine "Pain_current" ---
        # create an object to store info about Routine Pain_current
        Pain_current = data.Routine(
            name='Pain_current',
            components=[textCurrentPain, slider_curr_pain_level, tb_continue_sm, key_resp_pain_curr, slider_curr_pain_generic, slider_curr_pain_conf, textLabelMin, textLabelMax],
        )
        Pain_current.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        textCurrentPain.setText(rating_message)
        slider_curr_pain_level.reset()
        slider_curr_pain_level.setPos((0, 0))
        slider_curr_pain_level.setSize((slider_gen_width2, 0.1))
        tb_continue_sm.reset()
        # create starting attributes for key_resp_pain_curr
        key_resp_pain_curr.keys = []
        key_resp_pain_curr.rt = []
        _key_resp_pain_curr_allKeys = []
        # Run 'Begin Routine' code from currPainHelper
        allowContinue = False
        slider_curr_pain_generic.reset()
        slider_curr_pain_generic.setPos((0, 0))
        slider_curr_pain_generic.setSize((slider_gen_width, 0.1))
        slider_curr_pain_conf.reset()
        slider_curr_pain_conf.setPos((0, 0))
        slider_curr_pain_conf.setSize((slider_gen_width2, 0.1))
        textLabelMin.setPos((tick_positions[0]-0.075, label_y))
        textLabelMin.setText(rating_min_label)
        textLabelMax.setPos((tick_positions[-1], label_y))
        textLabelMax.setText(rating_max_label)
        # store start times for Pain_current
        Pain_current.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Pain_current.tStart = globalClock.getTime(format='float')
        Pain_current.status = STARTED
        thisExp.addData('Pain_current.started', Pain_current.tStart)
        Pain_current.maxDuration = None
        # keep track of which components have finished
        Pain_currentComponents = Pain_current.components
        for thisComponent in Pain_current.components:
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
        
        # --- Run Routine "Pain_current" ---
        thisExp.currentRoutine = Pain_current
        Pain_current.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisPain_current_loop, 'status') and thisPain_current_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *textCurrentPain* updates
            
            # if textCurrentPain is starting this frame...
            if textCurrentPain.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                textCurrentPain.frameNStart = frameN  # exact frame index
                textCurrentPain.tStart = t  # local t and not account for scr refresh
                textCurrentPain.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(textCurrentPain, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textCurrentPain.started')
                # update status
                textCurrentPain.status = STARTED
                textCurrentPain.setAutoDraw(True)
            
            # if textCurrentPain is active this frame...
            if textCurrentPain.status == STARTED:
                # update params
                pass
            
            # *slider_curr_pain_level* updates
            
            # if slider_curr_pain_level is starting this frame...
            if slider_curr_pain_level.status == NOT_STARTED and rating_type == 'currPainLevel':
                # keep track of start time/frame for later
                slider_curr_pain_level.frameNStart = frameN  # exact frame index
                slider_curr_pain_level.tStart = t  # local t and not account for scr refresh
                slider_curr_pain_level.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(slider_curr_pain_level, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'slider_curr_pain_level.started')
                # update status
                slider_curr_pain_level.status = STARTED
                slider_curr_pain_level.setAutoDraw(True)
            
            # if slider_curr_pain_level is active this frame...
            if slider_curr_pain_level.status == STARTED:
                # update params
                pass
            
            # if slider_curr_pain_level is stopping this frame...
            if slider_curr_pain_level.status == STARTED:
                if bool(rating_type != 'currPainLevel'):
                    # keep track of stop time/frame for later
                    slider_curr_pain_level.tStop = t  # not accounting for scr refresh
                    slider_curr_pain_level.tStopRefresh = tThisFlipGlobal  # on global time
                    slider_curr_pain_level.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'slider_curr_pain_level.stopped')
                    # update status
                    slider_curr_pain_level.status = FINISHED
                    slider_curr_pain_level.setAutoDraw(False)
            
            # *tb_continue_sm* updates
            
            # if tb_continue_sm is starting this frame...
            if tb_continue_sm.status == NOT_STARTED and allowContinue == True:
                # keep track of start time/frame for later
                tb_continue_sm.frameNStart = frameN  # exact frame index
                tb_continue_sm.tStart = t  # local t and not account for scr refresh
                tb_continue_sm.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(tb_continue_sm, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'tb_continue_sm.started')
                # update status
                tb_continue_sm.status = STARTED
                tb_continue_sm.setAutoDraw(True)
            
            # if tb_continue_sm is active this frame...
            if tb_continue_sm.status == STARTED:
                # update params
                pass
            
            # *key_resp_pain_curr* updates
            waitOnFlip = False
            
            # if key_resp_pain_curr is starting this frame...
            if key_resp_pain_curr.status == NOT_STARTED and allowContinue == True:
                # keep track of start time/frame for later
                key_resp_pain_curr.frameNStart = frameN  # exact frame index
                key_resp_pain_curr.tStart = t  # local t and not account for scr refresh
                key_resp_pain_curr.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_pain_curr, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_pain_curr.started')
                # update status
                key_resp_pain_curr.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_pain_curr.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_pain_curr.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_pain_curr.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_pain_curr.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_pain_curr_allKeys.extend(theseKeys)
                if len(_key_resp_pain_curr_allKeys):
                    key_resp_pain_curr.keys = _key_resp_pain_curr_allKeys[-1].name  # just the last key pressed
                    key_resp_pain_curr.rt = _key_resp_pain_curr_allKeys[-1].rt
                    key_resp_pain_curr.duration = _key_resp_pain_curr_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            # Run 'Each Frame' code from currPainHelper
            # Check if either slider has been rated
            rated = (slider_curr_pain_level.getRating() is not None) or (slider_curr_pain_generic.getRating() is not None) or (slider_curr_pain_conf.getRating() is not None)
            
            # If rated, allow spacebar to end the routine
            if rated:
                allowContinue = True
            else:
                allowContinue = False
            
            # *slider_curr_pain_generic* updates
            
            # if slider_curr_pain_generic is starting this frame...
            if slider_curr_pain_generic.status == NOT_STARTED and rating_type == 'generic':
                # keep track of start time/frame for later
                slider_curr_pain_generic.frameNStart = frameN  # exact frame index
                slider_curr_pain_generic.tStart = t  # local t and not account for scr refresh
                slider_curr_pain_generic.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(slider_curr_pain_generic, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'slider_curr_pain_generic.started')
                # update status
                slider_curr_pain_generic.status = STARTED
                slider_curr_pain_generic.setAutoDraw(True)
            
            # if slider_curr_pain_generic is active this frame...
            if slider_curr_pain_generic.status == STARTED:
                # update params
                pass
            
            # if slider_curr_pain_generic is stopping this frame...
            if slider_curr_pain_generic.status == STARTED:
                if bool(rating_type != 'generic'):
                    # keep track of stop time/frame for later
                    slider_curr_pain_generic.tStop = t  # not accounting for scr refresh
                    slider_curr_pain_generic.tStopRefresh = tThisFlipGlobal  # on global time
                    slider_curr_pain_generic.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'slider_curr_pain_generic.stopped')
                    # update status
                    slider_curr_pain_generic.status = FINISHED
                    slider_curr_pain_generic.setAutoDraw(False)
            
            # *slider_curr_pain_conf* updates
            
            # if slider_curr_pain_conf is starting this frame...
            if slider_curr_pain_conf.status == NOT_STARTED and rating_type == 'currPainConf':
                # keep track of start time/frame for later
                slider_curr_pain_conf.frameNStart = frameN  # exact frame index
                slider_curr_pain_conf.tStart = t  # local t and not account for scr refresh
                slider_curr_pain_conf.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(slider_curr_pain_conf, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'slider_curr_pain_conf.started')
                # update status
                slider_curr_pain_conf.status = STARTED
                slider_curr_pain_conf.setAutoDraw(True)
            
            # if slider_curr_pain_conf is active this frame...
            if slider_curr_pain_conf.status == STARTED:
                # update params
                pass
            
            # if slider_curr_pain_conf is stopping this frame...
            if slider_curr_pain_conf.status == STARTED:
                if bool(rating_type != 'currPainConf'):
                    # keep track of stop time/frame for later
                    slider_curr_pain_conf.tStop = t  # not accounting for scr refresh
                    slider_curr_pain_conf.tStopRefresh = tThisFlipGlobal  # on global time
                    slider_curr_pain_conf.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'slider_curr_pain_conf.stopped')
                    # update status
                    slider_curr_pain_conf.status = FINISHED
                    slider_curr_pain_conf.setAutoDraw(False)
            
            # *textLabelMin* updates
            
            # if textLabelMin is starting this frame...
            if textLabelMin.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                textLabelMin.frameNStart = frameN  # exact frame index
                textLabelMin.tStart = t  # local t and not account for scr refresh
                textLabelMin.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(textLabelMin, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textLabelMin.started')
                # update status
                textLabelMin.status = STARTED
                textLabelMin.setAutoDraw(True)
            
            # if textLabelMin is active this frame...
            if textLabelMin.status == STARTED:
                # update params
                pass
            
            # *textLabelMax* updates
            
            # if textLabelMax is starting this frame...
            if textLabelMax.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                textLabelMax.frameNStart = frameN  # exact frame index
                textLabelMax.tStart = t  # local t and not account for scr refresh
                textLabelMax.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(textLabelMax, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textLabelMax.started')
                # update status
                textLabelMax.status = STARTED
                textLabelMax.setAutoDraw(True)
            
            # if textLabelMax is active this frame...
            if textLabelMax.status == STARTED:
                # update params
                pass
            
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
                    currentRoutine=Pain_current,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                Pain_current.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if Pain_current.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in Pain_current.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Pain_current" ---
        for thisComponent in Pain_current.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Pain_current
        Pain_current.tStop = globalClock.getTime(format='float')
        Pain_current.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Pain_current.stopped', Pain_current.tStop)
        pain_current_loop.addData('slider_curr_pain_level.response', slider_curr_pain_level.getRating())
        pain_current_loop.addData('slider_curr_pain_level.rt', slider_curr_pain_level.getRT())
        # check responses
        if key_resp_pain_curr.keys in ['', [], None]:  # No response was made
            key_resp_pain_curr.keys = None
        pain_current_loop.addData('key_resp_pain_curr.keys',key_resp_pain_curr.keys)
        if key_resp_pain_curr.keys != None:  # we had a response
            pain_current_loop.addData('key_resp_pain_curr.rt', key_resp_pain_curr.rt)
            pain_current_loop.addData('key_resp_pain_curr.duration', key_resp_pain_curr.duration)
        pain_current_loop.addData('slider_curr_pain_generic.response', slider_curr_pain_generic.getRating())
        pain_current_loop.addData('slider_curr_pain_generic.rt', slider_curr_pain_generic.getRT())
        pain_current_loop.addData('slider_curr_pain_conf.response', slider_curr_pain_conf.getRating())
        pain_current_loop.addData('slider_curr_pain_conf.rt', slider_curr_pain_conf.getRT())
        # the Routine "Pain_current" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisPain_current_loop as finished
        if hasattr(thisPain_current_loop, 'status'):
            thisPain_current_loop.status = FINISHED
        # if awaiting a pause, pause now
        if pain_current_loop.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            pain_current_loop.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'pain_current_loop'
    pain_current_loop.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # set up handler to look after randomisation of conditions etc
    state_measure_end = data.TrialHandler2(
        name='state_measure_end',
        nReps=1.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions(
        '../../shared/loop-templates/loopStateMeasure.csv', 
        selection='[0, 1, 2, 3, 4, 5, 6]'
    )
    , 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(state_measure_end)  # add the loop to the experiment
    thisState_measure_end = state_measure_end.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisState_measure_end.rgb)
    if thisState_measure_end != None:
        for paramName in thisState_measure_end:
            globals()[paramName] = thisState_measure_end[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisState_measure_end in state_measure_end:
        state_measure_end.status = STARTED
        if hasattr(thisState_measure_end, 'status'):
            thisState_measure_end.status = STARTED
        currentLoop = state_measure_end
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisState_measure_end.rgb)
        if thisState_measure_end != None:
            for paramName in thisState_measure_end:
                globals()[paramName] = thisState_measure_end[paramName]
        
        # --- Prepare to start Routine "stateMeasure" ---
        # create an object to store info about Routine stateMeasure
        stateMeasure = data.Routine(
            name='stateMeasure',
            components=[t_sm_continue, key_resp_sm],
        )
        stateMeasure.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_sm_helper
        # --- Create a fresh slider for this trial ---
        allowContinue = False
        sm.begin_routine_from_category(rating_category)
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
            if hasattr(thisState_measure_end, 'status') and thisState_measure_end.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from code_sm_helper
            # If rated, allow spacebar to end the routine
            if sm.has_rating():
                allowContinue = True
            
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
        # Log rating, rt, and turn off AutoDraw
        sm.end_routine(currentLoop)
        # the Routine "stateMeasure" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisState_measure_end as finished
        if hasattr(thisState_measure_end, 'status'):
            thisState_measure_end.status = FINISHED
        # if awaiting a pause, pause now
        if state_measure_end.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            state_measure_end.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'state_measure_end'
    state_measure_end.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "End" ---
    # create an object to store info about Routine End
    End = data.Routine(
        name='End',
        components=[end_text],
    )
    End.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for End
    End.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    End.tStart = globalClock.getTime(format='float')
    End.status = STARTED
    thisExp.addData('End.started', End.tStart)
    End.maxDuration = None
    # keep track of which components have finished
    EndComponents = End.components
    for thisComponent in End.components:
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
    
    # --- Run Routine "End" ---
    thisExp.currentRoutine = End
    End.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *end_text* updates
        
        # if end_text is starting this frame...
        if end_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            end_text.frameNStart = frameN  # exact frame index
            end_text.tStart = t  # local t and not account for scr refresh
            end_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(end_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'end_text.started')
            # update status
            end_text.status = STARTED
            end_text.setAutoDraw(True)
        
        # if end_text is active this frame...
        if end_text.status == STARTED:
            # update params
            pass
        
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
                currentRoutine=End,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            End.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if End.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in End.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "End" ---
    for thisComponent in End.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for End
    End.tStop = globalClock.getTime(format='float')
    End.tStopRefresh = tThisFlipGlobal
    thisExp.addData('End.stopped', End.tStop)
    thisExp.nextEntry()
    # the Routine "End" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    # Run 'End Experiment' code from code_trigger_setup
    # Close connection and send reset byte.
    tb.end_experiment()
    
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
