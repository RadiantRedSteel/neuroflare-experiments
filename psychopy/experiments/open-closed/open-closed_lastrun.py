#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2025.1.1),
    on December 04, 2025, at 23:05
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
prefs.hardware['audioLib'] = 'ptb'
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

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2025.1.1'
expName = 'open-closed'  # from the Builder filename that created this script
expVersion = '1.0'
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
        originPath='D:\\Github\\neuroflare-experiments\\psychopy\\experiments\\open-closed\\open-closed_lastrun.py',
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
            winType='pyglet', allowGUI=True, allowStencil=False,
            monitor='testMonitor', color=(-0.5000, -0.5000, -0.5000), colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = (-0.5000, -0.5000, -0.5000)
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
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('key_resp_welcome') is None:
        # initialise key_resp_welcome
        key_resp_welcome = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_welcome',
        )
    if deviceManager.getDevice('key_resp_sm') is None:
        # initialise key_resp_sm
        key_resp_sm = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_sm',
        )
    if deviceManager.getDevice('key_resp_2') is None:
        # initialise key_resp_2
        key_resp_2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_2',
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
            backend='ioHub',
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
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
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
    
    # --- Initialize components for Routine "welcome" ---
    key_resp_welcome = keyboard.Keyboard(deviceName='key_resp_welcome')
    t_body = visual.TextStim(win=win, name='t_body',
        text='',
        font='Arial',
        units='height', pos=(0.0, 0.05), draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    t_continue = visual.TextStim(win=win, name='t_continue',
        text='Press the SPACEBAR to continue',
        font='Arial',
        units='height', pos=(0, -0.35), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "stateMeasure" ---
    slider_generic = visual.Slider(win=win, name='slider_generic',
        startValue=None, size=1.0, pos=(0, 0), units=win.units,
        labels=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), ticks=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), granularity=1.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Noto Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=0, readOnly=False)
    slider_SAM = visual.Slider(win=win, name='slider_SAM',
        startValue=None, size=1.0, pos=[0,0], units=win.units,
        labels=[1, 2, 3, 4, 5, 6, 7, 8, 9],ticks=None, granularity=1,
        style='radio', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Noto Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-1, readOnly=False)
    image_SAM = visual.ImageStim(
        win=win,
        name='image_SAM', units='height', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=1.0,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    t_sm_message = visual.TextStim(win=win, name='t_sm_message',
        text='',
        font='Arial',
        pos=(0, .35), draggable=False, height=0.07, wrapWidth=1.5, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    t_label_min = visual.TextStim(win=win, name='t_label_min',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=0.25, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    t_label_max = visual.TextStim(win=win, name='t_label_max',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=0.25, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    t_continue_sm = visual.TextStim(win=win, name='t_continue_sm',
        text='Press the SPACEBAR to continue',
        font='Arial',
        pos=(0, -0.35), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    # Run 'Begin Experiment' code from code_sm_helper
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
    slider_gen_width = aspect * 0.6   # 60% of screen width
    
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
    
    # Message Layout
    message_y = sam_y + 0.2
    
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
    
    key_resp_sm = keyboard.Keyboard(deviceName='key_resp_sm')
    
    # --- Initialize components for Routine "instruction" ---
    key_resp_2 = keyboard.Keyboard(deviceName='key_resp_2')
    t_instruction = visual.TextStim(win=win, name='t_instruction',
        text='',
        font='Arial',
        pos=(0.0, 0.05), draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    t_continue_2 = visual.TextStim(win=win, name='t_continue_2',
        text='Press the SPACEBAR to begin',
        font='Arial',
        pos=(0, -0.35), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "triggerStart" ---
    p_port_start = parallel.ParallelPort(address='0x0378')
    
    # --- Initialize components for Routine "fixation" ---
    cross_fixation = visual.ShapeStim(
        win=win, name='cross_fixation', vertices='cross',units='height', 
        size=(0.15, 0.15),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "triggerStop" ---
    p_port_stop = parallel.ParallelPort(address='0x0378')
    
    # --- Initialize components for Routine "stateMeasure" ---
    slider_generic = visual.Slider(win=win, name='slider_generic',
        startValue=None, size=1.0, pos=(0, 0), units=win.units,
        labels=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), ticks=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), granularity=1.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Noto Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=0, readOnly=False)
    slider_SAM = visual.Slider(win=win, name='slider_SAM',
        startValue=None, size=1.0, pos=[0,0], units=win.units,
        labels=[1, 2, 3, 4, 5, 6, 7, 8, 9],ticks=None, granularity=1,
        style='radio', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Noto Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-1, readOnly=False)
    image_SAM = visual.ImageStim(
        win=win,
        name='image_SAM', units='height', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=1.0,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    t_sm_message = visual.TextStim(win=win, name='t_sm_message',
        text='',
        font='Arial',
        pos=(0, .35), draggable=False, height=0.07, wrapWidth=1.5, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    t_label_min = visual.TextStim(win=win, name='t_label_min',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=0.25, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    t_label_max = visual.TextStim(win=win, name='t_label_max',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=0.25, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    t_continue_sm = visual.TextStim(win=win, name='t_continue_sm',
        text='Press the SPACEBAR to continue',
        font='Arial',
        pos=(0, -0.35), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    # Run 'Begin Experiment' code from code_sm_helper
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
    slider_gen_width = aspect * 0.6   # 60% of screen width
    
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
    
    # Message Layout
    message_y = sam_y + 0.2
    
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
    
    key_resp_sm = keyboard.Keyboard(deviceName='key_resp_sm')
    
    # --- Initialize components for Routine "goodbye" ---
    t_goodbye = visual.TextStim(win=win, name='t_goodbye',
        text='You have completed the trial. Thank you!',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
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
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # set up handler to look after randomisation of conditions etc
    intro_prompts = data.TrialHandler2(
        name='intro_prompts',
        nReps=1.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('../../shared/loop-templates/loopOpenClosedIntro.xlsx'), 
        seed=None, 
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
            components=[key_resp_welcome, t_body, t_continue],
        )
        welcome.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for key_resp_welcome
        key_resp_welcome.keys = []
        key_resp_welcome.rt = []
        _key_resp_welcome_allKeys = []
        t_body.setText(Message)
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
            
            # *key_resp_welcome* updates
            waitOnFlip = False
            
            # if key_resp_welcome is starting this frame...
            if key_resp_welcome.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_welcome.frameNStart = frameN  # exact frame index
                key_resp_welcome.tStart = t  # local t and not account for scr refresh
                key_resp_welcome.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_welcome, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_welcome.started')
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
            
            # *t_body* updates
            
            # if t_body is starting this frame...
            if t_body.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                t_body.frameNStart = frameN  # exact frame index
                t_body.tStart = t  # local t and not account for scr refresh
                t_body.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(t_body, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 't_body.started')
                # update status
                t_body.status = STARTED
                t_body.setAutoDraw(True)
            
            # if t_body is active this frame...
            if t_body.status == STARTED:
                # update params
                pass
            
            # *t_continue* updates
            
            # if t_continue is starting this frame...
            if t_continue.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                t_continue.frameNStart = frameN  # exact frame index
                t_continue.tStart = t  # local t and not account for scr refresh
                t_continue.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(t_continue, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 't_continue.started')
                # update status
                t_continue.status = STARTED
                t_continue.setAutoDraw(True)
            
            # if t_continue is active this frame...
            if t_continue.status == STARTED:
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
                    currentRoutine=welcome,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                welcome.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
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
        # check responses
        if key_resp_welcome.keys in ['', [], None]:  # No response was made
            key_resp_welcome.keys = None
        intro_prompts.addData('key_resp_welcome.keys',key_resp_welcome.keys)
        if key_resp_welcome.keys != None:  # we had a response
            intro_prompts.addData('key_resp_welcome.rt', key_resp_welcome.rt)
            intro_prompts.addData('key_resp_welcome.duration', key_resp_welcome.duration)
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
        trialList=data.importConditions('../../shared/loop-templates/loopStateMeasure.xlsx'), 
        seed=None, 
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
            components=[slider_generic, slider_SAM, image_SAM, t_sm_message, t_label_min, t_label_max, t_continue_sm, key_resp_sm],
        )
        stateMeasure.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        slider_generic.reset()
        slider_generic.setSize((slider_gen_width, 0.1))
        slider_SAM.reset()
        slider_SAM.setPos((0, slider_y))
        slider_SAM.setSize((slider_width, slider_height))
        image_SAM.setPos((0, sam_y))
        image_SAM.setSize((sam_width, sam_height))
        image_SAM.setImage(picture_path)
        t_sm_message.setText(rating_message)
        t_label_min.setPos((tick_positions[0], label_y))
        t_label_min.setText(rating_min_label)
        t_label_max.setPos((tick_positions[-1], label_y))
        t_label_max.setText(rating_max_label)
        # Run 'Begin Routine' code from code_sm_helper
        allowContinue = False
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
            
            # *slider_generic* updates
            
            # if slider_generic is starting this frame...
            if slider_generic.status == NOT_STARTED and rating_type == 'Generic':
                # keep track of start time/frame for later
                slider_generic.frameNStart = frameN  # exact frame index
                slider_generic.tStart = t  # local t and not account for scr refresh
                slider_generic.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(slider_generic, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'slider_generic.started')
                # update status
                slider_generic.status = STARTED
                slider_generic.setAutoDraw(True)
            
            # if slider_generic is active this frame...
            if slider_generic.status == STARTED:
                # update params
                pass
            
            # if slider_generic is stopping this frame...
            if slider_generic.status == STARTED:
                if bool(rating_type != 'Generic'):
                    # keep track of stop time/frame for later
                    slider_generic.tStop = t  # not accounting for scr refresh
                    slider_generic.tStopRefresh = tThisFlipGlobal  # on global time
                    slider_generic.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'slider_generic.stopped')
                    # update status
                    slider_generic.status = FINISHED
                    slider_generic.setAutoDraw(False)
            
            # *slider_SAM* updates
            
            # if slider_SAM is starting this frame...
            if slider_SAM.status == NOT_STARTED and rating_type == 'SAM':
                # keep track of start time/frame for later
                slider_SAM.frameNStart = frameN  # exact frame index
                slider_SAM.tStart = t  # local t and not account for scr refresh
                slider_SAM.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(slider_SAM, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'slider_SAM.started')
                # update status
                slider_SAM.status = STARTED
                slider_SAM.setAutoDraw(True)
            
            # if slider_SAM is active this frame...
            if slider_SAM.status == STARTED:
                # update params
                pass
            
            # if slider_SAM is stopping this frame...
            if slider_SAM.status == STARTED:
                if bool(rating_type != 'SAM'):
                    # keep track of stop time/frame for later
                    slider_SAM.tStop = t  # not accounting for scr refresh
                    slider_SAM.tStopRefresh = tThisFlipGlobal  # on global time
                    slider_SAM.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'slider_SAM.stopped')
                    # update status
                    slider_SAM.status = FINISHED
                    slider_SAM.setAutoDraw(False)
            
            # *image_SAM* updates
            
            # if image_SAM is starting this frame...
            if image_SAM.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                image_SAM.frameNStart = frameN  # exact frame index
                image_SAM.tStart = t  # local t and not account for scr refresh
                image_SAM.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_SAM, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_SAM.started')
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
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 't_sm_message.started')
                # update status
                t_sm_message.status = STARTED
                t_sm_message.setAutoDraw(True)
            
            # if t_sm_message is active this frame...
            if t_sm_message.status == STARTED:
                # update params
                pass
            
            # *t_label_min* updates
            
            # if t_label_min is starting this frame...
            if t_label_min.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                t_label_min.frameNStart = frameN  # exact frame index
                t_label_min.tStart = t  # local t and not account for scr refresh
                t_label_min.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(t_label_min, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 't_label_min.started')
                # update status
                t_label_min.status = STARTED
                t_label_min.setAutoDraw(True)
            
            # if t_label_min is active this frame...
            if t_label_min.status == STARTED:
                # update params
                pass
            
            # *t_label_max* updates
            
            # if t_label_max is starting this frame...
            if t_label_max.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                t_label_max.frameNStart = frameN  # exact frame index
                t_label_max.tStart = t  # local t and not account for scr refresh
                t_label_max.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(t_label_max, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 't_label_max.started')
                # update status
                t_label_max.status = STARTED
                t_label_max.setAutoDraw(True)
            
            # if t_label_max is active this frame...
            if t_label_max.status == STARTED:
                # update params
                pass
            
            # *t_continue_sm* updates
            
            # if t_continue_sm is starting this frame...
            if t_continue_sm.status == NOT_STARTED and allowContinue == True:
                # keep track of start time/frame for later
                t_continue_sm.frameNStart = frameN  # exact frame index
                t_continue_sm.tStart = t  # local t and not account for scr refresh
                t_continue_sm.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(t_continue_sm, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 't_continue_sm.started')
                # update status
                t_continue_sm.status = STARTED
                t_continue_sm.setAutoDraw(True)
            
            # if t_continue_sm is active this frame...
            if t_continue_sm.status == STARTED:
                # update params
                pass
            # Run 'Each Frame' code from code_sm_helper
            # Check if either slider has been rated
            rated = (slider_generic.getRating() is not None) or (slider_SAM.getRating() is not None)
            
            # If rated, allow spacebar to end the routine
            if rated:
                allowContinue = True
            else:
                allowContinue = False
            
            
            # *key_resp_sm* updates
            waitOnFlip = False
            
            # if key_resp_sm is starting this frame...
            if key_resp_sm.status == NOT_STARTED and allowContinue == True:
                # keep track of start time/frame for later
                key_resp_sm.frameNStart = frameN  # exact frame index
                key_resp_sm.tStart = t  # local t and not account for scr refresh
                key_resp_sm.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_sm, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_sm.started')
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
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                stateMeasure.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
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
        state_measure_pretrial.addData('slider_generic.response', slider_generic.getRating())
        state_measure_pretrial.addData('slider_generic.rt', slider_generic.getRT())
        state_measure_pretrial.addData('slider_SAM.response', slider_SAM.getRating())
        state_measure_pretrial.addData('slider_SAM.rt', slider_SAM.getRT())
        # check responses
        if key_resp_sm.keys in ['', [], None]:  # No response was made
            key_resp_sm.keys = None
        state_measure_pretrial.addData('key_resp_sm.keys',key_resp_sm.keys)
        if key_resp_sm.keys != None:  # we had a response
            state_measure_pretrial.addData('key_resp_sm.rt', key_resp_sm.rt)
            state_measure_pretrial.addData('key_resp_sm.duration', key_resp_sm.duration)
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
    
    # set up handler to look after randomisation of conditions etc
    openclosed_trials = data.TrialHandler2(
        name='openclosed_trials',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('../../shared/loop-templates/loopOpenClosedTrial.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(openclosed_trials)  # add the loop to the experiment
    thisOpenclosed_trial = openclosed_trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisOpenclosed_trial.rgb)
    if thisOpenclosed_trial != None:
        for paramName in thisOpenclosed_trial:
            globals()[paramName] = thisOpenclosed_trial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisOpenclosed_trial in openclosed_trials:
        openclosed_trials.status = STARTED
        if hasattr(thisOpenclosed_trial, 'status'):
            thisOpenclosed_trial.status = STARTED
        currentLoop = openclosed_trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisOpenclosed_trial.rgb)
        if thisOpenclosed_trial != None:
            for paramName in thisOpenclosed_trial:
                globals()[paramName] = thisOpenclosed_trial[paramName]
        
        # --- Prepare to start Routine "instruction" ---
        # create an object to store info about Routine instruction
        instruction = data.Routine(
            name='instruction',
            components=[key_resp_2, t_instruction, t_continue_2],
        )
        instruction.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for key_resp_2
        key_resp_2.keys = []
        key_resp_2.rt = []
        _key_resp_2_allKeys = []
        t_instruction.setText(Prompt
        )
        # store start times for instruction
        instruction.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        instruction.tStart = globalClock.getTime(format='float')
        instruction.status = STARTED
        thisExp.addData('instruction.started', instruction.tStart)
        instruction.maxDuration = None
        # keep track of which components have finished
        instructionComponents = instruction.components
        for thisComponent in instruction.components:
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
        
        # --- Run Routine "instruction" ---
        instruction.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisOpenclosed_trial, 'status') and thisOpenclosed_trial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *key_resp_2* updates
            waitOnFlip = False
            
            # if key_resp_2 is starting this frame...
            if key_resp_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_2.frameNStart = frameN  # exact frame index
                key_resp_2.tStart = t  # local t and not account for scr refresh
                key_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_2.started')
                # update status
                key_resp_2.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_2.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_2.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_2.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_2_allKeys.extend(theseKeys)
                if len(_key_resp_2_allKeys):
                    key_resp_2.keys = _key_resp_2_allKeys[-1].name  # just the last key pressed
                    key_resp_2.rt = _key_resp_2_allKeys[-1].rt
                    key_resp_2.duration = _key_resp_2_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *t_instruction* updates
            
            # if t_instruction is starting this frame...
            if t_instruction.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                t_instruction.frameNStart = frameN  # exact frame index
                t_instruction.tStart = t  # local t and not account for scr refresh
                t_instruction.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(t_instruction, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 't_instruction.started')
                # update status
                t_instruction.status = STARTED
                t_instruction.setAutoDraw(True)
            
            # if t_instruction is active this frame...
            if t_instruction.status == STARTED:
                # update params
                pass
            
            # *t_continue_2* updates
            
            # if t_continue_2 is starting this frame...
            if t_continue_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                t_continue_2.frameNStart = frameN  # exact frame index
                t_continue_2.tStart = t  # local t and not account for scr refresh
                t_continue_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(t_continue_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 't_continue_2.started')
                # update status
                t_continue_2.status = STARTED
                t_continue_2.setAutoDraw(True)
            
            # if t_continue_2 is active this frame...
            if t_continue_2.status == STARTED:
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
                    currentRoutine=instruction,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                instruction.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in instruction.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "instruction" ---
        for thisComponent in instruction.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for instruction
        instruction.tStop = globalClock.getTime(format='float')
        instruction.tStopRefresh = tThisFlipGlobal
        thisExp.addData('instruction.stopped', instruction.tStop)
        # check responses
        if key_resp_2.keys in ['', [], None]:  # No response was made
            key_resp_2.keys = None
        openclosed_trials.addData('key_resp_2.keys',key_resp_2.keys)
        if key_resp_2.keys != None:  # we had a response
            openclosed_trials.addData('key_resp_2.rt', key_resp_2.rt)
            openclosed_trials.addData('key_resp_2.duration', key_resp_2.duration)
        # the Routine "instruction" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "triggerStart" ---
        # create an object to store info about Routine triggerStart
        triggerStart = data.Routine(
            name='triggerStart',
            components=[p_port_start],
        )
        triggerStart.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for triggerStart
        triggerStart.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        triggerStart.tStart = globalClock.getTime(format='float')
        triggerStart.status = STARTED
        thisExp.addData('triggerStart.started', triggerStart.tStart)
        triggerStart.maxDuration = None
        # keep track of which components have finished
        triggerStartComponents = triggerStart.components
        for thisComponent in triggerStart.components:
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
        
        # --- Run Routine "triggerStart" ---
        triggerStart.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # if trial has changed, end Routine now
            if hasattr(thisOpenclosed_trial, 'status') and thisOpenclosed_trial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # *p_port_start* updates
            
            # if p_port_start is starting this frame...
            if p_port_start.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_port_start.frameNStart = frameN  # exact frame index
                p_port_start.tStart = t  # local t and not account for scr refresh
                p_port_start.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_port_start, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_port_start.started')
                # update status
                p_port_start.status = STARTED
                p_port_start.status = STARTED
                win.callOnFlip(p_port_start.setData, int(1))
            
            # if p_port_start is stopping this frame...
            if p_port_start.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > p_port_start.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    p_port_start.tStop = t  # not accounting for scr refresh
                    p_port_start.tStopRefresh = tThisFlipGlobal  # on global time
                    p_port_start.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'p_port_start.stopped')
                    # update status
                    p_port_start.status = FINISHED
                    win.callOnFlip(p_port_start.setData, int(0))
            
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
                    currentRoutine=triggerStart,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                triggerStart.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in triggerStart.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "triggerStart" ---
        for thisComponent in triggerStart.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for triggerStart
        triggerStart.tStop = globalClock.getTime(format='float')
        triggerStart.tStopRefresh = tThisFlipGlobal
        thisExp.addData('triggerStart.stopped', triggerStart.tStop)
        if p_port_start.status == STARTED:
            win.callOnFlip(p_port_start.setData, int(0))
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if triggerStart.maxDurationReached:
            routineTimer.addTime(-triggerStart.maxDuration)
        elif triggerStart.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        
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
        fixation.maxDuration = None
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
        fixation.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 300.0:
            # if trial has changed, end Routine now
            if hasattr(thisOpenclosed_trial, 'status') and thisOpenclosed_trial.status == STOPPING:
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
                if tThisFlipGlobal > cross_fixation.tStartRefresh + 300-frameTolerance:
                    # keep track of stop time/frame for later
                    cross_fixation.tStop = t  # not accounting for scr refresh
                    cross_fixation.tStopRefresh = tThisFlipGlobal  # on global time
                    cross_fixation.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cross_fixation.stopped')
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
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                fixation.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
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
            routineTimer.addTime(-300.000000)
        
        # --- Prepare to start Routine "triggerStop" ---
        # create an object to store info about Routine triggerStop
        triggerStop = data.Routine(
            name='triggerStop',
            components=[p_port_stop],
        )
        triggerStop.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for triggerStop
        triggerStop.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        triggerStop.tStart = globalClock.getTime(format='float')
        triggerStop.status = STARTED
        thisExp.addData('triggerStop.started', triggerStop.tStart)
        triggerStop.maxDuration = None
        # keep track of which components have finished
        triggerStopComponents = triggerStop.components
        for thisComponent in triggerStop.components:
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
        
        # --- Run Routine "triggerStop" ---
        triggerStop.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # if trial has changed, end Routine now
            if hasattr(thisOpenclosed_trial, 'status') and thisOpenclosed_trial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # *p_port_stop* updates
            
            # if p_port_stop is starting this frame...
            if p_port_stop.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_port_stop.frameNStart = frameN  # exact frame index
                p_port_stop.tStart = t  # local t and not account for scr refresh
                p_port_stop.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_port_stop, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_port_stop.started')
                # update status
                p_port_stop.status = STARTED
                p_port_stop.status = STARTED
                win.callOnFlip(p_port_stop.setData, int(2))
            
            # if p_port_stop is stopping this frame...
            if p_port_stop.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > p_port_stop.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    p_port_stop.tStop = t  # not accounting for scr refresh
                    p_port_stop.tStopRefresh = tThisFlipGlobal  # on global time
                    p_port_stop.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'p_port_stop.stopped')
                    # update status
                    p_port_stop.status = FINISHED
                    win.callOnFlip(p_port_stop.setData, int(0))
            
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
                    currentRoutine=triggerStop,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                triggerStop.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in triggerStop.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "triggerStop" ---
        for thisComponent in triggerStop.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for triggerStop
        triggerStop.tStop = globalClock.getTime(format='float')
        triggerStop.tStopRefresh = tThisFlipGlobal
        thisExp.addData('triggerStop.stopped', triggerStop.tStop)
        if p_port_stop.status == STARTED:
            win.callOnFlip(p_port_stop.setData, int(0))
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if triggerStop.maxDurationReached:
            routineTimer.addTime(-triggerStop.maxDuration)
        elif triggerStop.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        
        # set up handler to look after randomisation of conditions etc
        state_measure_trials = data.TrialHandler2(
            name='state_measure_trials',
            nReps=1.0, 
            method='sequential', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions('../../shared/loop-templates/loopStateMeasure.xlsx'), 
            seed=None, 
        )
        thisExp.addLoop(state_measure_trials)  # add the loop to the experiment
        thisState_measure_trial = state_measure_trials.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisState_measure_trial.rgb)
        if thisState_measure_trial != None:
            for paramName in thisState_measure_trial:
                globals()[paramName] = thisState_measure_trial[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisState_measure_trial in state_measure_trials:
            state_measure_trials.status = STARTED
            if hasattr(thisState_measure_trial, 'status'):
                thisState_measure_trial.status = STARTED
            currentLoop = state_measure_trials
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisState_measure_trial.rgb)
            if thisState_measure_trial != None:
                for paramName in thisState_measure_trial:
                    globals()[paramName] = thisState_measure_trial[paramName]
            
            # --- Prepare to start Routine "stateMeasure" ---
            # create an object to store info about Routine stateMeasure
            stateMeasure = data.Routine(
                name='stateMeasure',
                components=[slider_generic, slider_SAM, image_SAM, t_sm_message, t_label_min, t_label_max, t_continue_sm, key_resp_sm],
            )
            stateMeasure.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            slider_generic.reset()
            slider_generic.setSize((slider_gen_width, 0.1))
            slider_SAM.reset()
            slider_SAM.setPos((0, slider_y))
            slider_SAM.setSize((slider_width, slider_height))
            image_SAM.setPos((0, sam_y))
            image_SAM.setSize((sam_width, sam_height))
            image_SAM.setImage(picture_path)
            t_sm_message.setText(rating_message)
            t_label_min.setPos((tick_positions[0], label_y))
            t_label_min.setText(rating_min_label)
            t_label_max.setPos((tick_positions[-1], label_y))
            t_label_max.setText(rating_max_label)
            # Run 'Begin Routine' code from code_sm_helper
            allowContinue = False
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
            stateMeasure.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisState_measure_trial, 'status') and thisState_measure_trial.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *slider_generic* updates
                
                # if slider_generic is starting this frame...
                if slider_generic.status == NOT_STARTED and rating_type == 'Generic':
                    # keep track of start time/frame for later
                    slider_generic.frameNStart = frameN  # exact frame index
                    slider_generic.tStart = t  # local t and not account for scr refresh
                    slider_generic.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(slider_generic, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'slider_generic.started')
                    # update status
                    slider_generic.status = STARTED
                    slider_generic.setAutoDraw(True)
                
                # if slider_generic is active this frame...
                if slider_generic.status == STARTED:
                    # update params
                    pass
                
                # if slider_generic is stopping this frame...
                if slider_generic.status == STARTED:
                    if bool(rating_type != 'Generic'):
                        # keep track of stop time/frame for later
                        slider_generic.tStop = t  # not accounting for scr refresh
                        slider_generic.tStopRefresh = tThisFlipGlobal  # on global time
                        slider_generic.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'slider_generic.stopped')
                        # update status
                        slider_generic.status = FINISHED
                        slider_generic.setAutoDraw(False)
                
                # *slider_SAM* updates
                
                # if slider_SAM is starting this frame...
                if slider_SAM.status == NOT_STARTED and rating_type == 'SAM':
                    # keep track of start time/frame for later
                    slider_SAM.frameNStart = frameN  # exact frame index
                    slider_SAM.tStart = t  # local t and not account for scr refresh
                    slider_SAM.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(slider_SAM, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'slider_SAM.started')
                    # update status
                    slider_SAM.status = STARTED
                    slider_SAM.setAutoDraw(True)
                
                # if slider_SAM is active this frame...
                if slider_SAM.status == STARTED:
                    # update params
                    pass
                
                # if slider_SAM is stopping this frame...
                if slider_SAM.status == STARTED:
                    if bool(rating_type != 'SAM'):
                        # keep track of stop time/frame for later
                        slider_SAM.tStop = t  # not accounting for scr refresh
                        slider_SAM.tStopRefresh = tThisFlipGlobal  # on global time
                        slider_SAM.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'slider_SAM.stopped')
                        # update status
                        slider_SAM.status = FINISHED
                        slider_SAM.setAutoDraw(False)
                
                # *image_SAM* updates
                
                # if image_SAM is starting this frame...
                if image_SAM.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image_SAM.frameNStart = frameN  # exact frame index
                    image_SAM.tStart = t  # local t and not account for scr refresh
                    image_SAM.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_SAM, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_SAM.started')
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
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 't_sm_message.started')
                    # update status
                    t_sm_message.status = STARTED
                    t_sm_message.setAutoDraw(True)
                
                # if t_sm_message is active this frame...
                if t_sm_message.status == STARTED:
                    # update params
                    pass
                
                # *t_label_min* updates
                
                # if t_label_min is starting this frame...
                if t_label_min.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    t_label_min.frameNStart = frameN  # exact frame index
                    t_label_min.tStart = t  # local t and not account for scr refresh
                    t_label_min.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(t_label_min, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 't_label_min.started')
                    # update status
                    t_label_min.status = STARTED
                    t_label_min.setAutoDraw(True)
                
                # if t_label_min is active this frame...
                if t_label_min.status == STARTED:
                    # update params
                    pass
                
                # *t_label_max* updates
                
                # if t_label_max is starting this frame...
                if t_label_max.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    t_label_max.frameNStart = frameN  # exact frame index
                    t_label_max.tStart = t  # local t and not account for scr refresh
                    t_label_max.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(t_label_max, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 't_label_max.started')
                    # update status
                    t_label_max.status = STARTED
                    t_label_max.setAutoDraw(True)
                
                # if t_label_max is active this frame...
                if t_label_max.status == STARTED:
                    # update params
                    pass
                
                # *t_continue_sm* updates
                
                # if t_continue_sm is starting this frame...
                if t_continue_sm.status == NOT_STARTED and allowContinue == True:
                    # keep track of start time/frame for later
                    t_continue_sm.frameNStart = frameN  # exact frame index
                    t_continue_sm.tStart = t  # local t and not account for scr refresh
                    t_continue_sm.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(t_continue_sm, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 't_continue_sm.started')
                    # update status
                    t_continue_sm.status = STARTED
                    t_continue_sm.setAutoDraw(True)
                
                # if t_continue_sm is active this frame...
                if t_continue_sm.status == STARTED:
                    # update params
                    pass
                # Run 'Each Frame' code from code_sm_helper
                # Check if either slider has been rated
                rated = (slider_generic.getRating() is not None) or (slider_SAM.getRating() is not None)
                
                # If rated, allow spacebar to end the routine
                if rated:
                    allowContinue = True
                else:
                    allowContinue = False
                
                
                # *key_resp_sm* updates
                waitOnFlip = False
                
                # if key_resp_sm is starting this frame...
                if key_resp_sm.status == NOT_STARTED and allowContinue == True:
                    # keep track of start time/frame for later
                    key_resp_sm.frameNStart = frameN  # exact frame index
                    key_resp_sm.tStart = t  # local t and not account for scr refresh
                    key_resp_sm.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp_sm, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp_sm.started')
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
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    stateMeasure.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
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
            state_measure_trials.addData('slider_generic.response', slider_generic.getRating())
            state_measure_trials.addData('slider_generic.rt', slider_generic.getRT())
            state_measure_trials.addData('slider_SAM.response', slider_SAM.getRating())
            state_measure_trials.addData('slider_SAM.rt', slider_SAM.getRT())
            # check responses
            if key_resp_sm.keys in ['', [], None]:  # No response was made
                key_resp_sm.keys = None
            state_measure_trials.addData('key_resp_sm.keys',key_resp_sm.keys)
            if key_resp_sm.keys != None:  # we had a response
                state_measure_trials.addData('key_resp_sm.rt', key_resp_sm.rt)
                state_measure_trials.addData('key_resp_sm.duration', key_resp_sm.duration)
            # the Routine "stateMeasure" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            # mark thisState_measure_trial as finished
            if hasattr(thisState_measure_trial, 'status'):
                thisState_measure_trial.status = FINISHED
            # if awaiting a pause, pause now
            if state_measure_trials.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                state_measure_trials.status = STARTED
            thisExp.nextEntry()
            
        # completed 1.0 repeats of 'state_measure_trials'
        state_measure_trials.status = FINISHED
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # get names of stimulus parameters
        if state_measure_trials.trialList in ([], [None], None):
            params = []
        else:
            params = state_measure_trials.trialList[0].keys()
        # save data for this loop
        state_measure_trials.saveAsExcel(filename + '.xlsx', sheetName='state_measure_trials',
            stimOut=params,
            dataOut=['n','all_mean','all_std', 'all_raw'])
        # mark thisOpenclosed_trial as finished
        if hasattr(thisOpenclosed_trial, 'status'):
            thisOpenclosed_trial.status = FINISHED
        # if awaiting a pause, pause now
        if openclosed_trials.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            openclosed_trials.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'openclosed_trials'
    openclosed_trials.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    # get names of stimulus parameters
    if openclosed_trials.trialList in ([], [None], None):
        params = []
    else:
        params = openclosed_trials.trialList[0].keys()
    # save data for this loop
    openclosed_trials.saveAsExcel(filename + '.xlsx', sheetName='openclosed_trials',
        stimOut=params,
        dataOut=['n','all_mean','all_std', 'all_raw'])
    
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
    goodbye.maxDuration = None
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
    goodbye.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 4.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *t_goodbye* updates
        
        # if t_goodbye is starting this frame...
        if t_goodbye.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            t_goodbye.frameNStart = frameN  # exact frame index
            t_goodbye.tStart = t  # local t and not account for scr refresh
            t_goodbye.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(t_goodbye, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 't_goodbye.started')
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
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 't_goodbye.stopped')
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
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            goodbye.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
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
