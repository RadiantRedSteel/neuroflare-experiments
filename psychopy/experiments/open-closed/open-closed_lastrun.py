#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2025.1.1),
    on November 25, 2025, at 02:25
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
expVersion = ''
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
            logging.getLevel('warning')
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
    if deviceManager.getDevice('key_resp_state_measure') is None:
        # initialise key_resp_state_measure
        key_resp_state_measure = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_state_measure',
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
    tb_body = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Arial',
         ori=0.0, pos=(0.0, 0.05), draggable=False,      letterHeight=0.07,
         size=(1.5, 0.5), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='tb_body',
         depth=0, autoLog=True,
    )
    tb_continue = visual.TextBox2(
         win, text='Press the SPACEBAR to continue', placeholder='Type here...', font='Arial',
         ori=0.0, pos=(0, -0.4), draggable=False,      letterHeight=0.03,
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
         name='tb_continue',
         depth=-1, autoLog=True,
    )
    key_resp_welcome = keyboard.Keyboard(deviceName='key_resp_welcome')
    
    # --- Initialize components for Routine "stateMeasure" ---
    sliderGeneric = visual.Slider(win=win, name='sliderGeneric',
        startValue=None, size=1.0, pos=(0, 0), units=win.units,
        labels=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), ticks=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), granularity=1.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Noto Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=0, readOnly=False)
    sliderSAM = visual.Slider(win=win, name='sliderSAM',
        startValue=None, size=1.0, pos=[0,0], units=win.units,
        labels=[1, 2, 3, 4, 5, 6, 7, 8, 9], ticks=(1, 2, 3, 4, 5, 6, 7, 8, 9), granularity=1.0,
        style='scrollbar', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Noto Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-1, readOnly=False)
    imageSAM = visual.ImageStim(
        win=win,
        name='imageSAM', units='height', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=1.0,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    textStateMeasure = visual.TextStim(win=win, name='textStateMeasure',
        text='',
        font='Arial',
        pos=(0, .35), draggable=False, height=0.07, wrapWidth=1.5, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    textLabelMin = visual.TextStim(win=win, name='textLabelMin',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=0.5, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    textLabelMax = visual.TextStim(win=win, name='textLabelMax',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=0.5, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    tb_continue_sm = visual.TextBox2(
         win, text='Press the SPACEBAR to continue', placeholder='Type here...', font='Arial',
         ori=0.0, pos=(0, -0.4), draggable=False,      letterHeight=0.03,
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
         depth=-6, autoLog=True,
    )
    # Run 'Begin Experiment' code from stateMeasureHelper
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
    
    key_resp_state_measure = keyboard.Keyboard(deviceName='key_resp_state_measure')
    
    # --- Initialize components for Routine "instruction" ---
    tb_instruction = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Arial',
         ori=0.0, pos=(0.0, 0.05), draggable=False,      letterHeight=0.07,
         size=(1.5, 0.5), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=True, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='tb_instruction',
         depth=0, autoLog=True,
    )
    tb_continue_2 = visual.TextBox2(
         win, text='Press the SPACEBAR to begin', placeholder='Type here...', font='Arial',
         ori=0.0, pos=(0, -0.4), draggable=False,      letterHeight=0.03,
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
         name='tb_continue_2',
         depth=-1, autoLog=True,
    )
    key_resp_2 = keyboard.Keyboard(deviceName='key_resp_2')
    
    # --- Initialize components for Routine "triggerStart" ---
    p_port_start = parallel.ParallelPort(address='0x0378')
    
    # --- Initialize components for Routine "fixation" ---
    cross_fixation = visual.ShapeStim(
        win=win, name='cross_fixation', vertices='cross',
        size=(0.15, 0.15),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "triggerStop" ---
    p_port_stop = parallel.ParallelPort(address='0x0378')
    
    # --- Initialize components for Routine "stateMeasure" ---
    sliderGeneric = visual.Slider(win=win, name='sliderGeneric',
        startValue=None, size=1.0, pos=(0, 0), units=win.units,
        labels=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), ticks=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), granularity=1.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Noto Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=0, readOnly=False)
    sliderSAM = visual.Slider(win=win, name='sliderSAM',
        startValue=None, size=1.0, pos=[0,0], units=win.units,
        labels=[1, 2, 3, 4, 5, 6, 7, 8, 9], ticks=(1, 2, 3, 4, 5, 6, 7, 8, 9), granularity=1.0,
        style='scrollbar', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Noto Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-1, readOnly=False)
    imageSAM = visual.ImageStim(
        win=win,
        name='imageSAM', units='height', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=1.0,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    textStateMeasure = visual.TextStim(win=win, name='textStateMeasure',
        text='',
        font='Arial',
        pos=(0, .35), draggable=False, height=0.07, wrapWidth=1.5, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    textLabelMin = visual.TextStim(win=win, name='textLabelMin',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=0.5, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    textLabelMax = visual.TextStim(win=win, name='textLabelMax',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=0.5, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    tb_continue_sm = visual.TextBox2(
         win, text='Press the SPACEBAR to continue', placeholder='Type here...', font='Arial',
         ori=0.0, pos=(0, -0.4), draggable=False,      letterHeight=0.03,
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
         depth=-6, autoLog=True,
    )
    # Run 'Begin Experiment' code from stateMeasureHelper
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
    
    key_resp_state_measure = keyboard.Keyboard(deviceName='key_resp_state_measure')
    
    # --- Initialize components for Routine "goodbye" ---
    tb_goodbye = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Arial',
         ori=0.0, pos=(0.0, 0.0), draggable=False,      letterHeight=0.07,
         size=(1.5, 0.5), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='tb_goodbye',
         depth=0, autoLog=True,
    )
    
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
    intro_prompt_loop = data.TrialHandler2(
        name='intro_prompt_loop',
        nReps=1.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('../../shared/loop-templates/loopOpenClosedIntro.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(intro_prompt_loop)  # add the loop to the experiment
    thisIntro_prompt_loop = intro_prompt_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisIntro_prompt_loop.rgb)
    if thisIntro_prompt_loop != None:
        for paramName in thisIntro_prompt_loop:
            globals()[paramName] = thisIntro_prompt_loop[paramName]
    
    for thisIntro_prompt_loop in intro_prompt_loop:
        intro_prompt_loop.status = STARTED
        if hasattr(thisIntro_prompt_loop, 'status'):
            thisIntro_prompt_loop.status = STARTED
        currentLoop = intro_prompt_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # abbreviate parameter names if possible (e.g. rgb = thisIntro_prompt_loop.rgb)
        if thisIntro_prompt_loop != None:
            for paramName in thisIntro_prompt_loop:
                globals()[paramName] = thisIntro_prompt_loop[paramName]
        
        # --- Prepare to start Routine "welcome" ---
        # create an object to store info about Routine welcome
        welcome = data.Routine(
            name='welcome',
            components=[tb_body, tb_continue, key_resp_welcome],
        )
        welcome.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        tb_body.reset()
        tb_body.setText(Message)
        tb_continue.reset()
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
        welcome.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisIntro_prompt_loop, 'status') and thisIntro_prompt_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *tb_body* updates
            
            # if tb_body is starting this frame...
            if tb_body.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                tb_body.frameNStart = frameN  # exact frame index
                tb_body.tStart = t  # local t and not account for scr refresh
                tb_body.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(tb_body, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'tb_body.started')
                # update status
                tb_body.status = STARTED
                tb_body.setAutoDraw(True)
            
            # if tb_body is active this frame...
            if tb_body.status == STARTED:
                # update params
                pass
            
            # *tb_continue* updates
            
            # if tb_continue is starting this frame...
            if tb_continue.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                tb_continue.frameNStart = frameN  # exact frame index
                tb_continue.tStart = t  # local t and not account for scr refresh
                tb_continue.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(tb_continue, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'tb_continue.started')
                # update status
                tb_continue.status = STARTED
                tb_continue.setAutoDraw(True)
            
            # if tb_continue is active this frame...
            if tb_continue.status == STARTED:
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
        intro_prompt_loop.addData('key_resp_welcome.keys',key_resp_welcome.keys)
        if key_resp_welcome.keys != None:  # we had a response
            intro_prompt_loop.addData('key_resp_welcome.rt', key_resp_welcome.rt)
            intro_prompt_loop.addData('key_resp_welcome.duration', key_resp_welcome.duration)
        # the Routine "welcome" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisIntro_prompt_loop as finished
        if hasattr(thisIntro_prompt_loop, 'status'):
            thisIntro_prompt_loop.status = FINISHED
        # if awaiting a pause, pause now
        if intro_prompt_loop.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            intro_prompt_loop.status = STARTED
    # completed 1.0 repeats of 'intro_prompt_loop'
    intro_prompt_loop.status = FINISHED
    
    
    # set up handler to look after randomisation of conditions etc
    state_measure_loop1 = data.TrialHandler2(
        name='state_measure_loop1',
        nReps=1.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('../../shared/loop-templates/loopStateMeasure.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(state_measure_loop1)  # add the loop to the experiment
    thisState_measure_loop1 = state_measure_loop1.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisState_measure_loop1.rgb)
    if thisState_measure_loop1 != None:
        for paramName in thisState_measure_loop1:
            globals()[paramName] = thisState_measure_loop1[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisState_measure_loop1 in state_measure_loop1:
        state_measure_loop1.status = STARTED
        if hasattr(thisState_measure_loop1, 'status'):
            thisState_measure_loop1.status = STARTED
        currentLoop = state_measure_loop1
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisState_measure_loop1.rgb)
        if thisState_measure_loop1 != None:
            for paramName in thisState_measure_loop1:
                globals()[paramName] = thisState_measure_loop1[paramName]
        
        # --- Prepare to start Routine "stateMeasure" ---
        # create an object to store info about Routine stateMeasure
        stateMeasure = data.Routine(
            name='stateMeasure',
            components=[sliderGeneric, sliderSAM, imageSAM, textStateMeasure, textLabelMin, textLabelMax, tb_continue_sm, key_resp_state_measure],
        )
        stateMeasure.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        sliderGeneric.reset()
        sliderGeneric.setSize((slider_gen_width, 0.1))
        sliderSAM.reset()
        sliderSAM.setPos((0, slider_y))
        sliderSAM.setSize((slider_width, slider_height))
        imageSAM.setPos((0, sam_y))
        imageSAM.setSize((sam_width, sam_height))
        imageSAM.setImage(picture_path)
        textStateMeasure.setText(rating_message)
        textLabelMin.setPos((tick_positions[0], label_y))
        textLabelMin.setText(rating_min_label)
        textLabelMax.setPos((tick_positions[-1], label_y))
        textLabelMax.setText(rating_max_label)
        tb_continue_sm.reset()
        # Run 'Begin Routine' code from stateMeasureHelper
        allowContinue = False
        # create starting attributes for key_resp_state_measure
        key_resp_state_measure.keys = []
        key_resp_state_measure.rt = []
        _key_resp_state_measure_allKeys = []
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
            if hasattr(thisState_measure_loop1, 'status') and thisState_measure_loop1.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *sliderGeneric* updates
            
            # if sliderGeneric is starting this frame...
            if sliderGeneric.status == NOT_STARTED and rating_type == 'Generic':
                # keep track of start time/frame for later
                sliderGeneric.frameNStart = frameN  # exact frame index
                sliderGeneric.tStart = t  # local t and not account for scr refresh
                sliderGeneric.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(sliderGeneric, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'sliderGeneric.started')
                # update status
                sliderGeneric.status = STARTED
                sliderGeneric.setAutoDraw(True)
            
            # if sliderGeneric is active this frame...
            if sliderGeneric.status == STARTED:
                # update params
                pass
            
            # if sliderGeneric is stopping this frame...
            if sliderGeneric.status == STARTED:
                if bool(rating_type != 'Generic'):
                    # keep track of stop time/frame for later
                    sliderGeneric.tStop = t  # not accounting for scr refresh
                    sliderGeneric.tStopRefresh = tThisFlipGlobal  # on global time
                    sliderGeneric.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'sliderGeneric.stopped')
                    # update status
                    sliderGeneric.status = FINISHED
                    sliderGeneric.setAutoDraw(False)
            
            # *sliderSAM* updates
            
            # if sliderSAM is starting this frame...
            if sliderSAM.status == NOT_STARTED and rating_type == 'SAM':
                # keep track of start time/frame for later
                sliderSAM.frameNStart = frameN  # exact frame index
                sliderSAM.tStart = t  # local t and not account for scr refresh
                sliderSAM.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(sliderSAM, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'sliderSAM.started')
                # update status
                sliderSAM.status = STARTED
                sliderSAM.setAutoDraw(True)
            
            # if sliderSAM is active this frame...
            if sliderSAM.status == STARTED:
                # update params
                pass
            
            # if sliderSAM is stopping this frame...
            if sliderSAM.status == STARTED:
                if bool(rating_type != 'SAM'):
                    # keep track of stop time/frame for later
                    sliderSAM.tStop = t  # not accounting for scr refresh
                    sliderSAM.tStopRefresh = tThisFlipGlobal  # on global time
                    sliderSAM.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'sliderSAM.stopped')
                    # update status
                    sliderSAM.status = FINISHED
                    sliderSAM.setAutoDraw(False)
            
            # *imageSAM* updates
            
            # if imageSAM is starting this frame...
            if imageSAM.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                imageSAM.frameNStart = frameN  # exact frame index
                imageSAM.tStart = t  # local t and not account for scr refresh
                imageSAM.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(imageSAM, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'imageSAM.started')
                # update status
                imageSAM.status = STARTED
                imageSAM.setAutoDraw(True)
            
            # if imageSAM is active this frame...
            if imageSAM.status == STARTED:
                # update params
                pass
            
            # *textStateMeasure* updates
            
            # if textStateMeasure is starting this frame...
            if textStateMeasure.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                textStateMeasure.frameNStart = frameN  # exact frame index
                textStateMeasure.tStart = t  # local t and not account for scr refresh
                textStateMeasure.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(textStateMeasure, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textStateMeasure.started')
                # update status
                textStateMeasure.status = STARTED
                textStateMeasure.setAutoDraw(True)
            
            # if textStateMeasure is active this frame...
            if textStateMeasure.status == STARTED:
                # update params
                pass
            
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
            # Run 'Each Frame' code from stateMeasureHelper
            # Check if either slider has been rated
            rated = (sliderGeneric.getRating() is not None) or (sliderSAM.getRating() is not None)
            
            # If rated, allow spacebar to end the routine
            if rated:
                allowContinue = True
            else:
                allowContinue = False
            
            
            # *key_resp_state_measure* updates
            waitOnFlip = False
            
            # if key_resp_state_measure is starting this frame...
            if key_resp_state_measure.status == NOT_STARTED and allowContinue == True:
                # keep track of start time/frame for later
                key_resp_state_measure.frameNStart = frameN  # exact frame index
                key_resp_state_measure.tStart = t  # local t and not account for scr refresh
                key_resp_state_measure.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_state_measure, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_state_measure.started')
                # update status
                key_resp_state_measure.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_state_measure.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_state_measure.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_state_measure.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_state_measure.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_state_measure_allKeys.extend(theseKeys)
                if len(_key_resp_state_measure_allKeys):
                    key_resp_state_measure.keys = _key_resp_state_measure_allKeys[-1].name  # just the last key pressed
                    key_resp_state_measure.rt = _key_resp_state_measure_allKeys[-1].rt
                    key_resp_state_measure.duration = _key_resp_state_measure_allKeys[-1].duration
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
        state_measure_loop1.addData('sliderGeneric.response', sliderGeneric.getRating())
        state_measure_loop1.addData('sliderGeneric.rt', sliderGeneric.getRT())
        state_measure_loop1.addData('sliderSAM.response', sliderSAM.getRating())
        state_measure_loop1.addData('sliderSAM.rt', sliderSAM.getRT())
        # check responses
        if key_resp_state_measure.keys in ['', [], None]:  # No response was made
            key_resp_state_measure.keys = None
        state_measure_loop1.addData('key_resp_state_measure.keys',key_resp_state_measure.keys)
        if key_resp_state_measure.keys != None:  # we had a response
            state_measure_loop1.addData('key_resp_state_measure.rt', key_resp_state_measure.rt)
            state_measure_loop1.addData('key_resp_state_measure.duration', key_resp_state_measure.duration)
        # the Routine "stateMeasure" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisState_measure_loop1 as finished
        if hasattr(thisState_measure_loop1, 'status'):
            thisState_measure_loop1.status = FINISHED
        # if awaiting a pause, pause now
        if state_measure_loop1.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            state_measure_loop1.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'state_measure_loop1'
    state_measure_loop1.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    # get names of stimulus parameters
    if state_measure_loop1.trialList in ([], [None], None):
        params = []
    else:
        params = state_measure_loop1.trialList[0].keys()
    # save data for this loop
    state_measure_loop1.saveAsExcel(filename + '.xlsx', sheetName='state_measure_loop1',
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
            components=[tb_instruction, tb_continue_2, key_resp_2],
        )
        instruction.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        tb_instruction.reset()
        tb_instruction.setText(Prompt
        )
        tb_continue_2.reset()
        # create starting attributes for key_resp_2
        key_resp_2.keys = []
        key_resp_2.rt = []
        _key_resp_2_allKeys = []
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
            
            # *tb_instruction* updates
            
            # if tb_instruction is starting this frame...
            if tb_instruction.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                tb_instruction.frameNStart = frameN  # exact frame index
                tb_instruction.tStart = t  # local t and not account for scr refresh
                tb_instruction.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(tb_instruction, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'tb_instruction.started')
                # update status
                tb_instruction.status = STARTED
                tb_instruction.setAutoDraw(True)
            
            # if tb_instruction is active this frame...
            if tb_instruction.status == STARTED:
                # update params
                pass
            
            # *tb_continue_2* updates
            
            # if tb_continue_2 is starting this frame...
            if tb_continue_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                tb_continue_2.frameNStart = frameN  # exact frame index
                tb_continue_2.tStart = t  # local t and not account for scr refresh
                tb_continue_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(tb_continue_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'tb_continue_2.started')
                # update status
                tb_continue_2.status = STARTED
                tb_continue_2.setAutoDraw(True)
            
            # if tb_continue_2 is active this frame...
            if tb_continue_2.status == STARTED:
                # update params
                pass
            
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
        while continueRoutine and routineTimer.getTime() < 5.0:
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
                if tThisFlipGlobal > cross_fixation.tStartRefresh + 5-frameTolerance:
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
            routineTimer.addTime(-5.000000)
        
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
        state_measure_loop2 = data.TrialHandler2(
            name='state_measure_loop2',
            nReps=1.0, 
            method='sequential', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions('../../shared/loop-templates/loopStateMeasure.xlsx'), 
            seed=None, 
        )
        thisExp.addLoop(state_measure_loop2)  # add the loop to the experiment
        thisState_measure_loop2 = state_measure_loop2.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisState_measure_loop2.rgb)
        if thisState_measure_loop2 != None:
            for paramName in thisState_measure_loop2:
                globals()[paramName] = thisState_measure_loop2[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisState_measure_loop2 in state_measure_loop2:
            state_measure_loop2.status = STARTED
            if hasattr(thisState_measure_loop2, 'status'):
                thisState_measure_loop2.status = STARTED
            currentLoop = state_measure_loop2
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisState_measure_loop2.rgb)
            if thisState_measure_loop2 != None:
                for paramName in thisState_measure_loop2:
                    globals()[paramName] = thisState_measure_loop2[paramName]
            
            # --- Prepare to start Routine "stateMeasure" ---
            # create an object to store info about Routine stateMeasure
            stateMeasure = data.Routine(
                name='stateMeasure',
                components=[sliderGeneric, sliderSAM, imageSAM, textStateMeasure, textLabelMin, textLabelMax, tb_continue_sm, key_resp_state_measure],
            )
            stateMeasure.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            sliderGeneric.reset()
            sliderGeneric.setSize((slider_gen_width, 0.1))
            sliderSAM.reset()
            sliderSAM.setPos((0, slider_y))
            sliderSAM.setSize((slider_width, slider_height))
            imageSAM.setPos((0, sam_y))
            imageSAM.setSize((sam_width, sam_height))
            imageSAM.setImage(picture_path)
            textStateMeasure.setText(rating_message)
            textLabelMin.setPos((tick_positions[0], label_y))
            textLabelMin.setText(rating_min_label)
            textLabelMax.setPos((tick_positions[-1], label_y))
            textLabelMax.setText(rating_max_label)
            tb_continue_sm.reset()
            # Run 'Begin Routine' code from stateMeasureHelper
            allowContinue = False
            # create starting attributes for key_resp_state_measure
            key_resp_state_measure.keys = []
            key_resp_state_measure.rt = []
            _key_resp_state_measure_allKeys = []
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
                if hasattr(thisState_measure_loop2, 'status') and thisState_measure_loop2.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *sliderGeneric* updates
                
                # if sliderGeneric is starting this frame...
                if sliderGeneric.status == NOT_STARTED and rating_type == 'Generic':
                    # keep track of start time/frame for later
                    sliderGeneric.frameNStart = frameN  # exact frame index
                    sliderGeneric.tStart = t  # local t and not account for scr refresh
                    sliderGeneric.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(sliderGeneric, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'sliderGeneric.started')
                    # update status
                    sliderGeneric.status = STARTED
                    sliderGeneric.setAutoDraw(True)
                
                # if sliderGeneric is active this frame...
                if sliderGeneric.status == STARTED:
                    # update params
                    pass
                
                # if sliderGeneric is stopping this frame...
                if sliderGeneric.status == STARTED:
                    if bool(rating_type != 'Generic'):
                        # keep track of stop time/frame for later
                        sliderGeneric.tStop = t  # not accounting for scr refresh
                        sliderGeneric.tStopRefresh = tThisFlipGlobal  # on global time
                        sliderGeneric.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'sliderGeneric.stopped')
                        # update status
                        sliderGeneric.status = FINISHED
                        sliderGeneric.setAutoDraw(False)
                
                # *sliderSAM* updates
                
                # if sliderSAM is starting this frame...
                if sliderSAM.status == NOT_STARTED and rating_type == 'SAM':
                    # keep track of start time/frame for later
                    sliderSAM.frameNStart = frameN  # exact frame index
                    sliderSAM.tStart = t  # local t and not account for scr refresh
                    sliderSAM.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(sliderSAM, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'sliderSAM.started')
                    # update status
                    sliderSAM.status = STARTED
                    sliderSAM.setAutoDraw(True)
                
                # if sliderSAM is active this frame...
                if sliderSAM.status == STARTED:
                    # update params
                    pass
                
                # if sliderSAM is stopping this frame...
                if sliderSAM.status == STARTED:
                    if bool(rating_type != 'SAM'):
                        # keep track of stop time/frame for later
                        sliderSAM.tStop = t  # not accounting for scr refresh
                        sliderSAM.tStopRefresh = tThisFlipGlobal  # on global time
                        sliderSAM.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'sliderSAM.stopped')
                        # update status
                        sliderSAM.status = FINISHED
                        sliderSAM.setAutoDraw(False)
                
                # *imageSAM* updates
                
                # if imageSAM is starting this frame...
                if imageSAM.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    imageSAM.frameNStart = frameN  # exact frame index
                    imageSAM.tStart = t  # local t and not account for scr refresh
                    imageSAM.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(imageSAM, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'imageSAM.started')
                    # update status
                    imageSAM.status = STARTED
                    imageSAM.setAutoDraw(True)
                
                # if imageSAM is active this frame...
                if imageSAM.status == STARTED:
                    # update params
                    pass
                
                # *textStateMeasure* updates
                
                # if textStateMeasure is starting this frame...
                if textStateMeasure.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    textStateMeasure.frameNStart = frameN  # exact frame index
                    textStateMeasure.tStart = t  # local t and not account for scr refresh
                    textStateMeasure.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(textStateMeasure, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'textStateMeasure.started')
                    # update status
                    textStateMeasure.status = STARTED
                    textStateMeasure.setAutoDraw(True)
                
                # if textStateMeasure is active this frame...
                if textStateMeasure.status == STARTED:
                    # update params
                    pass
                
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
                # Run 'Each Frame' code from stateMeasureHelper
                # Check if either slider has been rated
                rated = (sliderGeneric.getRating() is not None) or (sliderSAM.getRating() is not None)
                
                # If rated, allow spacebar to end the routine
                if rated:
                    allowContinue = True
                else:
                    allowContinue = False
                
                
                # *key_resp_state_measure* updates
                waitOnFlip = False
                
                # if key_resp_state_measure is starting this frame...
                if key_resp_state_measure.status == NOT_STARTED and allowContinue == True:
                    # keep track of start time/frame for later
                    key_resp_state_measure.frameNStart = frameN  # exact frame index
                    key_resp_state_measure.tStart = t  # local t and not account for scr refresh
                    key_resp_state_measure.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp_state_measure, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp_state_measure.started')
                    # update status
                    key_resp_state_measure.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp_state_measure.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp_state_measure.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if key_resp_state_measure.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp_state_measure.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                    _key_resp_state_measure_allKeys.extend(theseKeys)
                    if len(_key_resp_state_measure_allKeys):
                        key_resp_state_measure.keys = _key_resp_state_measure_allKeys[-1].name  # just the last key pressed
                        key_resp_state_measure.rt = _key_resp_state_measure_allKeys[-1].rt
                        key_resp_state_measure.duration = _key_resp_state_measure_allKeys[-1].duration
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
            state_measure_loop2.addData('sliderGeneric.response', sliderGeneric.getRating())
            state_measure_loop2.addData('sliderGeneric.rt', sliderGeneric.getRT())
            state_measure_loop2.addData('sliderSAM.response', sliderSAM.getRating())
            state_measure_loop2.addData('sliderSAM.rt', sliderSAM.getRT())
            # check responses
            if key_resp_state_measure.keys in ['', [], None]:  # No response was made
                key_resp_state_measure.keys = None
            state_measure_loop2.addData('key_resp_state_measure.keys',key_resp_state_measure.keys)
            if key_resp_state_measure.keys != None:  # we had a response
                state_measure_loop2.addData('key_resp_state_measure.rt', key_resp_state_measure.rt)
                state_measure_loop2.addData('key_resp_state_measure.duration', key_resp_state_measure.duration)
            # the Routine "stateMeasure" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            # mark thisState_measure_loop2 as finished
            if hasattr(thisState_measure_loop2, 'status'):
                thisState_measure_loop2.status = FINISHED
            # if awaiting a pause, pause now
            if state_measure_loop2.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                state_measure_loop2.status = STARTED
            thisExp.nextEntry()
            
        # completed 1.0 repeats of 'state_measure_loop2'
        state_measure_loop2.status = FINISHED
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # get names of stimulus parameters
        if state_measure_loop2.trialList in ([], [None], None):
            params = []
        else:
            params = state_measure_loop2.trialList[0].keys()
        # save data for this loop
        state_measure_loop2.saveAsExcel(filename + '.xlsx', sheetName='state_measure_loop2',
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
        components=[tb_goodbye],
    )
    goodbye.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    tb_goodbye.reset()
    tb_goodbye.setText('You have completed the trial. Thank you!')
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
    while continueRoutine and routineTimer.getTime() < 5.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *tb_goodbye* updates
        
        # if tb_goodbye is starting this frame...
        if tb_goodbye.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            tb_goodbye.frameNStart = frameN  # exact frame index
            tb_goodbye.tStart = t  # local t and not account for scr refresh
            tb_goodbye.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(tb_goodbye, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'tb_goodbye.started')
            # update status
            tb_goodbye.status = STARTED
            tb_goodbye.setAutoDraw(True)
        
        # if tb_goodbye is active this frame...
        if tb_goodbye.status == STARTED:
            # update params
            pass
        
        # if tb_goodbye is stopping this frame...
        if tb_goodbye.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > tb_goodbye.tStartRefresh + 5-frameTolerance:
                # keep track of stop time/frame for later
                tb_goodbye.tStop = t  # not accounting for scr refresh
                tb_goodbye.tStopRefresh = tThisFlipGlobal  # on global time
                tb_goodbye.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'tb_goodbye.stopped')
                # update status
                tb_goodbye.status = FINISHED
                tb_goodbye.setAutoDraw(False)
        
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
        routineTimer.addTime(-5.000000)
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
