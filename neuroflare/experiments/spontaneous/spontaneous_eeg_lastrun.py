#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2025.2.3),
    on March 17, 2026, at 23:27
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
expName = 'spontaneous_eeg'  # from the Builder filename that created this script
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
        originPath='D:\\Github\\neuroflare-experiments\\neuroflare\\experiments\\spontaneous\\spontaneous_eeg_lastrun.py',
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
            logging.getLevel('info')
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
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
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
    
    # --- Initialize components for Routine "welcome" ---
    text = visual.TextStim(win=win, name='text',
        text="Please relax and remain as still as possible while we record your brain activity.\n\nFollow the on-screen instructions to assess your body pain at present.\n\nWhen you're ready, press the spacebar to begin.",
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "state_measure_intro_2" ---
    slider_generic0 = visual.Slider(win=win, name='slider_generic0',
        startValue=None, size=1.0, pos=(0, 0), units=win.units,
        labels=[0,1,2,3,4,5,6,7,8,9,10], ticks=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), granularity=1.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Noto Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=0, readOnly=False)
    slider_SAM0 = visual.Slider(win=win, name='slider_SAM0',
        startValue=None, size=1.0, pos=[0,0], units=win.units,
        labels=[1,2,3,4,5,6,7,8,9],ticks=None, granularity=1,
        style='radio', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Noto Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-1, readOnly=False)
    textStateMeasure0 = visual.TextStim(win=win, name='textStateMeasure0',
        text='',
        font='Arial',
        pos=(0, 0.35), draggable=False, height=0.05, wrapWidth=1.5, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    textLabelMin0 = visual.TextStim(win=win, name='textLabelMin0',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=0.5, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    textLabelMax0 = visual.TextStim(win=win, name='textLabelMax0',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=0.5, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    tb_continue_sm0 = visual.TextBox2(
         win, text='Press SPACEBAR to continue', placeholder='Type here...', font='Arial',
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
         name='tb_continue_sm0',
         depth=-5, autoLog=True,
    )
    key_resp_state_measure0 = keyboard.Keyboard(deviceName='defaultKeyboard')
    imageSAM0 = visual.ImageStim(
        win=win,
        name='imageSAM0', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=1.0,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-7.0)
    # Run 'Begin Experiment' code from stateMeasureHelper0
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
    
    # --- Initialize components for Routine "Intro_01" ---
    text_7 = visual.TextStim(win=win, name='text_7',
        text="We are now going to ask you to rate different words that describe pain. Click the number that best matches your experience for each word.\u200b\n\n\nFor each word shown, rate how well it describes your current pain using the following scale: \u200b\n\n0 – none, 1 – Mild, 2 – Moderate, 3 – Severe\u200b\n\n\u200b\n\n**When you're ready, press the spacebar to begin.**",
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_7 = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "X" ---
    polygon = visual.ShapeStim(
        win=win, name='polygon', vertices='cross',
        size=(0.5, 0.5),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "pain_adjectives01" ---
    pain_adjectives_text = visual.TextStim(win=win, name='pain_adjectives_text',
        text='',
        font='Arial',
        pos=(0, 0.25), draggable=False, height=0.15, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    pain_adjectives_kb_response = keyboard.Keyboard(deviceName='defaultKeyboard')
    pain_slider_2 = visual.Slider(win=win, name='pain_slider_2',
        startValue=None, size=1.0, pos=[0,0], units=win.units,
        labels=["None","Mild","Moderate","Severe"],ticks=None, granularity=1,
        style='radio', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Noto Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-2, readOnly=False)
    pain_tb_continue_sm = visual.TextBox2(
         win, text='Press the SPACEBAR to continue', placeholder='Type here...', font='Arial',
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
         name='pain_tb_continue_sm',
         depth=-3, autoLog=True,
    )
    # Run 'Begin Experiment' code from painHelperCode
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
    
    # --- Initialize components for Routine "X2" ---
    polygon_3 = visual.ShapeStim(
        win=win, name='polygon_3', vertices='cross',
        size=(0.5, 0.5),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "pain_adjectives02" ---
    pain_adjectives2_text = visual.TextStim(win=win, name='pain_adjectives2_text',
        text='',
        font='Arial',
        pos=(0, 0.25), draggable=False, height=0.15, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    pain_adjectives_key_resp02 = keyboard.Keyboard(deviceName='defaultKeyboard')
    pain_continue_tb2 = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Arial',
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
         name='pain_continue_tb2',
         depth=-2, autoLog=True,
    )
    
    # --- Initialize components for Routine "X_2" ---
    polygon_2 = visual.ShapeStim(
        win=win, name='polygon_2', vertices='cross',
        size=(0.5, 0.5),
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
    
    # --- Initialize components for Routine "state_measure_outro" ---
    sliderGeneric2 = visual.Slider(win=win, name='sliderGeneric2',
        startValue=None, size=1.0, pos=(0, 0), units=win.units,
        labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ticks=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), granularity=1.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Noto Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=0, readOnly=False)
    sliderSAM2 = visual.Slider(win=win, name='sliderSAM2',
        startValue=None, size=1.0, pos=[0,0], units=win.units,
        labels=[1, 2, 3, 4, 5, 6, 7, 8, 9],ticks=None, granularity=1,
        style='radio', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Noto Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-1, readOnly=False)
    imageSAM2 = visual.ImageStim(
        win=win,
        name='imageSAM2', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=1.0,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    textStateMeasure2 = visual.TextStim(win=win, name='textStateMeasure2',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=1.5, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    textLabelMin2 = visual.TextStim(win=win, name='textLabelMin2',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=0.5, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    textLabelMax2 = visual.TextStim(win=win, name='textLabelMax2',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=0.5, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    tb_continue_02 = visual.TextBox2(
         win, text='Press the SPACEBAR to continue', placeholder='Type here...', font='Arial',
         ori=0.0, pos=[0,0], draggable=False,      letterHeight=0.05,
         size=1.0, borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='tb_continue_02',
         depth=-6, autoLog=True,
    )
    # Run 'Begin Experiment' code from stateMeasureHelper2
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
    key_resp_state_measure2 = keyboard.Keyboard(deviceName='defaultKeyboard')
    
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
    
    # --- Prepare to start Routine "welcome" ---
    # create an object to store info about Routine welcome
    welcome = data.Routine(
        name='welcome',
        components=[text, key_resp],
    )
    welcome.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp
    key_resp.keys = []
    key_resp.rt = []
    _key_resp_allKeys = []
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
        
        # *text* updates
        
        # if text is starting this frame...
        if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text.frameNStart = frameN  # exact frame index
            text.tStart = t  # local t and not account for scr refresh
            text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text.started')
            # update status
            text.status = STARTED
            text.setAutoDraw(True)
        
        # if text is active this frame...
        if text.status == STARTED:
            # update params
            pass
        
        # *key_resp* updates
        waitOnFlip = False
        
        # if key_resp is starting this frame...
        if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp.frameNStart = frameN  # exact frame index
            key_resp.tStart = t  # local t and not account for scr refresh
            key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp.started')
            # update status
            key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp.status == STARTED and not waitOnFlip:
            theseKeys = key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_allKeys.extend(theseKeys)
            if len(_key_resp_allKeys):
                key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                key_resp.rt = _key_resp_allKeys[-1].rt
                key_resp.duration = _key_resp_allKeys[-1].duration
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
    # check responses
    if key_resp.keys in ['', [], None]:  # No response was made
        key_resp.keys = None
    thisExp.addData('key_resp.keys',key_resp.keys)
    if key_resp.keys != None:  # we had a response
        thisExp.addData('key_resp.rt', key_resp.rt)
        thisExp.addData('key_resp.duration', key_resp.duration)
    thisExp.nextEntry()
    # the Routine "welcome" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    state_measure_01 = data.TrialHandler2(
        name='state_measure_01',
        nReps=1.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('D:/Github/neuroflare-experiments/neuroflare/shared/loop-templates/loopStateMeasure.xlsx'), 
        seed=None, 
        isTrials=False, 
    )
    thisExp.addLoop(state_measure_01)  # add the loop to the experiment
    thisState_measure_01 = state_measure_01.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisState_measure_01.rgb)
    if thisState_measure_01 != None:
        for paramName in thisState_measure_01:
            globals()[paramName] = thisState_measure_01[paramName]
    
    for thisState_measure_01 in state_measure_01:
        state_measure_01.status = STARTED
        if hasattr(thisState_measure_01, 'status'):
            thisState_measure_01.status = STARTED
        currentLoop = state_measure_01
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # abbreviate parameter names if possible (e.g. rgb = thisState_measure_01.rgb)
        if thisState_measure_01 != None:
            for paramName in thisState_measure_01:
                globals()[paramName] = thisState_measure_01[paramName]
        
        # --- Prepare to start Routine "state_measure_intro_2" ---
        # create an object to store info about Routine state_measure_intro_2
        state_measure_intro_2 = data.Routine(
            name='state_measure_intro_2',
            components=[slider_generic0, slider_SAM0, textStateMeasure0, textLabelMin0, textLabelMax0, tb_continue_sm0, key_resp_state_measure0, imageSAM0],
        )
        state_measure_intro_2.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        slider_generic0.reset()
        slider_generic0.setSize((slider_gen_width, 0.1))
        slider_SAM0.reset()
        slider_SAM0.setPos((0, slider_y))
        slider_SAM0.setSize((slider_width, slider_height))
        textStateMeasure0.setText(rating_message)
        textLabelMin0.setPos((tick_positions[0], label_y))
        textLabelMin0.setText(rating_min_label)
        textLabelMax0.setPos((tick_positions[-1], label_y))
        textLabelMax0.setText(rating_max_label)
        tb_continue_sm0.reset()
        # create starting attributes for key_resp_state_measure0
        key_resp_state_measure0.keys = []
        key_resp_state_measure0.rt = []
        _key_resp_state_measure0_allKeys = []
        imageSAM0.setPos((0, sam_y))
        imageSAM0.setSize((sam_width, sam_height))
        imageSAM0.setImage(picture_path)
        # Run 'Begin Routine' code from stateMeasureHelper0
        allowContinue = False
        # store start times for state_measure_intro_2
        state_measure_intro_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        state_measure_intro_2.tStart = globalClock.getTime(format='float')
        state_measure_intro_2.status = STARTED
        thisExp.addData('state_measure_intro_2.started', state_measure_intro_2.tStart)
        state_measure_intro_2.maxDuration = None
        # keep track of which components have finished
        state_measure_intro_2Components = state_measure_intro_2.components
        for thisComponent in state_measure_intro_2.components:
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
        
        # --- Run Routine "state_measure_intro_2" ---
        thisExp.currentRoutine = state_measure_intro_2
        state_measure_intro_2.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisState_measure_01, 'status') and thisState_measure_01.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *slider_generic0* updates
            
            # if slider_generic0 is starting this frame...
            if slider_generic0.status == NOT_STARTED and rating_type == 'Generic':
                # keep track of start time/frame for later
                slider_generic0.frameNStart = frameN  # exact frame index
                slider_generic0.tStart = t  # local t and not account for scr refresh
                slider_generic0.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(slider_generic0, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'slider_generic0.started')
                # update status
                slider_generic0.status = STARTED
                slider_generic0.setAutoDraw(True)
            
            # if slider_generic0 is active this frame...
            if slider_generic0.status == STARTED:
                # update params
                pass
            
            # if slider_generic0 is stopping this frame...
            if slider_generic0.status == STARTED:
                if bool(rating_type != 'Generic'):
                    # keep track of stop time/frame for later
                    slider_generic0.tStop = t  # not accounting for scr refresh
                    slider_generic0.tStopRefresh = tThisFlipGlobal  # on global time
                    slider_generic0.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'slider_generic0.stopped')
                    # update status
                    slider_generic0.status = FINISHED
                    slider_generic0.setAutoDraw(False)
            
            # *slider_SAM0* updates
            
            # if slider_SAM0 is starting this frame...
            if slider_SAM0.status == NOT_STARTED and rating_type == 'SAM':
                # keep track of start time/frame for later
                slider_SAM0.frameNStart = frameN  # exact frame index
                slider_SAM0.tStart = t  # local t and not account for scr refresh
                slider_SAM0.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(slider_SAM0, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'slider_SAM0.started')
                # update status
                slider_SAM0.status = STARTED
                slider_SAM0.setAutoDraw(True)
            
            # if slider_SAM0 is active this frame...
            if slider_SAM0.status == STARTED:
                # update params
                pass
            
            # if slider_SAM0 is stopping this frame...
            if slider_SAM0.status == STARTED:
                if bool(rating_type != 'SAM'):
                    # keep track of stop time/frame for later
                    slider_SAM0.tStop = t  # not accounting for scr refresh
                    slider_SAM0.tStopRefresh = tThisFlipGlobal  # on global time
                    slider_SAM0.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'slider_SAM0.stopped')
                    # update status
                    slider_SAM0.status = FINISHED
                    slider_SAM0.setAutoDraw(False)
            
            # *textStateMeasure0* updates
            
            # if textStateMeasure0 is starting this frame...
            if textStateMeasure0.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                textStateMeasure0.frameNStart = frameN  # exact frame index
                textStateMeasure0.tStart = t  # local t and not account for scr refresh
                textStateMeasure0.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(textStateMeasure0, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textStateMeasure0.started')
                # update status
                textStateMeasure0.status = STARTED
                textStateMeasure0.setAutoDraw(True)
            
            # if textStateMeasure0 is active this frame...
            if textStateMeasure0.status == STARTED:
                # update params
                pass
            
            # *textLabelMin0* updates
            
            # if textLabelMin0 is starting this frame...
            if textLabelMin0.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                textLabelMin0.frameNStart = frameN  # exact frame index
                textLabelMin0.tStart = t  # local t and not account for scr refresh
                textLabelMin0.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(textLabelMin0, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textLabelMin0.started')
                # update status
                textLabelMin0.status = STARTED
                textLabelMin0.setAutoDraw(True)
            
            # if textLabelMin0 is active this frame...
            if textLabelMin0.status == STARTED:
                # update params
                pass
            
            # *textLabelMax0* updates
            
            # if textLabelMax0 is starting this frame...
            if textLabelMax0.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                textLabelMax0.frameNStart = frameN  # exact frame index
                textLabelMax0.tStart = t  # local t and not account for scr refresh
                textLabelMax0.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(textLabelMax0, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textLabelMax0.started')
                # update status
                textLabelMax0.status = STARTED
                textLabelMax0.setAutoDraw(True)
            
            # if textLabelMax0 is active this frame...
            if textLabelMax0.status == STARTED:
                # update params
                pass
            
            # *tb_continue_sm0* updates
            
            # if tb_continue_sm0 is starting this frame...
            if tb_continue_sm0.status == NOT_STARTED and allowContinue == True:
                # keep track of start time/frame for later
                tb_continue_sm0.frameNStart = frameN  # exact frame index
                tb_continue_sm0.tStart = t  # local t and not account for scr refresh
                tb_continue_sm0.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(tb_continue_sm0, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'tb_continue_sm0.started')
                # update status
                tb_continue_sm0.status = STARTED
                tb_continue_sm0.setAutoDraw(True)
            
            # if tb_continue_sm0 is active this frame...
            if tb_continue_sm0.status == STARTED:
                # update params
                pass
            
            # *key_resp_state_measure0* updates
            waitOnFlip = False
            
            # if key_resp_state_measure0 is starting this frame...
            if key_resp_state_measure0.status == NOT_STARTED and allowContinue == True:
                # keep track of start time/frame for later
                key_resp_state_measure0.frameNStart = frameN  # exact frame index
                key_resp_state_measure0.tStart = t  # local t and not account for scr refresh
                key_resp_state_measure0.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_state_measure0, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_state_measure0.started')
                # update status
                key_resp_state_measure0.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_state_measure0.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_state_measure0.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_state_measure0.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_state_measure0.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_state_measure0_allKeys.extend(theseKeys)
                if len(_key_resp_state_measure0_allKeys):
                    key_resp_state_measure0.keys = _key_resp_state_measure0_allKeys[-1].name  # just the last key pressed
                    key_resp_state_measure0.rt = _key_resp_state_measure0_allKeys[-1].rt
                    key_resp_state_measure0.duration = _key_resp_state_measure0_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *imageSAM0* updates
            
            # if imageSAM0 is starting this frame...
            if imageSAM0.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                imageSAM0.frameNStart = frameN  # exact frame index
                imageSAM0.tStart = t  # local t and not account for scr refresh
                imageSAM0.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(imageSAM0, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'imageSAM0.started')
                # update status
                imageSAM0.status = STARTED
                imageSAM0.setAutoDraw(True)
            
            # if imageSAM0 is active this frame...
            if imageSAM0.status == STARTED:
                # update params
                pass
            # Run 'Each Frame' code from stateMeasureHelper0
            # Check if either slider has been rated
            rated = (slider_generic0.getRating() is not None) or (slider_SAM0.getRating() is not None)
            
            # If rated, allow spacebar to end the routine
            if rated:
                allowContinue = True
            else:
                allowContinue = False
            
            
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
                    currentRoutine=state_measure_intro_2,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                state_measure_intro_2.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if state_measure_intro_2.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in state_measure_intro_2.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "state_measure_intro_2" ---
        for thisComponent in state_measure_intro_2.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for state_measure_intro_2
        state_measure_intro_2.tStop = globalClock.getTime(format='float')
        state_measure_intro_2.tStopRefresh = tThisFlipGlobal
        thisExp.addData('state_measure_intro_2.stopped', state_measure_intro_2.tStop)
        state_measure_01.addData('slider_generic0.response', slider_generic0.getRating())
        state_measure_01.addData('slider_generic0.rt', slider_generic0.getRT())
        state_measure_01.addData('slider_SAM0.response', slider_SAM0.getRating())
        state_measure_01.addData('slider_SAM0.rt', slider_SAM0.getRT())
        # check responses
        if key_resp_state_measure0.keys in ['', [], None]:  # No response was made
            key_resp_state_measure0.keys = None
        state_measure_01.addData('key_resp_state_measure0.keys',key_resp_state_measure0.keys)
        if key_resp_state_measure0.keys != None:  # we had a response
            state_measure_01.addData('key_resp_state_measure0.rt', key_resp_state_measure0.rt)
            state_measure_01.addData('key_resp_state_measure0.duration', key_resp_state_measure0.duration)
        # the Routine "state_measure_intro_2" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisState_measure_01 as finished
        if hasattr(thisState_measure_01, 'status'):
            thisState_measure_01.status = FINISHED
        # if awaiting a pause, pause now
        if state_measure_01.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            state_measure_01.status = STARTED
    # completed 1.0 repeats of 'state_measure_01'
    state_measure_01.status = FINISHED
    
    
    # --- Prepare to start Routine "Intro_01" ---
    # create an object to store info about Routine Intro_01
    Intro_01 = data.Routine(
        name='Intro_01',
        components=[text_7, key_resp_7],
    )
    Intro_01.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_7
    key_resp_7.keys = []
    key_resp_7.rt = []
    _key_resp_7_allKeys = []
    # store start times for Intro_01
    Intro_01.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Intro_01.tStart = globalClock.getTime(format='float')
    Intro_01.status = STARTED
    thisExp.addData('Intro_01.started', Intro_01.tStart)
    Intro_01.maxDuration = None
    # keep track of which components have finished
    Intro_01Components = Intro_01.components
    for thisComponent in Intro_01.components:
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
    
    # --- Run Routine "Intro_01" ---
    thisExp.currentRoutine = Intro_01
    Intro_01.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_7* updates
        
        # if text_7 is starting this frame...
        if text_7.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            text_7.frameNStart = frameN  # exact frame index
            text_7.tStart = t  # local t and not account for scr refresh
            text_7.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_7, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_7.started')
            # update status
            text_7.status = STARTED
            text_7.setAutoDraw(True)
        
        # if text_7 is active this frame...
        if text_7.status == STARTED:
            # update params
            pass
        
        # *key_resp_7* updates
        waitOnFlip = False
        
        # if key_resp_7 is starting this frame...
        if key_resp_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_7.frameNStart = frameN  # exact frame index
            key_resp_7.tStart = t  # local t and not account for scr refresh
            key_resp_7.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_7, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_7.started')
            # update status
            key_resp_7.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_7.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_7.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_7.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_7.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_7_allKeys.extend(theseKeys)
            if len(_key_resp_7_allKeys):
                key_resp_7.keys = _key_resp_7_allKeys[-1].name  # just the last key pressed
                key_resp_7.rt = _key_resp_7_allKeys[-1].rt
                key_resp_7.duration = _key_resp_7_allKeys[-1].duration
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
                currentRoutine=Intro_01,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            Intro_01.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if Intro_01.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in Intro_01.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Intro_01" ---
    for thisComponent in Intro_01.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Intro_01
    Intro_01.tStop = globalClock.getTime(format='float')
    Intro_01.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Intro_01.stopped', Intro_01.tStop)
    # check responses
    if key_resp_7.keys in ['', [], None]:  # No response was made
        key_resp_7.keys = None
    thisExp.addData('key_resp_7.keys',key_resp_7.keys)
    if key_resp_7.keys != None:  # we had a response
        thisExp.addData('key_resp_7.rt', key_resp_7.rt)
        thisExp.addData('key_resp_7.duration', key_resp_7.duration)
    thisExp.nextEntry()
    # the Routine "Intro_01" was not non-slip safe, so reset the non-slip timer
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
        
        # --- Prepare to start Routine "X" ---
        # create an object to store info about Routine X
        X = data.Routine(
            name='X',
            components=[polygon],
        )
        X.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for X
        X.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        X.tStart = globalClock.getTime(format='float')
        X.status = STARTED
        thisExp.addData('X.started', X.tStart)
        X.maxDuration = None
        # keep track of which components have finished
        XComponents = X.components
        for thisComponent in X.components:
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
        
        # --- Run Routine "X" ---
        thisExp.currentRoutine = X
        X.forceEnded = routineForceEnded = not continueRoutine
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
            
            # *polygon* updates
            
            # if polygon is starting this frame...
            if polygon.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                polygon.frameNStart = frameN  # exact frame index
                polygon.tStart = t  # local t and not account for scr refresh
                polygon.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(polygon, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'polygon.started')
                # update status
                polygon.status = STARTED
                polygon.setAutoDraw(True)
            
            # if polygon is active this frame...
            if polygon.status == STARTED:
                # update params
                pass
            
            # if polygon is stopping this frame...
            if polygon.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > polygon.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    polygon.tStop = t  # not accounting for scr refresh
                    polygon.tStopRefresh = tThisFlipGlobal  # on global time
                    polygon.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'polygon.stopped')
                    # update status
                    polygon.status = FINISHED
                    polygon.setAutoDraw(False)
            
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
                    currentRoutine=X,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                X.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if X.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in X.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "X" ---
        for thisComponent in X.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for X
        X.tStop = globalClock.getTime(format='float')
        X.tStopRefresh = tThisFlipGlobal
        thisExp.addData('X.stopped', X.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if X.maxDurationReached:
            routineTimer.addTime(-X.maxDuration)
        elif X.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        
        # --- Prepare to start Routine "pain_adjectives01" ---
        # create an object to store info about Routine pain_adjectives01
        pain_adjectives01 = data.Routine(
            name='pain_adjectives01',
            components=[pain_adjectives_text, pain_adjectives_kb_response, pain_slider_2, pain_tb_continue_sm],
        )
        pain_adjectives01.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for pain_adjectives_kb_response
        pain_adjectives_kb_response.keys = []
        pain_adjectives_kb_response.rt = []
        _pain_adjectives_kb_response_allKeys = []
        pain_slider_2.reset()
        pain_slider_2.setPos((0, slider_y))
        pain_slider_2.setSize((slider_width, slider_height))
        pain_tb_continue_sm.reset()
        # Run 'Begin Routine' code from painHelperCode
        allowContinue = False
        
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
            
            # *pain_adjectives_text* updates
            
            # if pain_adjectives_text is starting this frame...
            if pain_adjectives_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                pain_adjectives_text.frameNStart = frameN  # exact frame index
                pain_adjectives_text.tStart = t  # local t and not account for scr refresh
                pain_adjectives_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(pain_adjectives_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'pain_adjectives_text.started')
                # update status
                pain_adjectives_text.status = STARTED
                pain_adjectives_text.setAutoDraw(True)
            
            # if pain_adjectives_text is active this frame...
            if pain_adjectives_text.status == STARTED:
                # update params
                pain_adjectives_text.setText(pain_adjs, log=False)
            
            # *pain_adjectives_kb_response* updates
            waitOnFlip = False
            
            # if pain_adjectives_kb_response is starting this frame...
            if pain_adjectives_kb_response.status == NOT_STARTED and allowContinue == True:
                # keep track of start time/frame for later
                pain_adjectives_kb_response.frameNStart = frameN  # exact frame index
                pain_adjectives_kb_response.tStart = t  # local t and not account for scr refresh
                pain_adjectives_kb_response.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(pain_adjectives_kb_response, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'pain_adjectives_kb_response.started')
                # update status
                pain_adjectives_kb_response.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(pain_adjectives_kb_response.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(pain_adjectives_kb_response.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if pain_adjectives_kb_response.status == STARTED and not waitOnFlip:
                theseKeys = pain_adjectives_kb_response.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _pain_adjectives_kb_response_allKeys.extend(theseKeys)
                if len(_pain_adjectives_kb_response_allKeys):
                    pain_adjectives_kb_response.keys = _pain_adjectives_kb_response_allKeys[-1].name  # just the last key pressed
                    pain_adjectives_kb_response.rt = _pain_adjectives_kb_response_allKeys[-1].rt
                    pain_adjectives_kb_response.duration = _pain_adjectives_kb_response_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *pain_slider_2* updates
            
            # if pain_slider_2 is starting this frame...
            if pain_slider_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                pain_slider_2.frameNStart = frameN  # exact frame index
                pain_slider_2.tStart = t  # local t and not account for scr refresh
                pain_slider_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(pain_slider_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'pain_slider_2.started')
                # update status
                pain_slider_2.status = STARTED
                pain_slider_2.setAutoDraw(True)
            
            # if pain_slider_2 is active this frame...
            if pain_slider_2.status == STARTED:
                # update params
                pass
            
            # *pain_tb_continue_sm* updates
            
            # if pain_tb_continue_sm is starting this frame...
            if pain_tb_continue_sm.status == NOT_STARTED and allowContinue == True:
                # keep track of start time/frame for later
                pain_tb_continue_sm.frameNStart = frameN  # exact frame index
                pain_tb_continue_sm.tStart = t  # local t and not account for scr refresh
                pain_tb_continue_sm.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(pain_tb_continue_sm, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'pain_tb_continue_sm.started')
                # update status
                pain_tb_continue_sm.status = STARTED
                pain_tb_continue_sm.setAutoDraw(True)
            
            # if pain_tb_continue_sm is active this frame...
            if pain_tb_continue_sm.status == STARTED:
                # update params
                pass
            # Run 'Each Frame' code from painHelperCode
            # Check if either slider has been rated
            rated = (pain_slider_2.getRating() is not None) 
            
            # If rated, allow spacebar to end the routine
            if rated:
                allowContinue = True
            else:
                allowContinue = False
            
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
        # check responses
        if pain_adjectives_kb_response.keys in ['', [], None]:  # No response was made
            pain_adjectives_kb_response.keys = None
        pain_adjectives_loop.addData('pain_adjectives_kb_response.keys',pain_adjectives_kb_response.keys)
        if pain_adjectives_kb_response.keys != None:  # we had a response
            pain_adjectives_loop.addData('pain_adjectives_kb_response.rt', pain_adjectives_kb_response.rt)
            pain_adjectives_loop.addData('pain_adjectives_kb_response.duration', pain_adjectives_kb_response.duration)
        pain_adjectives_loop.addData('pain_slider_2.response', pain_slider_2.getRating())
        pain_adjectives_loop.addData('pain_slider_2.rt', pain_slider_2.getRT())
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
        
        # --- Prepare to start Routine "X2" ---
        # create an object to store info about Routine X2
        X2 = data.Routine(
            name='X2',
            components=[polygon_3],
        )
        X2.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for X2
        X2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        X2.tStart = globalClock.getTime(format='float')
        X2.status = STARTED
        thisExp.addData('X2.started', X2.tStart)
        X2.maxDuration = None
        # keep track of which components have finished
        X2Components = X2.components
        for thisComponent in X2.components:
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
        
        # --- Run Routine "X2" ---
        thisExp.currentRoutine = X2
        X2.forceEnded = routineForceEnded = not continueRoutine
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
            
            # *polygon_3* updates
            
            # if polygon_3 is starting this frame...
            if polygon_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                polygon_3.frameNStart = frameN  # exact frame index
                polygon_3.tStart = t  # local t and not account for scr refresh
                polygon_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(polygon_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'polygon_3.started')
                # update status
                polygon_3.status = STARTED
                polygon_3.setAutoDraw(True)
            
            # if polygon_3 is active this frame...
            if polygon_3.status == STARTED:
                # update params
                pass
            
            # if polygon_3 is stopping this frame...
            if polygon_3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > polygon_3.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    polygon_3.tStop = t  # not accounting for scr refresh
                    polygon_3.tStopRefresh = tThisFlipGlobal  # on global time
                    polygon_3.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'polygon_3.stopped')
                    # update status
                    polygon_3.status = FINISHED
                    polygon_3.setAutoDraw(False)
            
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
                    currentRoutine=X2,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                X2.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if X2.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in X2.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "X2" ---
        for thisComponent in X2.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for X2
        X2.tStop = globalClock.getTime(format='float')
        X2.tStopRefresh = tThisFlipGlobal
        thisExp.addData('X2.stopped', X2.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if X2.maxDurationReached:
            routineTimer.addTime(-X2.maxDuration)
        elif X2.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        
        # --- Prepare to start Routine "pain_adjectives02" ---
        # create an object to store info about Routine pain_adjectives02
        pain_adjectives02 = data.Routine(
            name='pain_adjectives02',
            components=[pain_adjectives2_text, pain_adjectives_key_resp02, pain_continue_tb2],
        )
        pain_adjectives02.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        pain_adjectives2_text.setText(pain_adjs)
        # create starting attributes for pain_adjectives_key_resp02
        pain_adjectives_key_resp02.keys = []
        pain_adjectives_key_resp02.rt = []
        _pain_adjectives_key_resp02_allKeys = []
        pain_continue_tb2.reset()
        pain_continue_tb2.setText('Press SPACEBAR to continue')
        # store start times for pain_adjectives02
        pain_adjectives02.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        pain_adjectives02.tStart = globalClock.getTime(format='float')
        pain_adjectives02.status = STARTED
        thisExp.addData('pain_adjectives02.started', pain_adjectives02.tStart)
        pain_adjectives02.maxDuration = None
        # keep track of which components have finished
        pain_adjectives02Components = pain_adjectives02.components
        for thisComponent in pain_adjectives02.components:
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
        
        # --- Run Routine "pain_adjectives02" ---
        thisExp.currentRoutine = pain_adjectives02
        pain_adjectives02.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisPain_adjectives_loop2, 'status') and thisPain_adjectives_loop2.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *pain_adjectives2_text* updates
            
            # if pain_adjectives2_text is starting this frame...
            if pain_adjectives2_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                pain_adjectives2_text.frameNStart = frameN  # exact frame index
                pain_adjectives2_text.tStart = t  # local t and not account for scr refresh
                pain_adjectives2_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(pain_adjectives2_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'pain_adjectives2_text.started')
                # update status
                pain_adjectives2_text.status = STARTED
                pain_adjectives2_text.setAutoDraw(True)
            
            # if pain_adjectives2_text is active this frame...
            if pain_adjectives2_text.status == STARTED:
                # update params
                pass
            
            # *pain_adjectives_key_resp02* updates
            waitOnFlip = False
            
            # if pain_adjectives_key_resp02 is starting this frame...
            if pain_adjectives_key_resp02.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                pain_adjectives_key_resp02.frameNStart = frameN  # exact frame index
                pain_adjectives_key_resp02.tStart = t  # local t and not account for scr refresh
                pain_adjectives_key_resp02.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(pain_adjectives_key_resp02, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'pain_adjectives_key_resp02.started')
                # update status
                pain_adjectives_key_resp02.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(pain_adjectives_key_resp02.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(pain_adjectives_key_resp02.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if pain_adjectives_key_resp02.status == STARTED and not waitOnFlip:
                theseKeys = pain_adjectives_key_resp02.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _pain_adjectives_key_resp02_allKeys.extend(theseKeys)
                if len(_pain_adjectives_key_resp02_allKeys):
                    pain_adjectives_key_resp02.keys = _pain_adjectives_key_resp02_allKeys[-1].name  # just the last key pressed
                    pain_adjectives_key_resp02.rt = _pain_adjectives_key_resp02_allKeys[-1].rt
                    pain_adjectives_key_resp02.duration = _pain_adjectives_key_resp02_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *pain_continue_tb2* updates
            
            # if pain_continue_tb2 is starting this frame...
            if pain_continue_tb2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                pain_continue_tb2.frameNStart = frameN  # exact frame index
                pain_continue_tb2.tStart = t  # local t and not account for scr refresh
                pain_continue_tb2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(pain_continue_tb2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'pain_continue_tb2.started')
                # update status
                pain_continue_tb2.status = STARTED
                pain_continue_tb2.setAutoDraw(True)
            
            # if pain_continue_tb2 is active this frame...
            if pain_continue_tb2.status == STARTED:
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
                    currentRoutine=pain_adjectives02,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                pain_adjectives02.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if pain_adjectives02.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in pain_adjectives02.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "pain_adjectives02" ---
        for thisComponent in pain_adjectives02.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for pain_adjectives02
        pain_adjectives02.tStop = globalClock.getTime(format='float')
        pain_adjectives02.tStopRefresh = tThisFlipGlobal
        thisExp.addData('pain_adjectives02.stopped', pain_adjectives02.tStop)
        # check responses
        if pain_adjectives_key_resp02.keys in ['', [], None]:  # No response was made
            pain_adjectives_key_resp02.keys = None
        pain_adjectives_loop2.addData('pain_adjectives_key_resp02.keys',pain_adjectives_key_resp02.keys)
        if pain_adjectives_key_resp02.keys != None:  # we had a response
            pain_adjectives_loop2.addData('pain_adjectives_key_resp02.rt', pain_adjectives_key_resp02.rt)
            pain_adjectives_loop2.addData('pain_adjectives_key_resp02.duration', pain_adjectives_key_resp02.duration)
        # the Routine "pain_adjectives02" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
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
        'C:/Users/colin/Desktop/OUPI/Research Projects/Sophia_You/You_Pain/shared/loop-templates/loopSpontaneousPainExpected.xlsx', 
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
        
        # --- Prepare to start Routine "X_2" ---
        # create an object to store info about Routine X_2
        X_2 = data.Routine(
            name='X_2',
            components=[polygon_2],
        )
        X_2.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for X_2
        X_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        X_2.tStart = globalClock.getTime(format='float')
        X_2.status = STARTED
        thisExp.addData('X_2.started', X_2.tStart)
        X_2.maxDuration = None
        # keep track of which components have finished
        X_2Components = X_2.components
        for thisComponent in X_2.components:
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
        
        # --- Run Routine "X_2" ---
        thisExp.currentRoutine = X_2
        X_2.forceEnded = routineForceEnded = not continueRoutine
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
            
            # *polygon_2* updates
            
            # if polygon_2 is starting this frame...
            if polygon_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                polygon_2.frameNStart = frameN  # exact frame index
                polygon_2.tStart = t  # local t and not account for scr refresh
                polygon_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(polygon_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'polygon_2.started')
                # update status
                polygon_2.status = STARTED
                polygon_2.setAutoDraw(True)
            
            # if polygon_2 is active this frame...
            if polygon_2.status == STARTED:
                # update params
                pass
            
            # if polygon_2 is stopping this frame...
            if polygon_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > polygon_2.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    polygon_2.tStop = t  # not accounting for scr refresh
                    polygon_2.tStopRefresh = tThisFlipGlobal  # on global time
                    polygon_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'polygon_2.stopped')
                    # update status
                    polygon_2.status = FINISHED
                    polygon_2.setAutoDraw(False)
            
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
                    currentRoutine=X_2,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                X_2.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if X_2.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in X_2.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "X_2" ---
        for thisComponent in X_2.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for X_2
        X_2.tStop = globalClock.getTime(format='float')
        X_2.tStopRefresh = tThisFlipGlobal
        thisExp.addData('X_2.stopped', X_2.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if X_2.maxDurationReached:
            routineTimer.addTime(-X_2.maxDuration)
        elif X_2.forceEnded:
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
    state_measure_02 = data.TrialHandler2(
        name='state_measure_02',
        nReps=1.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('D:/Github/neuroflare-experiments/neuroflare/shared/loop-templates/loopStateMeasureOutro.xlsx'), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(state_measure_02)  # add the loop to the experiment
    thisState_measure_02 = state_measure_02.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisState_measure_02.rgb)
    if thisState_measure_02 != None:
        for paramName in thisState_measure_02:
            globals()[paramName] = thisState_measure_02[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisState_measure_02 in state_measure_02:
        state_measure_02.status = STARTED
        if hasattr(thisState_measure_02, 'status'):
            thisState_measure_02.status = STARTED
        currentLoop = state_measure_02
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisState_measure_02.rgb)
        if thisState_measure_02 != None:
            for paramName in thisState_measure_02:
                globals()[paramName] = thisState_measure_02[paramName]
        
        # --- Prepare to start Routine "state_measure_outro" ---
        # create an object to store info about Routine state_measure_outro
        state_measure_outro = data.Routine(
            name='state_measure_outro',
            components=[sliderGeneric2, sliderSAM2, imageSAM2, textStateMeasure2, textLabelMin2, textLabelMax2, tb_continue_02, key_resp_state_measure2],
        )
        state_measure_outro.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        sliderGeneric2.reset()
        sliderGeneric2.setSize((slider_gen_width, 0.1))
        sliderSAM2.reset()
        sliderSAM2.setPos((0, slider_y))
        sliderSAM2.setSize((slider_width, slider_height))
        imageSAM2.setPos((0, sam_y))
        imageSAM2.setSize((sam_width, sam_height))
        imageSAM2.setImage(picture_path)
        textStateMeasure2.setPos((0, 0.35))
        textStateMeasure2.setText(rating_message)
        textLabelMin2.setPos((tick_positions[0], label_y))
        textLabelMin2.setText(rating_min_label)
        textLabelMax2.setPos((tick_positions[-1], label_y))
        textLabelMax2.setText(rating_max_label)
        tb_continue_02.reset()
        tb_continue_02.setPos((0, -0.40))
        tb_continue_02.setSize((0.5, 0.25))
        # Run 'Begin Routine' code from stateMeasureHelper2
        allowContinue = False
        # create starting attributes for key_resp_state_measure2
        key_resp_state_measure2.keys = []
        key_resp_state_measure2.rt = []
        _key_resp_state_measure2_allKeys = []
        # store start times for state_measure_outro
        state_measure_outro.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        state_measure_outro.tStart = globalClock.getTime(format='float')
        state_measure_outro.status = STARTED
        thisExp.addData('state_measure_outro.started', state_measure_outro.tStart)
        state_measure_outro.maxDuration = None
        # keep track of which components have finished
        state_measure_outroComponents = state_measure_outro.components
        for thisComponent in state_measure_outro.components:
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
        
        # --- Run Routine "state_measure_outro" ---
        thisExp.currentRoutine = state_measure_outro
        state_measure_outro.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisState_measure_02, 'status') and thisState_measure_02.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *sliderGeneric2* updates
            
            # if sliderGeneric2 is starting this frame...
            if sliderGeneric2.status == NOT_STARTED and rating_type == 'Generic':
                # keep track of start time/frame for later
                sliderGeneric2.frameNStart = frameN  # exact frame index
                sliderGeneric2.tStart = t  # local t and not account for scr refresh
                sliderGeneric2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(sliderGeneric2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'sliderGeneric2.started')
                # update status
                sliderGeneric2.status = STARTED
                sliderGeneric2.setAutoDraw(True)
            
            # if sliderGeneric2 is active this frame...
            if sliderGeneric2.status == STARTED:
                # update params
                pass
            
            # if sliderGeneric2 is stopping this frame...
            if sliderGeneric2.status == STARTED:
                if bool(rating_type != 'Generic'):
                    # keep track of stop time/frame for later
                    sliderGeneric2.tStop = t  # not accounting for scr refresh
                    sliderGeneric2.tStopRefresh = tThisFlipGlobal  # on global time
                    sliderGeneric2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'sliderGeneric2.stopped')
                    # update status
                    sliderGeneric2.status = FINISHED
                    sliderGeneric2.setAutoDraw(False)
            
            # *sliderSAM2* updates
            
            # if sliderSAM2 is starting this frame...
            if sliderSAM2.status == NOT_STARTED and rating_type == 'SAM':
                # keep track of start time/frame for later
                sliderSAM2.frameNStart = frameN  # exact frame index
                sliderSAM2.tStart = t  # local t and not account for scr refresh
                sliderSAM2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(sliderSAM2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'sliderSAM2.started')
                # update status
                sliderSAM2.status = STARTED
                sliderSAM2.setAutoDraw(True)
            
            # if sliderSAM2 is active this frame...
            if sliderSAM2.status == STARTED:
                # update params
                pass
            
            # if sliderSAM2 is stopping this frame...
            if sliderSAM2.status == STARTED:
                if bool(rating_type != 'SAM'):
                    # keep track of stop time/frame for later
                    sliderSAM2.tStop = t  # not accounting for scr refresh
                    sliderSAM2.tStopRefresh = tThisFlipGlobal  # on global time
                    sliderSAM2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'sliderSAM2.stopped')
                    # update status
                    sliderSAM2.status = FINISHED
                    sliderSAM2.setAutoDraw(False)
            
            # *imageSAM2* updates
            
            # if imageSAM2 is starting this frame...
            if imageSAM2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                imageSAM2.frameNStart = frameN  # exact frame index
                imageSAM2.tStart = t  # local t and not account for scr refresh
                imageSAM2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(imageSAM2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'imageSAM2.started')
                # update status
                imageSAM2.status = STARTED
                imageSAM2.setAutoDraw(True)
            
            # if imageSAM2 is active this frame...
            if imageSAM2.status == STARTED:
                # update params
                pass
            
            # *textStateMeasure2* updates
            
            # if textStateMeasure2 is starting this frame...
            if textStateMeasure2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                textStateMeasure2.frameNStart = frameN  # exact frame index
                textStateMeasure2.tStart = t  # local t and not account for scr refresh
                textStateMeasure2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(textStateMeasure2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textStateMeasure2.started')
                # update status
                textStateMeasure2.status = STARTED
                textStateMeasure2.setAutoDraw(True)
            
            # if textStateMeasure2 is active this frame...
            if textStateMeasure2.status == STARTED:
                # update params
                pass
            
            # *textLabelMin2* updates
            
            # if textLabelMin2 is starting this frame...
            if textLabelMin2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                textLabelMin2.frameNStart = frameN  # exact frame index
                textLabelMin2.tStart = t  # local t and not account for scr refresh
                textLabelMin2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(textLabelMin2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textLabelMin2.started')
                # update status
                textLabelMin2.status = STARTED
                textLabelMin2.setAutoDraw(True)
            
            # if textLabelMin2 is active this frame...
            if textLabelMin2.status == STARTED:
                # update params
                pass
            
            # *textLabelMax2* updates
            
            # if textLabelMax2 is starting this frame...
            if textLabelMax2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                textLabelMax2.frameNStart = frameN  # exact frame index
                textLabelMax2.tStart = t  # local t and not account for scr refresh
                textLabelMax2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(textLabelMax2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textLabelMax2.started')
                # update status
                textLabelMax2.status = STARTED
                textLabelMax2.setAutoDraw(True)
            
            # if textLabelMax2 is active this frame...
            if textLabelMax2.status == STARTED:
                # update params
                pass
            
            # *tb_continue_02* updates
            
            # if tb_continue_02 is starting this frame...
            if tb_continue_02.status == NOT_STARTED and allowContinue == True:
                # keep track of start time/frame for later
                tb_continue_02.frameNStart = frameN  # exact frame index
                tb_continue_02.tStart = t  # local t and not account for scr refresh
                tb_continue_02.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(tb_continue_02, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'tb_continue_02.started')
                # update status
                tb_continue_02.status = STARTED
                tb_continue_02.setAutoDraw(True)
            
            # if tb_continue_02 is active this frame...
            if tb_continue_02.status == STARTED:
                # update params
                pass
            # Run 'Each Frame' code from stateMeasureHelper2
            # Check if either slider has been rated
            rated = (sliderGeneric2.getRating() is not None) or (sliderSAM2.getRating() is not None)
            
            # If rated, allow spacebar to end the routine
            if rated:
                allowContinue = True
            else:
                allowContinue = False
            
            # *key_resp_state_measure2* updates
            waitOnFlip = False
            
            # if key_resp_state_measure2 is starting this frame...
            if key_resp_state_measure2.status == NOT_STARTED and allowContinue == True:
                # keep track of start time/frame for later
                key_resp_state_measure2.frameNStart = frameN  # exact frame index
                key_resp_state_measure2.tStart = t  # local t and not account for scr refresh
                key_resp_state_measure2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_state_measure2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_state_measure2.started')
                # update status
                key_resp_state_measure2.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_state_measure2.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_state_measure2.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_state_measure2.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_state_measure2.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_state_measure2_allKeys.extend(theseKeys)
                if len(_key_resp_state_measure2_allKeys):
                    key_resp_state_measure2.keys = _key_resp_state_measure2_allKeys[-1].name  # just the last key pressed
                    key_resp_state_measure2.rt = _key_resp_state_measure2_allKeys[-1].rt
                    key_resp_state_measure2.duration = _key_resp_state_measure2_allKeys[-1].duration
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
                    currentRoutine=state_measure_outro,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                state_measure_outro.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if state_measure_outro.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in state_measure_outro.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "state_measure_outro" ---
        for thisComponent in state_measure_outro.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for state_measure_outro
        state_measure_outro.tStop = globalClock.getTime(format='float')
        state_measure_outro.tStopRefresh = tThisFlipGlobal
        thisExp.addData('state_measure_outro.stopped', state_measure_outro.tStop)
        state_measure_02.addData('sliderGeneric2.response', sliderGeneric2.getRating())
        state_measure_02.addData('sliderGeneric2.rt', sliderGeneric2.getRT())
        state_measure_02.addData('sliderSAM2.response', sliderSAM2.getRating())
        state_measure_02.addData('sliderSAM2.rt', sliderSAM2.getRT())
        # check responses
        if key_resp_state_measure2.keys in ['', [], None]:  # No response was made
            key_resp_state_measure2.keys = None
        state_measure_02.addData('key_resp_state_measure2.keys',key_resp_state_measure2.keys)
        if key_resp_state_measure2.keys != None:  # we had a response
            state_measure_02.addData('key_resp_state_measure2.rt', key_resp_state_measure2.rt)
            state_measure_02.addData('key_resp_state_measure2.duration', key_resp_state_measure2.duration)
        # the Routine "state_measure_outro" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisState_measure_02 as finished
        if hasattr(thisState_measure_02, 'status'):
            thisState_measure_02.status = FINISHED
        # if awaiting a pause, pause now
        if state_measure_02.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            state_measure_02.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'state_measure_02'
    state_measure_02.status = FINISHED
    
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
