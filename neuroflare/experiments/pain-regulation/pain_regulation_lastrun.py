#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2025.2.3),
    on March 17, 2026, at 23:37
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
expName = 'pain_regulation'  # from the Builder filename that created this script
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
        originPath='D:\\Github\\neuroflare-experiments\\neuroflare\\experiments\\pain-regulation\\pain_regulation_lastrun.py',
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
    welcome_text = visual.TextBox2(
         win, text='Pain Regulation Task', placeholder='Type here...', font='Arial',
         ori=0.0, pos=(0, 0.05), draggable=False,      letterHeight=0.07,
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
         name='welcome_text',
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
    key_resp_welcome = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "state_measure1" ---
    sliderGeneric = visual.Slider(win=win, name='sliderGeneric',
        startValue=None, size=1.0, pos=(0, 0), units=win.units,
        labels=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), ticks=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), granularity=1.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Noto Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=0, readOnly=False)
    sliderSAM = visual.Slider(win=win, name='sliderSAM',
        startValue=None, size=1.0, pos=[0,0], units=win.units,
        labels=[1, 2, 3, 4, 5, 6, 7, 8, 9],ticks=None, granularity=1,
        style='radio', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Noto Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-1, readOnly=False)
    textStateMeasure = visual.TextStim(win=win, name='textStateMeasure',
        text='',
        font='Arial',
        pos=(0, 0.35), draggable=False, height=0.07, wrapWidth=1.5, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    textLabelMin = visual.TextStim(win=win, name='textLabelMin',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=0.5, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    textLabelMax = visual.TextStim(win=win, name='textLabelMax',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=0.5, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    tb_continue_sm = visual.TextBox2(
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
         name='tb_continue_sm',
         depth=-5, autoLog=True,
    )
    key_resp_state_measure = keyboard.Keyboard(deviceName='defaultKeyboard')
    imageSAM = visual.ImageStim(
        win=win,
        name='imageSAM', units='height', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=1.0,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-7.0)
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
    
    
    # --- Initialize components for Routine "Instruc01" ---
    text_2 = visual.TextStim(win=win, name='text_2',
        text='Please relax and remain as still as possitble while we record your brain activity.\n\nFollow the on-screen instructions.\n\nPress the SPACEBAR to continue.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_01 = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "Instruct02" ---
    text_3 = visual.TextStim(win=win, name='text_3',
        text='You will be asked to imagine the content of a written script and use the suggested mental images to modulate your body pain:\n\nPress the SPACEBAR to continue.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "cross" ---
    polygon = visual.ShapeStim(
        win=win, name='polygon', vertices='cross',
        size=(0.35, 0.35),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "training_instructions" ---
    train_instr_text = visual.TextStim(win=win, name='train_instr_text',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "train_stimulus" ---
    polygon_2 = visual.ShapeStim(
        win=win, name='polygon_2',
        size=(0.35, 0.35), vertices='triangle',
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "train_resp" ---
    text_4 = visual.TextStim(win=win, name='text_4',
        text='Please rate how you felt the stimulus as Painful (P) or Not Painful (N)',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_train_resp = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "end_train" ---
    text_5 = visual.TextStim(win=win, name='text_5',
        text='This concludes the training session.\n\nNow we begin the trials.\n\nPress the SPACEBAR to continue.',
        font='Arial',
        pos=(0, 0.15), draggable=False, height=0.05, wrapWidth=1.5, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    end_train_resp = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "pain_reg_instr01" ---
    pain_reg_instr_text = visual.TextStim(win=win, name='pain_reg_instr_text',
        text='',
        font='Arial',
        pos=(0, 0.15), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "pain_reg_stimulus" ---
    # Run 'Begin Experiment' code from codehelper_timing
    x = 0
    polygon_3 = visual.ShapeStim(
        win=win, name='polygon_3',
        size=[1.0, 1.0], vertices='triangle',
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    
    # --- Initialize components for Routine "pain_reg_resp" ---
    pain_reg_resp_text = visual.TextStim(win=win, name='pain_reg_resp_text',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_2 = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "pain_reg_state_measure" ---
    sliderGenericPainReg = visual.Slider(win=win, name='sliderGenericPainReg',
        startValue=None, size=1.0, pos=(0, 0), units=win.units,
        labels=[0,1,2,3,4,5,6,7,8,9,10], ticks=( 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), granularity=1.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Noto Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=0, readOnly=False)
    sliderSAMPainReg = visual.Slider(win=win, name='sliderSAMPainReg',
        startValue=None, size=1.0, pos=[0,0], units=win.units,
        labels=[1, 2, 3, 4, 5, 6, 7, 8, 9],ticks=None, granularity=1,
        style='radio', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Noto Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-1, readOnly=False)
    imageSAMPainReg = visual.ImageStim(
        win=win,
        name='imageSAMPainReg', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=1.0,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    textStateMeasurePainReg = visual.TextStim(win=win, name='textStateMeasurePainReg',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=1.5, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    textStateMeasureLabelMin = visual.TextStim(win=win, name='textStateMeasureLabelMin',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=0.5, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    textStateMeasureLabelMax = visual.TextStim(win=win, name='textStateMeasureLabelMax',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=0.5, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    tb_continue_painreg = visual.TextBox2(
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
         name='tb_continue_painreg',
         depth=-6, autoLog=True,
    )
    key_resp_state_measure_pain_reg = keyboard.Keyboard(deviceName='defaultKeyboard')
    # Run 'Begin Experiment' code from codepainreghelper
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
    
    # --- Initialize components for Routine "routine_5sec_block_pause" ---
    pause = visual.TextStim(win=win, name='pause',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "state_measure_end" ---
    sliderStateMeasureEnd = visual.Slider(win=win, name='sliderStateMeasureEnd',
        startValue=None, size=1.0, pos=(0,0), units=win.units,
        labels=[0,1,2,3,4,5,6,7,8,9,10], ticks=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), granularity=1.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Noto Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=0, readOnly=False)
    sliderSAMStateMeasureEnd = visual.Slider(win=win, name='sliderSAMStateMeasureEnd',
        startValue=None, size=1.0, pos=[0,0], units=win.units,
        labels=None,ticks=None, granularity=1,
        style='radio', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Noto Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-1, readOnly=False)
    imageSAMStateMeasureEnd = visual.ImageStim(
        win=win,
        name='imageSAMStateMeasureEnd', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=1.0,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    textStateMeasurePainRegEnd = visual.TextStim(win=win, name='textStateMeasurePainRegEnd',
        text='',
        font='Arial',
        pos=(0, 0.35), draggable=False, height=0.05, wrapWidth=1.5, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    textLabelMinStateMeasureEnd = visual.TextStim(win=win, name='textLabelMinStateMeasureEnd',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    textLabelMaxStateMeasureEnd = visual.TextStim(win=win, name='textLabelMaxStateMeasureEnd',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    textbox_continue_State_measure_End = visual.TextBox2(
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
         name='textbox_continue_State_measure_End',
         depth=-6, autoLog=True,
    )
    # Run 'Begin Experiment' code from codeStateMeasureEndHelper
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
    
    key_resp_StateMeasureEnd = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "end" ---
    text = visual.TextStim(win=win, name='text',
        text='End of this section.\nThank you.',
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
        components=[welcome_text, tb_continue, key_resp_welcome],
    )
    welcome.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    welcome_text.reset()
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
    thisExp.currentRoutine = welcome
    welcome.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *welcome_text* updates
        
        # if welcome_text is starting this frame...
        if welcome_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            welcome_text.frameNStart = frameN  # exact frame index
            welcome_text.tStart = t  # local t and not account for scr refresh
            welcome_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(welcome_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'welcome_text.started')
            # update status
            welcome_text.status = STARTED
            welcome_text.setAutoDraw(True)
        
        # if welcome_text is active this frame...
        if welcome_text.status == STARTED:
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
    if key_resp_welcome.keys in ['', [], None]:  # No response was made
        key_resp_welcome.keys = None
    thisExp.addData('key_resp_welcome.keys',key_resp_welcome.keys)
    if key_resp_welcome.keys != None:  # we had a response
        thisExp.addData('key_resp_welcome.rt', key_resp_welcome.rt)
        thisExp.addData('key_resp_welcome.duration', key_resp_welcome.duration)
    thisExp.nextEntry()
    # the Routine "welcome" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    state_measure_loop1 = data.TrialHandler2(
        name='state_measure_loop1',
        nReps=1.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions(
        'loopStateMeasurePainReg.xlsx', 
        selection='1:8'
    )
    , 
        seed=None, 
        isTrials=True, 
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
        
        # --- Prepare to start Routine "state_measure1" ---
        # create an object to store info about Routine state_measure1
        state_measure1 = data.Routine(
            name='state_measure1',
            components=[sliderGeneric, sliderSAM, textStateMeasure, textLabelMin, textLabelMax, tb_continue_sm, key_resp_state_measure, imageSAM],
        )
        state_measure1.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        sliderGeneric.reset()
        sliderGeneric.setSize((slider_gen_width, 0.1))
        sliderSAM.reset()
        sliderSAM.setPos((0, slider_y))
        sliderSAM.setSize((slider_width, slider_height))
        textStateMeasure.setText(rating_message)
        textLabelMin.setPos((tick_positions[0],label_y))
        textLabelMin.setText(rating_min_label)
        textLabelMax.setPos((tick_positions[-1], label_y))
        textLabelMax.setText(rating_max_label)
        tb_continue_sm.reset()
        # create starting attributes for key_resp_state_measure
        key_resp_state_measure.keys = []
        key_resp_state_measure.rt = []
        _key_resp_state_measure_allKeys = []
        imageSAM.setPos((0, sam_y))
        imageSAM.setSize((sam_width, sam_height))
        imageSAM.setImage(picture_path)
        # Run 'Begin Routine' code from stateMeasureHelper
        allowContinue = False
        # store start times for state_measure1
        state_measure1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        state_measure1.tStart = globalClock.getTime(format='float')
        state_measure1.status = STARTED
        thisExp.addData('state_measure1.started', state_measure1.tStart)
        state_measure1.maxDuration = None
        # keep track of which components have finished
        state_measure1Components = state_measure1.components
        for thisComponent in state_measure1.components:
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
        
        # --- Run Routine "state_measure1" ---
        thisExp.currentRoutine = state_measure1
        state_measure1.forceEnded = routineForceEnded = not continueRoutine
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
            # Run 'Each Frame' code from stateMeasureHelper
            # Check if either slider has been rated
            rated = (sliderGeneric.getRating() is not None) or (sliderSAM.getRating() is not None)
            
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
                    currentRoutine=state_measure1,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                state_measure1.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if state_measure1.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in state_measure1.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "state_measure1" ---
        for thisComponent in state_measure1.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for state_measure1
        state_measure1.tStop = globalClock.getTime(format='float')
        state_measure1.tStopRefresh = tThisFlipGlobal
        thisExp.addData('state_measure1.stopped', state_measure1.tStop)
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
        # the Routine "state_measure1" was not non-slip safe, so reset the non-slip timer
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
    
    # --- Prepare to start Routine "Instruc01" ---
    # create an object to store info about Routine Instruc01
    Instruc01 = data.Routine(
        name='Instruc01',
        components=[text_2, key_resp_01],
    )
    Instruc01.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_01
    key_resp_01.keys = []
    key_resp_01.rt = []
    _key_resp_01_allKeys = []
    # store start times for Instruc01
    Instruc01.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Instruc01.tStart = globalClock.getTime(format='float')
    Instruc01.status = STARTED
    thisExp.addData('Instruc01.started', Instruc01.tStart)
    Instruc01.maxDuration = None
    # keep track of which components have finished
    Instruc01Components = Instruc01.components
    for thisComponent in Instruc01.components:
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
    
    # --- Run Routine "Instruc01" ---
    thisExp.currentRoutine = Instruc01
    Instruc01.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_2* updates
        
        # if text_2 is starting this frame...
        if text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_2.frameNStart = frameN  # exact frame index
            text_2.tStart = t  # local t and not account for scr refresh
            text_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_2.started')
            # update status
            text_2.status = STARTED
            text_2.setAutoDraw(True)
        
        # if text_2 is active this frame...
        if text_2.status == STARTED:
            # update params
            pass
        
        # *key_resp_01* updates
        waitOnFlip = False
        
        # if key_resp_01 is starting this frame...
        if key_resp_01.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_01.frameNStart = frameN  # exact frame index
            key_resp_01.tStart = t  # local t and not account for scr refresh
            key_resp_01.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_01, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_01.started')
            # update status
            key_resp_01.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_01.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_01.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_01.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_01.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_01_allKeys.extend(theseKeys)
            if len(_key_resp_01_allKeys):
                key_resp_01.keys = _key_resp_01_allKeys[-1].name  # just the last key pressed
                key_resp_01.rt = _key_resp_01_allKeys[-1].rt
                key_resp_01.duration = _key_resp_01_allKeys[-1].duration
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
                currentRoutine=Instruc01,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            Instruc01.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if Instruc01.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in Instruc01.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Instruc01" ---
    for thisComponent in Instruc01.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Instruc01
    Instruc01.tStop = globalClock.getTime(format='float')
    Instruc01.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Instruc01.stopped', Instruc01.tStop)
    # check responses
    if key_resp_01.keys in ['', [], None]:  # No response was made
        key_resp_01.keys = None
    thisExp.addData('key_resp_01.keys',key_resp_01.keys)
    if key_resp_01.keys != None:  # we had a response
        thisExp.addData('key_resp_01.rt', key_resp_01.rt)
        thisExp.addData('key_resp_01.duration', key_resp_01.duration)
    thisExp.nextEntry()
    # the Routine "Instruc01" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "Instruct02" ---
    # create an object to store info about Routine Instruct02
    Instruct02 = data.Routine(
        name='Instruct02',
        components=[text_3, key_resp],
    )
    Instruct02.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp
    key_resp.keys = []
    key_resp.rt = []
    _key_resp_allKeys = []
    # store start times for Instruct02
    Instruct02.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Instruct02.tStart = globalClock.getTime(format='float')
    Instruct02.status = STARTED
    thisExp.addData('Instruct02.started', Instruct02.tStart)
    Instruct02.maxDuration = None
    # keep track of which components have finished
    Instruct02Components = Instruct02.components
    for thisComponent in Instruct02.components:
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
    
    # --- Run Routine "Instruct02" ---
    thisExp.currentRoutine = Instruct02
    Instruct02.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_3* updates
        
        # if text_3 is starting this frame...
        if text_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_3.frameNStart = frameN  # exact frame index
            text_3.tStart = t  # local t and not account for scr refresh
            text_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_3.started')
            # update status
            text_3.status = STARTED
            text_3.setAutoDraw(True)
        
        # if text_3 is active this frame...
        if text_3.status == STARTED:
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
                currentRoutine=Instruct02,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            Instruct02.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if Instruct02.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in Instruct02.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Instruct02" ---
    for thisComponent in Instruct02.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Instruct02
    Instruct02.tStop = globalClock.getTime(format='float')
    Instruct02.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Instruct02.stopped', Instruct02.tStop)
    # check responses
    if key_resp.keys in ['', [], None]:  # No response was made
        key_resp.keys = None
    thisExp.addData('key_resp.keys',key_resp.keys)
    if key_resp.keys != None:  # we had a response
        thisExp.addData('key_resp.rt', key_resp.rt)
        thisExp.addData('key_resp.duration', key_resp.duration)
    thisExp.nextEntry()
    # the Routine "Instruct02" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "cross" ---
    # create an object to store info about Routine cross
    cross = data.Routine(
        name='cross',
        components=[polygon],
    )
    cross.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for cross
    cross.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    cross.tStart = globalClock.getTime(format='float')
    cross.status = STARTED
    thisExp.addData('cross.started', cross.tStart)
    cross.maxDuration = None
    # keep track of which components have finished
    crossComponents = cross.components
    for thisComponent in cross.components:
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
    
    # --- Run Routine "cross" ---
    thisExp.currentRoutine = cross
    cross.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 1.0:
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
                currentRoutine=cross,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            cross.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if cross.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in cross.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "cross" ---
    for thisComponent in cross.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for cross
    cross.tStop = globalClock.getTime(format='float')
    cross.tStopRefresh = tThisFlipGlobal
    thisExp.addData('cross.stopped', cross.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if cross.maxDurationReached:
        routineTimer.addTime(-cross.maxDuration)
    elif cross.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-1.000000)
    thisExp.nextEntry()
    
    # set up handler to look after randomisation of conditions etc
    training_loop = data.TrialHandler2(
        name='training_loop',
        nReps=1.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('loopPainRegulationTrainingInstr.xlsx'), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(training_loop)  # add the loop to the experiment
    thisTraining_loop = training_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTraining_loop.rgb)
    if thisTraining_loop != None:
        for paramName in thisTraining_loop:
            globals()[paramName] = thisTraining_loop[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTraining_loop in training_loop:
        training_loop.status = STARTED
        if hasattr(thisTraining_loop, 'status'):
            thisTraining_loop.status = STARTED
        currentLoop = training_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTraining_loop.rgb)
        if thisTraining_loop != None:
            for paramName in thisTraining_loop:
                globals()[paramName] = thisTraining_loop[paramName]
        
        # --- Prepare to start Routine "training_instructions" ---
        # create an object to store info about Routine training_instructions
        training_instructions = data.Routine(
            name='training_instructions',
            components=[train_instr_text],
        )
        training_instructions.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        train_instr_text.setPos((0, 0.15))
        train_instr_text.setText(instr_message)
        # store start times for training_instructions
        training_instructions.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        training_instructions.tStart = globalClock.getTime(format='float')
        training_instructions.status = STARTED
        thisExp.addData('training_instructions.started', training_instructions.tStart)
        training_instructions.maxDuration = None
        # keep track of which components have finished
        training_instructionsComponents = training_instructions.components
        for thisComponent in training_instructions.components:
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
        
        # --- Run Routine "training_instructions" ---
        thisExp.currentRoutine = training_instructions
        training_instructions.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 15.0:
            # if trial has changed, end Routine now
            if hasattr(thisTraining_loop, 'status') and thisTraining_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *train_instr_text* updates
            
            # if train_instr_text is starting this frame...
            if train_instr_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                train_instr_text.frameNStart = frameN  # exact frame index
                train_instr_text.tStart = t  # local t and not account for scr refresh
                train_instr_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(train_instr_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'train_instr_text.started')
                # update status
                train_instr_text.status = STARTED
                train_instr_text.setAutoDraw(True)
            
            # if train_instr_text is active this frame...
            if train_instr_text.status == STARTED:
                # update params
                pass
            
            # if train_instr_text is stopping this frame...
            if train_instr_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > train_instr_text.tStartRefresh + 15-frameTolerance:
                    # keep track of stop time/frame for later
                    train_instr_text.tStop = t  # not accounting for scr refresh
                    train_instr_text.tStopRefresh = tThisFlipGlobal  # on global time
                    train_instr_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'train_instr_text.stopped')
                    # update status
                    train_instr_text.status = FINISHED
                    train_instr_text.setAutoDraw(False)
            
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
                    currentRoutine=training_instructions,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                training_instructions.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if training_instructions.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in training_instructions.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "training_instructions" ---
        for thisComponent in training_instructions.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for training_instructions
        training_instructions.tStop = globalClock.getTime(format='float')
        training_instructions.tStopRefresh = tThisFlipGlobal
        thisExp.addData('training_instructions.stopped', training_instructions.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if training_instructions.maxDurationReached:
            routineTimer.addTime(-training_instructions.maxDuration)
        elif training_instructions.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-15.000000)
        
        # --- Prepare to start Routine "train_stimulus" ---
        # create an object to store info about Routine train_stimulus
        train_stimulus = data.Routine(
            name='train_stimulus',
            components=[polygon_2],
        )
        train_stimulus.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for train_stimulus
        train_stimulus.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        train_stimulus.tStart = globalClock.getTime(format='float')
        train_stimulus.status = STARTED
        thisExp.addData('train_stimulus.started', train_stimulus.tStart)
        train_stimulus.maxDuration = None
        # keep track of which components have finished
        train_stimulusComponents = train_stimulus.components
        for thisComponent in train_stimulus.components:
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
        
        # --- Run Routine "train_stimulus" ---
        thisExp.currentRoutine = train_stimulus
        train_stimulus.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # if trial has changed, end Routine now
            if hasattr(thisTraining_loop, 'status') and thisTraining_loop.status == STOPPING:
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
                if tThisFlipGlobal > polygon_2.tStartRefresh + 2.0-frameTolerance:
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
                    currentRoutine=train_stimulus,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                train_stimulus.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if train_stimulus.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in train_stimulus.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "train_stimulus" ---
        for thisComponent in train_stimulus.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for train_stimulus
        train_stimulus.tStop = globalClock.getTime(format='float')
        train_stimulus.tStopRefresh = tThisFlipGlobal
        thisExp.addData('train_stimulus.stopped', train_stimulus.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if train_stimulus.maxDurationReached:
            routineTimer.addTime(-train_stimulus.maxDuration)
        elif train_stimulus.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "train_resp" ---
        # create an object to store info about Routine train_resp
        train_resp = data.Routine(
            name='train_resp',
            components=[text_4, key_train_resp],
        )
        train_resp.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for key_train_resp
        key_train_resp.keys = []
        key_train_resp.rt = []
        _key_train_resp_allKeys = []
        # store start times for train_resp
        train_resp.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        train_resp.tStart = globalClock.getTime(format='float')
        train_resp.status = STARTED
        thisExp.addData('train_resp.started', train_resp.tStart)
        train_resp.maxDuration = None
        # keep track of which components have finished
        train_respComponents = train_resp.components
        for thisComponent in train_resp.components:
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
        
        # --- Run Routine "train_resp" ---
        thisExp.currentRoutine = train_resp
        train_resp.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # if trial has changed, end Routine now
            if hasattr(thisTraining_loop, 'status') and thisTraining_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_4* updates
            
            # if text_4 is starting this frame...
            if text_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_4.frameNStart = frameN  # exact frame index
                text_4.tStart = t  # local t and not account for scr refresh
                text_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_4.started')
                # update status
                text_4.status = STARTED
                text_4.setAutoDraw(True)
            
            # if text_4 is active this frame...
            if text_4.status == STARTED:
                # update params
                pass
            
            # if text_4 is stopping this frame...
            if text_4.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_4.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    text_4.tStop = t  # not accounting for scr refresh
                    text_4.tStopRefresh = tThisFlipGlobal  # on global time
                    text_4.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_4.stopped')
                    # update status
                    text_4.status = FINISHED
                    text_4.setAutoDraw(False)
            
            # *key_train_resp* updates
            waitOnFlip = False
            
            # if key_train_resp is starting this frame...
            if key_train_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_train_resp.frameNStart = frameN  # exact frame index
                key_train_resp.tStart = t  # local t and not account for scr refresh
                key_train_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_train_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_train_resp.started')
                # update status
                key_train_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_train_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_train_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if key_train_resp is stopping this frame...
            if key_train_resp.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > key_train_resp.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    key_train_resp.tStop = t  # not accounting for scr refresh
                    key_train_resp.tStopRefresh = tThisFlipGlobal  # on global time
                    key_train_resp.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_train_resp.stopped')
                    # update status
                    key_train_resp.status = FINISHED
                    key_train_resp.status = FINISHED
            if key_train_resp.status == STARTED and not waitOnFlip:
                theseKeys = key_train_resp.getKeys(keyList=['p','n'], ignoreKeys=["escape"], waitRelease=False)
                _key_train_resp_allKeys.extend(theseKeys)
                if len(_key_train_resp_allKeys):
                    key_train_resp.keys = _key_train_resp_allKeys[-1].name  # just the last key pressed
                    key_train_resp.rt = _key_train_resp_allKeys[-1].rt
                    key_train_resp.duration = _key_train_resp_allKeys[-1].duration
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
                    currentRoutine=train_resp,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                train_resp.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if train_resp.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in train_resp.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "train_resp" ---
        for thisComponent in train_resp.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for train_resp
        train_resp.tStop = globalClock.getTime(format='float')
        train_resp.tStopRefresh = tThisFlipGlobal
        thisExp.addData('train_resp.stopped', train_resp.tStop)
        # check responses
        if key_train_resp.keys in ['', [], None]:  # No response was made
            key_train_resp.keys = None
        training_loop.addData('key_train_resp.keys',key_train_resp.keys)
        if key_train_resp.keys != None:  # we had a response
            training_loop.addData('key_train_resp.rt', key_train_resp.rt)
            training_loop.addData('key_train_resp.duration', key_train_resp.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if train_resp.maxDurationReached:
            routineTimer.addTime(-train_resp.maxDuration)
        elif train_resp.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        # mark thisTraining_loop as finished
        if hasattr(thisTraining_loop, 'status'):
            thisTraining_loop.status = FINISHED
        # if awaiting a pause, pause now
        if training_loop.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            training_loop.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'training_loop'
    training_loop.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "end_train" ---
    # create an object to store info about Routine end_train
    end_train = data.Routine(
        name='end_train',
        components=[text_5, end_train_resp],
    )
    end_train.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for end_train_resp
    end_train_resp.keys = []
    end_train_resp.rt = []
    _end_train_resp_allKeys = []
    # store start times for end_train
    end_train.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    end_train.tStart = globalClock.getTime(format='float')
    end_train.status = STARTED
    thisExp.addData('end_train.started', end_train.tStart)
    end_train.maxDuration = None
    # keep track of which components have finished
    end_trainComponents = end_train.components
    for thisComponent in end_train.components:
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
    
    # --- Run Routine "end_train" ---
    thisExp.currentRoutine = end_train
    end_train.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_5* updates
        
        # if text_5 is starting this frame...
        if text_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_5.frameNStart = frameN  # exact frame index
            text_5.tStart = t  # local t and not account for scr refresh
            text_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_5.started')
            # update status
            text_5.status = STARTED
            text_5.setAutoDraw(True)
        
        # if text_5 is active this frame...
        if text_5.status == STARTED:
            # update params
            pass
        
        # *end_train_resp* updates
        waitOnFlip = False
        
        # if end_train_resp is starting this frame...
        if end_train_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            end_train_resp.frameNStart = frameN  # exact frame index
            end_train_resp.tStart = t  # local t and not account for scr refresh
            end_train_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(end_train_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'end_train_resp.started')
            # update status
            end_train_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(end_train_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(end_train_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if end_train_resp.status == STARTED and not waitOnFlip:
            theseKeys = end_train_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _end_train_resp_allKeys.extend(theseKeys)
            if len(_end_train_resp_allKeys):
                end_train_resp.keys = _end_train_resp_allKeys[-1].name  # just the last key pressed
                end_train_resp.rt = _end_train_resp_allKeys[-1].rt
                end_train_resp.duration = _end_train_resp_allKeys[-1].duration
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
                currentRoutine=end_train,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            end_train.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if end_train.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in end_train.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "end_train" ---
    for thisComponent in end_train.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for end_train
    end_train.tStop = globalClock.getTime(format='float')
    end_train.tStopRefresh = tThisFlipGlobal
    thisExp.addData('end_train.stopped', end_train.tStop)
    # check responses
    if end_train_resp.keys in ['', [], None]:  # No response was made
        end_train_resp.keys = None
    thisExp.addData('end_train_resp.keys',end_train_resp.keys)
    if end_train_resp.keys != None:  # we had a response
        thisExp.addData('end_train_resp.rt', end_train_resp.rt)
        thisExp.addData('end_train_resp.duration', end_train_resp.duration)
    thisExp.nextEntry()
    # the Routine "end_train" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    pain_reg_main_loop = data.TrialHandler2(
        name='pain_reg_main_loop',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('loopPainRegulationMain.xlsx'), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(pain_reg_main_loop)  # add the loop to the experiment
    thisPain_reg_main_loop = pain_reg_main_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPain_reg_main_loop.rgb)
    if thisPain_reg_main_loop != None:
        for paramName in thisPain_reg_main_loop:
            globals()[paramName] = thisPain_reg_main_loop[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisPain_reg_main_loop in pain_reg_main_loop:
        pain_reg_main_loop.status = STARTED
        if hasattr(thisPain_reg_main_loop, 'status'):
            thisPain_reg_main_loop.status = STARTED
        currentLoop = pain_reg_main_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisPain_reg_main_loop.rgb)
        if thisPain_reg_main_loop != None:
            for paramName in thisPain_reg_main_loop:
                globals()[paramName] = thisPain_reg_main_loop[paramName]
        
        # --- Prepare to start Routine "pain_reg_instr01" ---
        # create an object to store info about Routine pain_reg_instr01
        pain_reg_instr01 = data.Routine(
            name='pain_reg_instr01',
            components=[pain_reg_instr_text],
        )
        pain_reg_instr01.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        pain_reg_instr_text.setText(pain_instr_message)
        # store start times for pain_reg_instr01
        pain_reg_instr01.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        pain_reg_instr01.tStart = globalClock.getTime(format='float')
        pain_reg_instr01.status = STARTED
        thisExp.addData('pain_reg_instr01.started', pain_reg_instr01.tStart)
        pain_reg_instr01.maxDuration = None
        # keep track of which components have finished
        pain_reg_instr01Components = pain_reg_instr01.components
        for thisComponent in pain_reg_instr01.components:
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
        
        # --- Run Routine "pain_reg_instr01" ---
        thisExp.currentRoutine = pain_reg_instr01
        pain_reg_instr01.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 15.0:
            # if trial has changed, end Routine now
            if hasattr(thisPain_reg_main_loop, 'status') and thisPain_reg_main_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *pain_reg_instr_text* updates
            
            # if pain_reg_instr_text is starting this frame...
            if pain_reg_instr_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                pain_reg_instr_text.frameNStart = frameN  # exact frame index
                pain_reg_instr_text.tStart = t  # local t and not account for scr refresh
                pain_reg_instr_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(pain_reg_instr_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'pain_reg_instr_text.started')
                # update status
                pain_reg_instr_text.status = STARTED
                pain_reg_instr_text.setAutoDraw(True)
            
            # if pain_reg_instr_text is active this frame...
            if pain_reg_instr_text.status == STARTED:
                # update params
                pass
            
            # if pain_reg_instr_text is stopping this frame...
            if pain_reg_instr_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > pain_reg_instr_text.tStartRefresh + 15-frameTolerance:
                    # keep track of stop time/frame for later
                    pain_reg_instr_text.tStop = t  # not accounting for scr refresh
                    pain_reg_instr_text.tStopRefresh = tThisFlipGlobal  # on global time
                    pain_reg_instr_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'pain_reg_instr_text.stopped')
                    # update status
                    pain_reg_instr_text.status = FINISHED
                    pain_reg_instr_text.setAutoDraw(False)
            
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
                    currentRoutine=pain_reg_instr01,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                pain_reg_instr01.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if pain_reg_instr01.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in pain_reg_instr01.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "pain_reg_instr01" ---
        for thisComponent in pain_reg_instr01.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for pain_reg_instr01
        pain_reg_instr01.tStop = globalClock.getTime(format='float')
        pain_reg_instr01.tStopRefresh = tThisFlipGlobal
        thisExp.addData('pain_reg_instr01.stopped', pain_reg_instr01.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if pain_reg_instr01.maxDurationReached:
            routineTimer.addTime(-pain_reg_instr01.maxDuration)
        elif pain_reg_instr01.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-15.000000)
        
        # set up handler to look after randomisation of conditions etc
        inner_trial_loop = data.TrialHandler2(
            name='inner_trial_loop',
            nReps=12.0, 
            method='sequential', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=[None], 
            seed=None, 
            isTrials=False, 
        )
        thisExp.addLoop(inner_trial_loop)  # add the loop to the experiment
        thisInner_trial_loop = inner_trial_loop.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisInner_trial_loop.rgb)
        if thisInner_trial_loop != None:
            for paramName in thisInner_trial_loop:
                globals()[paramName] = thisInner_trial_loop[paramName]
        
        for thisInner_trial_loop in inner_trial_loop:
            inner_trial_loop.status = STARTED
            if hasattr(thisInner_trial_loop, 'status'):
                thisInner_trial_loop.status = STARTED
            currentLoop = inner_trial_loop
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            # abbreviate parameter names if possible (e.g. rgb = thisInner_trial_loop.rgb)
            if thisInner_trial_loop != None:
                for paramName in thisInner_trial_loop:
                    globals()[paramName] = thisInner_trial_loop[paramName]
            
            # --- Prepare to start Routine "pain_reg_stimulus" ---
            # create an object to store info about Routine pain_reg_stimulus
            pain_reg_stimulus = data.Routine(
                name='pain_reg_stimulus',
                components=[polygon_3],
            )
            pain_reg_stimulus.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from codehelper_timing
            import random 
            if x == 0:
                set_time = 2
                x = 1
            else:
                set_time = random.uniform(1.2, 1.8)
                x = 1
            polygon_3.setSize((0.35, 0.35))
            # store start times for pain_reg_stimulus
            pain_reg_stimulus.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            pain_reg_stimulus.tStart = globalClock.getTime(format='float')
            pain_reg_stimulus.status = STARTED
            thisExp.addData('pain_reg_stimulus.started', pain_reg_stimulus.tStart)
            pain_reg_stimulus.maxDuration = None
            # keep track of which components have finished
            pain_reg_stimulusComponents = pain_reg_stimulus.components
            for thisComponent in pain_reg_stimulus.components:
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
            
            # --- Run Routine "pain_reg_stimulus" ---
            thisExp.currentRoutine = pain_reg_stimulus
            pain_reg_stimulus.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisInner_trial_loop, 'status') and thisInner_trial_loop.status == STOPPING:
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
                    if tThisFlipGlobal > polygon_3.tStartRefresh + set_time-frameTolerance:
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
                        currentRoutine=pain_reg_stimulus,
                    )
                    # skip the frame we paused on
                    continue
                
                # has a Component requested the Routine to end?
                if not continueRoutine:
                    pain_reg_stimulus.forceEnded = routineForceEnded = True
                # has the Routine been forcibly ended?
                if pain_reg_stimulus.forceEnded or routineForceEnded:
                    break
                # has every Component finished?
                continueRoutine = False
                for thisComponent in pain_reg_stimulus.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "pain_reg_stimulus" ---
            for thisComponent in pain_reg_stimulus.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for pain_reg_stimulus
            pain_reg_stimulus.tStop = globalClock.getTime(format='float')
            pain_reg_stimulus.tStopRefresh = tThisFlipGlobal
            thisExp.addData('pain_reg_stimulus.stopped', pain_reg_stimulus.tStop)
            # the Routine "pain_reg_stimulus" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "pain_reg_resp" ---
            # create an object to store info about Routine pain_reg_resp
            pain_reg_resp = data.Routine(
                name='pain_reg_resp',
                components=[pain_reg_resp_text, key_resp_2],
            )
            pain_reg_resp.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            pain_reg_resp_text.setText('Please rate how you felt the stimulus as Painful (P) or Not Painful (N)')
            # create starting attributes for key_resp_2
            key_resp_2.keys = []
            key_resp_2.rt = []
            _key_resp_2_allKeys = []
            # store start times for pain_reg_resp
            pain_reg_resp.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            pain_reg_resp.tStart = globalClock.getTime(format='float')
            pain_reg_resp.status = STARTED
            thisExp.addData('pain_reg_resp.started', pain_reg_resp.tStart)
            pain_reg_resp.maxDuration = None
            # keep track of which components have finished
            pain_reg_respComponents = pain_reg_resp.components
            for thisComponent in pain_reg_resp.components:
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
            
            # --- Run Routine "pain_reg_resp" ---
            thisExp.currentRoutine = pain_reg_resp
            pain_reg_resp.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 1.0:
                # if trial has changed, end Routine now
                if hasattr(thisInner_trial_loop, 'status') and thisInner_trial_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *pain_reg_resp_text* updates
                
                # if pain_reg_resp_text is starting this frame...
                if pain_reg_resp_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    pain_reg_resp_text.frameNStart = frameN  # exact frame index
                    pain_reg_resp_text.tStart = t  # local t and not account for scr refresh
                    pain_reg_resp_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(pain_reg_resp_text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'pain_reg_resp_text.started')
                    # update status
                    pain_reg_resp_text.status = STARTED
                    pain_reg_resp_text.setAutoDraw(True)
                
                # if pain_reg_resp_text is active this frame...
                if pain_reg_resp_text.status == STARTED:
                    # update params
                    pass
                
                # if pain_reg_resp_text is stopping this frame...
                if pain_reg_resp_text.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > pain_reg_resp_text.tStartRefresh + 1.0-frameTolerance:
                        # keep track of stop time/frame for later
                        pain_reg_resp_text.tStop = t  # not accounting for scr refresh
                        pain_reg_resp_text.tStopRefresh = tThisFlipGlobal  # on global time
                        pain_reg_resp_text.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'pain_reg_resp_text.stopped')
                        # update status
                        pain_reg_resp_text.status = FINISHED
                        pain_reg_resp_text.setAutoDraw(False)
                
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
                
                # if key_resp_2 is stopping this frame...
                if key_resp_2.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > key_resp_2.tStartRefresh + 1.0-frameTolerance:
                        # keep track of stop time/frame for later
                        key_resp_2.tStop = t  # not accounting for scr refresh
                        key_resp_2.tStopRefresh = tThisFlipGlobal  # on global time
                        key_resp_2.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'key_resp_2.stopped')
                        # update status
                        key_resp_2.status = FINISHED
                        key_resp_2.status = FINISHED
                if key_resp_2.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp_2.getKeys(keyList=['p','n'], ignoreKeys=["escape"], waitRelease=False)
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
                        currentRoutine=pain_reg_resp,
                    )
                    # skip the frame we paused on
                    continue
                
                # has a Component requested the Routine to end?
                if not continueRoutine:
                    pain_reg_resp.forceEnded = routineForceEnded = True
                # has the Routine been forcibly ended?
                if pain_reg_resp.forceEnded or routineForceEnded:
                    break
                # has every Component finished?
                continueRoutine = False
                for thisComponent in pain_reg_resp.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "pain_reg_resp" ---
            for thisComponent in pain_reg_resp.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for pain_reg_resp
            pain_reg_resp.tStop = globalClock.getTime(format='float')
            pain_reg_resp.tStopRefresh = tThisFlipGlobal
            thisExp.addData('pain_reg_resp.stopped', pain_reg_resp.tStop)
            # check responses
            if key_resp_2.keys in ['', [], None]:  # No response was made
                key_resp_2.keys = None
            inner_trial_loop.addData('key_resp_2.keys',key_resp_2.keys)
            if key_resp_2.keys != None:  # we had a response
                inner_trial_loop.addData('key_resp_2.rt', key_resp_2.rt)
                inner_trial_loop.addData('key_resp_2.duration', key_resp_2.duration)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if pain_reg_resp.maxDurationReached:
                routineTimer.addTime(-pain_reg_resp.maxDuration)
            elif pain_reg_resp.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-1.000000)
            # mark thisInner_trial_loop as finished
            if hasattr(thisInner_trial_loop, 'status'):
                thisInner_trial_loop.status = FINISHED
            # if awaiting a pause, pause now
            if inner_trial_loop.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                inner_trial_loop.status = STARTED
        # completed 12.0 repeats of 'inner_trial_loop'
        inner_trial_loop.status = FINISHED
        
        
        # set up handler to look after randomisation of conditions etc
        pain_reg_state_measure_loop = data.TrialHandler2(
            name='pain_reg_state_measure_loop',
            nReps=1.0, 
            method='sequential', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions('loopPainRegStateMeasureMain.xlsx'), 
            seed=None, 
            isTrials=True, 
        )
        thisExp.addLoop(pain_reg_state_measure_loop)  # add the loop to the experiment
        thisPain_reg_state_measure_loop = pain_reg_state_measure_loop.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisPain_reg_state_measure_loop.rgb)
        if thisPain_reg_state_measure_loop != None:
            for paramName in thisPain_reg_state_measure_loop:
                globals()[paramName] = thisPain_reg_state_measure_loop[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisPain_reg_state_measure_loop in pain_reg_state_measure_loop:
            pain_reg_state_measure_loop.status = STARTED
            if hasattr(thisPain_reg_state_measure_loop, 'status'):
                thisPain_reg_state_measure_loop.status = STARTED
            currentLoop = pain_reg_state_measure_loop
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisPain_reg_state_measure_loop.rgb)
            if thisPain_reg_state_measure_loop != None:
                for paramName in thisPain_reg_state_measure_loop:
                    globals()[paramName] = thisPain_reg_state_measure_loop[paramName]
            
            # --- Prepare to start Routine "pain_reg_state_measure" ---
            # create an object to store info about Routine pain_reg_state_measure
            pain_reg_state_measure = data.Routine(
                name='pain_reg_state_measure',
                components=[sliderGenericPainReg, sliderSAMPainReg, imageSAMPainReg, textStateMeasurePainReg, textStateMeasureLabelMin, textStateMeasureLabelMax, tb_continue_painreg, key_resp_state_measure_pain_reg],
            )
            pain_reg_state_measure.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            sliderGenericPainReg.reset()
            sliderGenericPainReg.setSize((slider_gen_width, 0.1))
            sliderSAMPainReg.reset()
            sliderSAMPainReg.setPos((0, slider_y))
            sliderSAMPainReg.setSize((slider_width, slider_height))
            imageSAMPainReg.setPos((0, sam_y))
            imageSAMPainReg.setSize((sam_width, sam_height))
            imageSAMPainReg.setImage(picture_path)
            textStateMeasurePainReg.setPos((0, 0.35))
            textStateMeasurePainReg.setText(rating_message)
            textStateMeasureLabelMin.setPos((tick_positions[0], label_y))
            textStateMeasureLabelMin.setText(rating_min_label)
            textStateMeasureLabelMax.setPos((tick_positions[-1], label_y))
            textStateMeasureLabelMax.setText(rating_max_label)
            tb_continue_painreg.reset()
            # create starting attributes for key_resp_state_measure_pain_reg
            key_resp_state_measure_pain_reg.keys = []
            key_resp_state_measure_pain_reg.rt = []
            _key_resp_state_measure_pain_reg_allKeys = []
            # Run 'Begin Routine' code from codepainreghelper
            allowContinue = False
            
            # store start times for pain_reg_state_measure
            pain_reg_state_measure.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            pain_reg_state_measure.tStart = globalClock.getTime(format='float')
            pain_reg_state_measure.status = STARTED
            thisExp.addData('pain_reg_state_measure.started', pain_reg_state_measure.tStart)
            pain_reg_state_measure.maxDuration = None
            # keep track of which components have finished
            pain_reg_state_measureComponents = pain_reg_state_measure.components
            for thisComponent in pain_reg_state_measure.components:
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
            
            # --- Run Routine "pain_reg_state_measure" ---
            thisExp.currentRoutine = pain_reg_state_measure
            pain_reg_state_measure.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisPain_reg_state_measure_loop, 'status') and thisPain_reg_state_measure_loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *sliderGenericPainReg* updates
                
                # if sliderGenericPainReg is starting this frame...
                if sliderGenericPainReg.status == NOT_STARTED and rating_type == 'Generic':
                    # keep track of start time/frame for later
                    sliderGenericPainReg.frameNStart = frameN  # exact frame index
                    sliderGenericPainReg.tStart = t  # local t and not account for scr refresh
                    sliderGenericPainReg.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(sliderGenericPainReg, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'sliderGenericPainReg.started')
                    # update status
                    sliderGenericPainReg.status = STARTED
                    sliderGenericPainReg.setAutoDraw(True)
                
                # if sliderGenericPainReg is active this frame...
                if sliderGenericPainReg.status == STARTED:
                    # update params
                    pass
                
                # if sliderGenericPainReg is stopping this frame...
                if sliderGenericPainReg.status == STARTED:
                    if bool(rating_type != 'Generic'):
                        # keep track of stop time/frame for later
                        sliderGenericPainReg.tStop = t  # not accounting for scr refresh
                        sliderGenericPainReg.tStopRefresh = tThisFlipGlobal  # on global time
                        sliderGenericPainReg.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'sliderGenericPainReg.stopped')
                        # update status
                        sliderGenericPainReg.status = FINISHED
                        sliderGenericPainReg.setAutoDraw(False)
                
                # *sliderSAMPainReg* updates
                
                # if sliderSAMPainReg is starting this frame...
                if sliderSAMPainReg.status == NOT_STARTED and rating_type == 'SAM':
                    # keep track of start time/frame for later
                    sliderSAMPainReg.frameNStart = frameN  # exact frame index
                    sliderSAMPainReg.tStart = t  # local t and not account for scr refresh
                    sliderSAMPainReg.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(sliderSAMPainReg, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'sliderSAMPainReg.started')
                    # update status
                    sliderSAMPainReg.status = STARTED
                    sliderSAMPainReg.setAutoDraw(True)
                
                # if sliderSAMPainReg is active this frame...
                if sliderSAMPainReg.status == STARTED:
                    # update params
                    pass
                
                # if sliderSAMPainReg is stopping this frame...
                if sliderSAMPainReg.status == STARTED:
                    if bool(rating_type != 'SAM'):
                        # keep track of stop time/frame for later
                        sliderSAMPainReg.tStop = t  # not accounting for scr refresh
                        sliderSAMPainReg.tStopRefresh = tThisFlipGlobal  # on global time
                        sliderSAMPainReg.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'sliderSAMPainReg.stopped')
                        # update status
                        sliderSAMPainReg.status = FINISHED
                        sliderSAMPainReg.setAutoDraw(False)
                
                # *imageSAMPainReg* updates
                
                # if imageSAMPainReg is starting this frame...
                if imageSAMPainReg.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    imageSAMPainReg.frameNStart = frameN  # exact frame index
                    imageSAMPainReg.tStart = t  # local t and not account for scr refresh
                    imageSAMPainReg.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(imageSAMPainReg, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'imageSAMPainReg.started')
                    # update status
                    imageSAMPainReg.status = STARTED
                    imageSAMPainReg.setAutoDraw(True)
                
                # if imageSAMPainReg is active this frame...
                if imageSAMPainReg.status == STARTED:
                    # update params
                    pass
                
                # *textStateMeasurePainReg* updates
                
                # if textStateMeasurePainReg is starting this frame...
                if textStateMeasurePainReg.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    textStateMeasurePainReg.frameNStart = frameN  # exact frame index
                    textStateMeasurePainReg.tStart = t  # local t and not account for scr refresh
                    textStateMeasurePainReg.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(textStateMeasurePainReg, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'textStateMeasurePainReg.started')
                    # update status
                    textStateMeasurePainReg.status = STARTED
                    textStateMeasurePainReg.setAutoDraw(True)
                
                # if textStateMeasurePainReg is active this frame...
                if textStateMeasurePainReg.status == STARTED:
                    # update params
                    pass
                
                # *textStateMeasureLabelMin* updates
                
                # if textStateMeasureLabelMin is starting this frame...
                if textStateMeasureLabelMin.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    textStateMeasureLabelMin.frameNStart = frameN  # exact frame index
                    textStateMeasureLabelMin.tStart = t  # local t and not account for scr refresh
                    textStateMeasureLabelMin.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(textStateMeasureLabelMin, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'textStateMeasureLabelMin.started')
                    # update status
                    textStateMeasureLabelMin.status = STARTED
                    textStateMeasureLabelMin.setAutoDraw(True)
                
                # if textStateMeasureLabelMin is active this frame...
                if textStateMeasureLabelMin.status == STARTED:
                    # update params
                    pass
                
                # *textStateMeasureLabelMax* updates
                
                # if textStateMeasureLabelMax is starting this frame...
                if textStateMeasureLabelMax.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    textStateMeasureLabelMax.frameNStart = frameN  # exact frame index
                    textStateMeasureLabelMax.tStart = t  # local t and not account for scr refresh
                    textStateMeasureLabelMax.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(textStateMeasureLabelMax, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'textStateMeasureLabelMax.started')
                    # update status
                    textStateMeasureLabelMax.status = STARTED
                    textStateMeasureLabelMax.setAutoDraw(True)
                
                # if textStateMeasureLabelMax is active this frame...
                if textStateMeasureLabelMax.status == STARTED:
                    # update params
                    pass
                
                # *tb_continue_painreg* updates
                
                # if tb_continue_painreg is starting this frame...
                if tb_continue_painreg.status == NOT_STARTED and allowContinue == True:
                    # keep track of start time/frame for later
                    tb_continue_painreg.frameNStart = frameN  # exact frame index
                    tb_continue_painreg.tStart = t  # local t and not account for scr refresh
                    tb_continue_painreg.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(tb_continue_painreg, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'tb_continue_painreg.started')
                    # update status
                    tb_continue_painreg.status = STARTED
                    tb_continue_painreg.setAutoDraw(True)
                
                # if tb_continue_painreg is active this frame...
                if tb_continue_painreg.status == STARTED:
                    # update params
                    pass
                
                # *key_resp_state_measure_pain_reg* updates
                waitOnFlip = False
                
                # if key_resp_state_measure_pain_reg is starting this frame...
                if key_resp_state_measure_pain_reg.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp_state_measure_pain_reg.frameNStart = frameN  # exact frame index
                    key_resp_state_measure_pain_reg.tStart = t  # local t and not account for scr refresh
                    key_resp_state_measure_pain_reg.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp_state_measure_pain_reg, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp_state_measure_pain_reg.started')
                    # update status
                    key_resp_state_measure_pain_reg.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp_state_measure_pain_reg.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp_state_measure_pain_reg.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if key_resp_state_measure_pain_reg.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp_state_measure_pain_reg.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                    _key_resp_state_measure_pain_reg_allKeys.extend(theseKeys)
                    if len(_key_resp_state_measure_pain_reg_allKeys):
                        key_resp_state_measure_pain_reg.keys = _key_resp_state_measure_pain_reg_allKeys[-1].name  # just the last key pressed
                        key_resp_state_measure_pain_reg.rt = _key_resp_state_measure_pain_reg_allKeys[-1].rt
                        key_resp_state_measure_pain_reg.duration = _key_resp_state_measure_pain_reg_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                # Run 'Each Frame' code from codepainreghelper
                # Check if either slider has been rated
                rated = (sliderGenericPainReg.getRating() is not None) or (sliderSAMPainReg.getRating() is not None)
                
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
                        currentRoutine=pain_reg_state_measure,
                    )
                    # skip the frame we paused on
                    continue
                
                # has a Component requested the Routine to end?
                if not continueRoutine:
                    pain_reg_state_measure.forceEnded = routineForceEnded = True
                # has the Routine been forcibly ended?
                if pain_reg_state_measure.forceEnded or routineForceEnded:
                    break
                # has every Component finished?
                continueRoutine = False
                for thisComponent in pain_reg_state_measure.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "pain_reg_state_measure" ---
            for thisComponent in pain_reg_state_measure.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for pain_reg_state_measure
            pain_reg_state_measure.tStop = globalClock.getTime(format='float')
            pain_reg_state_measure.tStopRefresh = tThisFlipGlobal
            thisExp.addData('pain_reg_state_measure.stopped', pain_reg_state_measure.tStop)
            pain_reg_state_measure_loop.addData('sliderGenericPainReg.response', sliderGenericPainReg.getRating())
            pain_reg_state_measure_loop.addData('sliderGenericPainReg.rt', sliderGenericPainReg.getRT())
            pain_reg_state_measure_loop.addData('sliderSAMPainReg.response', sliderSAMPainReg.getRating())
            pain_reg_state_measure_loop.addData('sliderSAMPainReg.rt', sliderSAMPainReg.getRT())
            # check responses
            if key_resp_state_measure_pain_reg.keys in ['', [], None]:  # No response was made
                key_resp_state_measure_pain_reg.keys = None
            pain_reg_state_measure_loop.addData('key_resp_state_measure_pain_reg.keys',key_resp_state_measure_pain_reg.keys)
            if key_resp_state_measure_pain_reg.keys != None:  # we had a response
                pain_reg_state_measure_loop.addData('key_resp_state_measure_pain_reg.rt', key_resp_state_measure_pain_reg.rt)
                pain_reg_state_measure_loop.addData('key_resp_state_measure_pain_reg.duration', key_resp_state_measure_pain_reg.duration)
            # the Routine "pain_reg_state_measure" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            # mark thisPain_reg_state_measure_loop as finished
            if hasattr(thisPain_reg_state_measure_loop, 'status'):
                thisPain_reg_state_measure_loop.status = FINISHED
            # if awaiting a pause, pause now
            if pain_reg_state_measure_loop.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                pain_reg_state_measure_loop.status = STARTED
            thisExp.nextEntry()
            
        # completed 1.0 repeats of 'pain_reg_state_measure_loop'
        pain_reg_state_measure_loop.status = FINISHED
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        # --- Prepare to start Routine "routine_5sec_block_pause" ---
        # create an object to store info about Routine routine_5sec_block_pause
        routine_5sec_block_pause = data.Routine(
            name='routine_5sec_block_pause',
            components=[pause],
        )
        routine_5sec_block_pause.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for routine_5sec_block_pause
        routine_5sec_block_pause.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        routine_5sec_block_pause.tStart = globalClock.getTime(format='float')
        routine_5sec_block_pause.status = STARTED
        thisExp.addData('routine_5sec_block_pause.started', routine_5sec_block_pause.tStart)
        routine_5sec_block_pause.maxDuration = None
        # keep track of which components have finished
        routine_5sec_block_pauseComponents = routine_5sec_block_pause.components
        for thisComponent in routine_5sec_block_pause.components:
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
        
        # --- Run Routine "routine_5sec_block_pause" ---
        thisExp.currentRoutine = routine_5sec_block_pause
        routine_5sec_block_pause.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 5.0:
            # if trial has changed, end Routine now
            if hasattr(thisPain_reg_main_loop, 'status') and thisPain_reg_main_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *pause* updates
            
            # if pause is starting this frame...
            if pause.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                pause.frameNStart = frameN  # exact frame index
                pause.tStart = t  # local t and not account for scr refresh
                pause.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(pause, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'pause.started')
                # update status
                pause.status = STARTED
                pause.setAutoDraw(True)
            
            # if pause is active this frame...
            if pause.status == STARTED:
                # update params
                pass
            
            # if pause is stopping this frame...
            if pause.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > pause.tStartRefresh + 5.0-frameTolerance:
                    # keep track of stop time/frame for later
                    pause.tStop = t  # not accounting for scr refresh
                    pause.tStopRefresh = tThisFlipGlobal  # on global time
                    pause.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'pause.stopped')
                    # update status
                    pause.status = FINISHED
                    pause.setAutoDraw(False)
            
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
                    currentRoutine=routine_5sec_block_pause,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                routine_5sec_block_pause.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if routine_5sec_block_pause.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in routine_5sec_block_pause.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "routine_5sec_block_pause" ---
        for thisComponent in routine_5sec_block_pause.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for routine_5sec_block_pause
        routine_5sec_block_pause.tStop = globalClock.getTime(format='float')
        routine_5sec_block_pause.tStopRefresh = tThisFlipGlobal
        thisExp.addData('routine_5sec_block_pause.stopped', routine_5sec_block_pause.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routine_5sec_block_pause.maxDurationReached:
            routineTimer.addTime(-routine_5sec_block_pause.maxDuration)
        elif routine_5sec_block_pause.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-5.000000)
        # mark thisPain_reg_main_loop as finished
        if hasattr(thisPain_reg_main_loop, 'status'):
            thisPain_reg_main_loop.status = FINISHED
        # if awaiting a pause, pause now
        if pain_reg_main_loop.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            pain_reg_main_loop.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'pain_reg_main_loop'
    pain_reg_main_loop.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # set up handler to look after randomisation of conditions etc
    state_measure_loop2 = data.TrialHandler2(
        name='state_measure_loop2',
        nReps=1.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('../../shared/loop-templates/loopStateMeasureOutro.xlsx'), 
        seed=None, 
        isTrials=True, 
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
        
        # --- Prepare to start Routine "state_measure_end" ---
        # create an object to store info about Routine state_measure_end
        state_measure_end = data.Routine(
            name='state_measure_end',
            components=[sliderStateMeasureEnd, sliderSAMStateMeasureEnd, imageSAMStateMeasureEnd, textStateMeasurePainRegEnd, textLabelMinStateMeasureEnd, textLabelMaxStateMeasureEnd, textbox_continue_State_measure_End, key_resp_StateMeasureEnd],
        )
        state_measure_end.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        sliderStateMeasureEnd.reset()
        sliderStateMeasureEnd.setSize((slider_gen_width, 0.1))
        sliderSAMStateMeasureEnd.reset()
        sliderSAMStateMeasureEnd.setPos((0, slider_y))
        sliderSAMStateMeasureEnd.setSize((slider_width, slider_height))
        imageSAMStateMeasureEnd.setPos((0, sam_y))
        imageSAMStateMeasureEnd.setSize((sam_width, sam_height))
        imageSAMStateMeasureEnd.setImage(picture_path)
        textStateMeasurePainRegEnd.setText(rating_message)
        textLabelMinStateMeasureEnd.setPos((tick_positions[0], label_y))
        textLabelMinStateMeasureEnd.setText(rating_min_label)
        textLabelMaxStateMeasureEnd.setPos((tick_positions[-1], label_y))
        textLabelMaxStateMeasureEnd.setText(rating_max_label)
        textbox_continue_State_measure_End.reset()
        textbox_continue_State_measure_End.setText('Press the SPACEBAR to continue')
        # Run 'Begin Routine' code from codeStateMeasureEndHelper
        allowContinue = False
        # create starting attributes for key_resp_StateMeasureEnd
        key_resp_StateMeasureEnd.keys = []
        key_resp_StateMeasureEnd.rt = []
        _key_resp_StateMeasureEnd_allKeys = []
        # store start times for state_measure_end
        state_measure_end.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        state_measure_end.tStart = globalClock.getTime(format='float')
        state_measure_end.status = STARTED
        thisExp.addData('state_measure_end.started', state_measure_end.tStart)
        state_measure_end.maxDuration = None
        # keep track of which components have finished
        state_measure_endComponents = state_measure_end.components
        for thisComponent in state_measure_end.components:
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
        
        # --- Run Routine "state_measure_end" ---
        thisExp.currentRoutine = state_measure_end
        state_measure_end.forceEnded = routineForceEnded = not continueRoutine
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
            
            # *sliderStateMeasureEnd* updates
            
            # if sliderStateMeasureEnd is starting this frame...
            if sliderStateMeasureEnd.status == NOT_STARTED and rating_type == 'Generic':
                # keep track of start time/frame for later
                sliderStateMeasureEnd.frameNStart = frameN  # exact frame index
                sliderStateMeasureEnd.tStart = t  # local t and not account for scr refresh
                sliderStateMeasureEnd.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(sliderStateMeasureEnd, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'sliderStateMeasureEnd.started')
                # update status
                sliderStateMeasureEnd.status = STARTED
                sliderStateMeasureEnd.setAutoDraw(True)
            
            # if sliderStateMeasureEnd is active this frame...
            if sliderStateMeasureEnd.status == STARTED:
                # update params
                pass
            
            # if sliderStateMeasureEnd is stopping this frame...
            if sliderStateMeasureEnd.status == STARTED:
                if bool(rating_type != 'Generic'):
                    # keep track of stop time/frame for later
                    sliderStateMeasureEnd.tStop = t  # not accounting for scr refresh
                    sliderStateMeasureEnd.tStopRefresh = tThisFlipGlobal  # on global time
                    sliderStateMeasureEnd.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'sliderStateMeasureEnd.stopped')
                    # update status
                    sliderStateMeasureEnd.status = FINISHED
                    sliderStateMeasureEnd.setAutoDraw(False)
            
            # *sliderSAMStateMeasureEnd* updates
            
            # if sliderSAMStateMeasureEnd is starting this frame...
            if sliderSAMStateMeasureEnd.status == NOT_STARTED and rating_type == 'SAM':
                # keep track of start time/frame for later
                sliderSAMStateMeasureEnd.frameNStart = frameN  # exact frame index
                sliderSAMStateMeasureEnd.tStart = t  # local t and not account for scr refresh
                sliderSAMStateMeasureEnd.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(sliderSAMStateMeasureEnd, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'sliderSAMStateMeasureEnd.started')
                # update status
                sliderSAMStateMeasureEnd.status = STARTED
                sliderSAMStateMeasureEnd.setAutoDraw(True)
            
            # if sliderSAMStateMeasureEnd is active this frame...
            if sliderSAMStateMeasureEnd.status == STARTED:
                # update params
                pass
            
            # if sliderSAMStateMeasureEnd is stopping this frame...
            if sliderSAMStateMeasureEnd.status == STARTED:
                if bool(rating_type != 'SAM'):
                    # keep track of stop time/frame for later
                    sliderSAMStateMeasureEnd.tStop = t  # not accounting for scr refresh
                    sliderSAMStateMeasureEnd.tStopRefresh = tThisFlipGlobal  # on global time
                    sliderSAMStateMeasureEnd.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'sliderSAMStateMeasureEnd.stopped')
                    # update status
                    sliderSAMStateMeasureEnd.status = FINISHED
                    sliderSAMStateMeasureEnd.setAutoDraw(False)
            
            # *imageSAMStateMeasureEnd* updates
            
            # if imageSAMStateMeasureEnd is starting this frame...
            if imageSAMStateMeasureEnd.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                imageSAMStateMeasureEnd.frameNStart = frameN  # exact frame index
                imageSAMStateMeasureEnd.tStart = t  # local t and not account for scr refresh
                imageSAMStateMeasureEnd.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(imageSAMStateMeasureEnd, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'imageSAMStateMeasureEnd.started')
                # update status
                imageSAMStateMeasureEnd.status = STARTED
                imageSAMStateMeasureEnd.setAutoDraw(True)
            
            # if imageSAMStateMeasureEnd is active this frame...
            if imageSAMStateMeasureEnd.status == STARTED:
                # update params
                pass
            
            # *textStateMeasurePainRegEnd* updates
            
            # if textStateMeasurePainRegEnd is starting this frame...
            if textStateMeasurePainRegEnd.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                textStateMeasurePainRegEnd.frameNStart = frameN  # exact frame index
                textStateMeasurePainRegEnd.tStart = t  # local t and not account for scr refresh
                textStateMeasurePainRegEnd.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(textStateMeasurePainRegEnd, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textStateMeasurePainRegEnd.started')
                # update status
                textStateMeasurePainRegEnd.status = STARTED
                textStateMeasurePainRegEnd.setAutoDraw(True)
            
            # if textStateMeasurePainRegEnd is active this frame...
            if textStateMeasurePainRegEnd.status == STARTED:
                # update params
                pass
            
            # *textLabelMinStateMeasureEnd* updates
            
            # if textLabelMinStateMeasureEnd is starting this frame...
            if textLabelMinStateMeasureEnd.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                textLabelMinStateMeasureEnd.frameNStart = frameN  # exact frame index
                textLabelMinStateMeasureEnd.tStart = t  # local t and not account for scr refresh
                textLabelMinStateMeasureEnd.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(textLabelMinStateMeasureEnd, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textLabelMinStateMeasureEnd.started')
                # update status
                textLabelMinStateMeasureEnd.status = STARTED
                textLabelMinStateMeasureEnd.setAutoDraw(True)
            
            # if textLabelMinStateMeasureEnd is active this frame...
            if textLabelMinStateMeasureEnd.status == STARTED:
                # update params
                pass
            
            # *textLabelMaxStateMeasureEnd* updates
            
            # if textLabelMaxStateMeasureEnd is starting this frame...
            if textLabelMaxStateMeasureEnd.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                textLabelMaxStateMeasureEnd.frameNStart = frameN  # exact frame index
                textLabelMaxStateMeasureEnd.tStart = t  # local t and not account for scr refresh
                textLabelMaxStateMeasureEnd.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(textLabelMaxStateMeasureEnd, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textLabelMaxStateMeasureEnd.started')
                # update status
                textLabelMaxStateMeasureEnd.status = STARTED
                textLabelMaxStateMeasureEnd.setAutoDraw(True)
            
            # if textLabelMaxStateMeasureEnd is active this frame...
            if textLabelMaxStateMeasureEnd.status == STARTED:
                # update params
                pass
            
            # *textbox_continue_State_measure_End* updates
            
            # if textbox_continue_State_measure_End is starting this frame...
            if textbox_continue_State_measure_End.status == NOT_STARTED and allowContinue == True:
                # keep track of start time/frame for later
                textbox_continue_State_measure_End.frameNStart = frameN  # exact frame index
                textbox_continue_State_measure_End.tStart = t  # local t and not account for scr refresh
                textbox_continue_State_measure_End.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(textbox_continue_State_measure_End, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textbox_continue_State_measure_End.started')
                # update status
                textbox_continue_State_measure_End.status = STARTED
                textbox_continue_State_measure_End.setAutoDraw(True)
            
            # if textbox_continue_State_measure_End is active this frame...
            if textbox_continue_State_measure_End.status == STARTED:
                # update params
                pass
            # Run 'Each Frame' code from codeStateMeasureEndHelper
            # Check if either slider has been rated
            rated = (sliderStateMeasureEnd.getRating() is not None) or (sliderSAMStateMeasureEnd.getRating() is not None)
            
            # If rated, allow spacebar to end the routine
            if rated:
                allowContinue = True
            else:
                allowContinue = False
            
            # *key_resp_StateMeasureEnd* updates
            waitOnFlip = False
            
            # if key_resp_StateMeasureEnd is starting this frame...
            if key_resp_StateMeasureEnd.status == NOT_STARTED and allowContinue == True:
                # keep track of start time/frame for later
                key_resp_StateMeasureEnd.frameNStart = frameN  # exact frame index
                key_resp_StateMeasureEnd.tStart = t  # local t and not account for scr refresh
                key_resp_StateMeasureEnd.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_StateMeasureEnd, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_StateMeasureEnd.started')
                # update status
                key_resp_StateMeasureEnd.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_StateMeasureEnd.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_StateMeasureEnd.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_StateMeasureEnd.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_StateMeasureEnd.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_StateMeasureEnd_allKeys.extend(theseKeys)
                if len(_key_resp_StateMeasureEnd_allKeys):
                    key_resp_StateMeasureEnd.keys = _key_resp_StateMeasureEnd_allKeys[-1].name  # just the last key pressed
                    key_resp_StateMeasureEnd.rt = _key_resp_StateMeasureEnd_allKeys[-1].rt
                    key_resp_StateMeasureEnd.duration = _key_resp_StateMeasureEnd_allKeys[-1].duration
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
                    currentRoutine=state_measure_end,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                state_measure_end.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if state_measure_end.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in state_measure_end.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "state_measure_end" ---
        for thisComponent in state_measure_end.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for state_measure_end
        state_measure_end.tStop = globalClock.getTime(format='float')
        state_measure_end.tStopRefresh = tThisFlipGlobal
        thisExp.addData('state_measure_end.stopped', state_measure_end.tStop)
        state_measure_loop2.addData('sliderStateMeasureEnd.response', sliderStateMeasureEnd.getRating())
        state_measure_loop2.addData('sliderStateMeasureEnd.rt', sliderStateMeasureEnd.getRT())
        state_measure_loop2.addData('sliderSAMStateMeasureEnd.response', sliderSAMStateMeasureEnd.getRating())
        state_measure_loop2.addData('sliderSAMStateMeasureEnd.rt', sliderSAMStateMeasureEnd.getRT())
        # check responses
        if key_resp_StateMeasureEnd.keys in ['', [], None]:  # No response was made
            key_resp_StateMeasureEnd.keys = None
        state_measure_loop2.addData('key_resp_StateMeasureEnd.keys',key_resp_StateMeasureEnd.keys)
        if key_resp_StateMeasureEnd.keys != None:  # we had a response
            state_measure_loop2.addData('key_resp_StateMeasureEnd.rt', key_resp_StateMeasureEnd.rt)
            state_measure_loop2.addData('key_resp_StateMeasureEnd.duration', key_resp_StateMeasureEnd.duration)
        # the Routine "state_measure_end" was not non-slip safe, so reset the non-slip timer
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
    
    # --- Prepare to start Routine "end" ---
    # create an object to store info about Routine end
    end = data.Routine(
        name='end',
        components=[text],
    )
    end.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for end
    end.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    end.tStart = globalClock.getTime(format='float')
    end.status = STARTED
    thisExp.addData('end.started', end.tStart)
    end.maxDuration = None
    # keep track of which components have finished
    endComponents = end.components
    for thisComponent in end.components:
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
    
    # --- Run Routine "end" ---
    thisExp.currentRoutine = end
    end.forceEnded = routineForceEnded = not continueRoutine
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
                currentRoutine=end,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            end.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if end.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in end.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "end" ---
    for thisComponent in end.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for end
    end.tStop = globalClock.getTime(format='float')
    end.tStopRefresh = tThisFlipGlobal
    thisExp.addData('end.stopped', end.tStop)
    thisExp.nextEntry()
    # the Routine "end" was not non-slip safe, so reset the non-slip timer
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
