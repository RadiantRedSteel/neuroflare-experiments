#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2025.2.3),
    on January 05, 2026, at 19:38
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

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2025.2.3'
expName = 'phoda'  # from the Builder filename that created this script
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
        originPath='D:\\Github\\neuroflare-experiments\\psychopy\\experiments\\phoda\\phoda.py',
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
    # Also includes imports for later components
    # ------------------------------------------------------------
    
    import random # for t_blank_delayer
    
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
    # Run 'Begin Experiment' code from code_phoda_slider_setup
    # ============================================================
    # =================== CONFIGURATION ==========================
    # ============================================================
    
    # PHODA image settings
    PS_PHODA_IMG_PATH = ''
    PS_PHODA_IMG_PX_WIDTH = 283
    PS_PHODA_IMG_PX_HEIGHT = 425
    PS_PHODA_IMG_HEIGHT_RATIO = 0.62  # Percentage of screen height
    PS_PHODA_IMG_POS_Y = 0.13  # Vertical position (0 = center, positive = above center)
    
    # Slider settings
    PS_SLIDER_WIDTH_RATIO = 0.8  # Percentage of screen width
    PS_SLIDER_HEIGHT = 0.08
    PS_SLIDER_MARKER_WIDTH = 0.02
    PS_SLIDER_GRANULARITY = 0  # 0 for continuous (VAS), 1 for integers
    PS_SLIDER_TICKS = range(0, 101)
    
    # Spacing
    PS_IMAGE_SLIDER_GAP = 0.08  # Vertical space between image and slider
    PS_SLIDER_LABELS_GAP = 0.03  # Vertical space between slider and numeric labels
    
    # Label settings
    PS_LABEL_VALUES = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    PS_LABEL_HEIGHT = 0.03
    PS_LABEL_FONT = 'Noto Sans'
    PS_LABEL_COLOR = 'white'
    
    # Endpoint labels
    PS_LABEL_LEFT_TEXT = 'Not at all\nharmful'
    PS_LABEL_RIGHT_TEXT = 'Extremely\nharmful'
    PS_LABEL_ENDPOINT_HEIGHT = 0.025
    PS_LABEL_ENDPOINT_OFFSET = 0.055 # subtracted from label y-pos
    
    # Rating display
    PS_RATING_TEXT_HEIGHT = 0.04
    PS_RATING_TEXT_DEFAULT = 'Click or drag the slider to rate'
    
    # Auto logging for all components
    PS_AUTO_LOG = False
    PS_AUTO_LOG_SLIDER = False
    
    # ============================================================
    # =================== HELPER FUNCTIONS =======================
    # ============================================================
    
    def calculate_phoda_geometry():
        """
        Calculate all geometry for PHODA image and slider layout.
        Returns a dictionary with all positioning and sizing values.
        """
        # Image geometry
        img_aspect = PS_PHODA_IMG_PX_WIDTH / PS_PHODA_IMG_PX_HEIGHT
        img_height = PS_PHODA_IMG_HEIGHT_RATIO
        img_width = img_height * img_aspect
        
        # Slider geometry
        slider_width = aspect * PS_SLIDER_WIDTH_RATIO
        slider_height = PS_SLIDER_HEIGHT
        
        # Vertical positioning
        img_pos_y = PS_PHODA_IMG_POS_Y  # Use the defined position for the image
        img_bottom = img_pos_y - img_height / 2
        slider_pos_y = img_bottom - PS_IMAGE_SLIDER_GAP
        label_pos_y = slider_pos_y - slider_height / 2 - PS_SLIDER_LABELS_GAP
        
        # Slider boundaries
        slider_left = -slider_width / 2
        slider_right = slider_width / 2
        
        return {
            'img_width': img_width,
            'img_height': img_height,
            'img_pos_y': img_pos_y,
            'slider_width': slider_width,
            'slider_height': slider_height,
            'slider_pos_y': slider_pos_y,
            'slider_left': slider_left,
            'slider_right': slider_right,
            'label_pos_y': label_pos_y
        }
    
    def create_custom_slider(geom):
        """
        Create and customize a PHODA slider with narrow marker.
        """
        slider = visual.Slider(
            win=win,
            size=(geom['slider_width'], geom['slider_height']),
            pos=(0, geom['slider_pos_y']),
            ticks=PS_SLIDER_TICKS,
            labels=None,
            granularity=PS_SLIDER_GRANULARITY,
            style='scrollbar',
            flip=False,
            name="ps_phoda_slider",
            autoLog=PS_AUTO_LOG_SLIDER
    
        )
        
        # Customize marker width
        slider.marker.size = (PS_SLIDER_MARKER_WIDTH, slider.marker.size[1])
        
        # Adjust line size (compensate for scrollbar's 20% extra width)
        slider.line.size = (geom['slider_width'] + PS_SLIDER_MARKER_WIDTH, slider.line.size[1])
        
        return slider
    
    def create_slider_labels(geom):
        """
        Create numeric labels positioned along the slider.
        """
        labels = []
        slider_left = geom['slider_left']
        slider_right = geom['slider_right']
        label_pos_y = geom['label_pos_y']
        
        for label_val in PS_LABEL_VALUES:
            label_ratio = label_val / 100
            x_pos = slider_left + (slider_right - slider_left) * label_ratio
            
            label_obj = visual.TextStim(
                win=win,
                text=str(label_val),
                pos=(x_pos, label_pos_y),
                height=PS_LABEL_HEIGHT,
                anchorVert='top',
                color=PS_LABEL_COLOR,
                font=PS_LABEL_FONT,
                name=f"ps_slider_label_{label_val}",
                autoLog=PS_AUTO_LOG
            )
            labels.append(label_obj)
        
        return labels
    
    def create_endpoint_labels(geom):
        """
        Create left and right endpoint description labels.
        """
        label_left = visual.TextStim(
            win=win,
            text=PS_LABEL_LEFT_TEXT,
            pos=(geom['slider_left'], geom['label_pos_y'] - PS_LABEL_ENDPOINT_OFFSET),
            height=PS_LABEL_ENDPOINT_HEIGHT,
            anchorVert='top',
            color=PS_LABEL_COLOR,
            font=PS_LABEL_FONT,
            name="ps_label_left",
            autoLog=PS_AUTO_LOG
        )
        
        label_right = visual.TextStim(
            win=win,
            text=PS_LABEL_RIGHT_TEXT,
            pos=(geom['slider_right'], geom['label_pos_y'] - PS_LABEL_ENDPOINT_OFFSET),
            height=PS_LABEL_ENDPOINT_HEIGHT,
            anchorVert='top',
            color=PS_LABEL_COLOR,
            font=PS_LABEL_FONT,
            name="ps_label_right",
            autoLog=PS_AUTO_LOG
        )
        
        return label_left, label_right
    
    def run_phoda_trial():
        """
        Draw all PHODA components on the window.
        """
        ps_phoda_slider.draw()
        for label in ps_manual_labels:
            label.draw()
        ps_phoda_label_left.draw()
        ps_phoda_label_right.draw()
        
        # Update rating display in real-time
        if ps_phoda_slider.markerPos is not None:
            if PS_SLIDER_GRANULARITY == 0:
                # Show float for VAS (continuous)
                ps_phoda_rating_text.text = f'Current rating: {ps_phoda_slider.markerPos:.1f}'
            else:
                # Show integer for discrete scale
                ps_phoda_rating_text.text = f'Current rating: {int(ps_phoda_slider.markerPos)}'
        else:
            ps_phoda_rating_text.text = PS_RATING_TEXT_DEFAULT
        
        ps_phoda_rating_text.draw()
    
    def phoda_autodraw_on():
        """
        Turn on AutoDraw for all PHODA components on the window.
        """
        ps_phoda_slider.setAutoDraw(True)
        for label in ps_manual_labels:
            label.setAutoDraw(True)
        ps_phoda_label_left.setAutoDraw(True)
        ps_phoda_label_right.setAutoDraw(True)
        ps_phoda_rating_text.setAutoDraw(True)
        logging.exp("PHODA components: AutoDraw ON")
    
    def phoda_autodraw_off():
        """
        Turn off AutoDraw for all PHODA components on the window.
        """
        ps_phoda_slider.setAutoDraw(False)
        for label in ps_manual_labels:
            label.setAutoDraw(False)
        ps_phoda_label_left.setAutoDraw(False)
        ps_phoda_label_right.setAutoDraw(False)
        ps_phoda_rating_text.setAutoDraw(False)
        logging.exp("PHODA components: AutoDraw OFF")
    
    def update_phoda_rating_text(last_rating_value):
        """
        Update the PHODA rating display text if the slider value has changed.
        """
        current = ps_phoda_slider.markerPos
    
        # Only update when the value actually changes
        if current != last_rating_value:
            if current is None:
                ps_phoda_rating_text.text = PS_RATING_TEXT_DEFAULT
            else:
                if PS_SLIDER_GRANULARITY == 0:
                    ps_phoda_rating_text.text = f'Current rating: {current:.1f}'
                else:
                    ps_phoda_rating_text.text = f'Current rating: {int(current)}'
    
            # Update the flag
            last_rating_value = current
    
        return last_rating_value
    
    # Calculate all geometry
    ps_geom = calculate_phoda_geometry()
    
    # ============================================================
    # =================== CREATE COMPONENTS ======================
    # ============================================================
    
    # PHODA image
    ps_phoda_comp_img_pos_y = ps_geom['img_pos_y']
    ps_phoda_comp_img_size = (ps_geom['img_width'], ps_geom['img_height'])
    
    # Custom slider
    ps_phoda_slider = create_custom_slider(ps_geom)
    
    # Slider labels
    ps_manual_labels = create_slider_labels(ps_geom)
    
    # Endpoint labels
    ps_phoda_label_left, ps_phoda_label_right = create_endpoint_labels(ps_geom)
    t_phoda_continue_pos_y = ps_geom['label_pos_y'] - PS_LABEL_ENDPOINT_OFFSET -0.02
    
    # Rating display text
    ps_phoda_rating_text = visual.TextStim(
        win=win,
        text=PS_RATING_TEXT_DEFAULT,
        pos=(0, ps_geom['slider_pos_y']),
        height=PS_RATING_TEXT_HEIGHT,
        bold=True,
        color=PS_LABEL_COLOR,
        font=PS_LABEL_FONT,
        name="ps_phoda_rating_text",
        autoLog=PS_AUTO_LOG
    )
    
    
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
    
    # --- Initialize components for Routine "instruction" ---
    # Run 'Begin Experiment' code from code_instruction_helper
    view_mode_text = 'You will see photographs of daily activities.\n\nIn this block, simply view each picture without making a rating.'
    rate_mode_text = 'You will see photographs of daily activities.\n\nFor each picture, please rate how harmful you feel the activity is to your body.'
    t_instruction_body = visual.TextStim(win=win, name='t_instruction_body',
        text='',
        font='Arial',
        units='height', pos=(0.0, 0.05), draggable=False, height=0.07, wrapWidth=comp_wrap_width_body, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    t_instruction_continue = visual.TextStim(win=win, name='t_instruction_continue',
        text='Press the SPACEBAR to begin',
        font='Arial',
        pos=(0, -0.35), draggable=False, height=0.04, wrapWidth=comp_wrap_width_continue, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    key_resp_instruction = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "phodaDelay" ---
    t_blank_delayer = visual.TextStim(win=win, name='t_blank_delayer',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "phodaView" ---
    image_phoda = visual.ImageStim(
        win=win,
        name='image_phoda', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=1.0,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    p_port_phoda = parallel.ParallelPort(address='0x0378')
    t_phoda_continue = visual.TextStim(win=win, name='t_phoda_continue',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.04, wrapWidth=comp_wrap_width_continue, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    key_resp_phoda = keyboard.Keyboard(deviceName='defaultKeyboard')
    
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
        units='height', pos=(0, 0), draggable=False, height=0.07, wrapWidth=comp_wrap_width_header, ori=0.0, 
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
        trialList=data.importConditions('loopPhodaIntro.csv'), 
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
        #thisExp.addData('rating', sm_slider.getRating())
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
    
    # set up handler to look after randomisation of conditions etc
    phoda_mode_loop = data.TrialHandler2(
        name='phoda_mode_loop',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('loopPhodaMode.csv'), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(phoda_mode_loop)  # add the loop to the experiment
    thisPhoda_mode_loop = phoda_mode_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPhoda_mode_loop.rgb)
    if thisPhoda_mode_loop != None:
        for paramName in thisPhoda_mode_loop:
            globals()[paramName] = thisPhoda_mode_loop[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisPhoda_mode_loop in phoda_mode_loop:
        phoda_mode_loop.status = STARTED
        if hasattr(thisPhoda_mode_loop, 'status'):
            thisPhoda_mode_loop.status = STARTED
        currentLoop = phoda_mode_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisPhoda_mode_loop.rgb)
        if thisPhoda_mode_loop != None:
            for paramName in thisPhoda_mode_loop:
                globals()[paramName] = thisPhoda_mode_loop[paramName]
        
        # --- Prepare to start Routine "instruction" ---
        # create an object to store info about Routine instruction
        instruction = data.Routine(
            name='instruction',
            components=[t_instruction_body, t_instruction_continue, key_resp_instruction],
        )
        instruction.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_instruction_helper
        # ------------------------------------------------------------
        # code_instruction_helper
        
        # Select the appropriate instruction text based 
        # on the current trial's mode value
        # ------------------------------------------------------------
        # Determine whether this PHODA trial is in view or rate mode.
        # In view mode, components must be allowed to start immediately.
        isModeView = (mode == 'view')
        isModeRate = (mode == 'rate')
        
        if isModeView:
            instruction_text = view_mode_text
        elif isModeRate:
            instruction_text = rate_mode_text
        else:
            # Fallback message in case the condition file contains an
            # unexpected category. This helps catch typos or file issues.
            instruction_text = "Unknown mode. Please contact the experimenter."
        t_instruction_body.setText(instruction_text)
        # create starting attributes for key_resp_instruction
        key_resp_instruction.keys = []
        key_resp_instruction.rt = []
        _key_resp_instruction_allKeys = []
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
        thisExp.currentRoutine = instruction
        instruction.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisPhoda_mode_loop, 'status') and thisPhoda_mode_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *t_instruction_body* updates
            
            # if t_instruction_body is starting this frame...
            if t_instruction_body.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                t_instruction_body.frameNStart = frameN  # exact frame index
                t_instruction_body.tStart = t  # local t and not account for scr refresh
                t_instruction_body.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(t_instruction_body, 'tStartRefresh')  # time at next scr refresh
                # update status
                t_instruction_body.status = STARTED
                t_instruction_body.setAutoDraw(True)
            
            # if t_instruction_body is active this frame...
            if t_instruction_body.status == STARTED:
                # update params
                pass
            
            # *t_instruction_continue* updates
            
            # if t_instruction_continue is starting this frame...
            if t_instruction_continue.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                t_instruction_continue.frameNStart = frameN  # exact frame index
                t_instruction_continue.tStart = t  # local t and not account for scr refresh
                t_instruction_continue.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(t_instruction_continue, 'tStartRefresh')  # time at next scr refresh
                # update status
                t_instruction_continue.status = STARTED
                t_instruction_continue.setAutoDraw(True)
            
            # if t_instruction_continue is active this frame...
            if t_instruction_continue.status == STARTED:
                # update params
                pass
            
            # *key_resp_instruction* updates
            waitOnFlip = False
            
            # if key_resp_instruction is starting this frame...
            if key_resp_instruction.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_instruction.frameNStart = frameN  # exact frame index
                key_resp_instruction.tStart = t  # local t and not account for scr refresh
                key_resp_instruction.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_instruction, 'tStartRefresh')  # time at next scr refresh
                # update status
                key_resp_instruction.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_instruction.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_instruction.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_instruction.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_instruction.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_instruction_allKeys.extend(theseKeys)
                if len(_key_resp_instruction_allKeys):
                    key_resp_instruction.keys = _key_resp_instruction_allKeys[-1].name  # just the last key pressed
                    key_resp_instruction.rt = _key_resp_instruction_allKeys[-1].rt
                    key_resp_instruction.duration = _key_resp_instruction_allKeys[-1].duration
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
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                instruction.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if instruction.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
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
        # the Routine "instruction" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        phoda_trials = data.TrialHandler2(
            name='phoda_trials',
            nReps=1.0, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions('phodaPhotoList.csv'), 
            seed=None, 
            isTrials=True, 
        )
        thisExp.addLoop(phoda_trials)  # add the loop to the experiment
        thisPhoda_trial = phoda_trials.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisPhoda_trial.rgb)
        if thisPhoda_trial != None:
            for paramName in thisPhoda_trial:
                globals()[paramName] = thisPhoda_trial[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisPhoda_trial in phoda_trials:
            phoda_trials.status = STARTED
            if hasattr(thisPhoda_trial, 'status'):
                thisPhoda_trial.status = STARTED
            currentLoop = phoda_trials
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisPhoda_trial.rgb)
            if thisPhoda_trial != None:
                for paramName in thisPhoda_trial:
                    globals()[paramName] = thisPhoda_trial[paramName]
            
            # --- Prepare to start Routine "phodaDelay" ---
            # create an object to store info about Routine phodaDelay
            phodaDelay = data.Routine(
                name='phodaDelay',
                components=[t_blank_delayer],
            )
            phodaDelay.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from code_delay_calculator
            # Gives a random float between 1.5 and 2.5 seconds (i.e., 1500–2500 ms)
            # Used for t_blank_delayer
            delay_time = random.uniform(1.5, 2.5)
            
            t_blank_delayer.setText('')
            # store start times for phodaDelay
            phodaDelay.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            phodaDelay.tStart = globalClock.getTime(format='float')
            phodaDelay.status = STARTED
            thisExp.addData('phodaDelay.started', phodaDelay.tStart)
            phodaDelay.maxDuration = None
            # keep track of which components have finished
            phodaDelayComponents = phodaDelay.components
            for thisComponent in phodaDelay.components:
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
            
            # --- Run Routine "phodaDelay" ---
            thisExp.currentRoutine = phodaDelay
            phodaDelay.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisPhoda_trial, 'status') and thisPhoda_trial.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *t_blank_delayer* updates
                
                # if t_blank_delayer is starting this frame...
                if t_blank_delayer.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    t_blank_delayer.frameNStart = frameN  # exact frame index
                    t_blank_delayer.tStart = t  # local t and not account for scr refresh
                    t_blank_delayer.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(t_blank_delayer, 'tStartRefresh')  # time at next scr refresh
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
                        currentRoutine=phodaDelay,
                    )
                    # skip the frame we paused on
                    continue
                
                # has a Component requested the Routine to end?
                if not continueRoutine:
                    phodaDelay.forceEnded = routineForceEnded = True
                # has the Routine been forcibly ended?
                if phodaDelay.forceEnded or routineForceEnded:
                    break
                # has every Component finished?
                continueRoutine = False
                for thisComponent in phodaDelay.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "phodaDelay" ---
            for thisComponent in phodaDelay.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for phodaDelay
            phodaDelay.tStop = globalClock.getTime(format='float')
            phodaDelay.tStopRefresh = tThisFlipGlobal
            thisExp.addData('phodaDelay.stopped', phodaDelay.tStop)
            # the Routine "phodaDelay" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "phodaView" ---
            # create an object to store info about Routine phodaView
            phodaView = data.Routine(
                name='phodaView',
                components=[image_phoda, p_port_phoda, t_phoda_continue, key_resp_phoda],
            )
            phodaView.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from code_phoda_helper
            # PsychoPy Builder evaluates component start conditions before any code runs.
            # We compute a stable flag here so both t_phoda_continue and key_resp_phoda
            # can rely on it without hitting race‑condition issues.
            canContinue = isModeView
            
            # Tracks the last displayed slider value so we only update the rating text
            # when the marker actually changes. This prevents unnecessary property updates
            # and keeps the log file clean.
            last_rating_value = "INIT"
            
            # Only draw slider components in rate mode. View mode hides all rating UI.
            if isModeRate:
                phoda_autodraw_on()
            
            # Reset the slider so markerPos doesn't carry over between trials.
            ps_phoda_slider.reset()
            
            # Set the image path for this trial. Used by image_phoda to load the correct photo.
            ps_phoda_img_path = f'phoda-stimuli\\{phoda_photo}'
            
            image_phoda.setPos((0, ps_phoda_comp_img_pos_y))
            image_phoda.setSize(ps_phoda_comp_img_size)
            image_phoda.setImage(ps_phoda_img_path)
            t_phoda_continue.setPos((0, t_phoda_continue_pos_y))
            t_phoda_continue.setText('Press the SPACEBAR to continue')
            # create starting attributes for key_resp_phoda
            key_resp_phoda.keys = []
            key_resp_phoda.rt = []
            _key_resp_phoda_allKeys = []
            # store start times for phodaView
            phodaView.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            phodaView.tStart = globalClock.getTime(format='float')
            phodaView.status = STARTED
            thisExp.addData('phodaView.started', phodaView.tStart)
            phodaView.maxDuration = None
            # keep track of which components have finished
            phodaViewComponents = phodaView.components
            for thisComponent in phodaView.components:
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
            
            # --- Run Routine "phodaView" ---
            thisExp.currentRoutine = phodaView
            phodaView.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # if trial has changed, end Routine now
                if hasattr(thisPhoda_trial, 'status') and thisPhoda_trial.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # Run 'Each Frame' code from code_phoda_helper
                # Draws the slider, labels, and rating
                if isModeRate:
                    last_rating_value = update_phoda_rating_text(last_rating_value)
                    # If rated, allow spacebar to end the routine
                    if ps_phoda_slider.getRating() is not None:
                        canContinue = True
                
                
                
                # *image_phoda* updates
                
                # if image_phoda is starting this frame...
                if image_phoda.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image_phoda.frameNStart = frameN  # exact frame index
                    image_phoda.tStart = t  # local t and not account for scr refresh
                    image_phoda.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_phoda, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_phoda.started')
                    # update status
                    image_phoda.status = STARTED
                    image_phoda.setAutoDraw(True)
                
                # if image_phoda is active this frame...
                if image_phoda.status == STARTED:
                    # update params
                    pass
                
                # if image_phoda is stopping this frame...
                if image_phoda.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > image_phoda.tStartRefresh + 6-frameTolerance:
                        # keep track of stop time/frame for later
                        image_phoda.tStop = t  # not accounting for scr refresh
                        image_phoda.tStopRefresh = tThisFlipGlobal  # on global time
                        image_phoda.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_phoda.stopped')
                        # update status
                        image_phoda.status = FINISHED
                        image_phoda.setAutoDraw(False)
                # *p_port_phoda* updates
                
                # if p_port_phoda is starting this frame...
                if p_port_phoda.status == NOT_STARTED and image_phoda.status == STARTED:
                    # keep track of start time/frame for later
                    p_port_phoda.frameNStart = frameN  # exact frame index
                    p_port_phoda.tStart = t  # local t and not account for scr refresh
                    p_port_phoda.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(p_port_phoda, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.addData('p_port_phoda.started', t)
                    # update status
                    p_port_phoda.status = STARTED
                    p_port_phoda.status = STARTED
                    win.callOnFlip(p_port_phoda.setData, int(1))
                
                # if p_port_phoda is stopping this frame...
                if p_port_phoda.status == STARTED:
                    if bool(image_phoda.status == STOPPED):
                        # keep track of stop time/frame for later
                        p_port_phoda.tStop = t  # not accounting for scr refresh
                        p_port_phoda.tStopRefresh = tThisFlipGlobal  # on global time
                        p_port_phoda.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.addData('p_port_phoda.stopped', t)
                        # update status
                        p_port_phoda.status = FINISHED
                        win.callOnFlip(p_port_phoda.setData, int(0))
                
                # *t_phoda_continue* updates
                
                # if t_phoda_continue is starting this frame...
                if t_phoda_continue.status == NOT_STARTED and canContinue:
                    # keep track of start time/frame for later
                    t_phoda_continue.frameNStart = frameN  # exact frame index
                    t_phoda_continue.tStart = t  # local t and not account for scr refresh
                    t_phoda_continue.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(t_phoda_continue, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    t_phoda_continue.status = STARTED
                    t_phoda_continue.setAutoDraw(True)
                
                # if t_phoda_continue is active this frame...
                if t_phoda_continue.status == STARTED:
                    # update params
                    pass
                
                # if t_phoda_continue is stopping this frame...
                if t_phoda_continue.status == STARTED:
                    if bool(isModeView):
                        # keep track of stop time/frame for later
                        t_phoda_continue.tStop = t  # not accounting for scr refresh
                        t_phoda_continue.tStopRefresh = tThisFlipGlobal  # on global time
                        t_phoda_continue.frameNStop = frameN  # exact frame index
                        # update status
                        t_phoda_continue.status = FINISHED
                        t_phoda_continue.setAutoDraw(False)
                
                # *key_resp_phoda* updates
                waitOnFlip = False
                
                # if key_resp_phoda is starting this frame...
                if key_resp_phoda.status == NOT_STARTED and canContinue:
                    # keep track of start time/frame for later
                    key_resp_phoda.frameNStart = frameN  # exact frame index
                    key_resp_phoda.tStart = t  # local t and not account for scr refresh
                    key_resp_phoda.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp_phoda, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    key_resp_phoda.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp_phoda.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp_phoda.clearEvents, eventType='keyboard')  # clear events on next screen flip
                
                # if key_resp_phoda is stopping this frame...
                if key_resp_phoda.status == STARTED:
                    if bool(isModeView):
                        # keep track of stop time/frame for later
                        key_resp_phoda.tStop = t  # not accounting for scr refresh
                        key_resp_phoda.tStopRefresh = tThisFlipGlobal  # on global time
                        key_resp_phoda.frameNStop = frameN  # exact frame index
                        # update status
                        key_resp_phoda.status = FINISHED
                        key_resp_phoda.status = FINISHED
                if key_resp_phoda.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp_phoda.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                    _key_resp_phoda_allKeys.extend(theseKeys)
                    if len(_key_resp_phoda_allKeys):
                        key_resp_phoda.keys = _key_resp_phoda_allKeys[-1].name  # just the last key pressed
                        key_resp_phoda.rt = _key_resp_phoda_allKeys[-1].rt
                        key_resp_phoda.duration = _key_resp_phoda_allKeys[-1].duration
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
                        currentRoutine=phodaView,
                    )
                    # skip the frame we paused on
                    continue
                
                # has a Component requested the Routine to end?
                if not continueRoutine:
                    phodaView.forceEnded = routineForceEnded = True
                # has the Routine been forcibly ended?
                if phodaView.forceEnded or routineForceEnded:
                    break
                # has every Component finished?
                continueRoutine = False
                for thisComponent in phodaView.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "phodaView" ---
            for thisComponent in phodaView.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for phodaView
            phodaView.tStop = globalClock.getTime(format='float')
            phodaView.tStopRefresh = tThisFlipGlobal
            thisExp.addData('phodaView.stopped', phodaView.tStop)
            # Run 'End Routine' code from code_phoda_helper
            phoda_autodraw_off()
            
            if isModeRate:
                currentLoop.addData('rating', ps_phoda_slider.getRating())
                currentLoop.addData('rating_rt', ps_phoda_slider.getRT())
                currentLoop.addData('delay_time', delay_time)
                try:
                    print(f'{phoda_photo} rating: {ps_phoda_slider.getRating()}')
                    logging.data(f'{phoda_photo} rating: {ps_phoda_slider.getRating()}')
                    logging.data(f'{phoda_photo} rating_rt: {ps_phoda_slider.getRT()}')
                except:
                    logging.error("Error printing phoda rating")
            if p_port_phoda.status == STARTED:
                win.callOnFlip(p_port_phoda.setData, int(0))
            # the Routine "phodaView" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            # mark thisPhoda_trial as finished
            if hasattr(thisPhoda_trial, 'status'):
                thisPhoda_trial.status = FINISHED
            # if awaiting a pause, pause now
            if phoda_trials.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                phoda_trials.status = STARTED
            thisExp.nextEntry()
            
        # completed 1.0 repeats of 'phoda_trials'
        phoda_trials.status = FINISHED
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # get names of stimulus parameters
        if phoda_trials.trialList in ([], [None], None):
            params = []
        else:
            params = phoda_trials.trialList[0].keys()
        # save data for this loop
        phoda_trials.saveAsExcel(filename + '.xlsx', sheetName='phoda_trials',
            stimOut=params,
            dataOut=['n','all_mean','all_std', 'all_raw'])
        
        # set up handler to look after randomisation of conditions etc
        state_measure_trials = data.TrialHandler2(
            name='state_measure_trials',
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
                if hasattr(thisState_measure_trial, 'status') and thisState_measure_trial.status == STOPPING:
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
            #thisExp.addData('rating', sm_slider.getRating())
            currentLoop.addData('rating', sm_slider.getRating())
            currentLoop.addData('rating_rt', sm_slider.getRT())
            
            try:
                logging.data(f"State-measure rating: {rating_category}, {sm_slider.getRating()}")
            except:
                logging.error("Error printing state-measure rating")
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
        # mark thisPhoda_mode_loop as finished
        if hasattr(thisPhoda_mode_loop, 'status'):
            thisPhoda_mode_loop.status = FINISHED
        # if awaiting a pause, pause now
        if phoda_mode_loop.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            phoda_mode_loop.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'phoda_mode_loop'
    phoda_mode_loop.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    # get names of stimulus parameters
    if phoda_mode_loop.trialList in ([], [None], None):
        params = []
    else:
        params = phoda_mode_loop.trialList[0].keys()
    # save data for this loop
    phoda_mode_loop.saveAsExcel(filename + '.xlsx', sheetName='phoda_mode_loop',
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
