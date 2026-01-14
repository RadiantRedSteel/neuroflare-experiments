#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2025.2.3),
    on January 13, 2026, at 22:52
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
expName = 'open-closed'  # from the Builder filename that created this script
expVersion = '1.02'
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
        originPath='D:\\Github\\neuroflare-experiments\\neuroflare\\experiments\\open-closed\\open-closed_lastrun.py',
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
    # Run 'Begin Experiment' code from code_open_closed_setup
    # ------------------------------------------------------------
    # code_open_closed_setup
    # Define the instruction text for each eye-state condition.
    # ------------------------------------------------------------
    # These strings are kept in-code (rather than in the CSV) so the
    # condition file only needs to specify the Category. This keeps
    # the output cleaner and prevents text duplication across files.
    open_eyes_text = "Look at the + mark and stay still for 5 minutes."
    closed_eyes_text = "Close your eyes and stay still for 5 minutes."
    
    # This variable will be updated on each loop iteration by the
    # instruction helper code. It provides the text shown in the
    # instruction routine.
    instruction_text = "Default message."
    # Run 'Begin Experiment' code from code_sound_setup
    # ------------------------------------------------------------
    # code_sound_setup
    # Create a sound that will be played after the fixation routine
    # Also used during testing to ensure sound backend is working
    # ------------------------------------------------------------
    
    test_sound = sound.Sound(
        value=330,
        volume=1.0,
        secs=1.0,
        hamming=True,
        stereo=True,
        autoLog=False,
        name='test_sound'
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
    
    # --- Initialize components for Routine "testSound" ---
    t_test_sound = visual.TextStim(win=win, name='t_test_sound',
        text='At the end of each round, a brief sound will play to let you know it is complete. \n\nPress T to test the sound.\n\nIf you do not hear it, try pressing T again or adjust your volume.',
        font='Arial',
        pos=(0.0, 0.05), draggable=False, height=0.07, wrapWidth=comp_wrap_width_body, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    t_test_sound_continue = visual.TextStim(win=win, name='t_test_sound_continue',
        text='Press the SPACEBAR to continue',
        font='Arial',
        pos=(0, -0.35), draggable=False, height=0.04, wrapWidth=comp_wrap_width_continue, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    key_resp_test_sound = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "stateMeasure" ---
    t_sm_continue = visual.TextStim(win=win, name='t_sm_continue',
        text='Press the SPACEBAR to continue',
        font='Arial',
        pos=(0, -0.35), draggable=False, height=0.04, wrapWidth=comp_wrap_width_continue, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_sm = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "instruction" ---
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
    
    # --- Initialize components for Routine "fixation" ---
    cross_fixation = visual.ShapeStim(
        win=win, name='cross_fixation', vertices='cross',units='height', 
        size=(0.2, 0.2),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    t_fixation_closed = visual.TextStim(win=win, name='t_fixation_closed',
        text='',
        font='Arial',
        units='height', pos=(0.0, -0.4), draggable=False, height=0.03, wrapWidth=comp_wrap_width_body, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    p_port_fixation = parallel.ParallelPort(address='0x0378')
    
    # --- Initialize components for Routine "fixationFinish" ---
    t_fixation_finish = visual.TextStim(win=win, name='t_fixation_finish',
        text='This round is complete.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.07, wrapWidth=comp_wrap_width_body, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "stateMeasure" ---
    t_sm_continue = visual.TextStim(win=win, name='t_sm_continue',
        text='Press the SPACEBAR to continue',
        font='Arial',
        pos=(0, -0.35), draggable=False, height=0.04, wrapWidth=comp_wrap_width_continue, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
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
        trialList=data.importConditions('loopOpenClosedIntro.csv'), 
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
    
    
    # --- Prepare to start Routine "testSound" ---
    # create an object to store info about Routine testSound
    testSound = data.Routine(
        name='testSound',
        components=[t_test_sound, t_test_sound_continue, key_resp_test_sound],
    )
    testSound.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from code_test_sound_helper
    # Clear any recorded keys so that the sound doesn't play early
    event.clearEvents(eventType='keyboard')
    
    # If sound_tested is true, then the user can continue to the next routine
    sound_tested = False
    # create starting attributes for key_resp_test_sound
    key_resp_test_sound.keys = []
    key_resp_test_sound.rt = []
    _key_resp_test_sound_allKeys = []
    # store start times for testSound
    testSound.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    testSound.tStart = globalClock.getTime(format='float')
    testSound.status = STARTED
    thisExp.addData('testSound.started', testSound.tStart)
    testSound.maxDuration = None
    # keep track of which components have finished
    testSoundComponents = testSound.components
    for thisComponent in testSound.components:
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
    
    # --- Run Routine "testSound" ---
    thisExp.currentRoutine = testSound
    testSound.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *t_test_sound* updates
        
        # if t_test_sound is starting this frame...
        if t_test_sound.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            t_test_sound.frameNStart = frameN  # exact frame index
            t_test_sound.tStart = t  # local t and not account for scr refresh
            t_test_sound.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(t_test_sound, 'tStartRefresh')  # time at next scr refresh
            # update status
            t_test_sound.status = STARTED
            t_test_sound.setAutoDraw(True)
        
        # if t_test_sound is active this frame...
        if t_test_sound.status == STARTED:
            # update params
            pass
        # Run 'Each Frame' code from code_test_sound_helper
        # Check to see if the key is pressed
        # If it is, play the sound and let the user continue
        if 't' in event.getKeys() and not test_sound.isPlaying:
            test_sound.play()
            sound_tested = True
        
        
        # *t_test_sound_continue* updates
        
        # if t_test_sound_continue is starting this frame...
        if t_test_sound_continue.status == NOT_STARTED and sound_tested:
            # keep track of start time/frame for later
            t_test_sound_continue.frameNStart = frameN  # exact frame index
            t_test_sound_continue.tStart = t  # local t and not account for scr refresh
            t_test_sound_continue.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(t_test_sound_continue, 'tStartRefresh')  # time at next scr refresh
            # update status
            t_test_sound_continue.status = STARTED
            t_test_sound_continue.setAutoDraw(True)
        
        # if t_test_sound_continue is active this frame...
        if t_test_sound_continue.status == STARTED:
            # update params
            pass
        
        # *key_resp_test_sound* updates
        waitOnFlip = False
        
        # if key_resp_test_sound is starting this frame...
        if key_resp_test_sound.status == NOT_STARTED and sound_tested:
            # keep track of start time/frame for later
            key_resp_test_sound.frameNStart = frameN  # exact frame index
            key_resp_test_sound.tStart = t  # local t and not account for scr refresh
            key_resp_test_sound.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_test_sound, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_resp_test_sound.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_test_sound.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_test_sound.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_test_sound.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_test_sound.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_test_sound_allKeys.extend(theseKeys)
            if len(_key_resp_test_sound_allKeys):
                key_resp_test_sound.keys = _key_resp_test_sound_allKeys[-1].name  # just the last key pressed
                key_resp_test_sound.rt = _key_resp_test_sound_allKeys[-1].rt
                key_resp_test_sound.duration = _key_resp_test_sound_allKeys[-1].duration
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
                currentRoutine=testSound,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            testSound.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if testSound.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in testSound.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "testSound" ---
    for thisComponent in testSound.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for testSound
    testSound.tStop = globalClock.getTime(format='float')
    testSound.tStopRefresh = tThisFlipGlobal
    thisExp.addData('testSound.stopped', testSound.tStop)
    thisExp.nextEntry()
    # the Routine "testSound" was not non-slip safe, so reset the non-slip timer
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
        trialList=data.importConditions('loopOpenClosedTrial.csv'), 
        seed=None, 
        isTrials=True, 
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
            components=[t_instruction_body, t_instruction_continue, key_resp_instruction],
        )
        instruction.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_instruction_helper
        # ------------------------------------------------------------
        # code_instruction_helper
        
        # Select the appropriate instruction text based on the current
        # trial's Category value. This keeps the logic centralized and
        # avoids storing long text strings in the condition file.
        # ------------------------------------------------------------
        if Category == "Open-Eyes":
            instruction_text = open_eyes_text
        elif Category == "Closed-Eyes":
            instruction_text = closed_eyes_text
        else:
            # Fallback message in case the condition file contains an
            # unexpected category. This helps catch typos or file issues.
            instruction_text = "Unknown category. Please contact the experimenter."
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
            if hasattr(thisOpenclosed_trial, 'status') and thisOpenclosed_trial.status == STOPPING:
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
        
        # --- Prepare to start Routine "fixation" ---
        # create an object to store info about Routine fixation
        fixation = data.Routine(
            name='fixation',
            components=[cross_fixation, t_fixation_closed, p_port_fixation],
        )
        fixation.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        t_fixation_closed.setText('Please keep your eyes closed during this block.')
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
        thisExp.currentRoutine = fixation
        fixation.forceEnded = routineForceEnded = not continueRoutine
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
            
            # *t_fixation_closed* updates
            
            # if t_fixation_closed is starting this frame...
            if t_fixation_closed.status == NOT_STARTED and Category == "Closed-Eyes":
                # keep track of start time/frame for later
                t_fixation_closed.frameNStart = frameN  # exact frame index
                t_fixation_closed.tStart = t  # local t and not account for scr refresh
                t_fixation_closed.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(t_fixation_closed, 'tStartRefresh')  # time at next scr refresh
                # update status
                t_fixation_closed.status = STARTED
                t_fixation_closed.setAutoDraw(True)
            
            # if t_fixation_closed is active this frame...
            if t_fixation_closed.status == STARTED:
                # update params
                pass
            
            # if t_fixation_closed is stopping this frame...
            if t_fixation_closed.status == STARTED:
                if bool(Category != "Closed-Eyes"):
                    # keep track of stop time/frame for later
                    t_fixation_closed.tStop = t  # not accounting for scr refresh
                    t_fixation_closed.tStopRefresh = tThisFlipGlobal  # on global time
                    t_fixation_closed.frameNStop = frameN  # exact frame index
                    # update status
                    t_fixation_closed.status = FINISHED
                    t_fixation_closed.setAutoDraw(False)
            # *p_port_fixation* updates
            
            # if p_port_fixation is starting this frame...
            if p_port_fixation.status == NOT_STARTED and cross_fixation.status == STARTED:
                # keep track of start time/frame for later
                p_port_fixation.frameNStart = frameN  # exact frame index
                p_port_fixation.tStart = t  # local t and not account for scr refresh
                p_port_fixation.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_port_fixation, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'p_port_fixation.started')
                # update status
                p_port_fixation.status = STARTED
                p_port_fixation.status = STARTED
                win.callOnFlip(p_port_fixation.setData, int(1))
            
            # if p_port_fixation is stopping this frame...
            if p_port_fixation.status == STARTED:
                if bool(cross_fixation.status == STOPPED):
                    # keep track of stop time/frame for later
                    p_port_fixation.tStop = t  # not accounting for scr refresh
                    p_port_fixation.tStopRefresh = tThisFlipGlobal  # on global time
                    p_port_fixation.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'p_port_fixation.stopped')
                    # update status
                    p_port_fixation.status = FINISHED
                    win.callOnFlip(p_port_fixation.setData, int(0))
            
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
        if p_port_fixation.status == STARTED:
            win.callOnFlip(p_port_fixation.setData, int(0))
        # the Routine "fixation" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "fixationFinish" ---
        # create an object to store info about Routine fixationFinish
        fixationFinish = data.Routine(
            name='fixationFinish',
            components=[t_fixation_finish],
        )
        fixationFinish.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_fixation_finish_helper
        # Plays a sound created in code_sound_setup to let 
        # the user know that the fixatio nroutine has ended
        test_sound.play()
        
        # store start times for fixationFinish
        fixationFinish.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        fixationFinish.tStart = globalClock.getTime(format='float')
        fixationFinish.status = STARTED
        thisExp.addData('fixationFinish.started', fixationFinish.tStart)
        fixationFinish.maxDuration = None
        # keep track of which components have finished
        fixationFinishComponents = fixationFinish.components
        for thisComponent in fixationFinish.components:
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
        
        # --- Run Routine "fixationFinish" ---
        thisExp.currentRoutine = fixationFinish
        fixationFinish.forceEnded = routineForceEnded = not continueRoutine
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
            
            # *t_fixation_finish* updates
            
            # if t_fixation_finish is starting this frame...
            if t_fixation_finish.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                t_fixation_finish.frameNStart = frameN  # exact frame index
                t_fixation_finish.tStart = t  # local t and not account for scr refresh
                t_fixation_finish.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(t_fixation_finish, 'tStartRefresh')  # time at next scr refresh
                # update status
                t_fixation_finish.status = STARTED
                t_fixation_finish.setAutoDraw(True)
            
            # if t_fixation_finish is active this frame...
            if t_fixation_finish.status == STARTED:
                # update params
                pass
            
            # if t_fixation_finish is stopping this frame...
            if t_fixation_finish.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > t_fixation_finish.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    t_fixation_finish.tStop = t  # not accounting for scr refresh
                    t_fixation_finish.tStopRefresh = tThisFlipGlobal  # on global time
                    t_fixation_finish.frameNStop = frameN  # exact frame index
                    # update status
                    t_fixation_finish.status = FINISHED
                    t_fixation_finish.setAutoDraw(False)
            
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
                    currentRoutine=fixationFinish,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                fixationFinish.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if fixationFinish.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in fixationFinish.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "fixationFinish" ---
        for thisComponent in fixationFinish.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for fixationFinish
        fixationFinish.tStop = globalClock.getTime(format='float')
        fixationFinish.tStopRefresh = tThisFlipGlobal
        thisExp.addData('fixationFinish.stopped', fixationFinish.tStop)
        # Run 'End Routine' code from code_fixation_finish_helper
        ## Stop the sound, just in case it trys to escape the routine.
        #test_sound.stop()
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if fixationFinish.maxDurationReached:
            routineTimer.addTime(-fixationFinish.maxDuration)
        elif fixationFinish.forceEnded:
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
            trialList=data.importConditions(
            '../../shared/loop-templates/loopStateMeasure.csv', 
            selection='[0, 1, 2, 3, 4, 5, 6]'
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
                if hasattr(thisState_measure_trial, 'status') and thisState_measure_trial.status == STOPPING:
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
