from typing import Any, Dict, List, Optional, Union
from psychopy import visual, logging
from .state_measure_labels import resolve_spec, is_sam

# --- Public helpers ----------------------------------------------------------

def parse_tick_values(raw: Union[str, List[int]]) -> List[int]:
    """Parse tick values from PsychoPy's CSV/loop variable.
    Accepts a list or a comma-separated string of ints.
    """
    if isinstance(raw, list):
        return raw
    return [int(x.strip()) for x in str(raw).split(',')]


# --- StateMeasure core -------------------------------------------------------
class StateMeasure:
    """Encapsulates State-Measure logic for Builder experiments.

    - Expects Builder globals like `win`, `aspect`, `screen_width`, `screen_height` to exist,
      but allows explicit injection for testing or scripts.
    - Handles both Generic and SAM rating types.
    - Creates components (slider, min/max labels, optional SAM image) with stable names.
    - Provides utility methods to manage AutoDraw, check rating, and collect data.
    """

    # Generic defaults (height units)
    GEN_SLIDER_WIDTH_RATIO = 0.6   # 60% of screen width (multiplied by aspect)
    GEN_SLIDER_HEIGHT = 0.10
    GEN_SLIDER_POS_Y = 0.0
    GEN_LABEL_POS_Y_OFFSET = -0.20

    # SAM image (pixels)
    SAM_IMG_PX_WIDTH = 1168
    SAM_IMG_PX_HEIGHT = 231
    SAM_PICTOGRAM_PX_WIDTH = 215
    SAM_GAP_PX_WIDTH = 23

    # SAM layout (height units)
    SAM_IMG_SIZE_HEIGHT = 0.25      # 25% of screen height
    SAM_IMG_POS_Y = 0.10
    SAM_MESSAGE_POS_Y_OFFSET = 0.20
    SAM_SLIDER_POS_Y = -0.09
    SAM_SLIDER_HEIGHT = 0.06
    SAM_LABEL_POS_Y_OFFSET = -0.15

    # Labels and text
    LABEL_WRAP_WIDTH_RATIO = 0.20  # 20% of screen width (multiplied by aspect)

    # Message text (all types)
    MESSAGE_POS_Y = 0.35
    MESSAGE_WRAP_WIDTH_RATIO = 0.8  # 80% of screen width (multiplied by aspect)
    MESSAGE_HEIGHT = 0.07
    MESSAGE_FONT = 'Arial'
    MESSAGE_COLOR = 'white'

    def __init__(
        self,
        win: Optional[visual.Window] = None,
        aspect: Optional[float] = None,
        screen_width: Optional[int] = None,
        screen_height: Optional[int] = None,
    ) -> None:
        # Resolve environment from globals if not provided
        if win is None:
            try:
                from psychopy import visual as _v
                # Use global win if available
                win = globals().get('win', None)
                if win is None:
                    raise RuntimeError("StateMeasure requires a PsychoPy 'win'.")
            except Exception as e:
                raise RuntimeError("StateMeasure init failed: no window.") from e
        self.win = win
        self.units = win.units

        if aspect is None:
            aspect = globals().get('aspect', None)
            if aspect is None:
                # Fallback: compute from window size
                w, h = self.win.size
                aspect = w / h
        self.aspect = aspect

        if screen_width is None or screen_height is None:
            try:
                screen_width, screen_height = self.win.size
            except Exception:
                screen_width, screen_height = (1920, 1080)
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Runtime components
        self.slider: Optional[visual.Slider] = None
        self.label_min: Optional[visual.TextStim] = None
        self.label_max: Optional[visual.TextStim] = None
        self.message_text: Optional[visual.TextStim] = None
        self.sam_image: Optional[visual.ImageStim] = None
        self._last_rating_value: Any = None

        # Cached geometry
        self._sam_tick_positions: Optional[List[float]] = None

    # --- Geometry ------------------------------------------------------------
    def _compute_generic_geometry(self) -> Dict[str, Any]:
        slider_w = self.aspect * self.GEN_SLIDER_WIDTH_RATIO
        slider_h = self.GEN_SLIDER_HEIGHT
        pos_y = self.GEN_SLIDER_POS_Y
        label_pos_y = pos_y + self.GEN_LABEL_POS_Y_OFFSET
        return {
            'slider_size': (slider_w, slider_h),
            'slider_pos': (0.0, pos_y),
            'label_pos_y': label_pos_y,
            'half_width': slider_w / 2.0,
        }

    def _compute_sam_tick_positions(self, img_size_width: float) -> List[float]:
        """Compute 9 tick positions aligned to SAM pictograms within image width."""
        scale_factor = img_size_width / self.SAM_IMG_PX_WIDTH
        pict_w = self.SAM_PICTOGRAM_PX_WIDTH * scale_factor
        gap_w = self.SAM_GAP_PX_WIDTH * scale_factor

        tick_positions: List[float] = []
        x = -img_size_width / 2.0 + pict_w / 2.0
        for i in range(9):
            tick_positions.append(x)
            if i % 2 == 0:
                x += pict_w / 2.0 + gap_w / 2.0
            else:
                x += gap_w / 2.0 + pict_w / 2.0
        return tick_positions

    def _compute_sam_geometry(self) -> Dict[str, Any]:
        img_h = self.SAM_IMG_SIZE_HEIGHT
        img_w = img_h * (self.SAM_IMG_PX_WIDTH / self.SAM_IMG_PX_HEIGHT)
        slider_h = self.SAM_SLIDER_HEIGHT
        slider_w = img_w * ((self.SAM_IMG_PX_WIDTH - self.SAM_PICTOGRAM_PX_WIDTH) / self.SAM_IMG_PX_WIDTH)

        # Cache tick positions per geometry
        self._sam_tick_positions = self._compute_sam_tick_positions(img_w)

        return {
            'img_size': (img_w, img_h),
            'img_pos': (0.0, self.SAM_IMG_POS_Y),
            'slider_size': (slider_w, slider_h),
            'slider_pos': (0.0, self.SAM_SLIDER_POS_Y),
            'label_pos_y': self.SAM_SLIDER_POS_Y + self.SAM_LABEL_POS_Y_OFFSET,
        }

    def _compute_tick_positions_generic(self, tick_values: List[int], half_width: float, total_width: float) -> List[float]:
        if len(tick_values) <= 1:
            return [0.0]
        return [
            -half_width + (i / (len(tick_values) - 1)) * total_width
            for i in range(len(tick_values))
        ]

    # --- Component creation --------------------------------------------------
    def begin_routine(
        self,
        *,
        rating_type: str,
        rating_category: str,
        rating_message: str,
        rating_min_label: str,
        rating_max_label: str,
        picture_path: str,
        tick_values: Union[str, List[int]],
        granularity: int,
        style: str,
        auto_draw: bool = True,
    ) -> None:
        """Create components for the current routine row.

        - Builds a new slider each time (PsychoPy restriction on ticks).
        - Creates min/max labels under the slider.
        - For SAM, optionally creates and shows the image if picture_path is set.
        - Optionally enables AutoDraw for managed components.
        """
        # Parse ticks
        ticks = parse_tick_values(tick_values)

        # Decide geometry based on rating_type
        if str(rating_type).strip().upper() == 'SAM':
            geom = self._compute_sam_geometry()
            tick_positions = self._sam_tick_positions or []
            label_min_pos = (tick_positions[0], geom['label_pos_y'])
            label_max_pos = (tick_positions[-1], geom['label_pos_y'])
        else:
            geom = self._compute_generic_geometry()
            total_w = geom['half_width'] * 2.0
            tick_positions = self._compute_tick_positions_generic(ticks, geom['half_width'], total_w)
            label_min_pos = (tick_positions[0], geom['label_pos_y'])
            label_max_pos = (tick_positions[-1], geom['label_pos_y'])

        # Create slider
        self.slider = visual.Slider(
            win=self.win,
            ticks=ticks,
            labels=ticks,
            granularity=granularity,
            style=str(style).lower(),
            pos=geom['slider_pos'],
            size=geom['slider_size'],
            labelHeight=0.05,
            colorSpace='rgb',
            markerColor='Red',
            lineColor='White',
            labelColor='LightGray',
            font='Noto Sans',
            name='sm_slider',
            autoLog=False,
        )

        # Min/Max labels
        self.label_min = visual.TextStim(
            win=self.win,
            text=rating_min_label,
            pos=(label_min_pos[0], label_min_pos[1]),
            #wrapWidth=self.aspect * self.LABEL_WRAP_WIDTH_RATIO,
            anchorVert='top',
            height=0.05,
            color='white',
            font='Noto Sans',
            name='sm_label_min',
            autoLog=False,
        )
        self.label_max = visual.TextStim(
            win=self.win,
            text=rating_max_label,
            pos=(label_max_pos[0], label_max_pos[1]),
            #wrapWidth=self.aspect * self.LABEL_WRAP_WIDTH_RATIO,
            anchorVert='top',
            height=0.05,
            color='white',
            font='Noto Sans',
            name='sm_label_max',
            autoLog=False,
        )

        # Message text
        self.message_text = visual.TextStim(
            win=self.win,
            text=rating_message,
            pos=(0.0, self.MESSAGE_POS_Y),
            wrapWidth=self.aspect * self.MESSAGE_WRAP_WIDTH_RATIO,
            height=self.MESSAGE_HEIGHT,
            color=self.MESSAGE_COLOR,
            font=self.MESSAGE_FONT,
            name='sm_message_text',
            autoLog=False,
        )

        # SAM image (optional)
        self.sam_image = None
        if str(rating_type).strip().upper() == 'SAM' and picture_path:
            self.sam_image = visual.ImageStim(
                win=self.win,
                image=picture_path,
                units='height',
                pos=(0.0, geom['img_pos'][1]),
                size=geom['img_size'],
                name='sm_sam_image',
                autoLog=False,
            )

        if auto_draw:
            self.autodraw_on()

        logging.debug(f"StateMeasure: began routine for '{rating_category}' ({rating_type}).")

    def begin_routine_from_category(
        self,
        rating_category: str,
        overrides: Optional[Dict[str, Any]] = None,
        *,
        auto_draw: bool = True,
    ) -> None:
        """Convenience: resolve spec by category and start routine.

        - Infers SAM vs Generic from category name (prefix 'sam_').
        - Merges built-in defaults and optional per-row overrides.
        """
        cat = str(rating_category).strip()
        spec = resolve_spec(cat, overrides)
        rating_type = 'SAM' if is_sam(cat) else 'GENERIC'

        self.begin_routine(
            rating_type=rating_type,
            rating_category=cat,
            rating_message=spec.get('message', ''),
            rating_min_label=spec.get('min_label', ''),
            rating_max_label=spec.get('max_label', ''),
            picture_path=spec.get('picture_path', ''),
            tick_values=spec.get('tick_values', []),
            granularity=int(spec.get('granularity', 1)),
            style=str(spec.get('style', 'rating')),
            auto_draw=auto_draw,
        )

    # --- AutoDraw management -------------------------------------------------
    def autodraw_on(self) -> None:
        if self.message_text:
            self.message_text.setAutoDraw(True)
        if self.slider:
            self.slider.setAutoDraw(True)
        if self.label_min:
            self.label_min.setAutoDraw(True)
        if self.label_max:
            self.label_max.setAutoDraw(True)
        if self.sam_image:
            self.sam_image.setAutoDraw(True)
        logging.exp("StateMeasure components: AutoDraw ON")

    def autodraw_off(self) -> None:
        if self.message_text:
            self.message_text.setAutoDraw(False)
        if self.slider:
            self.slider.setAutoDraw(False)
        if self.label_min:
            self.label_min.setAutoDraw(False)
        if self.label_max:
            self.label_max.setAutoDraw(False)
        if self.sam_image:
            self.sam_image.setAutoDraw(False)
        logging.exp("StateMeasure components: AutoDraw OFF")

    # --- Routine helpers -----------------------------------------------------
    def has_rating(self) -> bool:
        return (self.slider is not None) and (self.slider.getRating() is not None)

    def get_rating(self) -> Optional[float]:
        return self.slider.getRating() if self.slider else None

    def get_rt(self) -> Optional[float]:
        return self.slider.getRT() if self.slider else None

    def end_routine(self, currentLoop: Optional[Any] = None) -> Dict[str, Any]:
        """Collect rating/RT and optionally add data to Builder's currentLoop."""
        rating = self.get_rating()
        rt = self.get_rt()

        if currentLoop is not None:
            try:
                currentLoop.addData('rating', rating)
                currentLoop.addData('rating_rt', rt)
            except Exception:
                logging.error("StateMeasure: failed to add data to currentLoop.")

        try:
            logging.data(f"State-measure rating collected: {rating}")
        except Exception:
            logging.error("StateMeasure: error logging rating.")

        # Turn off AutoDraw so next routines control their components cleanly
        self.autodraw_off()

        return {'rating': rating, 'rating_rt': rt}

    # --- Optional debug/demo -------------------------------------------------
    def demo_loop(self) -> Optional[float]:
        """Run a simple AutoDraw demo until SPACE when rated; ESC to cancel.
        Returns final rating or None.
        """
        from psychopy import event

        while True:
            keys = event.getKeys(['space', 'escape'])
            if 'space' in keys and self.has_rating():
                return self.get_rating()
            if 'escape' in keys:
                return None
            self.win.flip()
