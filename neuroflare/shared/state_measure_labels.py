from typing import Any, Dict, Optional

# Category naming convention: SAM items start with 'sam_'

def is_sam(category: str) -> bool:
    return str(category).lower().startswith('sam')

# Built-in defaults fallbacks
DEFAULT_GENERIC = {
    'message': '',
    'min_label': 'Minimum',
    'max_label': 'Maximum',
    'picture_path': '',
    'tick_values': [i for i in range(0, 11)],
    'granularity': 1,
    'style': 'rating',
}

DEFAULT_SAM = {
    'message': '',
    'min_label': 'Low',
    'max_label': 'High',
    'picture_path': '',
    'tick_values': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    'granularity': 1,
    'style': 'radio',
}

# Canonical specs keyed by rating_category
RATING_SPECS: Dict[str, Dict[str, Any]] = {
    # Generic
    'fatigue': {
        'message': 'Rate your current fatigue level',
        'min_label': 'No fatigue\nat all',
        'max_label': 'As bad as\nyou can imagine',
        'tick_values': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'granularity': 1,
        'style': 'rating',
    },
    'sleepiness': {
        'message': 'Rate your current sleepiness level',
        'min_label': 'Not at\nall sleepy',
        'max_label': 'As bad as\nyou can imagine',
        'tick_values': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'granularity': 1,
        'style': 'rating',
    },
    'pain': {
        'message': 'Rate your current pain level',
        'min_label': 'No pain',
        'max_label': 'Worst possible\npain',
        'tick_values': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'granularity': 1,
        'style': 'rating',
    },
    'pain_unpleasantness': {
        'message': 'Rate your current pain unpleasantness level',
        'min_label': 'Not at all\nunpleasant pain',
        'max_label': 'Worst possible\nunpleasant pain',
        'tick_values': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'granularity': 1,
        'style': 'rating',
    },
    'emotion_regulation': {
        'message': 'Rate how successful you were in doing the emotion regulation tasks',
        'min_label': 'Not at\nall successful',
        'max_label': 'Completely\nsuccessful',
        'tick_values': [0, 1, 2, 3, 4, 5, 6, 7],
        'granularity': 1,
        'style': 'rating',
    },

    # SAM
    'sam_valence': {
        'message': 'Rate how pleasant or unpleasant you feel',
        'min_label': 'Pleasant',
        'max_label': 'Unpleasant',
        'picture_path': '../../shared/pictures/sam-valence.png',
        'tick_values': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        'granularity': 1,
        'style': 'radio',
    },
    'sam_arousal': {
        'message': 'Rate how calm or excited you feel',
        'min_label': 'Calm',
        'max_label': 'Excited',
        'picture_path': '../../shared/pictures/sam-arousal.png',
        'tick_values': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        'granularity': 1,
        'style': 'radio',
    },
    'sam_dominance': {
        'message': 'Rate how dominated or in control you feel',
        'min_label': 'Dominated',
        'max_label': 'In Control',
        'picture_path': '../../shared/pictures/sam-dominance.png',
        'tick_values': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        'granularity': 1,
        'style': 'radio',
    },
}


def _base_for_category(category: str) -> Dict[str, Any]:
    return DEFAULT_SAM.copy() if is_sam(category) else DEFAULT_GENERIC.copy()


def resolve_spec(category: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Resolve a complete spec for a rating category.

    Merge order (later wins): defaults -> RATING_SPECS[category] -> overrides.
    Returns a dict with keys: message, min_label, max_label, picture_path,
    tick_values, granularity, style.
    """
    cat = str(category).lower().strip()
    spec = _base_for_category(cat)
    spec.update(RATING_SPECS.get(cat, {}))
    if overrides:
        spec.update(overrides)
    return spec
