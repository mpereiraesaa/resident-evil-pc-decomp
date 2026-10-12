"""Versioned reconstruction options; experimental profiles require evaluation."""

import copy

PROFILES = {
    'existing': {},
    'creative-v2': {
        'parametric_details': True, 'creative_fallback': True,
        'stair_refinement': True, 'joint_stair_supports': True,
        'constrain_stair_proportions': True, 'raised_support_occlusion': True,
        'feasible_stair_starts': True, 'suspended_stair_starts': True,
        'scene_depth_fit': True, 'scene_depth_polish': True,
        'constrain_stair_headroom': True, 'creative_stair_landings': True,
        'single_view_wall_fixtures': True, 'fixture_part_fit': True,
        'inset_fixture_anchors': True, 'multilight_wall_fixtures': True,
        'refine_fixture_anchor': True, 'local_fixture_appearance': True,
        'foreground_fixture_guard': True, 'handrail_candidates': True,
    },
}


def apply_profile(name, recipes, rooms):
    """Explicit room recipes override defaults; inputs are never mutated."""
    if name not in PROFILES:
        raise ValueError('Unknown detail profile: ' + name)
    if not isinstance(recipes, dict) or any(not isinstance(r, dict) for r in recipes.values()):
        raise ValueError('Recipes must map room names to objects')
    result = copy.deepcopy(recipes)
    if name != 'existing':
        for room in rooms:
            result[room] = {**copy.deepcopy(PROFILES[name]), **result.get(room, {}),
                            '_detail_profile': name}
    return result
