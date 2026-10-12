"""Additional fixed-floor depth veto; never replace raw geometric evidence."""

import json

import numpy as np
from room3d_preview import render
from room3d_refine import calibrate_floor_depth, depth_error


class FloorDepthGuard:
    """Lazily calibrate each view against the unchanged baseline scene.

    Call reserved views only after training selection. Candidate geometry never
    supplies calibration samples, and calibration is not ground-truth geometry.
    """

    def __init__(self, room, segments, depths, baseline_path, baseline_scenes):
        self.views = {
            camera["index"]: (camera, entries, observation)
            for camera, entries, observation in zip(room["cameras"], segments, depths, strict=True)
        }
        self.path = baseline_path
        self.scenes = baseline_scenes
        self.references = {}
        self.calibration = {}

    def compare(self, index, candidate_scene):
        if index not in self.references:
            camera, entries, observation = self.views[index]
            masks = [mask for item, mask in entries
                     if item["label"].strip().casefold() in ("floor", "wooden floor")
                     and item["score"] >= .3 and item["predicted_iou"] >= .9
                     and mask.sum() >= 500]
            mask = np.logical_or.reduce(masks) if masks else np.zeros(observation[0].shape, bool)
            floor = render(self.path, camera, return_depth=True, only_groups={"floor"})[2]
            corrected, evidence = calibrate_floor_depth(observation, floor, self.scenes[index], mask)
            # Own the reference arrays; neither caller mutation nor a later
            # candidate may change the evidence used for a previous comparison.
            target, valid = (array.copy() for array in corrected)
            self.references[index] = (target, valid, depth_error(self.scenes[index], target, valid))
            self.calibration[index] = evidence
        target, valid, before = self.references[index]
        after = depth_error(candidate_scene, target, valid)
        delta = after - before
        return {"camera": index, "depth_before": before, "depth_after": after,
                "depth_delta": delta, "safe": bool(delta <= .005),
                "calibration_accepted": self.calibration[index]["accepted"]}


def check_selected_scene(room, previous, candidate, segments, depths, out,
                         generate_mesh, heldout_cameras=()):
    """Veto one selected assembly; never search using reserved-view evidence."""
    heldout = set(heldout_cameras)
    known = {camera["index"] for camera in room["cameras"]}
    if not heldout <= known:
        raise ValueError("Expected known reserved cameras")
    out.mkdir(parents=True, exist_ok=True)
    report = {"accepted": False, "training": [], "heldout": [], "calibration": {},
              "qualification": "Fixed baseline floor-reference veto of selected geometry. No candidate reselection, ground-truth or navigation claim."}

    def finish(reason):
        report["reason"] = reason
        (out / "results.json").write_text(json.dumps(report, indent=2))
        return report

    if heldout == known:
        return finish("no_training_camera")
    before_path, after_path = out / "baseline.obj", out / "selected.obj"
    generate_mesh(room, before_path, previous)
    generate_mesh(room, after_path, candidate)
    scenes = {}
    guard = FloorDepthGuard(room, segments, depths, before_path, scenes)
    report["calibration"] = guard.calibration
    for reserved, key in ((False, "training"), (True, "heldout")):
        for camera in room["cameras"]:
            index = camera["index"]
            if (index in heldout) != reserved:
                continue
            scenes[index] = render(before_path, camera, return_depth=True)[2]
            selected = render(after_path, camera, return_depth=True)[2]
            report[key].append(guard.compare(index, selected))
        if not all(check["safe"] for check in report[key]):
            return finish("reserved_view_regressed" if reserved else "training_view_regressed")
    report["accepted"] = True
    return finish("selected_scene_depth_safe")
