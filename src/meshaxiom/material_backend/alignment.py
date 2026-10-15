"""Deterministic CPU camera registration; inference never invents a hidden pose."""
from __future__ import annotations

import math
import time

import numpy as np
from PIL import Image


def prepare_reference(path, size=1024):
    """Letterbox the supplied image once, sharing exactly that crop with fitting."""
    with Image.open(path) as source:
        rgba = source.convert("RGBA")
        has_alpha = np.any(np.asarray(rgba)[:, :, 3] != 255)
        scale = size / max(rgba.size)
        new_size = tuple(max(1, round(length * scale)) for length in rgba.size)
        rgba = rgba.resize(new_size, Image.Resampling.LANCZOS)
        canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        canvas.alpha_composite(rgba, ((size - new_size[0]) // 2, (size - new_size[1]) // 2))
        colors = np.asarray(canvas)
    black = Image.new("RGBA", canvas.size, (0, 0, 0, 255))
    reference = Image.alpha_composite(black, canvas).convert("RGB")
    if has_alpha:
        mask = colors[:, :, 3] > 128
        method = "supplied_alpha"
    else:
        # Infer only an approximately uniform boundary color; this is an
        # explicitly recorded image heuristic, not a hidden segmentation model.
        rgb = np.asarray(reference, dtype=np.float32)
        image_box = np.asarray(rgba.convert("RGB"), dtype=np.float32)
        edge = np.concatenate([image_box[0], image_box[-1], image_box[:, 0], image_box[:, -1]])
        background = np.median(edge, axis=0)
        if np.quantile(np.linalg.norm(edge - background, axis=1), .9) > 35:
            return reference, None, {"mask_source": "unavailable_complex_background", "letterbox_size": size, "crop": "none"}
        mask = np.linalg.norm(rgb - background, axis=2) > 30
        # Pixels outside the original rectangle are padding, not foreground.
        inside = np.zeros(mask.shape, dtype=bool)
        x = (size - new_size[0]) // 2; y = (size - new_size[1]) // 2
        inside[y:y + new_size[1], x:x + new_size[0]] = True
        mask &= inside
        method = "uniform_border_color_heuristic"
    if not .002 < mask.mean() < .97:
        mask = None
        method = "unavailable_empty_or_full_foreground"
    return reference, mask, {"mask_source": method, "letterbox_size": size, "crop": "none", "original_scaled_size": list(new_size)}


def orientation_matrix(up_axis="y"):
    if up_axis == "y":
        return np.eye(3)
    if up_axis == "z":
        # Z-up +X-front mesh -> camera X-right, Y-up, Z-forward.
        return np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
    raise ValueError("material camera up_axis must be 'y' or 'z'")


def align_mesh(vertices, faces, mask, supplied_camera=None, *, up_axis="y", min_iou=.25, fit_size=96):
    """Fit yaw/elevation/distance/offset by foreground silhouette when omitted.

    The temporary rigid transform and uniform normalization affect only neural
    conditioning. Final geometry is restored from exact source coordinates.
    """
    import cv2
    from scipy.optimize import minimize

    cv2.setNumThreads(1)
    started = time.monotonic()
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    supplied = supplied_camera is not None
    supplied_camera = supplied_camera or {}
    up_axis = str(supplied_camera.get("up_axis", up_axis)).lower().removesuffix("_up")
    rest = vertices @ orientation_matrix(up_axis).T
    fov = float(supplied_camera.get("camera_angle_x", math.radians(20)))
    if not math.radians(5) <= fov <= math.radians(120):
        raise ValueError("camera_angle_x must be radians within [5, 120] degrees")
    focal = .5 / math.tan(fov / 2)
    if supplied and "distance" not in supplied_camera:
        raise ValueError("An explicit camera requires distance as well as camera_angle_x")
    if supplied and "camera_angle_x" not in supplied_camera:
        raise ValueError("An explicit camera requires camera_angle_x in radians")
    if not 0 <= min_iou <= 1:
        raise ValueError("min_alignment_iou must be in [0, 1]")
    target = None if mask is None else cv2.resize(mask.astype(np.uint8), (fit_size, fit_size), interpolation=cv2.INTER_NEAREST)

    def transformed(parameters):
        azimuth, elevation, _, dx, dy = parameters
        azimuth, elevation = math.radians(azimuth), math.radians(elevation)
        yaw = np.array([[math.cos(azimuth), 0, -math.sin(azimuth)], [0, 1, 0], [math.sin(azimuth), 0, math.cos(azimuth)]])
        pitch = np.array([[1, 0, 0], [0, math.cos(elevation), -math.sin(elevation)], [0, math.sin(elevation), math.cos(elevation)]])
        rotation = pitch @ yaw
        rotated = rest @ rotation.T
        center = (rotated.min(axis=0) + rotated.max(axis=0)) / 2
        # Reserve room for registration offsets; never clip the voxel AABB.
        scale = .90 / np.ptp(rotated, axis=0).max()
        aligned = (rotated - center) * scale + [dx, dy, 0]
        return aligned, rotation, center, scale

    def silhouette(parameters):
        aligned, *_ = transformed(parameters)
        denominator = parameters[2] - aligned[:, 2]
        if denominator.min() <= 0:
            return np.zeros((fit_size, fit_size), dtype=np.uint8)
        points = np.column_stack([.5 + focal * aligned[:, 0] / denominator, .5 - focal * aligned[:, 1] / denominator]) * fit_size
        points = np.clip(np.rint(points), -10000, 10000).astype(np.int32)
        out = np.zeros((fit_size, fit_size), dtype=np.uint8)
        for triangle in points[faces]:
            cv2.fillConvexPoly(out, triangle, 1)
        return out

    def loss(parameters):
        rendered = silhouette(parameters).astype(bool)
        return 1 - np.count_nonzero(rendered & target.astype(bool)) / max(1, np.count_nonzero(rendered | target.astype(bool)))

    if supplied:
        parameters = [float(supplied_camera.get("azimuth", 0)), float(supplied_camera.get("elevation", 0)),
            float(supplied_camera["distance"]), float(supplied_camera.get("offset_x", 0)), float(supplied_camera.get("offset_y", 0))]
        if not all(math.isfinite(value) for value in parameters) or parameters[2] <= .51:
            raise ValueError("Camera parameters must be finite and distance must exceed normalized mesh depth")
        if abs(parameters[3]) > .049 or abs(parameters[4]) > .049:
            raise ValueError("Camera offsets must fit the voxel box (absolute offset < .05)")
        fit_kind = "supplied"
        iou = None if target is None else 1 - loss(parameters)
    else:
        if target is None:
            raise ValueError("Cannot derive a camera from this background; provide a foreground-alpha image or an explicit camera")
        fraction = max(.01, math.sqrt(float(target.mean())))
        distance = max(.9, min(9., .45 / (math.tan(fov / 2) * fraction)))
        starts = [[azimuth, elevation, distance, 0, 0] for azimuth in range(-180, 180, 45) for elevation in [-15, 15, 35]]
        best = min(starts, key=loss)
        fit = minimize(loss, best, method="Powell", bounds=[(-180, 180), (-45, 60), (.8, 12), (-.045, .045), (-.045, .045)],
            options={"maxiter": 12, "maxfev": 220, "xtol": .005, "ftol": .001})
        parameters = fit.x.tolist() if loss(fit.x) <= loss(best) else best
        iou = 1 - loss(parameters)
        if iou < min_iou:
            raise ValueError(f"Derived camera silhouette IoU {iou:.3f} is below {min_iou:.3f}; provide an explicit matched camera")
        fit_kind = "derived_silhouette_registration"
    aligned, rotation, center, scale = transformed(parameters)
    if np.abs(aligned).max() >= .5:
        raise ValueError("Aligned mesh exceeds the fixed shape encoder voxel box")
    report = {
        "source": fit_kind, "parameters": dict(zip(["azimuth", "elevation", "distance", "offset_x", "offset_y"], parameters)),
        "camera_angle_x": fov, "distance": parameters[2], "up_axis": up_axis, "mesh_scale": 1.,
        "orientation_matrix": orientation_matrix(up_axis).tolist(), "rotation_matrix": rotation.tolist(),
        "rotated_center": center.tolist(), "uniform_scale": float(scale), "silhouette_iou": iou,
        "silhouette_fit_resolution": fit_size, "min_alignment_iou": min_iou, "seconds": time.monotonic() - started,
        "reference_camera_semantically_verified": False,
        "note": "Single-view silhouette registration can be ambiguous. This rigid conditioning transform is undone using exact source vertices on export.",
    }
    return aligned.astype(np.float32), report
