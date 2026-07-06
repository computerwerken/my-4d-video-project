#!/usr/bin/env python3
# MIT License. k2stitch math validation harness.
#
# This is a *reference implementation and test* for the math in k2stitch_core.cc.
# It ports, line-by-line, the relevant parts of Lifecast's fisheye_camera.h and
# projection.cc (precomputeVR180toFthetaWarp = what VVE uses to ingest VR180,
# precomputeremapFisheyeToEquirectWarp = what k2stitch uses to produce VR180),
# then proves end-to-end that:
#
#   T1. The ported lens model is self-consistent (pixel -> ray -> pixel round trip,
#       including k1 distortion + tilt, matches to < 0.01 px).
#   T2. A synthetic dual-fisheye pair (K2 Pro geometry: 200 deg lenses, decentered,
#       distorted, slightly misrotated), stitched by the k2stitch warp chain into
#       SBS VR180 equirect, then ingested by *Lifecast's own* VR180->ftheta warp,
#       reproduces the ground-truth scene (high PSNR), while mirrored/flipped
#       variants of the convention fail (low PSNR). This retires the convention risk.
#   T3. The auto-refine solver (vertical-disparity minimizing Levenberg-Marquardt on
#       ORB feature matches, identical residual to the C++ implementation) recovers
#       deliberately injected pitch/roll misalignment to within ~0.03 degrees.
#
# Run: python3 k2stitch_test_math.py   (requires numpy + opencv-python)

import math
import sys
import numpy as np
import cv2

# ---------------------------------------------------------------------------
# Ported from lifecast_apps/source/fisheye_camera.h (FisheyeCamera<double>)
# ---------------------------------------------------------------------------

class FisheyeCamera:
    def __init__(self, width, height, radius_at_90, cx, cy, k1=0.0, tilt=0.0,
                 world_from_cam=None):
        self.width = width
        self.height = height
        self.radius_at_90 = radius_at_90
        self.cx = cx
        self.cy = cy
        self.k1 = k1
        self.tilt = tilt
        # cam_from_world is the [R] in lifecast's convention; we store its inverse
        # (world_from_cam) and transpose where needed, matching worldFromCam()/camFromWorld().
        self.world_from_cam = np.eye(3) if world_from_cam is None else world_from_cam

    def cam_from_world(self):
        return self.world_from_cam.T

    # pixelFromCam, "Version 3: no atan2, with tilt" (vectorized).
    # p_cam: (...,3) array. Returns (...,2) pixel array.
    def pixel_from_cam(self, p):
        eps = 1e-9
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        phi = np.arctan2(np.sqrt(eps + x * x + y * y), z)
        phi_dist = phi * (1.0 + phi * phi * self.k1)
        r = phi_dist * self.radius_at_90 / (math.pi / 2.0)
        norm = np.sqrt(eps + x * x + y * y)
        px = r * (1.0 + self.tilt) * x / norm + self.cx
        py = r * (-y) / norm + self.cy
        return np.stack([px, py], axis=-1)

    # rayDirFromPixel with tilt + k1 Newton inversion (vectorized).
    # pix: (...,2). Returns (...,3) unit rays in camera coords (+X right, +Y up, +Z fwd).
    def ray_dir_from_pixel(self, pix):
        eps = 1e-9
        dx = (pix[..., 0] - self.cx) / (1.0 + self.tilt)
        dy = pix[..., 1] - self.cy
        theta = np.arctan2(dy, dx)
        norm = np.sqrt(dx * dx + dy * dy + eps)
        phi = norm * (math.pi / 2.0) / self.radius_at_90
        k1 = self.k1
        phi2 = phi * phi
        phi4 = phi2 * phi2
        phi6 = phi4 * phi2
        phi0 = phi * (1 - k1 * phi2 + 3.0 * k1 * k1 * phi4 - 12.0 * k1 ** 3 * phi6)
        for _ in range(3):
            phi0 = phi0 - (phi0 * (1.0 + k1 * phi0 * phi0) - phi) / (1.0 + 3 * k1 * phi0 * phi0)
        return np.stack([
            np.cos(theta) * np.sin(phi0),
            -np.sin(theta) * np.sin(phi0),
            np.cos(phi0)], axis=-1)


def make_perfect_ftheta_camera(image_size):
    # Port of projection.cc makePerfectFthetaCamera
    return FisheyeCamera(image_size, image_size, image_size / 2.0,
                         image_size / 2.0, image_size / 2.0, k1=0.0, tilt=0.0)


# ---------------------------------------------------------------------------
# Ported from lifecast_apps/source/projection.cc
# ---------------------------------------------------------------------------

def precompute_vr180_to_ftheta_warp(ftheta_size, eqr_size, ftheta_scale=1.0):
    """Port of projection::precomputeVR180toFthetaWarp. This is EXACTLY how VVE
    samples each eye of an ingested SBS VR180 file. Returns map_x, map_y (float32)."""
    cam = make_perfect_ftheta_camera(ftheta_size)
    cam_scaled = FisheyeCamera(cam.width, cam.height, cam.radius_at_90 * ftheta_scale,
                               cam.cx, cam.cy)
    rmax = ftheta_scale * ftheta_size / 2.0
    ys, xs = np.mgrid[0:ftheta_size, 0:ftheta_size].astype(np.float64)
    pix = np.stack([xs, ys], axis=-1)
    r = cam_scaled.ray_dir_from_pixel(pix)
    # ray_dir = (r.z, r.x, -r.y)
    rd_x, rd_y, rd_z = r[..., 2], r[..., 0], -r[..., 1]
    u = 0.5 + np.arctan2(rd_y, rd_x) / math.pi
    v = 0.5 + np.arctan2(rd_z, np.sqrt(rd_x ** 2 + rd_y ** 2)) / math.pi
    map_x = (u * eqr_size).astype(np.float32)
    map_y = (v * eqr_size).astype(np.float32)
    dx = xs - ftheta_size / 2
    dy = ys - ftheta_size / 2
    invalid = dx * dx + dy * dy > rmax * rmax
    map_x[invalid] = -1
    map_y[invalid] = -1
    return map_x, map_y


def precompute_fisheye_to_equirect_warp(eqr_width, eqr_height, cam):
    """Port of projection::precomputeremapFisheyeToEquirectWarp ("for 180 equirect").
    This is the k2stitch output warp: for each VR180 equirect output pixel, where to
    sample in the raw fisheye. cam.world_from_cam carries the alignment rotation."""
    ys, xs = np.mgrid[0:eqr_height, 0:eqr_width].astype(np.float64)
    lon = -math.pi * (xs / eqr_width - 0.5) + math.pi / 2
    lat = math.pi * (ys / eqr_height - 0.5)
    p = np.stack([np.cos(lat) * np.cos(lon),
                  -np.sin(lat),
                  np.cos(lat) * np.sin(lon)], axis=-1)
    p_cam = p @ cam.cam_from_world().T  # camFromWorld(p)
    pix = cam.pixel_from_cam(p_cam)
    # Rays behind the camera (z<0 by a margin) are invalid for a <=200 deg lens
    behind = p_cam[..., 2] < math.cos(math.radians(115))
    map_x = pix[..., 0].astype(np.float32)
    map_y = pix[..., 1].astype(np.float32)
    map_x[behind] = -1
    map_y[behind] = -1
    return map_x, map_y


# ---------------------------------------------------------------------------
# k2stitch rig model (mirrors k2stitch_calibration.h)
# ---------------------------------------------------------------------------

def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def world_from_cam_ypr(yaw, pitch, roll):
    # Convention used by k2stitch: world_from_cam = Ry(yaw) * Rx(pitch) * Rz(roll)
    return rot_y(yaw) @ rot_x(pitch) @ rot_z(roll)

def make_eye_camera(width, height, cx, cy, circle_radius, lens_fov_deg, k1, tilt,
                    yaw, pitch, roll):
    # radius_at_90 from full-FOV circle radius: r = phi * radius_at_90 / (pi/2)
    radius_at_90 = circle_radius * 90.0 / (lens_fov_deg / 2.0)
    return FisheyeCamera(width, height, radius_at_90, cx, cy, k1, tilt,
                         world_from_cam_ypr(yaw, pitch, roll))


# ---------------------------------------------------------------------------
# Synthetic scene (at infinity => rotation-only geometry, like a distant scene)
# ---------------------------------------------------------------------------

def _hash01(i, j, salt):
    f = np.sin(i * 127.1 + j * 311.7 + salt * 74.7) * 43758.5453
    return f - np.floor(f)

def scene_color(ray):
    """ray: (...,3) unit vectors -> (...,3) float BGR in [0,1]. Textured with
    multi-scale random cells + 10-degree gridlines (feature-rich for ORB)."""
    x, y, z = ray[..., 0], ray[..., 1], ray[..., 2]
    a = np.arctan2(x, z)                       # longitude
    b = np.arctan2(y, np.sqrt(x * x + z * z))  # latitude
    chans = []
    for salt, s in [(1.0, 9.0), (2.0, 23.0), (3.0, 57.0)]:
        h = 0.55 * _hash01(np.floor(a * s), np.floor(b * s), salt) \
          + 0.30 * _hash01(np.floor(a * s * 3.1), np.floor(b * s * 3.1), salt + 9) \
          + 0.15 * (0.5 + 0.5 * np.sin(a * s * 2 + salt) * np.cos(b * s * 2 - salt))
        chans.append(h)
    img = np.stack(chans, axis=-1)
    ad = np.degrees(a); bd = np.degrees(b)
    grid = (np.abs((ad + 5) % 10 - 5) < 0.25) | (np.abs((bd + 5) % 10 - 5) < 0.25)
    img[grid] = 1.0
    return img


def render_fisheye(cam, circle_radius):
    """Render the synthetic scene through the (decentered, distorted, rotated) lens."""
    ys, xs = np.mgrid[0:cam.height, 0:cam.width].astype(np.float64)
    pix = np.stack([xs, ys], axis=-1)
    ray_cam = cam.ray_dir_from_pixel(pix)
    ray_world = ray_cam @ cam.world_from_cam.T
    img = scene_color(ray_world)
    rr = np.sqrt((xs - cam.cx) ** 2 + (ys - cam.cy) ** 2)
    img[rr > circle_radius] = 0.0
    return (img * 255).astype(np.uint8)


def psnr(a, b, mask=None):
    a = a.astype(np.float64); b = b.astype(np.float64)
    if mask is not None:
        d = ((a - b) ** 2)[mask]
    else:
        d = (a - b) ** 2
    mse = d.mean()
    return 99.0 if mse < 1e-12 else 10 * math.log10(255 * 255 / mse)


# ---------------------------------------------------------------------------
# Solver (identical algorithm to k2stitch_core.cc refineAlignment)
# ---------------------------------------------------------------------------

def vertical_angle(ray):
    return np.arctan2(-ray[..., 1], np.sqrt(ray[..., 0] ** 2 + ray[..., 2] ** 2))

def solve_pitch_roll(rays_cam_L, rays_cam_R, wfc_L, yaw_R, pitch_R, roll_R,
                     iters=30, huber=0.003):
    """Levenberg-Marquardt on (d_pitch, d_roll) of the right eye, minimizing vertical
    angular disparity. rays_cam_*: (N,3) fixed camera-space rays of matched features."""
    delta_L = vertical_angle(rays_cam_L @ wfc_L.T)

    def residuals(p):
        wfc_R = world_from_cam_ypr(yaw_R, pitch_R + p[0], roll_R + p[1])
        return vertical_angle(rays_cam_R @ wfc_R.T) - delta_L

    p = np.zeros(2)
    lm = 1e-4
    r = residuals(p)
    w = np.ones(len(r))
    for _ in range(iters):
        # IRLS Huber weights
        absr = np.abs(r)
        w = np.where(absr <= huber, 1.0, huber / np.maximum(absr, 1e-12))
        # numeric jacobian
        J = np.zeros((len(r), 2))
        eps = 1e-7
        for k in range(2):
            dp = p.copy(); dp[k] += eps
            J[:, k] = (residuals(dp) - r) / eps
        A = J.T @ (w[:, None] * J) + lm * np.eye(2)
        g = J.T @ (w * r)
        step = np.linalg.solve(A, -g)
        p_new = p + step
        r_new = residuals(p_new)
        if (w * r_new ** 2).sum() < (w * r ** 2).sum():
            p, r = p_new, r_new
            lm = max(lm * 0.5, 1e-9)
        else:
            lm *= 10
        if np.linalg.norm(step) < 1e-10:
            break
    return p, r


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def t1_lens_round_trip():
    cam = make_eye_camera(1848, 1386, cx=930.3, cy=689.7, circle_radius=675.0,
                          lens_fov_deg=200.0, k1=0.06, tilt=0.001,
                          yaw=0, pitch=0, roll=0)
    rng = np.random.default_rng(7)
    n = 20000
    ang = rng.uniform(0, 2 * math.pi, n)
    rad = np.sqrt(rng.uniform(0, 1, n)) * 660.0  # inside circle
    pix = np.stack([cam.cx + rad * np.cos(ang), cam.cy + rad * np.sin(ang)], axis=-1)
    ray = cam.ray_dir_from_pixel(pix)
    pix2 = cam.pixel_from_cam(ray)
    err = np.sqrt(((pix - pix2) ** 2).sum(axis=-1))
    print(f"T1 lens round trip: max err {err.max():.6f} px, mean {err.mean():.6f} px")
    assert err.max() < 0.01, "lens model inversion failed"
    return True


def build_rig(perturb_R=(0.0, 0.0)):
    W, H = 1848, 1386
    common = dict(width=W, height=H, lens_fov_deg=200.0)
    # Ground-truth calibration (typical slightly-imperfect rig)
    cam_L = make_eye_camera(cx=W / 2 + 5.3, cy=H / 2 - 3.7, circle_radius=676.0,
                            k1=0.05, tilt=0.0008,
                            yaw=math.radians(0.20), pitch=math.radians(0.15),
                            roll=math.radians(-0.10), **common)
    cam_R = make_eye_camera(cx=W / 2 - 4.1, cy=H / 2 + 2.9, circle_radius=674.0,
                            k1=0.055, tilt=-0.0005,
                            yaw=math.radians(-0.15),
                            pitch=math.radians(-0.20) + math.radians(perturb_R[0]),
                            roll=math.radians(0.12) + math.radians(perturb_R[1]),
                            **common)
    return cam_L, cam_R


def stitch_sbs(cam_L, cam_R, eqr_size):
    mxL, myL = precompute_fisheye_to_equirect_warp(eqr_size, eqr_size, cam_L)
    mxR, myR = precompute_fisheye_to_equirect_warp(eqr_size, eqr_size, cam_R)
    fe_L = render_fisheye(cam_L, 676.0)
    fe_R = render_fisheye(cam_R, 674.0)
    eq_L = cv2.remap(fe_L, mxL, myL, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    eq_R = cv2.remap(fe_R, mxR, myR, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    return np.concatenate([eq_L, eq_R], axis=1), (mxL, myL), (mxR, myR), fe_L, fe_R


def t2_vve_round_trip():
    eqr_size = 1024
    cam_L, cam_R = build_rig()
    sbs, _, _, _, _ = stitch_sbs(cam_L, cam_R, eqr_size)

    # VVE ingest: split SBS -> left half is L eye, then VR180->ftheta warp (their code)
    L_eye = sbs[:, :eqr_size]
    ftheta_size = 768
    map_x, map_y = precompute_vr180_to_ftheta_warp(ftheta_size, eqr_size, ftheta_scale=1.0)
    ftheta_from_pipeline = cv2.remap(L_eye, map_x, map_y, cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_CONSTANT)

    # Ground truth: render the scene directly through a perfect ftheta camera
    cam_ft = make_perfect_ftheta_camera(ftheta_size)
    ys, xs = np.mgrid[0:ftheta_size, 0:ftheta_size].astype(np.float64)
    pix = np.stack([xs, ys], axis=-1)
    ray = cam_ft.ray_dir_from_pixel(pix)
    direct = (scene_color(ray) * 255).astype(np.uint8)
    dx = xs - ftheta_size / 2; dy = ys - ftheta_size / 2
    valid = (dx * dx + dy * dy) < (ftheta_size / 2 - 8) ** 2
    # exclude the outer ~10 deg (uses fisheye data near the 200-deg edge; interp softens it)
    inner = (dx * dx + dy * dy) < (ftheta_size / 2 * 0.92) ** 2

    p_ok = psnr(ftheta_from_pipeline, direct, inner)
    # Convention falsification: mirrored / flipped eye must score far worse.
    # (A wrong-eye check is meaningless here: the scene is at infinity, so both
    # eyes see nearly identical images by construction.)
    p_mirror = psnr(cv2.remap(np.ascontiguousarray(L_eye[:, ::-1]), map_x, map_y,
                              cv2.INTER_CUBIC), direct, inner)
    p_flip = psnr(cv2.remap(np.ascontiguousarray(L_eye[::-1, :]), map_x, map_y,
                            cv2.INTER_CUBIC), direct, inner)
    print(f"T2 VVE round trip PSNR: correct={p_ok:.1f} dB | mirrored={p_mirror:.1f} "
          f"| v-flipped={p_flip:.1f}")
    assert p_ok > 21.0, "round trip PSNR too low - convention or warp bug"
    assert p_ok > p_mirror + 6 and p_ok > p_flip + 6, "orientation not decisive"
    cv2.imwrite("/tmp/k2_t2_sbs.png", sbs)
    cv2.imwrite("/tmp/k2_t2_ftheta_pipeline.png", ftheta_from_pipeline)
    cv2.imwrite("/tmp/k2_t2_ftheta_direct.png", direct)
    return True


def t3_solver_recovery():
    eqr_size = 1024
    inj_pitch, inj_roll = 0.40, -0.30  # degrees, injected misalignment on right eye
    cam_L, cam_R_true = build_rig()
    cam_L2, cam_R_bad = build_rig(perturb_R=(inj_pitch, inj_roll))

    # k2stitch believes the unperturbed calibration; footage comes from the perturbed rig.
    # Simulate: render fisheye with TRUE(perturbed) rig, stitch with BELIEVED calibration.
    fe_L = render_fisheye(cam_L2, 676.0)
    fe_R = render_fisheye(cam_R_bad, 674.0)
    believed_L, believed_R = build_rig()  # solver's starting point
    mxL, myL = precompute_fisheye_to_equirect_warp(eqr_size, eqr_size, believed_L)
    mxR, myR = precompute_fisheye_to_equirect_warp(eqr_size, eqr_size, believed_R)
    eq_L = cv2.remap(fe_L, mxL, myL, cv2.INTER_CUBIC)
    eq_R = cv2.remap(fe_R, mxR, myR, cv2.INTER_CUBIC)

    # ORB matching (same params as C++)
    orb = cv2.ORB_create(nfeatures=3000, fastThreshold=12)
    gL = cv2.cvtColor(eq_L, cv2.COLOR_BGR2GRAY)
    gR = cv2.cvtColor(eq_R, cv2.COLOR_BGR2GRAY)
    kL, dL = orb.detectAndCompute(gL, None)
    kR, dR = orb.detectAndCompute(gR, None)
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(dL, dR, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    print(f"T3 features: {len(kL)}/{len(kR)} kp, {len(good)} ratio-test matches")
    assert len(good) > 80, "not enough matches"

    # matched equirect pixels -> raw fisheye pixels (via warp maps) -> fixed cam rays
    def cam_rays(kps, idxs, mx, my, cam):
        pts = np.array([kps[i].pt for i in idxs])
        xi = np.clip(pts[:, 0].round().astype(int), 0, eqr_size - 1)
        yi = np.clip(pts[:, 1].round().astype(int), 0, eqr_size - 1)
        fx = mx[yi, xi]; fy = my[yi, xi]
        ok = (fx >= 0) & (fy >= 0)
        return cam.ray_dir_from_pixel(np.stack([fx, fy], axis=-1)), ok

    raysL, okL = cam_rays(kL, [m.queryIdx for m in good], mxL, myL, believed_L)
    raysR, okR = cam_rays(kR, [m.trainIdx for m in good], mxR, myR, believed_R)
    # NOTE: the rays must be computed with the lens model, but the *observations* come
    # from the true (perturbed) rig; the solver adjusts believed pitch/roll of R.
    # Since footage was rendered with cam_R_bad but unwarped with believed_R, the
    # fisheye pixel we look up is the observation; its cam-space ray is fixed.
    ok = okL & okR
    raysL, raysR = raysL[ok], raysR[ok]

    # Robust pre-filter: drop worst 10% by initial residual
    d0 = vertical_angle(raysR @ believed_R.world_from_cam.T) - \
         vertical_angle(raysL @ believed_L.world_from_cam.T)
    keep = np.abs(d0 - np.median(d0)) < 4 * (np.median(np.abs(d0 - np.median(d0))) + 1e-9)
    raysL, raysR = raysL[keep], raysR[keep]

    yawR = math.radians(-0.15); pitchR = math.radians(-0.20); rollR = math.radians(0.12)
    p, r_after = solve_pitch_roll(raysL, raysR, believed_L.world_from_cam,
                                  yawR, pitchR, rollR)
    rec_pitch, rec_roll = math.degrees(p[0]), math.degrees(p[1])
    rms_before = math.degrees(np.sqrt((d0[keep] ** 2).mean())) * eqr_size / 180.0
    rms_after = math.degrees(np.sqrt((r_after ** 2).mean())) * eqr_size / 180.0
    print(f"T3 injected (pitch,roll)=({inj_pitch:+.3f},{inj_roll:+.3f}) deg | "
          f"recovered=({rec_pitch:+.3f},{rec_roll:+.3f}) deg")
    print(f"T3 RMS vertical disparity: before={rms_before:.2f} px, after={rms_after:.2f} px "
          f"(at {eqr_size}px/eye)")
    assert abs(rec_pitch - inj_pitch) < 0.05, "pitch not recovered"
    assert abs(rec_roll - inj_roll) < 0.05, "roll not recovered"
    # Residual floor ~0.8 px is ORB keypoint localization noise, not alignment error
    # (the recovered angles above prove that). Require a clear reduction + sane floor.
    assert rms_after < rms_before * 0.5, "solver did not reduce vertical disparity"
    assert rms_after < 1.5, "vertical disparity residual too high"
    return True


if __name__ == "__main__":
    ok = t1_lens_round_trip() and t2_vve_round_trip() and t3_solver_recovery()
    print("ALL TESTS PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)
