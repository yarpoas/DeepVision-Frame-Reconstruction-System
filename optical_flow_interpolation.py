import cv2
import numpy as np
from scipy.interpolate import RegularGridInterpolator

def interpolate_frame_optical_flow(frameA, frameC):
    """
    Interpolates an intermediate frame between frameA and frameC using optical flow.
    Based on motion compensation and pixel warping.
    
    Parameters:
        frameA (np.ndarray): First input frame (RGB).
        frameC (np.ndarray): Second input frame (RGB).
    
    Returns:
        np.ndarray: Interpolated frame (RGB, uint8).
    """
    # Convert to grayscale
    grayA = cv2.cvtColor(frameA, cv2.COLOR_RGB2GRAY)
    grayC = cv2.cvtColor(frameC, cv2.COLOR_RGB2GRAY)

    # Calculate optical flow from A to C using Farneback method
    flow = cv2.calcOpticalFlowFarneback(
        grayA, grayC,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    h, w = grayA.shape
    X, Y = np.meshgrid(np.arange(w), np.arange(h))

    half_vx = flow[..., 0] / 2
    half_vy = flow[..., 1] / 2

    # Warp both frames
    warpedA = np.zeros_like(frameA, dtype=np.float32)
    warpedC = np.zeros_like(frameC, dtype=np.float32)

    for k in range(3):
        channelA = frameA[:, :, k].astype(np.float32)
        channelC = frameC[:, :, k].astype(np.float32)

        interp_A = RegularGridInterpolator(
            (np.arange(h), np.arange(w)),
            channelA,
            bounds_error=False,
            fill_value=0
        )
        interp_C = RegularGridInterpolator(
            (np.arange(h), np.arange(w)),
            channelC,
            bounds_error=False,
            fill_value=0
        )

        coordsA = np.stack([Y + half_vy, X + half_vx], axis=-1)
        coordsC = np.stack([Y - half_vy, X - half_vx], axis=-1)

        warpedA[:, :, k] = interp_A(coordsA)
        warpedC[:, :, k] = interp_C(coordsC)

    blended = 0.5 * warpedA + 0.5 * warpedC
    return np.clip(blended, 0, 255).astype(np.uint8)
