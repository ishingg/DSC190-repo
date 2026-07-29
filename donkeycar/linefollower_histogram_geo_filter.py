import cv2
import numpy as np
import time
import logging
from simple_pid import PID

logger = logging.getLogger(__name__)


class HybridLineFollower:
    """
    Hybrid color+shape line follower: LAB b-channel mask picks color
    candidate pixels, a connected-component geometry filter (area +
    aspect ratio) drops non-line blobs (glare, shadow edges, speckle),
    then a column histogram + position-hold controller steers toward
    the surviving column. No line fitting, no extrapolation,
    self-calibrating target column. Adapted from DonkeyCar's upstream
    LineFollower.

    Calibration knobs live entirely on cfg as single DF_* values (see
    __init__) so tuning never requires touching this file:
      DF_SHAPE_FILTER    - True/False, flip off to fall back to the
                            pure color-histogram pipeline for A/B tests
      DF_MIN_BLOB_AREA / DF_MAX_BLOB_AREA   - px^2 size gate
      DF_MIN_BLOB_ASPECT / DF_MAX_BLOB_ASPECT - height/width gate
      DF_PID_P / DF_PID_I / DF_PID_D        - steering PID gains
      DF_PID_SAMPLE_TIME                    - None = update every frame

    Steering is driven by simple_pid.PID (same library upstream
    DonkeyCar's LineFollower uses). simple_pid computes
    error = setpoint - input, i.e. (target - col), the opposite sign of
    the (col - target) error used by the old plain-P controller here -
    that's why DF_PID_P defaults negative, matching the sign convention
    already used for PID_P in cfg_cv_control.py.
    """

    def __init__(self, cfg):
        g = lambda name, default: getattr(cfg, name, default)
        self.rw = int(g('DF_RESIZE_W', 320))
        self.rh = int(g('DF_RESIZE_H', 240))
        self.crop_top = float(g('DF_CROP_TOP', 0.45))
        self.b_offset = int(g('DF_B_OFFSET', 12))
        self.min_col_px = int(g('DF_MIN_COL_PX', 5))     # yellow px in winning column to trust it
        self.smooth = float(g('DF_STEERING_SMOOTH', 0.5))
        self.decay_rate = float(g('DF_DECAY_RATE', 0.85))
        self.lost_stop_frames = int(g('DF_LOST_STOP_FRAMES', 30))
        self.throttle_run = float(g('THROTTLE_INITIAL', 0.2))
        self.throttle_slow = float(g('THROTTLE_MIN', 0.12))
        tp = g('DF_TARGET_PIXEL', None)
        self.target = None if tp is None else float(tp)  # None -> auto-calibrate on first good frame

        # --- steering controller (PID, matches upstream LineFollower) ---
        self.pid = PID(
            Kp=float(g('DF_PID_P', -0.006)),
            Ki=float(g('DF_PID_I', 0.0)),
            Kd=float(g('DF_PID_D', 0.0)),
            setpoint=0.0 if self.target is None else self.target,
            sample_time=g('DF_PID_SAMPLE_TIME', None),  # None = recompute every call regardless of dt
            output_limits=(-1.0, 1.0),
        )

        # --- shape filter stage (color candidates -> geometry-plausible blobs) ---
        self.shape_filter_on = bool(g('DF_SHAPE_FILTER', True))  # False = pure color histogram (old DashFollower behavior)
        self.min_blob_area = int(g('DF_MIN_BLOB_AREA', 15))      # px^2; drops speckle/noise
        self.max_blob_area = int(g('DF_MAX_BLOB_AREA', 6000))    # px^2; drops glare/reflection patches
        self.min_aspect = float(g('DF_MIN_BLOB_ASPECT', 0.0))    # h/w; drops wide flat blobs (shadow edges)
        self.max_aspect = float(g('DF_MAX_BLOB_ASPECT', 12.0))   # h/w; drops thin noise spikes

        self.steering = 0.0
        self.lost_frames = 0
        self._frame_count = 0
        self._dbg = (0, 0.0, 0.0, 0, 0)

    def _apply_shape_filter(self, mask):
        """Keep only connected components whose size/aspect look like a line dash.
        Returns (filtered_mask, n_total_blobs, n_kept_blobs) for debug logging."""
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n_labels <= 1:
            return mask, 0, 0

        keep = np.zeros(n_labels, dtype=bool)
        n_total = n_labels - 1
        n_kept = 0
        for i in range(1, n_labels):  # skip label 0 (background)
            area = stats[i, cv2.CC_STAT_AREA]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            aspect = h / max(w, 1)
            if (self.min_blob_area <= area <= self.max_blob_area
                    and self.min_aspect <= aspect <= self.max_aspect):
                keep[i] = True
                n_kept += 1

        filtered = keep[labels].astype(np.uint8)
        return filtered, n_total, n_kept

    def run(self, cam_img):
        if cam_img is None:
            return 0.0, 0.0
        t0 = time.time()

        img = cv2.resize(cam_img, (self.rw, self.rh), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)       # OakD delivers RGB; keep BGR downstream (matches validated mask)
        roi = img[int(self.rh * self.crop_top):, :, :]

        # stage 1 - color: lighting-robust yellow via LAB b-channel (yellow high vs neutral pavement ~median)
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        b = lab[:, :, 2]
        b_thresh = int(np.median(b)) + self.b_offset
        mask = (b >= b_thresh).astype(np.uint8)

        # stage 2 - shape: drop color-candidate blobs that aren't line-shaped
        n_blobs_total, n_blobs_kept = 0, 0
        if self.shape_filter_on:
            mask, n_blobs_total, n_blobs_kept = self._apply_shape_filter(mask)

        # per-column yellow, summed over the whole ROI height so dashes across gaps accumulate
        hist = mask.sum(axis=0).astype(np.float32)
        if hist.size >= 5:
            hist = np.convolve(hist, np.ones(5, np.float32) / 5.0, mode='same')
        col = int(np.argmax(hist))
        col_px = float(hist[col])

        if self.target is None and col_px >= self.min_col_px:
            self.target = float(col)
            self.pid.setpoint = self.target
            logger.info(f"DF: auto-set target column = {col}")

        if col_px >= self.min_col_px and self.target is not None:
            error = col - self.target
            target_steer = float(self.pid(col))
            self.steering = self.smooth * self.steering + (1.0 - self.smooth) * target_steer
            self.lost_frames = 0
            throttle = self.throttle_run
            self._dbg = (col, col_px, error, n_blobs_total, n_blobs_kept)
        else:
            self.lost_frames += 1
            if self.lost_frames > 15:
                self.steering *= self.decay_rate
            throttle = self.throttle_slow if self.lost_frames < self.lost_stop_frames else 0.0
            self._dbg = (col, col_px, 0.0, n_blobs_total, n_blobs_kept)

        self._frame_count += 1
        if self._frame_count % 10 == 0:
            c, cpx, err, nb_tot, nb_kept = self._dbg
            tgt = -1 if self.target is None else int(self.target)
            logger.info(f"DF: col={c} px={cpx:.0f} target={tgt} err={err:+.0f} "
                        f"st={self.steering:+.3f} lost={self.lost_frames} "
                        f"blobs={nb_kept}/{nb_tot} "
                        f"dt={(time.time()-t0)*1000:.1f}ms")

        return float(np.clip(self.steering, -1.0, 1.0)), throttle
