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
      DF_SEARCH_WINDOW_PX                   - px radius (in resized-image
                                               coords) around the last
                                               known-good column to search
                                               first, before falling back
                                               to a full-frame search
      DF_REACQUIRE_LOST_FRAMES              - consecutive empty-window
                                               frames required before a
                                               full-frame re-scan is
                                               allowed (see below)
      DF_LOCAL_THRESH    - True/False, flip off to fall back to a single
                            whole-ROI median (old behavior) for A/B tests
      DF_LOCAL_TILE_H / DF_LOCAL_TILE_W     - px size (resized-image
                                               coords) of each grid tile
                                               used for local-median
                                               thresholding (see below)

    Color thresholding (stage 1) uses DF_LOCAL_THRESH to pick between a
    single whole-ROI median baseline (b_med + DF_B_OFFSET applied to
    every pixel) and a grid of tiles (DF_LOCAL_TILE_H x DF_LOCAL_TILE_W),
    each computing its own local median so a shaded patch and a sunny
    patch in the same frame get their own baselines instead of one
    baseline dominated by whichever lighting condition covers more of
    the ROI. The coarse per-tile median grid is bilinearly upsampled
    back to full ROI resolution so the threshold varies smoothly instead
    of jumping at tile borders. DF_B_OFFSET is then applied on top of
    whichever baseline (global or local) is active. Tile size must stay
    well above the line's on-screen pixel width (widest near the bottom
    of the ROI) or the line's own pixels start dragging up their tile's
    median and shrinking their own detection gap.

    Column selection is gated by DF_SEARCH_WINDOW_PX: once locked onto
    the line, only columns within that radius of the last known-good
    position are considered, so a transient noise blob (glare, a second
    dash, a cone edge) elsewhere in the frame can't hijack the winning
    column for a single frame and cause a sudden hard steer. Because the
    line is dashed, the window legitimately comes up empty for a few
    frames between dashes - those frames just hold the last known-good
    position (counted as lost, steering decays as usual) rather than
    triggering a full-frame search. The full frame is only searched again
    once the window has come up empty for DF_REACQUIRE_LOST_FRAMES
    consecutive frames (i.e. genuinely lost, not just a dash gap).

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
        self.throttle_run = float(g('DF_THROTTLE_RUN', 0.2))
        self.throttle_slow = float(g('DF_THROTTLE_SLOW', 0.12))
        tp = g('DF_TARGET_PIXEL', None)
        self.target = None if tp is None else float(tp)  # None -> auto-calibrate on first good frame
        self.search_window = int(g('DF_SEARCH_WINDOW_PX', 60))  # px radius around last-known-good col to search first
        self.reacquire_lost_frames = int(g('DF_REACQUIRE_LOST_FRAMES', 10))  # consecutive empty-window frames before allowing a full-frame re-scan
        self.last_good_col = None  # last confirmed line position; freezes while lost, resets on full re-acquire

        # --- color baseline stage: whole-ROI median vs per-tile local median ---
        self.local_thresh_on = bool(g('DF_LOCAL_THRESH', True))  # False = single whole-ROI median (old behavior)
        self.tile_h = int(g('DF_LOCAL_TILE_H', 33))  # px height per grid tile; keep small vs shadow-patch scale
        self.tile_w = int(g('DF_LOCAL_TILE_W', 64))  # px width per grid tile; keep well above line's on-screen width

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
        self._dbg = (0, 0.0, 0.0, 0, 0, 0)

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

    def _local_median_map(self, b):
        """Per-tile local median of the b-channel, upsampled back to full
        ROI resolution. Each tile is thresholded against its own
        background instead of one whole-ROI median, so a shaded patch
        and a sunny patch in the same frame each get a baseline that
        reflects their own lighting rather than one baseline dominated
        by whichever condition covers more of the frame."""
        h, w = b.shape
        th = max(1, min(self.tile_h, h))
        tw = max(1, min(self.tile_w, w))
        n_rows = int(np.ceil(h / th))
        n_cols = int(np.ceil(w / tw))
        med_grid = np.empty((n_rows, n_cols), dtype=np.float32)
        for r in range(n_rows):
            y0, y1 = r * th, min(h, (r + 1) * th)
            for c in range(n_cols):
                x0, x1 = c * tw, min(w, (c + 1) * tw)
                med_grid[r, c] = np.median(b[y0:y1, x0:x1])
        return cv2.resize(med_grid, (w, h), interpolation=cv2.INTER_LINEAR)

    def compute_mask(self, cam_img):
        """
        Run stage 1 (color) + stage 2 (shape) of the pipeline in isolation
        and return the resized/cropped ROI plus the resulting binary mask,
        without touching any PID/steering/lost-frame state. Shared by run()
        and by offline tooling (e.g. snapshot capture scripts) so both stay
        on identical mask logic.
        """
        img = cv2.resize(cam_img, (self.rw, self.rh), interpolation=cv2.INTER_AREA)
        roi = img[int(self.rh * self.crop_top):, :, :]       # OakD delivers BGR already; no conversion needed

        # stage 1 - color: lighting-robust yellow via LAB b-channel (yellow high vs neutral pavement ~median)
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        b = lab[:, :, 2]
        b_med = int(np.median(b))  # kept for debug logging regardless of which baseline mode is active
        if self.local_thresh_on:
            baseline = self._local_median_map(b)
        else:
            baseline = b_med
        mask = (b >= baseline + self.b_offset).astype(np.uint8)

        # stage 2 - shape: drop color-candidate blobs that aren't line-shaped
        n_blobs_total, n_blobs_kept = 0, 0
        if self.shape_filter_on:
            mask, n_blobs_total, n_blobs_kept = self._apply_shape_filter(mask)

        return roi, mask, n_blobs_total, n_blobs_kept, b_med

    def run(self, cam_img):
        if cam_img is None:
            return 0.0, 0.0
        t0 = time.time()

        roi, mask, n_blobs_total, n_blobs_kept, b_med = self.compute_mask(cam_img)

        # per-column yellow, summed over the whole ROI height so dashes across gaps accumulate
        hist = mask.sum(axis=0).astype(np.float32)
        if hist.size >= 5:
            hist = np.convolve(hist, np.ones(5, np.float32) / 5.0, mode='same')

        # search near the last known-good column first, so a transient blob
        # elsewhere in the frame can't hijack the winner for a single frame;
        # only fall back to a full-frame search once the window has come up
        # empty for DF_REACQUIRE_LOST_FRAMES straight frames (genuinely lost,
        # not just the gap between dashes)
        if self.last_good_col is not None:
            lo = max(0, int(self.last_good_col) - self.search_window)
            hi = min(hist.size, int(self.last_good_col) + self.search_window + 1)
            windowed = hist[lo:hi]
            win_col = int(np.argmax(windowed))
            win_px = float(windowed[win_col])
            if win_px >= self.min_col_px:
                col = lo + win_col
                col_px = win_px
            elif self.lost_frames >= self.reacquire_lost_frames:
                col = int(np.argmax(hist))
                col_px = float(hist[col])
            else:
                # dash-gap frame: hold last known-good position instead of
                # re-scanning the whole frame
                col = int(self.last_good_col)
                col_px = 0.0
        else:
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
            self.last_good_col = col
            throttle = self.throttle_run
            self._dbg = (col, col_px, error, n_blobs_total, n_blobs_kept, b_med)
        else:
            self.lost_frames += 1
            if self.lost_frames > 15:
                self.steering *= self.decay_rate
            throttle = self.throttle_slow if self.lost_frames < self.lost_stop_frames else 0.0
            self._dbg = (col, col_px, 0.0, n_blobs_total, n_blobs_kept, b_med)

        self._frame_count += 1
        if self._frame_count % 10 == 0:
            c, cpx, err, nb_tot, nb_kept, bmed = self._dbg
            tgt = -1 if self.target is None else int(self.target)
            logger.info(f"DF: col={c} px={cpx:.0f} target={tgt} err={err:+.0f} "
                        f"st={self.steering:+.3f} lost={self.lost_frames} "
                        f"blobs={nb_kept}/{nb_tot} bmed={bmed} "
                        f"dt={(time.time()-t0)*1000:.1f}ms")

        return float(np.clip(self.steering, -1.0, 1.0)), throttle
