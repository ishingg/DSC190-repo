import cv2
import numpy as np
import time
import logging
from simple_pid import PID

logger = logging.getLogger(__name__)


class ConeReverseLineFollower:
    """
    Line follower (LAB b-channel color mask + connected-component geometry
    filter + column histogram + PID steering) with two extra behaviors:

      (1) Reverse-on-lost -- DF_REVERSE_ON_LOST
          When the line is lost long enough to stop, drive straight back
          (steering 0, DF_REVERSE_THROTTLE) until it's re-seen for
          DF_RECOVER_FRAMES frames, then resume forward.

      (2) Cone-stop with hysteresis -- DF_CONE_STOP
          Count orange pixels in the lower frame. TRIP into a latched stop
          when the count crosses DF_CONE_TRIP_PX. RELEASE only after the
          count stays below DF_CONE_CLEAR_PX for DF_CONE_CLEAR_FRAMES
          consecutive frames -- i.e. the cone has been physically removed,
          not just momentarily read low. The two thresholds + the frame
          count make the stop "sticky": a single noisy frame dipping under
          the trip level can't restart the car. Release does NOT depend on
          the line, so a cone lying flat on the ground (a wide low orange
          band, not a tall shape) is handled the same way -- it's a pixel
          count, no shape assumption.

    Both behaviors default OFF, so this file is a drop-in that behaves like
    the plain follower until enabled in myconfig.

    REQUIREMENT: both need the follower to own throttle, which only happens
    in 'local' (full-auto) mode. In 'local_angle' throttle comes from the
    controller, so the car cannot stop or reverse itself. Verify a negative
    pilot throttle actually backs the car up on your VESC before enabling
    reverse-on-lost.

    Cone thresholds are tuned from the conepx= value in the debug log:
    roll the cone toward the car, note conepx where you want it to stop
    (-> DF_CONE_TRIP_PX just below that), and note conepx with the cone
    removed (-> DF_CONE_CLEAR_PX comfortably above that, well under trip).

    All other knobs (DF_*) are unchanged from the base follower; see below.
    """

    def __init__(self, cfg):
        g = lambda name, default: getattr(cfg, name, default)
        self.rw = int(g('DF_RESIZE_W', 320))
        self.rh = int(g('DF_RESIZE_H', 240))
        self.crop_top = float(g('DF_CROP_TOP', 0.45))
        self.b_offset = int(g('DF_B_OFFSET', 12))
        self.min_col_px = int(g('DF_MIN_COL_PX', 5))
        self.smooth = float(g('DF_STEERING_SMOOTH', 0.5))
        self.decay_rate = float(g('DF_DECAY_RATE', 0.85))
        self.lost_stop_frames = int(g('DF_LOST_STOP_FRAMES', 30))
        self.throttle_run = float(g('DF_THROTTLE_RUN', 0.2))
        self.throttle_slow = float(g('DF_THROTTLE_SLOW', 0.12))
        tp = g('DF_TARGET_PIXEL', None)
        self.target = None if tp is None else float(tp)
        self.search_window = int(g('DF_SEARCH_WINDOW_PX', 60))
        self.reacquire_lost_frames = int(g('DF_REACQUIRE_LOST_FRAMES', 10))
        self.last_good_col = None

        # --- color baseline: whole-ROI median vs per-tile local median ---
        self.local_thresh_on = bool(g('DF_LOCAL_THRESH', True))
        self.tile_h = int(g('DF_LOCAL_TILE_H', 33))
        self.tile_w = int(g('DF_LOCAL_TILE_W', 64))

        # --- steering PID ---
        self.pid = PID(
            Kp=float(g('DF_PID_P', -0.006)),
            Ki=float(g('DF_PID_I', 0.0)),
            Kd=float(g('DF_PID_D', 0.0)),
            setpoint=0.0 if self.target is None else self.target,
            sample_time=g('DF_PID_SAMPLE_TIME', None),
            output_limits=(-1.0, 1.0),
        )

        # --- shape filter ---
        self.shape_filter_on = bool(g('DF_SHAPE_FILTER', True))
        self.min_blob_area = int(g('DF_MIN_BLOB_AREA', 15))
        self.max_blob_area = int(g('DF_MAX_BLOB_AREA', 6000))
        self.min_aspect = float(g('DF_MIN_BLOB_ASPECT', 0.0))
        self.max_aspect = float(g('DF_MAX_BLOB_ASPECT', 12.0))

        # --- behavior (1): reverse straight when the line is lost ---
        self.reverse_on_lost = bool(g('DF_REVERSE_ON_LOST', False))
        self.reverse_throttle = float(g('DF_REVERSE_THROTTLE', -0.15))
        self.recover_needed = int(g('DF_RECOVER_FRAMES', 3))

        # --- behavior (2): cone-stop with hysteresis ---
        self.cone_stop_on = bool(g('DF_CONE_STOP', False))
        self.cone_crop_top = float(g('DF_CONE_CROP_TOP', 0.35))
        # TRIP defaults to the old DF_CONE_MIN_PX if you already set that, else 800.
        self.cone_trip_px = int(g('DF_CONE_TRIP_PX', g('DF_CONE_MIN_PX', 800)))
        # RELEASE threshold: well below trip. Cone gone reads ~0; a partial
        # glimpse can still read a few hundred, so keep clear < trip with margin.
        self.cone_clear_px = int(g('DF_CONE_CLEAR_PX', 300))
        # consecutive below-clear frames required before releasing the stop
        self.cone_clear_needed = int(g('DF_CONE_CLEAR_FRAMES', 8))
        self.cone_lo = np.array(g('DF_CONE_HSV_LO', (5, 120, 90)), dtype=np.uint8)
        self.cone_hi = np.array(g('DF_CONE_HSV_HI', (20, 255, 255)), dtype=np.uint8)

        # --- state machine ---
        self.state = 'follow'        # 'follow' | 'reverse' | 'blocked'
        self.recover_frames = 0      # consecutive line-found frames
        self.cone_clear_frames = 0   # consecutive cone-clear frames (for release)

        self.steering = 0.0
        self.lost_frames = 0
        self._frame_count = 0

    def _apply_shape_filter(self, mask):
        """Keep only connected components whose size/aspect look like a line dash.
        Returns (filtered_mask, n_total_blobs, n_kept_blobs)."""
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n_labels <= 1:
            return mask, 0, 0

        keep = np.zeros(n_labels, dtype=bool)
        n_total = n_labels - 1
        n_kept = 0
        for i in range(1, n_labels):
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
        """Per-tile local median of the b-channel, upsampled to full ROI res."""
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
        """Stage 1 (color) + stage 2 (shape) in isolation. Returns
        (roi, mask, n_total, n_kept, b_med). Shared with offline tooling."""
        img = cv2.resize(cam_img, (self.rw, self.rh), interpolation=cv2.INTER_AREA)
        roi = img[int(self.rh * self.crop_top):, :, :]       # OakD delivers BGR already

        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        b = lab[:, :, 2]
        b_med = int(np.median(b))
        baseline = self._local_median_map(b) if self.local_thresh_on else b_med
        mask = (b >= baseline + self.b_offset).astype(np.uint8)

        n_blobs_total, n_blobs_kept = 0, 0
        if self.shape_filter_on:
            mask, n_blobs_total, n_blobs_kept = self._apply_shape_filter(mask)

        return roi, mask, n_blobs_total, n_blobs_kept, b_med

    def _detect_cone(self, cam_img):
        """Return the orange-pixel count in the lower frame (raw, no threshold).
        HSV rather than the line's LAB b-channel because orange and yellow both
        read high-b; HSV hue separates them. Its own crop (DF_CONE_CROP_TOP)
        catches a cone low/near, including one lying flat on the ground."""
        img = cv2.resize(cam_img, (self.rw, self.rh), interpolation=cv2.INTER_AREA)
        region = img[int(self.rh * self.cone_crop_top):, :, :]
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        m = cv2.inRange(hsv, self.cone_lo, self.cone_hi)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        return int(cv2.countNonZero(m))

    def _winning_column(self, mask):
        """Column histogram + windowed winner selection. Returns (col, col_px)."""
        hist = mask.sum(axis=0).astype(np.float32)
        if hist.size >= 5:
            hist = np.convolve(hist, np.ones(5, np.float32) / 5.0, mode='same')

        if self.last_good_col is not None:
            lo = max(0, int(self.last_good_col) - self.search_window)
            hi = min(hist.size, int(self.last_good_col) + self.search_window + 1)
            windowed = hist[lo:hi]
            win_col = int(np.argmax(windowed))
            win_px = float(windowed[win_col])
            if win_px >= self.min_col_px:
                return lo + win_col, win_px
            elif self.lost_frames >= self.reacquire_lost_frames:
                col = int(np.argmax(hist))
                return col, float(hist[col])
            else:
                return int(self.last_good_col), 0.0
        else:
            col = int(np.argmax(hist))
            return col, float(hist[col])

    def run(self, cam_img):
        if cam_img is None:
            return 0.0, 0.0
        t0 = time.time()

        roi, mask, n_blobs_total, n_blobs_kept, b_med = self.compute_mask(cam_img)
        col, col_px = self._winning_column(mask)

        if self.target is None and col_px >= self.min_col_px:
            self.target = float(col)
            self.pid.setpoint = self.target
            logger.info(f"DF: auto-set target column = {col}")

        line_found = (col_px >= self.min_col_px and self.target is not None)
        self.recover_frames = self.recover_frames + 1 if line_found else 0
        recovered = self.recover_frames >= self.recover_needed

        cone_px = self._detect_cone(cam_img) if self.cone_stop_on else 0

        steering, throttle = self._drive_state(cone_px, line_found, recovered, col)

        self._frame_count += 1
        if self._frame_count % 10 == 0:
            tgt = -1 if self.target is None else int(self.target)
            logger.info(f"DF[{self.state}]: col={col} px={col_px:.0f} target={tgt} "
                        f"st={self.steering:+.3f} thr={throttle:+.2f} lost={self.lost_frames} "
                        f"blobs={n_blobs_kept}/{n_blobs_total} conepx={cone_px} "
                        f"clear={self.cone_clear_frames} dt={(time.time()-t0)*1000:.1f}ms")

        return float(np.clip(steering, -1.0, 1.0)), throttle

    def _drive_state(self, cone_px, line_found, recovered, col):
        """State machine. 'follow' is the plain follower; 'blocked' and
        'reverse' are the add-ons. Cone hysteresis is evaluated first so a
        cone always overrides following/reversing."""

        # --- CONE-STOP with hysteresis (independent of the line) ---
        if self.cone_stop_on:
            if self.state == 'blocked':
                # release only after the cone has been clearly gone for a while
                if cone_px < self.cone_clear_px:
                    self.cone_clear_frames += 1
                else:
                    self.cone_clear_frames = 0   # any resurgence resets the release timer
                if self.cone_clear_frames >= self.cone_clear_needed:
                    logger.info("DF: cone cleared -> resuming")
                    self.state = 'follow'
                    self.cone_clear_frames = 0
                    self.last_good_col = None     # re-acquire the line fresh after the stop
                    self.lost_frames = 0
                    # fall through into follow this frame
                else:
                    self.steering = 0.0
                    return 0.0, 0.0
            elif cone_px >= self.cone_trip_px:
                if self.state != 'blocked':
                    logger.info(f"DF: cone detected (conepx={cone_px}) -> stopping")
                self.state = 'blocked'
                self.cone_clear_frames = 0
                self.steering = 0.0
                return 0.0, 0.0

        # --- REVERSE: back straight until the line is re-acquired ---
        if self.state == 'reverse':
            if recovered:
                logger.info("DF: line re-acquired -> resuming forward")
                self.state = 'follow'
                self.lost_frames = 0
            else:
                self.steering = 0.0
                return 0.0, self.reverse_throttle

        # --- FOLLOW: plain line-following ---
        if line_found:
            target_steer = float(self.pid(col))
            self.steering = self.smooth * self.steering + (1.0 - self.smooth) * target_steer
            self.lost_frames = 0
            self.last_good_col = col
            return self.steering, self.throttle_run
        else:
            self.lost_frames += 1
            if self.lost_frames > 15:
                self.steering *= self.decay_rate
            if self.reverse_on_lost and self.lost_frames >= self.lost_stop_frames:
                if self.state != 'reverse':
                    logger.info("DF: line lost -> reversing to re-acquire")
                self.state = 'reverse'
                self.steering = 0.0
                return 0.0, self.reverse_throttle
            throttle = self.throttle_slow if self.lost_frames < self.lost_stop_frames else 0.0
            return self.steering, throttle
