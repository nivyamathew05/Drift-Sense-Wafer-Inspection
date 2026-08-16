import numpy as np
import cv2
import time

class DriftSenseLocalizer:
    """
    Scale-Aware Sub-Pixel Localizer for Wafer Navigation Error Recovery.
    Multi-Scale Normalized Cross-Correlation (NCC) with anti-aliased template scaling,
    parabolic peak fitting, and spatial tie-breaking for periodic semiconductor structures.
    """
    def __init__(self, nominal_scale=10.0, scale_step=0.25, rotation_range=(-2.0, 2.0), rotation_step=1.0):
        self.nominal_scale = nominal_scale
        self.scale_step = scale_step
        self.rotation_range = rotation_range
        self.rotation_step = rotation_step

    def localize(self, ref_img, search_img):
        t0 = time.perf_counter()

        search_h, search_w = search_img.shape[:2]
        center_x, center_y = search_w / 2.0, search_h / 2.0

        # Pre-smooth search image to suppress SEM shot noise
        search_blur = cv2.GaussianBlur(search_img, (5, 5), 1.0)

        # Scale sweep parameters around nominal scale (8.5:1 to 11.5:1)
        scales = np.arange(8.5, 11.51, self.scale_step)
        rotations = np.arange(self.rotation_range[0], self.rotation_range[1] + 0.1, self.rotation_step)

        best_score = -1.0
        best_cand = None

        for s in scales:
            tpl_w = int(round(ref_img.shape[1] / s))
            tpl_h = int(round(ref_img.shape[0] / s))

            if tpl_w < 10 or tpl_h < 10 or tpl_w >= search_w or tpl_h >= search_h:
                continue

            # Anti-aliasing Gaussian blur before multi-scale downsampling
            sigma = max(0.5, s / 4.0)
            ref_smooth = cv2.GaussianBlur(ref_img, (0, 0), sigmaX=sigma, sigmaY=sigma)
            ref_resized = cv2.resize(ref_smooth, (tpl_w, tpl_h), interpolation=cv2.INTER_AREA)

            for r in rotations:
                if abs(r) > 0.01:
                    M = cv2.getRotationMatrix2D((tpl_w / 2.0, tpl_h / 2.0), r, 1.0)
                    tpl = cv2.warpAffine(ref_resized, M, (tpl_w, tpl_h), borderMode=cv2.BORDER_REFLECT)
                else:
                    tpl = ref_resized

                res = cv2.matchTemplate(search_blur, tpl, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

                if max_val > best_score:
                    best_score = max_val
                    best_cand = {
                        "score": max_val,
                        "res_map": res,
                        "top_left": max_loc,
                        "tpl_w": tpl_w,
                        "tpl_h": tpl_h,
                        "scale": s,
                        "rotation": r
                    }

        # Spatial Tie-Breaking on top match map
        res_map = best_cand["res_map"]
        tpl_w, tpl_h = best_cand["tpl_w"], best_cand["tpl_h"]

        thresh = best_score - 0.008
        peak_y, peak_x = np.where(res_map >= thresh)

        cand_peaks = []
        for py, px in zip(peak_y, peak_x):
            c_x = px + tpl_w / 2.0
            c_y = py + tpl_h / 2.0
            cand_peaks.append((float(res_map[py, px]), c_x, c_y, px, py))

        if len(cand_peaks) > 1:
            selected_peak = min(cand_peaks, key=lambda c: (c[1] - center_x) ** 2 + (c[2] - center_y) ** 2)
        else:
            selected_peak = cand_peaks[0]

        sel_score, _, _, tl_x, tl_y = selected_peak

        # Sub-Pixel Parabolic Interpolation
        sub_x, sub_y = float(tl_x), float(tl_y)

        if 1 <= tl_x < res_map.shape[1] - 1 and 1 <= tl_y < res_map.shape[0] - 1:
            fx = res_map[tl_y, tl_x - 1:tl_x + 2]
            fy = res_map[tl_y - 1:tl_y + 2, tl_x]
            
            denom_x = (2.0 * fx[1] - fx[0] - fx[2] + 1e-6)
            denom_y = (2.0 * fy[1] - fy[0] - fy[2] + 1e-6)
            
            dx = (fx[2] - fx[0]) / (2.0 * denom_x)
            dy = (fy[2] - fy[0]) / (2.0 * denom_y)

            sub_x += np.clip(dx, -0.5, 0.5)
            sub_y += np.clip(dy, -0.5, 0.5)

        final_center_x = sub_x + tpl_w / 2.0
        final_center_y = sub_y + tpl_h / 2.0

        t1 = time.perf_counter()
        runtime = t1 - t0

        return final_center_x, final_center_y, sel_score, runtime
