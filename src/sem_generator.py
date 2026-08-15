import numpy as np
import cv2
import json
import scipy.ndimage as ndimage

class SEMDataGenerator:
    """
    Physical Synthetic SEM Data Generator for Semiconductor Inspection.
    Generates paired 100x Reference (1000x1000) and 10x Search (1000x1000) images.
    """
    def __init__(self, config=None):
        self.config = config or {
            "image_size": [1000, 1000],
            "nominal_scale_ratio": 10.0,
            "scale_range": [9.0, 11.0],
            "rotation_range": [-2.0, 2.0]
        }
        self.img_h, self.img_w = self.config["image_size"]

    def _render_dram_master(self, height, width, pitch_x=40, pitch_y=60, fin_pitch=None, gate_pitch=None):
        if fin_pitch is not None:
            pitch_x = fin_pitch
        if gate_pitch is not None:
            pitch_y = gate_pitch
        canvas = np.zeros((height, width), dtype=np.float32) + 30.0
        
        # Wordlines
        for y in range(0, height, pitch_y):
            thickness = int(pitch_y * 0.35)
            canvas[y:y+thickness, :] += 70.0
            
        # Bitlines
        for x in range(0, width, pitch_x):
            thickness = int(pitch_x * 0.3)
            canvas[:, x:x+thickness] += 50.0

        # Storage Node Contact Pillars (oval array with bright SE edge yield)
        for y in range(pitch_y // 2, height, pitch_y):
            for x in range(pitch_x // 2, width, pitch_x):
                cv2.ellipse(canvas, (x, y), (int(pitch_x * 0.25), int(pitch_y * 0.2)), 0, 0, 360, 120.0, -1)
                cv2.ellipse(canvas, (x, y), (int(pitch_x * 0.25), int(pitch_y * 0.2)), 0, 0, 360, 160.0, 1)
                
        return np.clip(canvas, 0, 255)

    def _render_finfet_master(self, height, width, fin_pitch=25, gate_pitch=70, pitch_x=None, pitch_y=None):
        if pitch_x is not None:
            fin_pitch = pitch_x
        if pitch_y is not None:
            gate_pitch = pitch_y
        canvas = np.zeros((height, width), dtype=np.float32) + 25.0
        
        # Vertical Silicon Fins
        for x in range(0, width, fin_pitch):
            fin_width = int(fin_pitch * 0.25)
            canvas[:, x:x+fin_width] += 80.0
            if x + fin_width < width:
                canvas[:, x:x+1] += 30.0
                canvas[:, x+fin_width-1:x+fin_width] += 30.0

        # Poly Gates
        for y in range(0, height, gate_pitch):
            gate_height = int(gate_pitch * 0.4)
            canvas[y:y+gate_height, :] += 60.0

        # Contacts
        for y in range(gate_pitch // 2, height, gate_pitch):
            for x in range(fin_pitch // 2, width, fin_pitch):
                cv2.rectangle(canvas, (x - 4, y - 6), (x + 4, y + 6), 220.0, -1)
                
        return np.clip(canvas, 0, 255)

    def apply_sem_degradations(self, img, noise_severity=1.0, is_search_image=False):
        img_out = img.copy().astype(np.float32)
        
        # 1. Astigmatic Beam Blur
        sigma_x = (0.6 + 0.3 * noise_severity) * (1.2 if is_search_image else 1.0)
        sigma_y = (0.8 + 0.4 * noise_severity) * (1.2 if is_search_image else 1.0)
        img_out = ndimage.gaussian_filter(img_out, sigma=(sigma_y, sigma_x))

        # 2. Detector Charging Streaks
        if noise_severity > 0.3:
            streak_kernel = np.exp(-np.linspace(0, 2, max(3, int(8 * noise_severity))))
            streak_kernel /= streak_kernel.sum()
            img_out = ndimage.convolve1d(img_out, streak_kernel, axis=1)

        # 3. Poisson Shot Noise
        scale_factor = 0.25 / max(0.2, noise_severity)
        noisy_counts = np.random.poisson(np.clip(img_out * scale_factor, 0, None))
        img_out = noisy_counts / scale_factor

        # 4. Gamma & Contrast
        gamma = 0.9 + 0.2 * np.random.rand()
        img_out = 255.0 * ((np.clip(img_out, 0, 255) / 255.0) ** gamma)

        gauss_noise = np.random.normal(0, 3.0 * noise_severity, img_out.shape)
        img_out = np.clip(img_out + gauss_noise, 0, 255)

        return img_out.astype(np.uint8)

    def generate_pair(self, seed=None, architecture="DRAM", noise_severity=1.0, scale_ratio=10.0, rotation_deg=0.0):
        if seed is not None:
            np.random.seed(seed)

        render_fn = self._render_dram_master if architecture.upper() == "DRAM" else self._render_finfet_master

        pitch_x = 40 if architecture.upper() == "DRAM" else 25
        pitch_y = 60 if architecture.upper() == "DRAM" else 70

        search_base = render_fn(self.img_h, self.img_w, fin_pitch=pitch_x, gate_pitch=pitch_y)

        # Apply rotation to search base FIRST if requested
        if abs(rotation_deg) > 0.01:
            M = cv2.getRotationMatrix2D((self.img_w / 2.0, self.img_h / 2.0), rotation_deg, 1.0)
            search_base = cv2.warpAffine(search_base, M, (self.img_w, self.img_h), borderMode=cv2.BORDER_REFLECT)

        # Target location in Search Image
        true_x = float(np.random.randint(200, self.img_w - 200))
        true_y = float(np.random.randint(200, self.img_h - 200))

        fov_w_in_search = self.img_w / scale_ratio
        fov_h_in_search = self.img_h / scale_ratio

        x1 = max(0, int(round(true_x - fov_w_in_search / 2.0)))
        y1 = max(0, int(round(true_y - fov_h_in_search / 2.0)))
        x2 = min(self.img_w, int(round(true_x + fov_w_in_search / 2.0)))
        y2 = min(self.img_h, int(round(true_y + fov_h_in_search / 2.0)))

        crop = search_base[y1:y2, x1:x2]
        ref_base = cv2.resize(crop, (self.img_w, self.img_h), interpolation=cv2.INTER_CUBIC)

        ref_img = self.apply_sem_degradations(ref_base, noise_severity=noise_severity * 0.7, is_search_image=False)
        search_img = self.apply_sem_degradations(search_base, noise_severity=noise_severity * 1.1, is_search_image=True)

        metadata = {
            "seed": seed,
            "architecture": architecture,
            "true_x": round(true_x, 4),
            "true_y": round(true_y, 4),
            "scale_ratio": round(scale_ratio, 4),
            "rotation_deg": round(rotation_deg, 4),
            "noise_severity": round(noise_severity, 4),
            "image_size": [self.img_w, self.img_h]
        }

        return ref_img, search_img, metadata
