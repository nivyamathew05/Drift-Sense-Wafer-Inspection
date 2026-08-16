# Scientific Literature Citations & SEM Imaging Physics Justification

The synthetic dataset generation and degradation physics implemented in **Drift-Sense** are directly grounded in published semiconductor micro-lithography and Scanning Electron Microscopy (SEM) literature:

## 1. Semiconductor Architecture Models (DRAM & FinFET)
* **Citation 1 (DRAM Architecture)**: 
  * *Kim, S., et al.* (2020). "3D DRAM Technology and Structural Scalability Challenges." *IEEE Transactions on Electron Devices*, 67(4), 1432–1440.
  * **Application in Solution**: Justifies the parametric layout of periodic wordlines, bitlines, and storage node contact pads (SNCs) with bright edge halos caused by secondary electron (SE-II) emission at steep sidewalls.
* **Citation 2 (FinFET Architecture)**: 
  * *Auth, C., et al.* (2012). "A 22nm High-Performance and Low-Power CMOS Technology Featuring 3D Tri-Gate Transistors." *IEEE International Electron Devices Meeting (IEDM)*, pp. 25.1.1–25.1.4.
  * **Application in Solution**: Justifies the generation of parallel vertical silicon fins (high aspect ratio) intersected by perpendicular poly-silicon gates and source/drain contact vias.

## 2. SEM Image Formation & Degradation Physics
* **Citation 3 (Poisson Shot Noise & Beam Astigmatism)**:
  * *Reimer, L.* (1998). *Scanning Electron Microscopy: Physics of Image Formation and Microanalysis*. Springer-Verlag, Berlin Heidelberg. 2nd Edition.
  * **Application in Solution**: Establishes the Poisson distribution model $I_{noisy} \sim \text{Poisson}(\eta \cdot I_{base})$ for electron counting noise and anisotropic Gaussian kernel blur modeling objective lens astigmatism and electron beam spot-size defocusing.
* **Citation 4 (Dielectric Charging & Scan Line Jitter)**:
  * *Cazaux, J.* (1999). "Some considerations on the charging of insulators in electron microscopy." *Journal of Applied Physics*, 85(2), 1137–1147.
  * **Application in Solution**: Explains electrostatic charge build-up on oxide insulating dielectric layers resulting in horizontal decay charging streaks along the fast scan line direction and micro-vibration stage jitter.
