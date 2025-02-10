# Off-Axis-Holography-and-Phase-Reconstruction
Off-Axis Holography and Phase Reconstruction
## Overview
This project focuses on creating and reconstructing phase information from an off-axis hologram. The methodology involves simulating the interference pattern between a reference beam and a sample beam, capturing the digital hologram, and reconstructing the phase. Additionally, we compare phase reconstruction from clean and noisy holograms to analyze noise effects.
## Project Goals
-	Simulate interference between a reference beam and a sample beam.
-	Capture and process off-axis holograms.
-	Reconstruct the phase from both clean and noisy holograms.
-	Analyze the impact of noise on phase reconstruction.
## Methodology
1.	Hologram Generation
-	Convert an image to grayscale.
-	Define a sample beam with phase values extracted from the image.
-	Set the amplitude to 1.
-	Combine the sample beam with a reference beam to create an interference pattern.
2.	Adding Noise
-	Generate a Gaussian noise matrix.
-	Add multiplicative noise to the interference pattern.
-	Obtain a noisy hologram.
3.	Phase Reconstruction
-	Apply Fourier Transform (FFT) to extract relevant frequency components.
-	Filter and isolate the desired signal.
-	Use inverse FFT to reconstruct the phase.
-	Apply phase unwrapping techniques to resolve discontinuities.
-	Implement Hilbert Transform for alternative phase reconstruction.
4.	Analysis
-	Compare phase reconstruction accuracy between clean and noisy holograms.
-	Evaluate the impact of noise on the phase structure.
-	Compute Structural Similarity Index (SSIM) to quantify differences.
-	Compare Fourier Transform and Hilbert Transform approaches.
## Results
-	Clean holograms produce high-fidelity phase reconstructions.
-	Noisy holograms introduce distortions, particularly in regions with fine details.
-	Higher noise levels reduce the Signal-to-Noise Ratio (SNR), leading to errors in phase reconstruction.
-	Hilbert Transform and Fourier Transform both allow phase reconstruction, but Fourier Transform enables better noise filtering.
## Files Included
-	image_off_axis.jpg - Original image used for simulation.
-	Off_Axis_hologram_processing.m - MATLAB script for hologram generation and phase reconstruction.
## Conclusion
This project successfully demonstrates phase reconstruction from off-axis holography. The analysis confirms that noise significantly impacts the reconstruction accuracy, and advanced filtering techniques are necessary to mitigate its effects.
## Acknowledgments
Authors: Shaked Dahan & Chen Atias
For questions and collaboration, feel free to contact us!

