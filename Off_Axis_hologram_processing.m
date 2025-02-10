close all; clear; clc;

%%

% Step 1: Sample Beam
img = imread('image_off_axis.jpg'); % Load the image
img_gray = double(rgb2gray(img)); % Convert the image to grayscale
img_phase = img_gray / max(img_gray(:)) * 2 * pi; % Normalize phase between 0 and 2pi
[Ny, Nx] = size(img_phase); % Get image dimensions
sample_wave = exp(1i * img_phase); % Sample beam: phase only

% Display the phase
figure;
imagesc(img_phase); % Show the phase
colorbar; % Add a color bar
title('Phase of the Loaded Image (0 to 2\pi)');
xlabel('X (pixels)');
ylabel('Y (pixels)');

% Step 2: Reference Beam
lambda = 500e-9; % Wavelength (500 nanometers)
pixel_size = 1e-6; % Pixel size (1 micrometer)
d_pixels = 3; % 3 pixels per period
d_meters = d_pixels * pixel_size; % Convert pixel period to meters
theta = asin(lambda / d_meters); % Cardinal angle (radians)
kx = 2 * pi / lambda * sin(theta); % Wave vector in the x direction

% Normalized pixel grid:
[x, y] = meshgrid((0:Nx-1)/Nx, (0:Ny-1)/Ny); % Normalized coordinates in the range [0,1]
x = x * (Nx * pixel_size); % Scale to full physical range
y = y * (Ny * pixel_size); % Scale to full physical range

% Normalized reference beam:
ref_wave = exp(1i * kx * x); % Plane wave normalized to image dimensions

% Step 3: Creating a Clean Hologram
hologram_clean = abs(sample_wave + ref_wave).^2; % Intensity from interference

% Adding multiplicative noise
noise_sigma = 0.5; % Standard deviation of noise
multiplicative_noise = 1 + noise_sigma * randn(size(hologram_clean)); % Generate multiplicative noise
hologram_noisy = hologram_clean .* multiplicative_noise; % Apply multiplicative noise

% Step 5: Saving the Holograms
imwrite(mat2gray(hologram_clean), 'hologram_clean.png'); % Save clean hologram
imwrite(mat2gray(hologram_noisy), 'hologram_noisy.png'); % Save noisy hologram

% Step 6: Displaying the Results
figure;
subplot(1, 2, 1);
imshow(hologram_clean, []);
title('Hologram (Clean)');
subplot(1, 2, 2);
imshow(hologram_noisy, []);
title('Hologram (Noisy)');

% Comparisons

% 3D Surface Plot of the Holograms
figure;

subplot(1, 2, 1);
surf(hologram_clean, 'EdgeColor', 'none');
title('Clean Hologram');
xlabel('X (pixels)');
ylabel('Y (pixels)');
zlabel('Pixel Value');
colorbar;
view(3); % 3D view

subplot(1, 2, 2);
surf(hologram_noisy, 'EdgeColor', 'none');
title('Noisy Hologram');
xlabel('X (pixels)');
ylabel('Y (pixels)');
zlabel('Pixel Value');
colorbar;
view(3); % 3D view

% Statistical Analysis of Pixels

% Select a data segment from the clean hologram with values starting from 1.5
hologram_clean_cut = hologram_clean(hologram_clean >= 1.5);
hologram_noisy_cut = hologram_noisy(hologram_noisy >= 1.5);

% Display Histograms
figure;

% Histogram for the clean hologram
subplot(2, 1, 1);
histogram(hologram_clean_cut, 50, 'FaceColor', 'b');
title('Clean Hologram Pixel Value Distribution');
xlabel('Pixel Value');
ylabel('Frequency');

% Histogram for the noisy hologram
subplot(2, 1, 2);
histogram(hologram_noisy_cut, 50, 'FaceColor', 'r');
title('Noisy Hologram Pixel Value Distribution');
xlabel('Pixel Value');
ylabel('Frequency');

%%
% Fourier Transform and Frequency Domain Visualization

% Perform 2D Fourier Transform (FFT) on the holograms
fft_clean = fftshift(fft2(fftshift(hologram_clean))); % FFT of the clean hologram
fft_noisy = fftshift(fft2(fftshift(hologram_noisy))); % FFT of the noisy hologram

% Step 1: Fourier Transform of the Reference Beam
fft_ref = fftshift(fft2(fftshift(ref_wave))); % Compute FFT of the reference beam

% Step 2: Identify the Peak in the Fourier Transform of the Reference
[max_val, max_idx] = max(abs(fft_ref(:))); % Find the location of the maximum value
[y_peak, x_peak] = ind2sub(size(fft_ref), max_idx); % Get the peak position in the matrix

% Step 3: Crop Around the Peak
window_size = 10; % Half of the window size around the peak
y_range = max(1, y_peak - window_size):min(size(fft_ref, 1), y_peak + window_size);
x_range = max(1, x_peak - window_size):min(size(fft_ref, 2), x_peak + window_size);
cropped_ref = fft_ref(y_range, x_range); % Crop the reference beam

% Compute absolute value and log contrast for enhancement
fft_clean_log = log(abs(fft_clean) + 1);
fft_noisy_log = log(abs(fft_noisy) + 1);

% Display results
figure;

% Clean hologram in the frequency domain
subplot(1, 2, 1);
imagesc(fft_clean_log); % Display Fourier Transform of the clean hologram
title('Fourier Transform of Clean Hologram');
xlabel('X (pixels)');
ylabel('Y (pixels)');
colorbar; % Add a color bar
axis on; % Display axis labels
axis image; % Maintain correct axis proportions

% Cropping the relevant region for the clean hologram
disp('Selected region for Clean Hologram:');
%rect_clean = [204, 121, 19, 16]; % Provided values
rect_clean = [209, 123, 12, 13]; % Provided values
disp(rect_clean);
cropped_clean = fft_clean(rect_clean(2):(rect_clean(2)+rect_clean(4)-1), rect_clean(1):(rect_clean(1)+rect_clean(3)-1));

% Noisy hologram in the frequency domain
subplot(1, 2, 2);
imagesc(fft_noisy_log); % Display Fourier Transform of the noisy hologram
title('Fourier Transform of Noisy Hologram');
xlabel('X (pixels)');
ylabel('Y (pixels)');
colorbar; % Add a color bar
axis on; % Display axis labels
axis image; % Maintain correct axis proportions

%%
% Cropping the relevant region for the noisy hologram
disp('Selected region for Noisy Hologram:');
rect_noisy = [209, 123, 12, 13]; % Provided values
disp(rect_noisy);
cropped_noisy = fft_noisy(rect_noisy(2):(rect_noisy(2)+rect_noisy(4)-1), rect_noisy(1):(rect_noisy(1)+rect_noisy(3)-1));

% Axis labels in Fourier domain
[Ny, Nx] = size(hologram_clean); % Image size
xticks = linspace(1, Nx, 5); % Divide X-axis into 5 regions
yticks = linspace(1, Ny, 5); % Divide Y-axis into 5 regions
subplot(1, 2, 1);
xticklabels(round(linspace(-Nx/2, Nx/2, 5))); % Update X-axis labels
yticklabels(round(linspace(-Ny/2, Ny/2, 5))); % Update Y-axis labels
subplot(1, 2, 2);
xticklabels(round(linspace(-Nx/2, Nx/2, 5))); % Same for noisy hologram
yticklabels(round(linspace(-Ny/2, Ny/2, 5)));

% Display cropped regions
figure;
subplot(1, 2, 1);
imagesc(log(abs(cropped_clean)));
title('Cropped Region (Clean Hologram)');
colorbar;

subplot(1, 2, 2);
imagesc(log(abs(cropped_noisy)));
title('Cropped Region (Noisy Hologram)');
colorbar;

% Display the cropped reference wave
figure;
imagesc(abs(cropped_ref)); % Show amplitude of cropped reference
colorbar; % Add a color bar
title('Cropped Reference Wave');
xlabel('X (pixels)');
ylabel('Y (pixels)');

%% Padding

% Step 9: Zero-padding to the original image size

% Padding for the clean hologram
padded_clean = zeros(Ny, Nx); % Create a zero-padded matrix
[y_c, x_c] = size(cropped_clean); % Get the size of the cropped region
y_start_clean = floor((Ny - y_c) / 2) + 1; % Starting position on Y-axis
x_start_clean = floor((Nx - x_c) / 2) + 1; % Starting position on X-axis
padded_clean(y_start_clean:y_start_clean+y_c-1, x_start_clean:x_start_clean+x_c-1) = cropped_clean; % Apply padding

% Padding for the noisy hologram
padded_noisy = zeros(Ny, Nx); % Create a zero-padded matrix
[y_n, x_n] = size(cropped_noisy); % Get the size of the cropped region
y_start_noisy = floor((Ny - y_n) / 2) + 1; % Starting position on Y-axis
x_start_noisy = floor((Nx - x_n) / 2) + 1; % Starting position on X-axis
padded_noisy(y_start_noisy:y_start_noisy+y_n-1, x_start_noisy:x_start_noisy+x_n-1) = cropped_noisy; % Apply padding

% Padding for the reference wave
padded_ref = zeros(Ny, Nx); % Create a zero-padded matrix
[y_n, x_n] = size(cropped_ref); % Get the size of the cropped region
y_start_ref = floor((Ny - y_n) / 2) + 1; % Starting position on Y-axis
x_start_ref = floor((Nx - x_n) / 2) + 1; % Starting position on X-axis
padded_ref(y_start_ref:y_start_ref+y_n-1, x_start_ref:x_start_ref+x_n-1) = cropped_ref; % Apply padding

% Display padded results
figure;
subplot(1, 2, 1);
imagesc(log(abs(padded_clean))); % Display padded clean hologram
title('Padded Region (Clean Hologram)');
colorbar;

subplot(1, 2, 2);
imagesc(log(abs(padded_noisy))); % Display padded noisy hologram
title('Padded Region (Noisy Hologram)');
colorbar;

% Step 10: Inverse Transform on the Padded Images
reconstructed_clean_before = ifftshift(ifft2(ifftshift(padded_clean))); % Inverse transform for the clean hologram
reconstructed_noisy_before = ifftshift(ifft2(ifftshift(padded_noisy))); % Inverse transform for the noisy hologram
reconstructed_ref = ifftshift(ifft2(ifftshift(padded_ref))); % Inverse transform for the reference wave

reconstructed_clean = reconstructed_clean_before ./ (reconstructed_ref + eps);
reconstructed_noisy = reconstructed_noisy_before ./ (reconstructed_ref + eps);

%%
% Step 11: Phase Reconstruction
phase_clean = angle(reconstructed_clean); % Compute phase from the clean image
phase_noisy = angle(reconstructed_noisy); % Compute phase from the noisy image

% Step 12: Display Phase Before Unwrapping with Inverted Contrast
figure;

% Display the reconstructed phase from the clean image with inverted contrast
subplot(1, 2, 1);
imagesc(imcomplement(phase_clean));
title('Phase Before Unwrapping (Clean Hologram)');
xlabel('X (pixels)');
ylabel('Y (pixels)');
colorbar;

% Display the reconstructed phase from the noisy image with inverted contrast
subplot(1, 2, 2);
imagesc(imcomplement(phase_noisy));
title('Phase Before Unwrapping (Noisy Hologram)');
xlabel('X (pixels)');
ylabel('Y (pixels)');
colorbar;

%%
% Step 13: Use Unwrapping Function
unwrapped_phase_clean = phase_unwrapping_2D(phase_clean); % Unwrapping for clean phase
unwrapped_phase_noisy = phase_unwrapping_2D(phase_noisy); % Unwrapping for noisy phase

% Step 14: Display Phase After Unwrapping with Inverted Contrast
figure;

% Unwrapped clean phase with inverted contrast
subplot(1, 2, 1);
imagesc(imcomplement(unwrapped_phase_clean)); % Invert contrast
title('Phase After Unwrapping (Clean Hologram)');
xlabel('X (pixels)');
ylabel('Y (pixels)');
colorbar;

% Unwrapped noisy phase with inverted contrast
subplot(1, 2, 2);
imagesc(imcomplement(unwrapped_phase_noisy)); % Invert contrast
title('Phase After Unwrapping (Noisy Hologram)');
xlabel('X (pixels)');
ylabel('Y (pixels)');
colorbar;

%% Comparison
[ssim_value, ssim_map] = ssim(unwrapped_phase_noisy, unwrapped_phase_clean);

% Display SSIM Map
figure;
imagesc(imcomplement(ssim_map));
title(sprintf('SSIM Map (SSIM: %.4f)', ssim_value));
xlabel('X (pixels)');
ylabel('Y (pixels)');
colorbar;

%%%%%%%%%%%%%%%%% Adding Phase Reconstruction Using Hilbert Transform %%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%% Hilbert Transform Only on Clean Hologram %%%%%%%%%%%%%%%%%%%%%%%%%

% Hologram I
I = hologram_clean;

% 1. Hilbert Calculation
FT_I = fft2(I); 
% sign(w=omega)
[Ny, Nx] = size(I);
kx = linspace(-Nx/2, Nx/2-1, Nx);
ky = linspace(-Ny/2, Ny/2-1, Ny);
[Kx, Ky] = meshgrid(kx, ky);
sign_w = double(Kx > 0) - double(Kx < 0); 
% (-j * sign(w) * FFT)
j_w_FT_I = -1j .* sign_w .* FT_I;
% IFFT
HT_I = ifft2(j_w_FT_I); 

% 2. OR*
OR = I + 1j .* HT_I;

% Compute R*
R = sqrt(ref_wave); % Reference beam (intensity from the square root of background values)
R_star = conj(R); % Compute the complex conjugate of the reference beam

% O without R*
O = OR ./ R_star;

% 3. Phase Calculation
H_phase = angle(O);

% 4. Phase Unwrapping
unwrapped_H_phase2 = phase_unwrapping_2D(H_phase);

% 4.5 Filtering for H_phase

% Step 1: Compute Fourier Transform of the Reconstructed Phase
FT_H_unwrapped = fftshift(fft2(unwrapped_H_phase2)); % FFT of the reconstructed phase

% Step 2: Create a Low-Pass Filter in the Frequency Domain
[Ny, Nx] = size(unwrapped_H_phase2);
[u, v] = meshgrid(-Nx/2:Nx/2-1, -Ny/2:Ny/2-1); % Frequency grid
D = sqrt(u.^2 + v.^2); % Distance from frequency center
D0 = 20; % Filtering radius (adjust as needed)
LP_filter = exp(-(D.^2) / (2 * D0^2)); % Gaussian filter

% Step 3: Apply the Filter
H_phase_unwrapped_Filtered = FT_H_unwrapped .* LP_filter;

% Step 4: Convert Back to Spatial Domain
H_phase_unwrapped_Filtered_ = ifft2(ifftshift(H_phase_unwrapped_Filtered)); % Convert back to image space

% 5. Display All Results

figure();

% Phase from the original hologram
subplot(1, 3, 1);
imagesc(img_phase);
title('Phase from Hologram');
xlabel('X (pixels)');
ylabel('Y (pixels)');
colorbar;

% Phase using Fourier Transform
subplot(1, 3, 2);
imagesc(imcomplement(unwrapped_phase_clean)); % Invert contrast
title('Phase from Clean Hologram (Fourier)');
xlabel('X (pixels)');
ylabel('Y (pixels)');
colorbar;

% Phase using Hilbert Transform
subplot(1, 3, 3);
imagesc(H_phase_unwrapped_Filtered_);
title('Phase from Clean Hologram (Hilbert)');
xlabel('X (pixels)');
ylabel('Y (pixels)');
colorbar;

%%
% phase_unwrapping_2D function
function [unwrapped_phase] = phase_unwrapping_2D(wrapped_phase)
    % discrete cosine transform (DCT) based un-weighted least squares (UWLS)
    % algorithm - based on the DCT method from page 200 in
    % two dimensional phase unwrapping / D.C. Ghiglia and M.D. Pritt

    [rows, cols] = size(wrapped_phase);

    temp = zeros(rows, cols);
    dif_x = zeros(rows, cols);
    dif_y = zeros(rows, cols);

    % calculate wrapped differences
    % dif_X
    for m = 1:rows % for the last column we have zeros due to mirror reflection causing replication
        for n = 1:cols-1
            temp(m, n) = wrapped_phase(m, n+1) - wrapped_phase(m, n);
            dif_x(m, n) = atan2(sin(temp(m, n)), cos(temp(m, n))); % wrap
        end
    end

    % dif_Y
    for m = 1:rows-1 % for the last row we have zeros due to mirror reflection causing replication
        for n = 1:cols
            temp(m, n) = wrapped_phase(m+1, n) - wrapped_phase(m, n);
            dif_y(m, n) = atan2(sin(temp(m, n)), cos(temp(m, n))); % wrap
        end
    end

    ro_x = zeros(rows, cols);
    ro_y = zeros(rows, cols);

    % ro_x
    for m = 1:rows
        ro_x(m, 1) = dif_x(m, 1); % dif_x(m, 0) is zero due to periodicity and mirroring
        for n = 2:cols
            ro_x(m, n) = dif_x(m, n) - dif_x(m, n-1);
        end
    end

    % ro_y
    for n = 1:cols
        ro_y(1, n) = dif_y(1, n); % dif_x(0, n) is zero due to periodicity and mirroring
    end

    for m = 2:rows
        for n = 1:cols
            ro_y(m, n) = dif_y(m, n) - dif_y(m-1, n);
        end
    end

    % ro
    ro = ro_x + ro_y;

    result = dct2(ro);

    for m = 1:rows
        for n = 1:cols
            if (n == 1 && m == 1)
                result(m, n) = 0;
            else
                result(m, n) = result(m, n) / ((2*cos(pi*(m-1)/rows) + 2*cos(pi*(n-1)/cols) - 4));
            end
        end
    end

    unwrapped_phase = idct2(result);
end
