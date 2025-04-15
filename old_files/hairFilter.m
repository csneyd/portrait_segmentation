function [output_img] = hairFilter(M, N, f0, sigma)
% isoGaussBandwidthFilter - Isotropic Gaussian Bandwidth Filter
%   M, N    - Dimensions of the filter
%   f0      - Center frequency
%   sigma   - Bandwidth (standard deviation)

H = zeros(M, N);  % Initialize filter

% Loop through each frequency pair
for u = 1:M
    for v = 1:N
        % Calculate distance from (u, v) to the center of the frequency domain
        D = sqrt((u - M/2)^2 + (v - N/2)^2);
        
        % Gaussian Bandpass Filter formula
        H(u, v) = exp(-((D - f0)^2) / (2 * sigma^2));
    end
end

output_img = H;
end
