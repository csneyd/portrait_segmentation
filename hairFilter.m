function [output_img] = hairFilter(M, N, C0, W)
% Gaussian Band-Pass Filter
%   M, N    - Dimensions of the filter
%   C0      - Center frequency
%   W   - Bandwidth (standard deviation)

H = zeros(M, N);  % Initialize filter

% Loop through each frequency pair
for u = 1:M
    for v = 1:N
        % Calculate distance from (u, v) to the center of the frequency domain
        D = sqrt((u - M/2)^2 + (v - N/2)^2);
        
        % Gaussian Bandpass Filter formula
        H(u, v) = exp(-((D^2-C0^2) / (D * W))^2);
    end
end

output_img = H;
end
