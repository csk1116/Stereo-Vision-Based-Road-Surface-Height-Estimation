% Auto-generated by stereoCalibrator app on 24-Jan-2019
%-------------------------------------------------------


% Define images to process
imageFileNames1 = {'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\left\v2_left03.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\left\v2_left04.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\left\v2_left09.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\left\v2_left10.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\left\v2_left100.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\left\v2_left102.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\left\v2_left104.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\left\v2_left107.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\left\v2_left14.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\left\v2_left18.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\left\v2_left19.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\left\v2_left20.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\left\v2_left21.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\left\v2_left25.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\left\v2_left26.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\left\v2_left28.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\left\v2_left30.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\left\v2_left32.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\left\v2_left36.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\left\v2_left41.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\left\v2_left42.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\left\v2_left47.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\left\v2_left48.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\left\v2_left49.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\left\v2_left50.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\left\v2_left52.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\left\v2_left53.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\left\v2_left54.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\left\v2_left55.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\left\v2_left56.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\left\v2_left58.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\left\v2_left59.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\left\v2_left61.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\left\v2_left62.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\left\v2_left63.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\left\v2_left67.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\left\v2_left68.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\left\v2_left69.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\left\v2_left72.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\left\v2_left73.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\left\v2_left75.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\left\v2_left77.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\left\v2_left78.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\left\v2_left79.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\left\v2_left83.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\left\v2_left84.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\left\v2_left85.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\left\v2_left87.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\left\v2_left88.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\left\v2_left90.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\left\v2_left91.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\left\v2_left93.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\left\v2_left97.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\left\v2_left98.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\left\v2_left99.png',...
    };
imageFileNames2 = {'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\right\v2_right03.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\right\v2_right04.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\right\v2_right09.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\right\v2_right10.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\right\v2_right100.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\right\v2_right102.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\right\v2_right104.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\right\v2_right107.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\right\v2_right14.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\right\v2_right18.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\right\v2_right19.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\right\v2_right20.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\right\v2_right21.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\right\v2_right25.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\right\v2_right26.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\right\v2_right28.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\right\v2_right30.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\right\v2_right32.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\right\v2_right36.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\right\v2_right41.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\right\v2_right42.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\right\v2_right47.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\right\v2_right48.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\right\v2_right49.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\right\v2_right50.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\right\v2_right52.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\right\v2_right53.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\right\v2_right54.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\right\v2_right55.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\right\v2_right56.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\right\v2_right58.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\right\v2_right59.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\right\v2_right61.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\right\v2_right62.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\right\v2_right63.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\right\v2_right67.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\right\v2_right68.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\right\v2_right69.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\right\v2_right72.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\right\v2_right73.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\right\v2_right75.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\right\v2_right77.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\right\v2_right78.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\right\v2_right79.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\right\v2_right83.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\right\v2_right84.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\right\v2_right85.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\right\v2_right87.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\right\v2_right88.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\right\v2_right90.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\right\v2_right91.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\right\v2_right93.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\right\v2_right97.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\right\v2_right98.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_2\right\v2_right99.png',...
    };

% Detect checkerboards in images
[imagePoints, boardSize, imagesUsed] = detectCheckerboardPoints(imageFileNames1, imageFileNames2);

% Generate world coordinates of the checkerboard keypoints
squareSize = 100;  % in units of 'millimeters'
worldPoints = generateCheckerboardPoints(boardSize, squareSize);

% Read one of the images from the first stereo pair
I1 = imread(imageFileNames1{1});
[mrows, ncols, ~] = size(I1);

% Calibrate the camera
[stereoParams, pairsUsed, estimationErrors] = estimateCameraParameters(imagePoints, worldPoints, ...
    'EstimateSkew', false, 'EstimateTangentialDistortion', false, ...
    'NumRadialDistortionCoefficients', 2, 'WorldUnits', 'millimeters', ...
    'InitialIntrinsicMatrix', [], 'InitialRadialDistortion', [], ...
    'ImageSize', [mrows, ncols]);

% View reprojection errors
h1=figure; showReprojectionErrors(stereoParams);

% Visualize pattern locations
h2=figure; showExtrinsics(stereoParams, 'CameraCentric');

% Display parameter estimation errors
displayErrors(estimationErrors, stereoParams);

% You can use the calibration data to rectify stereo images.
I2 = imread(imageFileNames2{1});
[J1, J2] = rectifyStereoImages(I1, I2, stereoParams);

% See additional examples of how to use the calibration data.  At the prompt type:
% showdemo('StereoCalibrationAndSceneReconstructionExample')
% showdemo('DepthEstimationFromStereoVideoExample')
