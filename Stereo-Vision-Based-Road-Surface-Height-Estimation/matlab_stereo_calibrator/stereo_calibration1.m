% Auto-generated by stereoCalibrator app on 17-Jan-2019
%-------------------------------------------------------


% Define images to process
imageFileNames1 = {'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left00.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left03.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left04.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left06.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left08.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left09.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left10.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left100.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left101.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left102.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left103.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left104.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left105.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left106.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left107.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left108.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left109.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left11.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left110.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left111.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left112.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left114.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left115.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left117.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left118.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left119.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left120.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left121.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left122.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left123.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left124.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left125.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left126.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left127.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left128.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left129.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left130.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left131.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left132.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left133.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left134.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left135.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left136.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left137.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left138.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left139.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left14.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left15.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left17.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left18.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left19.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left20.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left21.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left22.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left23.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left24.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left25.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left26.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left28.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left30.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left31.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left32.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left34.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left35.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left36.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left37.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left38.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left39.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left40.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left41.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left42.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left43.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left44.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left45.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left46.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left47.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left48.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left49.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left50.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left51.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left52.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left53.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left54.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left55.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left56.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left57.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left58.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left59.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left60.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left61.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left62.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left63.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left64.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left65.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left66.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left67.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left68.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left69.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left70.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left71.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left72.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left73.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left74.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left75.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left76.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left77.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left78.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left79.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left80.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left81.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left82.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left83.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left84.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left85.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left86.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left87.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left88.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left89.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left90.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left91.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left92.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left93.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left94.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left95.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left96.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left97.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left98.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\left\v2_left99.png',...
    };
imageFileNames2 = {'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right00.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right03.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right04.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right06.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right08.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right09.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right10.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right100.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right101.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right102.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right103.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right104.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right105.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right106.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right107.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right108.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right109.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right11.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right110.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right111.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right112.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right114.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right115.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right117.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right118.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right119.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right120.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right121.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right122.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right123.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right124.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right125.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right126.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right127.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right128.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right129.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right130.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right131.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right132.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right133.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right134.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right135.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right136.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right137.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right138.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right139.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right14.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right15.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right17.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right18.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right19.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right20.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right21.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right22.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right23.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right24.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right25.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right26.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right28.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right30.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right31.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right32.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right34.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right35.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right36.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right37.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right38.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right39.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right40.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right41.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right42.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right43.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right44.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right45.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right46.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right47.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right48.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right49.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right50.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right51.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right52.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right53.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right54.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right55.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right56.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right57.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right58.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right59.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right60.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right61.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right62.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right63.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right64.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right65.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right66.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right67.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right68.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right69.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right70.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right71.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right72.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right73.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right74.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right75.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right76.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right77.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right78.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right79.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right80.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right81.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right82.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right83.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right84.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right85.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right86.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right87.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right88.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right89.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right90.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right91.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right92.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right93.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right94.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right95.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right96.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right97.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right98.png',...
    'C:\Users\CSK\Desktop\matlab_stereo_calibrator\cali_1\right\v2_right99.png',...
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
