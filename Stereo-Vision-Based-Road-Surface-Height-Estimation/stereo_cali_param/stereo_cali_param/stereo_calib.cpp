/* This is sample from the OpenCV book. The copyright notice is below */

/* *************** License:**************************
   Oct. 3, 2008
   Right to use this code in any way you want without warranty, support or any guarantee of it working.

   BOOK: It would be nice if you cited it:
   Learning OpenCV: Computer Vision with the OpenCV Library
     by Gary Bradski and Adrian Kaehler
     Published by O'Reilly Media, October 3, 2008

   AVAILABLE AT:
     http://www.amazon.com/Learning-OpenCV-Computer-Vision-Library/dp/0596516134
     Or: http://oreilly.com/catalog/9780596516130/
     ISBN-10: 0596516134 or: ISBN-13: 978-0596516130

   OPENCV WEBSITES:
     Homepage:      http://opencv.org
     Online docs:   http://docs.opencv.org
     Q&A forum:     http://answers.opencv.org
     Issue tracker: http://code.opencv.org
     GitHub:        https://github.com/opencv/opencv/
   ************************************************** */

#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <fstream>
#include <time.h>


using namespace cv;
using namespace std;

static int print_help()
{
    cout <<
            " Given a list of chessboard images, the number of corners (nx, ny)\n"
            " on the chessboards, and a flag: useCalibrated for \n"
            "   calibrated (0) or\n"
            "   uncalibrated \n"
            "     (1: use cvStereoCalibrate(), 2: compute fundamental\n"
            "         matrix separately) stereo. \n"
            " Calibrate the cameras and display the\n"
            " rectified results along with the computed disparity images.   \n" << endl;
    cout << "Usage:\n ./stereo_calib -w=<board_width default=9> -h=<board_height default=6> -s=<square_size default=1.0> <image list XML/YML file default=../data/stereo_calib.xml>\n" << endl;
    return 0;
}


static void StereoCalib(const vector<string>& imagelist, Size boardSize, float squareSize, Mat& rmap_c00, Mat& rmap_c01, Mat& rmap_c10, Mat& rmap_c11, Mat& Q_c,
	                    bool displayCorners = false, bool useCalibrated = true, bool showRectified = true)
{
    if( imagelist.size() % 2 != 0 )
    {
        cout << "Error: the image list contains odd (non-even) number of elements\n";
		return;
    }

    const int maxScale = 2;

    // ARRAY AND VECTOR STORAGE:
    vector<vector<Point2f> > imagePoints[2];
    vector<vector<Point3f> > objectPoints;
    Size imageSize;

    int i, j, k, nimages = (int)imagelist.size()/2;
	//int shot = 0;

    imagePoints[0].resize(nimages);
    imagePoints[1].resize(nimages);
    vector<string> goodImageList;

    for( i = j = 0; i < nimages; i++ )
    {
        for( k = 0; k < 2; k++ )
        {
            const string& filename = imagelist[i*2+k];
            Mat img = imread("C:/Users/CSK/source/repos/stereo_cali_param/stereo_cali_param/stereo_calib2/1M~3M/" + filename,IMREAD_GRAYSCALE);
			//cout << "channels" << img.channels() << endl;
            if(img.empty())
                break;
            if( imageSize == Size() )
                imageSize = img.size();
            else if( img.size() != imageSize )
            {
                cout << "The image " << filename << " has the size different from the first image size. Skipping the pair\n";
                break;
            }
            bool found = false;
            vector<Point2f>& corners = imagePoints[k][j];
            for( int scale = 1; scale <= maxScale; scale++ )
            {
                Mat timg;
                if( scale == 1 )
                    timg = img;
                else
                    resize(img, timg, Size(), scale, scale, INTER_LINEAR_EXACT);
                found = findChessboardCorners(timg, boardSize, corners,
                    CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
                if( found )
                {
                    if( scale > 1 )
                    {
                        Mat cornersMat(corners);
                        cornersMat *= 1./scale;
                    }
                    break;
                }
            }
            if( displayCorners )
            {
                cout << filename << endl;
                Mat cimg, cimg1;
                cvtColor(img, cimg, COLOR_GRAY2BGR);
                drawChessboardCorners(cimg, boardSize, corners, found);
				cimg1 = cimg.clone();
                //double sf = 640./MAX(img.rows, img.cols);
                //resize(cimg, cimg1, Size(), sf, sf, INTER_LINEAR_EXACT);
				imshow("corners", cimg1);

				/*if (shot == 20) {
					stringstream l_name1;
					l_name1 << "corner_left" << i << ".png";
					imwrite(l_name1.str(), cimg1);
					
				}



				if (shot == 21) {
					stringstream r_name1;
					r_name1 << "corner_right" << i << ".png";
					imwrite(r_name1.str(), cimg1);
					
				}
				
				shot++;*/
				
                char c = (char)waitKey(1000);
                if( c == 27 || c == 'q' || c == 'Q' ) //Allow ESC to quit
                    exit(-1);
            }
            else
                putchar('.');
            if( !found )
                break;


            cornerSubPix(img, corners, Size(15,15), Size(-1,-1),
                         TermCriteria(TermCriteria::COUNT+TermCriteria::EPS,
                                      10000, 1e-5));
        }
        if( k == 2 )
        {
            goodImageList.push_back(imagelist[i*2]);
            goodImageList.push_back(imagelist[i*2+1]);
            j++;
        }
    }
    cout << j << " pairs have been successfully detected.\n";
    nimages = j;
    if( nimages < 2 )
    {
        cout << "Error: too little pairs to run the calibration\n";
		return;
    }

    imagePoints[0].resize(nimages);
    imagePoints[1].resize(nimages);
    objectPoints.resize(nimages);

    for( i = 0; i < nimages; i++ )
    {
        for( j = 0; j < boardSize.height; j++ )
            for( k = 0; k < boardSize.width; k++ )
                objectPoints[i].push_back(Point3f(k*squareSize, j*squareSize, 0));
    }

    cout << "Running stereo calibration ...\n";

	//IF WE DONT HAVE INITIAL CAMERA INTRINSIC AND EXTRINSIC GUESS
   /* Mat cameraMatrix[2], distCoeffs[2];
    cameraMatrix[0] = initCameraMatrix2D(objectPoints,imagePoints[0],imageSize,0);
    cameraMatrix[1] = initCameraMatrix2D(objectPoints,imagePoints[1],imageSize,0);*/

	// FACTORY CALIBRATION DATA 1280X720

	////720
	////left cam origin
	float o_left_cam_fx = 698.7140f;
	float o_left_cam_fy = 698.714f;
	float o_left_cam_cx = 627.074f;
	float o_left_cam_cy = 346.143f;
	float o_left_cam_k1 = -0.171376f;
	float o_left_cam_k2 = 0.0255988f;
	float o_left_cam_p1 = 0.f;
	float o_left_cam_p2 = 0.f;
	float o_left_cam_k3 = 0.f;

	//right cam origin
	float o_right_cam_fx = 699.066f;
	float o_right_cam_fy = 699.066f;
	float o_right_cam_cx = 621.208f;
	float o_right_cam_cy = 361.433f;
	float o_right_cam_k1 = -0.171887f;
	float o_right_cam_k2 = 0.0263327f;
	float o_right_cam_p1 = 0.f;
	float o_right_cam_p2 = 0.f;
	float o_right_cam_k3 = 0.f;

	// R & T origin
	float o_Tx = -120.f;
	float o_Ty = 0.f;
	float o_Tz = 0.f;
	float o_Rx = -0.00419028f;
	float o_CV = 0.0183696f;
	float o_Rz = 0.000714531f;

	////2k
	////left cam origin
	//float o_left_cam_fx = 1399.13f;
	//float o_left_cam_fy = 1399.13f;
	//float o_left_cam_cx = 1086.25f;
	//float o_left_cam_cy = 592.27f;
	//float o_left_cam_k1 = -0.169384f;
	//float o_left_cam_k2 = 0.0264289f;
	//float o_left_cam_p1 = 0.f;
	//float o_left_cam_p2 = 0.f;
	//float o_left_cam_k3 = 0.f;

	////right cam origin
	//float o_right_cam_fx = 1399.03f;
	//float o_right_cam_fy = 1399.03f;
	//float o_right_cam_cx = 1068.54f;
	//float o_right_cam_cy = 626.859f;
	//float o_right_cam_k1 = -0.167001f;
	//float o_right_cam_k2 = 0.0248225f;
	//float o_right_cam_p1 = 0.f;
	//float o_right_cam_p2 = 0.f;
	//float o_right_cam_k3 = 0.f;

	//// R & T origin
	//float o_Tx = -119.965f;
	//float o_Ty = 0.f;
	//float o_Tz = 0.f;
	//float o_Rx = 0.000160316f;
	//float o_CV = 0.0163856f;
	//float o_Rz = 0.000553166f;

	//1080
	////left cam origin
	//float o_left_cam_fx = 1399.13f;
	//float o_left_cam_fy = 1399.13f;
	//float o_left_cam_cx = 942.252f;
	//float o_left_cam_cy = 511.27f;
	//float o_left_cam_k1 = -0.169384f;
	//float o_left_cam_k2 = 0.0264289f;
	//float o_left_cam_p1 = 0.f;
	//float o_left_cam_p2 = 0.f;
	//float o_left_cam_k3 = 0.f;

	////right cam origin
	//float o_right_cam_fx = 1399.03f;
	//float o_right_cam_fy = 1399.03f;
	//float o_right_cam_cx = 924.544f;
	//float o_right_cam_cy = 545.859f;
	//float o_right_cam_k1 = -0.167001f;
	//float o_right_cam_k2 = 0.0248225f;
	//float o_right_cam_p1 = 0.f;
	//float o_right_cam_p2 = 0.f;
	//float o_right_cam_k3 = 0.f;

	//// R & T origin
	//float o_Tx = -119.965f;
	//float o_Ty = 0.f;
	//float o_Tz = 0.f;
	//float o_Rx = 0.000160316f;
	//float o_CV = 0.0163856f;
	//float o_Rz = 0.000553166f;


	Mat cameraMatrix[2], distCoeffs[2];
	cameraMatrix[0] = (cv::Mat_<double>(3, 3) << o_left_cam_fx, 0, o_left_cam_cx, 0, o_left_cam_fy, o_left_cam_cy, 0, 0, 1);
	cameraMatrix[1] = (cv::Mat_<double>(3, 3) << o_right_cam_fx, 0, o_right_cam_cx, 0, o_right_cam_fy, o_right_cam_cy, 0, 0, 1);
	distCoeffs[0] = (cv::Mat_<double>(5, 1) << o_left_cam_k1, o_left_cam_k2, o_left_cam_p1, o_left_cam_p2, o_left_cam_k3);
	distCoeffs[1] = (cv::Mat_<double>(5, 1) << o_right_cam_k1, o_right_cam_k2, o_right_cam_p1, o_right_cam_p2, o_right_cam_k3);

	Mat R, T;
	R = (cv::Mat_<double>(3, 1) << o_Rx, o_CV, o_Rz);
	T = (cv::Mat_<double>(3, 1) << o_Tx, o_Ty, o_Tz);

	Rodrigues(R /*in*/, R /*out*/);

	// save initial intrinsic parameters
	FileStorage fs("ini_intrinsics_720.yml", FileStorage::WRITE);
	if (fs.isOpened())
	{
		fs << "M1" << cameraMatrix[0] << "D1" << distCoeffs[0] <<
			"M2" << cameraMatrix[1] << "D2" << distCoeffs[1]<<"T"<<T<<"R"<<R;
		fs.release();
	}
	else
		cout << "Error: can not save the ini intrinsic parameters\n";

    Mat E, F, P;
	vector<vector<Point>> perview;

    double rms = stereoCalibrate(objectPoints, imagePoints[0], imagePoints[1],
                    cameraMatrix[0], distCoeffs[0],
                    cameraMatrix[1], distCoeffs[1],
                    imageSize, R, T, E, F, P,
		            CALIB_USE_INTRINSIC_GUESS + CALIB_USE_EXTRINSIC_GUESS,
                    TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 10000, 1e-5) );

	//compute the X Y residual of reprojection error of each chessboard view and each chessboard point

	vector<vector<Point2f>> rp_imagePoints[2];
	vector<vector<Point2f>> rs_imagePoints[2];
	rp_imagePoints[0].resize(nimages);
	rp_imagePoints[1].resize(nimages);
	rs_imagePoints[0].resize(nimages);
	rs_imagePoints[1].resize(nimages);
	Mat Rvec, Tvec;
	double er = 0;
	double n_tol = 0;

	for (i = 0; i < nimages; i++)
	{

		for (k = 0; k < 2; k++) {

			int npt = (int)imagePoints[k][i].size();
			for (j = 0; j < npt; j++) {
				solvePnP(objectPoints[i], imagePoints[k][i], cameraMatrix[k], distCoeffs[k], Rvec, Tvec);
				projectPoints(objectPoints[i], Rvec, Tvec, cameraMatrix[k], distCoeffs[k], rp_imagePoints[k][i]);
				rs_imagePoints[k][i].push_back(Point2f(rp_imagePoints[k][i][j].x - imagePoints[k][i][j].x, rp_imagePoints[k][i][j].y - imagePoints[k][i][j].y));
				
				double erp = (rs_imagePoints[k][i][j].x*rs_imagePoints[k][i][j].x + rs_imagePoints[k][i][j].y*rs_imagePoints[k][i][j].y);
				er += erp;
				n_tol ++;
			}
		}
		
	}
	er = sqrt( er / n_tol );

	//store reprojection xy residual points data
	ofstream myfile("reprojection xy residual points 720.txt");
	if (myfile.is_open())
	{
		for (i=0; i< nimages; i++)
		{
			for (k=0; k<2; k++)
			{
				myfile << "IMG" << k << i << "\n";
				int npt = (int)imagePoints[k][i].size();
				for (j = 0; j < npt; j++) {
					if (k == 0) {
						myfile << rs_imagePoints[k][i][j].x << " , " << rs_imagePoints[k][i][j].y << " , " << sqrt(rs_imagePoints[k][i][j].x*rs_imagePoints[k][i][j].x +
							rs_imagePoints[k][i][j].y*rs_imagePoints[k][i][j].y) << " , " << i << "\n";
					}
					if (k == 1) {
						myfile << rs_imagePoints[k][i][j].x << " , " << rs_imagePoints[k][i][j].y << " , " << sqrt(rs_imagePoints[k][i][j].x*rs_imagePoints[k][i][j].x +
							rs_imagePoints[k][i][j].y*rs_imagePoints[k][i][j].y) << " , " << i << "\n";
					}
				}
			}
		}
		myfile.close();
	}
	else
		cout << "Error: can not save the data\n";
	
	//SAVE EACH IMAGE'S REPROJECTION ERROR
	ofstream myfile1("720 perview reprojection error.txt");
	if (myfile1.is_open())
	{
		for (i = 0; i < nimages; i++)
		{
			for (k = 0; k < 2; k++)
			{
				myfile1 << "IMG " << k << i << " error " << P.at<double>(i, k) << "\n";
			}
		}
		myfile1.close();
	}
	else
		cout << "Error: can not save the data\n";
	
    cout << "done with RMS error=" << rms << " computed RMS error=" << er << endl;

// CALIBRATION QUALITY CHECK
// because the output fundamental matrix implicitly
// includes all the output information,
// we can check the quality of calibration using the
// epipolar geometry constraint: m2^t*F*m1=0
    double err = 0;
    int npoints = 0;
    vector<Vec3f> lines[2];
    for( i = 0; i < nimages; i++ )
    {
        int npt = (int)imagePoints[0][i].size();
        Mat imgpt[2];
        for( k = 0; k < 2; k++ )
        {
            imgpt[k] = Mat(imagePoints[k][i]);
            undistortPoints(imgpt[k], imgpt[k], cameraMatrix[k], distCoeffs[k], Mat(), cameraMatrix[k]);
            computeCorrespondEpilines(imgpt[k], k+1, F, lines[k]);
        }

		

        for( j = 0; j < npt; j++ )
        {
            double errij = fabs(imagePoints[0][i][j].x*lines[1][j][0] +
                                imagePoints[0][i][j].y*lines[1][j][1] + lines[1][j][2]) +
                           fabs(imagePoints[1][i][j].x*lines[0][j][0] +
                                imagePoints[1][i][j].y*lines[0][j][1] + lines[0][j][2]);

			

            err += errij;
        }
        npoints += npt;

		if (i == 10) {

			Mat leftepi = imread("C:/Users/CSK/source/repos/stereo_cali_param/stereo_cali_param/v2_left30.png", IMREAD_GRAYSCALE);
			Mat rightepi = imread("C:/Users/CSK/source/repos/stereo_cali_param/stereo_cali_param/v2_right30.png", IMREAD_GRAYSCALE);
			//Mat undisleftepi, undisrightepi;
			//undistort(leftepi, undisleftepi, cameraMatrix[0], distCoeffs[0]);
			//undistort(rightepi, undisrightepi, cameraMatrix[1], distCoeffs[1]);
			
			for (int a = 0; a < npt; a++)
			{

				line(rightepi, Point2f(0, -lines[0][a][2] / lines[0][a][1]), Point2f((float)rightepi.cols, -(lines[0][a][2] + lines[0][a][0] * (float)rightepi.cols)) / lines[0][a][1], Scalar(0, 0, 255), 1, 8);
				line(leftepi, Point2f(0, -lines[1][a][2] / lines[1][a][1]), Point2f((float)leftepi.cols, -(lines[1][a][2] + lines[1][a][0] * (float)leftepi.cols)) / lines[1][a][1], Scalar(0, 0, 255), 1, 8);

			}
			imshow("leftepi", leftepi);
			imshow("rightepi", rightepi);
			
		}
		

    }
    cout << "average epipolar err = " <<  err/npoints << endl;

    // save intrinsic parameters
    fs.open("intrinsics720.yml", FileStorage::WRITE);
    if( fs.isOpened() )
    {
        fs << "M1" << cameraMatrix[0] << "D1" << distCoeffs[0] <<
            "M2" << cameraMatrix[1] << "D2" << distCoeffs[1]<< "rms error" << rms << "avg epipolar err" << err/npoints << "perview error" << P << "image resolution" << imageSize << "F" << F;
        fs.release();
    }
    else
        cout << "Error: can not save the intrinsic parameters\n";

    Mat R1, R2, P1, P2, Q;
    Rect validRoi[2];

    stereoRectify(cameraMatrix[0], distCoeffs[0],
                  cameraMatrix[1], distCoeffs[1],
                  imageSize, R, T, R1, R2, P1, P2, Q,
                  CALIB_ZERO_DISPARITY, 1, imageSize, &validRoi[0], &validRoi[1]);

	cout << "validRoi\n\n"<<validRoi[0]<<"\n"<<validRoi[1];

	Mat RT;
	Rodrigues(R /*in*/, RT /*out*/);

	// save extrinsic parameters
    fs.open("extrinsics720.yml", FileStorage::WRITE);
    if( fs.isOpened() )
    {
        fs << "R" << R << "T" << T << "R1" << R1 << "R2" << R2 << "P1" << P1 << "P2" << P2 << "Q" << Q<< "RT" << RT << "validRoi left" << validRoi[0] << "validRoi right" << validRoi[1];
        fs.release();
    }
    else
        cout << "Error: can not save the extrinsic parameters\n";

    // OpenCV can handle left-right
    // or up-down camera arrangements
    //bool isVerticalStereo = fabs(P2.at<double>(1, 3)) > fabs(P2.at<double>(0, 3));
	bool isVerticalStereo = false;

// COMPUTE AND DISPLAY RECTIFICATION
    if( !showRectified )
		cout << "showRectified = false\n";;

    Mat rmap[2][2];

// IF BY CALIBRATED (BOUGUET'S METHOD)
    if( useCalibrated )
    {
        // we already computed everything
    }
// OR ELSE HARTLEY'S METHOD
    else
 // use intrinsic parameters of each camera, but
 // compute the rectification transformation directly
 // from the fundamental matrix
    {
        vector<Point2f> allimgpt[2];
        for( k = 0; k < 2; k++ )
        {
            for( i = 0; i < nimages; i++ )
                std::copy(imagePoints[k][i].begin(), imagePoints[k][i].end(), back_inserter(allimgpt[k]));
        }
        F = findFundamentalMat(Mat(allimgpt[0]), Mat(allimgpt[1]), FM_8POINT, 0, 0);
        Mat H1, H2;
        stereoRectifyUncalibrated(Mat(allimgpt[0]), Mat(allimgpt[1]), F, imageSize, H1, H2, 3);

        R1 = cameraMatrix[0].inv()*H1*cameraMatrix[0];
        R2 = cameraMatrix[1].inv()*H2*cameraMatrix[1];
        P1 = cameraMatrix[0];
        P2 = cameraMatrix[1];
    }

    //Precompute maps for cv::remap()
    initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize, CV_32FC1, rmap[0][0], rmap[0][1]);
    initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize, CV_32FC1, rmap[1][0], rmap[1][1]);

	//Mat rmap_c00, rmap_c10, rmap_c01, rmap_c11;
	rmap_c00 = rmap[0][0].clone();
	rmap_c01 = rmap[0][1].clone();
	rmap_c10 = rmap[1][0].clone();
	rmap_c11 = rmap[1][1].clone();
	Q_c = Q.clone();
	
	Mat canvas;
	if (!isVerticalStereo)
	{
		canvas.create(imageSize.height, imageSize.width * 2, CV_8UC3);
	}
	else
	{
	}

	for (i = 0; i < nimages; i++)
	{
		for (k = 0; k < 2; k++)
		{
			Mat img = imread("C:/Users/CSK/source/repos/stereo_cali_param/stereo_cali_param/stereo_calib2/1M~3M/" + goodImageList[i * 2 + k], 0), rimg, cimg;
			remap(img, rimg, rmap[k][0], rmap[k][1], INTER_LINEAR);
			//imshow("r", rimg);
			cvtColor(rimg, cimg, COLOR_GRAY2BGR,3);

			//save rectified image

			/*if (k == 0 && i == 10 ) {
				stringstream l_name;
				l_name << "rectified_left" << i << ".png";
				cimg = cimg(cv::Rect(57, 82, 1148, 550));
				imwrite(l_name.str(), cimg);
			}

			if (k == 1 && i == 10 ) {
				stringstream r_name;
				r_name << "rectified_right" << i << ".png";
				cimg = cimg(cv::Rect(57, 82, 1148, 550));
				imwrite(r_name.str(), cimg);
			}*/

			Mat canvasPart = canvas(Rect(imageSize.width*k, 0, imageSize.width, imageSize.height));
			resize(cimg, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);
			//imshow("raw", img);
			if (useCalibrated)
			{
				Rect vroi(cvRound(validRoi[k].x), cvRound(validRoi[k].y),
					cvRound(validRoi[k].width), cvRound(validRoi[k].height));
				rectangle(canvasPart, vroi, Scalar(255, 0, 0), 3, 8);
			}
		}

		if (!isVerticalStereo)
			for (j = 0; j < canvas.rows; j += 16)
				line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 0, 255), 1, 8);
		else
			for (j = 0; j < canvas.cols; j += 16)
				line(canvas, Point(j, 0), Point(j, canvas.rows), Scalar(0, 255, 0), 1, 8);

		imshow("rectified", canvas);
		/*if (i == 10 || i == 21 || i == 44) {
			stringstream l_name;
			l_name << "rectified_leftright" << i << ".png";
			imwrite(l_name.str(), canvas);
		}*/
		imwrite("rectified-pair 2k.png", canvas);
		char c = (char)waitKey(500);
		if (c == 27 || c == 'q' || c == 'Q')
			break;
	}
}


static bool readStringList( const string& filename, vector<string>& l )
{
    l.resize(0);
    FileStorage fs(filename, FileStorage::READ);
    if( !fs.isOpened() )
        return false;
    FileNode n = fs.getFirstTopLevelNode();
    if( n.type() != FileNode::SEQ )
        return false;
    FileNodeIterator it = n.begin(), it_end = n.end();
    for( ; it != it_end; ++it )
        l.push_back((string)*it);
    return true;
}

int main(int argc, char** argv)
{
    Size boardSize;
    string imagelistfn;
    bool showRectified;
   // cv::CommandLineParser parser(argc, argv, "{w|9|}{h|6|}{s|1.0|}{nr||}{help||}{@input|../data/stereo_calib.xml|}");
    //if (parser.has("help"))
        //return print_help();
    //showRectified = !parser.has("nr");
	showRectified = true;
	imagelistfn = "C:/Users/CSK/source/repos/stereo_cali_param/stereo_cali_param/stereo_calib2/stereo_calib2.xml";
	boardSize.width = 9;
	boardSize.height = 6;
	float squareSize = 100.0;
    //imagelistfn = parser.get<string>("@input");
    //boardSize.width = parser.get<int>("w");
    //boardSize.height = parser.get<int>("h");
    //float squareSize = parser.get<float>("s");
    /*if (!parser.check())
    {
        parser.printErrors();
        return 1;
    }*/
    vector<string> imagelist;
    bool ok = readStringList(imagelistfn, imagelist);
    if(!ok || imagelist.empty())
    {
        cout << "can not open " << imagelistfn << " or the string list is empty" << endl;
        return print_help();
    }

	Mat rmap00, rmap01, rmap10, rmap11, Q;

    StereoCalib(imagelist, boardSize, squareSize, rmap00, rmap01, rmap10, rmap11, Q, false, true, showRectified);



	// calibrate and rectify the camera and capture
	//video resolution [2k:15fps 1080p:30fps 720p:60fps wvga:100fps]
	cv::Size2i image_size = cv::Size2i(2208, 1242);

	//open the camera
	VideoCapture cap(1);
	if (!cap.isOpened())
		return -1;
	cap.grab();

	// Set the video resolution (2*Width * Height)
	cap.set(CV_CAP_PROP_FRAME_WIDTH, image_size.width * 2);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, image_size.height);
	cap.grab();

	Mat frame, left_raw, right_raw, left_rectified, right_rectified, left_cropped, right_cropped;

	char capture = 'r';

	cout << "Image resolution:" << image_size << endl;

	//capture image
	cout << "Press 'c' to capture ..." << endl;

	int Icount = 0;

	//timer
	/*time_t start, end;
	double elapsed;
	start = time(NULL);
	int frame_count = 0;*/

	while (capture != 'q') {

		//timer
		/*end = time(NULL);
		elapsed = difftime(end, start);
		frame_count += 1;
		cout << "fps = " << (double)frame_count/elapsed << endl;*/


		// Get a new frame from camera
		cap >> frame;
		// Extract left and right images from side-by-side
		left_raw = frame(cv::Rect(0, 0, frame.cols / 2, frame.rows));
		right_raw = frame(cv::Rect(frame.cols / 2, 0, frame.cols / 2, frame.rows));

		remap(left_raw, left_rectified, rmap00, rmap01, INTER_LINEAR);
		remap(right_raw, right_rectified, rmap10, rmap11, INTER_LINEAR);

		left_cropped = left_rectified.clone();
		right_cropped = right_rectified.clone();

		left_cropped = left_cropped(cv::Rect(57, 82, 1148, 550));
		right_cropped = right_cropped(cv::Rect(57, 82, 1148, 550));

		double C_x = -1 * Q.at<double>(0, 3) - 57;
		double C_y = -1 * Q.at<double>(1, 3) - 82;
		double B_v = left_cropped.rows * 1 / 4;

		line(left_cropped, Point2d(C_x, 0), Point2d(C_x, left_cropped.rows), Scalar(0, 0, 255), 2, 8);
		line(left_cropped, Point2d(0, C_y), Point2d(left_cropped.cols, C_y), Scalar(0, 0, 255), 2, 8);
		line(right_cropped, Point2d(C_x, 0), Point2d(C_x, right_cropped.rows), Scalar(0, 0, 255), 2, 8);
		line(right_cropped, Point2d(0, C_y), Point2d(right_cropped.cols, C_y), Scalar(0, 0, 255), 2, 8);
		line(left_cropped, Point2d(0, B_v), Point2d(left_cropped.cols, B_v), Scalar(0, 255, 0), 2, 8);
		line(right_cropped, Point2d(0, B_v), Point2d(right_cropped.cols, B_v), Scalar(0, 255, 0), 2, 8);

		imshow("left_cropped", left_cropped);
		imshow("right_cropped", right_cropped);

		imshow("left_rectified", left_rectified);
		imshow("right_rectified", right_rectified);

		/*Mat left_g, right_g, left_eq, right_eq;
		
		cvtColor(left_cropped, left_g, COLOR_BGR2GRAY);
		cvtColor(right_cropped, right_g, COLOR_BGR2GRAY);

		equalizeHist(left_g, left_eq);
		equalizeHist(right_g, right_eq);

		imshow("left_cropped", left_eq);
		imshow("right_cropped", right_eq);

		imshow("left", left_g);
		imshow("right", right_g);*/


		if (capture == 'c') {

			//save files at proper locations if user presses 'c'
			stringstream l_name, r_name, lr_name, rr_name;
			l_name << "left_road" << Icount << ".png";
			r_name << "right_road" << Icount << ".png";
			lr_name << "left_road_distance" << Icount << ".png";
			rr_name << "right_road_distance" << Icount << ".png";

			imwrite(l_name.str(), left_rectified);
			imwrite(r_name.str(), right_rectified);
			imwrite(lr_name.str(), left_cropped);
			imwrite(rr_name.str(), right_cropped);
			cout << "Saved set" << Icount << endl;
			Icount++;
		}

		if (capture == 'd')
		{
			imwrite("car_distance_test_left.png", left_cropped);
			imwrite("car_distance_test_right.png", right_cropped);
		}


		capture = waitKey(1);
	}
	cap.release();
	return 0;
}
