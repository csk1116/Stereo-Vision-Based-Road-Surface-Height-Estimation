#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2\viz.hpp"
#include "opencv2/core/matx.hpp"



#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <fstream>

using namespace cv;
using namespace std;


static void StereoCalib(const vector<string>& imagelist, Size boardSize, float squareSize, Mat& rmap_c00, Mat& rmap_c01, Mat& rmap_c10, Mat& rmap_c11,
	bool displayCorners = false, bool useCalibrated = true, bool showRectified = true)
{
	if (imagelist.size() % 2 != 0)
	{
		cout << "Error: the image list contains odd (non-even) number of elements\n";
		return;
	}

	const int maxScale = 2;

	// ARRAY AND VECTOR STORAGE:
	vector<vector<Point2f> > imagePoints[2];
	vector<vector<Point3f> > objectPoints;
	Size imageSize;

	int i, j, k, nimages = (int)imagelist.size() / 2;

	imagePoints[0].resize(nimages);
	imagePoints[1].resize(nimages);
	vector<string> goodImageList;

	for (i = j = 0; i < nimages; i++)
	{
		for (k = 0; k < 2; k++)
		{
			const string& filename = imagelist[i * 2 + k];
			Mat img = imread(filename, IMREAD_GRAYSCALE);
			//cout << "channels" << img.channels() << endl;
			if (img.empty())
				break;
			if (imageSize == Size())
				imageSize = img.size();
			else if (img.size() != imageSize)
			{
				cout << "The image " << filename << " has the size different from the first image size. Skipping the pair\n";
				break;
			}
			bool found = false;
			vector<Point2f>& corners = imagePoints[k][j];
			for (int scale = 1; scale <= maxScale; scale++)
			{
				Mat timg;
				if (scale == 1)
					timg = img;
				else
					resize(img, timg, Size(), scale, scale, INTER_LINEAR_EXACT);
				found = findChessboardCorners(timg, boardSize, corners,
					CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
				if (found)
				{
					if (scale > 1)
					{
						Mat cornersMat(corners);
						cornersMat *= 1. / scale;
					}
					break;
				}
			}
			if (displayCorners)
			{
				cout << filename << endl;
				Mat cimg, cimg1;
				cvtColor(img, cimg, COLOR_GRAY2BGR);
				drawChessboardCorners(cimg, boardSize, corners, found);
				//double sf = 640./MAX(img.rows, img.cols);
				//resize(cimg, cimg1, Size(), sf, sf, INTER_LINEAR_EXACT);
				imshow("corners", cimg);
				char c = (char)waitKey(500);
				if (c == 27 || c == 'q' || c == 'Q') //Allow ESC to quit
					exit(-1);
			}
			else
				putchar('.');
			if (!found)
				break;

			cornerSubPix(img, corners, Size(15, 15), Size(-1, -1),
				TermCriteria(TermCriteria::COUNT + TermCriteria::EPS,
					10000, 1e-5));
		}
		if (k == 2)
		{
			goodImageList.push_back(imagelist[i * 2]);
			goodImageList.push_back(imagelist[i * 2 + 1]);
			j++;
		}
	}
	cout << j << " pairs have been successfully detected.\n";
	nimages = j;
	if (nimages < 2)
	{
		cout << "Error: too little pairs to run the calibration\n";
		return;
	}

	imagePoints[0].resize(nimages);
	imagePoints[1].resize(nimages);
	objectPoints.resize(nimages);

	for (i = 0; i < nimages; i++)
	{
		for (j = 0; j < boardSize.height; j++)
			for (k = 0; k < boardSize.width; k++)
				objectPoints[i].push_back(Point3f(k*squareSize, j*squareSize, 0));
	}

	cout << "Running stereo calibration ...\n";


	// FACTORY CALIBRATION DATA 1280X720

	//left cam origin
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


	Mat cameraMatrix[2], distCoeffs[2];
	cameraMatrix[0] = (cv::Mat_<double>(3, 3) << o_left_cam_fx, 0, o_left_cam_cx, 0, o_left_cam_fy, o_left_cam_cy, 0, 0, 1);
	cameraMatrix[1] = (cv::Mat_<double>(3, 3) << o_right_cam_fx, 0, o_right_cam_cx, 0, o_right_cam_fy, o_right_cam_cy, 0, 0, 1);
	distCoeffs[0] = (cv::Mat_<double>(5, 1) << o_left_cam_k1, o_left_cam_k2, o_left_cam_p1, o_left_cam_p2, o_left_cam_k3);
	distCoeffs[1] = (cv::Mat_<double>(5, 1) << o_right_cam_k1, o_right_cam_k2, o_right_cam_p1, o_right_cam_p2, o_right_cam_k3);

	Mat R, T;
	R = (cv::Mat_<double>(3, 1) << o_Rx, o_CV, o_Rz);
	T = (cv::Mat_<double>(3, 1) << o_Tx, o_Ty, o_Tz);

	Rodrigues(R /*in*/, R /*out*/);

	Mat E, F, P;

	double rms = stereoCalibrate(objectPoints, imagePoints[0], imagePoints[1],
		cameraMatrix[0], distCoeffs[0],
		cameraMatrix[1], distCoeffs[1],
		imageSize, R, T, E, F, P,
		CALIB_USE_INTRINSIC_GUESS + CALIB_USE_EXTRINSIC_GUESS,
		TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 10000, 1e-5));


	// CALIBRATION QUALITY CHECK
	// because the output fundamental matrix implicitly
	// includes all the output information,
	// we can check the quality of calibration using the
	// epipolar geometry constraint: m2^t*F*m1=0
	double err = 0;
	int npoints = 0;
	vector<Vec3f> lines[2];
	for (i = 0; i < nimages; i++)
	{
		int npt = (int)imagePoints[0][i].size();
		Mat imgpt[2];
		for (k = 0; k < 2; k++)
		{
			imgpt[k] = Mat(imagePoints[k][i]);
			undistortPoints(imgpt[k], imgpt[k], cameraMatrix[k], distCoeffs[k], Mat(), cameraMatrix[k]);
			computeCorrespondEpilines(imgpt[k], k + 1, F, lines[k]);
		}
		for (j = 0; j < npt; j++)
		{
			double errij = fabs(imagePoints[0][i][j].x*lines[1][j][0] +
				imagePoints[0][i][j].y*lines[1][j][1] + lines[1][j][2]) +
				fabs(imagePoints[1][i][j].x*lines[0][j][0] +
					imagePoints[1][i][j].y*lines[0][j][1] + lines[0][j][2]);
			err += errij;
		}
		npoints += npt;
	}
	cout << "average epipolar err = " << err / npoints << endl;


	Mat R1, R2, P1, P2, Q;
	Rect validRoi[2];

	stereoRectify(cameraMatrix[0], distCoeffs[0],
		cameraMatrix[1], distCoeffs[1],
		imageSize, R, T, R1, R2, P1, P2, Q,
		CALIB_ZERO_DISPARITY, 1, imageSize, &validRoi[0], &validRoi[1]);

	cout << Q.rows << Q.cols << Q << Q.type() << Q.isContinuous() << endl;

	Mat rmap[2][2];

	//Precompute maps for cv::remap()
	initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize, CV_32FC1, rmap[0][0], rmap[0][1]);
	initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize, CV_32FC1, rmap[1][0], rmap[1][1]);

	//Mat rmap_c00, rmap_c10, rmap_c01, rmap_c11;
	rmap_c00 = rmap[0][0].clone();
	rmap_c01 = rmap[0][1].clone();
	rmap_c10 = rmap[1][0].clone();
	rmap_c11 = rmap[1][1].clone();
}


static void VarifyCalib(const vector<string>& imagelist_v, Size boardSize, Mat rmap_00, Mat rmap_01, Mat rmap_10, Mat rmap_11,
	bool displayCorners = false, bool displayRectify =false, bool display3D = false)
{
	if (imagelist_v.size() % 2 != 0)
	{
		cout << "Error: the image list contains odd (non-even) number of elements\n";
		return;
	}

	const int maxScale = 2;

	// ARRAY AND VECTOR STORAGE:
	vector<vector<Point2f> > imagePoints_v[2];
	Size imageSize_v;

	int i, j, k, nimages = (int)imagelist_v.size() / 2;

	imagePoints_v[0].resize(nimages);
	imagePoints_v[1].resize(nimages);
	vector<string> goodImageList_v;

	for (i = j = 0; i < nimages; i++)
	{
		for (k = 0; k < 2; k++)
		{
			const string& filename = imagelist_v[i * 2 + k];
			Mat img = imread(filename, IMREAD_GRAYSCALE);
			
			if (img.empty())
				break;
			
			//calibrate & rectify
			if (k == 0)
				remap(img, img, rmap_00, rmap_01, INTER_LINEAR);
			if (k == 1)
				remap(img, img, rmap_10, rmap_11, INTER_LINEAR);

			img = img(cv::Rect(57, 82, 1148, 550));

			if (imageSize_v == Size())
				imageSize_v = img.size();
			else if (img.size() != imageSize_v)
			{
				cout << "The image " << filename << " has the size different from the first image size. Skipping the pair\n";
				break;
			}

			bool found = false;
			vector<Point2f>& corners = imagePoints_v[k][j];
			for (int scale = 1; scale <= maxScale; scale++)
			{
				Mat timg;
				if (scale == 1)
					timg = img;
				else
					resize(img, timg, Size(), scale, scale, INTER_LINEAR_EXACT);
				found = findChessboardCorners(timg, boardSize, corners,
					CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
				if (found)
				{
					if (scale > 1)
					{
						Mat cornersMat(corners);
						cornersMat *= 1. / scale;
					}
					break;
				}
			}


			if (displayCorners)
			{
				cout << filename << endl;
				Mat cimg, cimg1;
				cvtColor(img, cimg, COLOR_GRAY2BGR);
				drawChessboardCorners(cimg, boardSize, corners, found);
				imshow("corners", cimg);
				char c = (char)waitKey(500);
				if (c == 27 || c == 'q' || c == 'Q') //Allow ESC to quit
					exit(-1);
			}
			else
				putchar('.');
			if (!found)
				break;

			cornerSubPix(img, corners, Size(15, 15), Size(-1, -1),
				TermCriteria(TermCriteria::COUNT + TermCriteria::EPS,
					10000, 1e-5));
		}
		if (k == 2)
		{
			goodImageList_v.push_back(imagelist_v[i * 2]);
			goodImageList_v.push_back(imagelist_v[i * 2 + 1]);
			j++;
		}

		

	}
	
	cout << j << " verified pairs have been successfully detected.\n";

	nimages = j;

	imagePoints_v[0].resize(nimages);
	imagePoints_v[1].resize(nimages);

	cout << "verifying ...\n";

	//load projection matrix Q
	FileStorage fs("C:/Users/CSK/source/repos/verify_calib/verify_calib/extrinsics.yml", FileStorage::READ);
	Mat Q, C;
	Matx33d Cam;
	fs["Q"] >> Q;
	fs["P1"] >> C;
	//-1/Tx = Q(4,3), f = Q(3,4), -Cx = Q(1,4) + 57, -Cy = Q(2,4) + 82
	C.at<double>(0, 2) -= 57;
	C.at<double>(1, 2) -= 82;
	C = C(cv::Rect(0, 0, 3, 3));
	C.convertTo(C, CV_64FC1);
	Cam = C;
	cout << C.size() << endl;
	double Qt = Q.at<double>(3, 2);
	double f = Q.at<double>(2, 3);
	double nCx = Q.at<double>(0, 3) + 57;
	double nCy = Q.at<double>(1, 3) + 82;
	//cout<< Q.at<double>(3, 2);

	//compute the X Y residual of reprojection error of each chessboard view and each chessboard point

	double er = 0;
	double n_tol = 0;

	for (i = 0; i < nimages; i++)
	{
			int npt = (int)imagePoints_v[0][i].size();
			for (j = 0; j < npt; j++) {
				double erp = fabs(imagePoints_v[0][i][j].y - imagePoints_v[1][i][j].y);
				er += erp;
				n_tol++;
			}
	}
	er = er / n_tol;

	


	//store rectification xy points data
	ofstream myfile("verify calibration and rectification results.txt");
	if (myfile.is_open())
	{
		for (i = 0; i < nimages; i++)
		{
				myfile << "IMG " << i << "\n" << " left_x, left_y, right_x, right_y, disparity, L_Y - R_Y, Depth " << "\n";
				int npt = (int)imagePoints_v[0][i].size();
				for (j = 0; j < npt; j++) {
					myfile << imagePoints_v[0][i][j].x << " , " << imagePoints_v[0][i][j].y << " , " << imagePoints_v[1][i][j].x << " , " << imagePoints_v[1][i][j].y
						<< " , " << imagePoints_v[0][i][j].x - imagePoints_v[1][i][j].x << " , " << imagePoints_v[0][i][j].y - imagePoints_v[1][i][j].y 
						<< " , " << f/((imagePoints_v[0][i][j].x - imagePoints_v[1][i][j].x) * Qt) << "\n";
				}
		}
		myfile << " AVG_ER = " << er;
		myfile.close();
	}
	else
		cout << "Error: can not save the data\n";

	//graph for depth vs disparity
	myfile.open("depth vs disparity 2k.txt");
	if (myfile.is_open())
	{
		myfile << " depth, " << " disparity " << "\n";
		for (i = 1; i < 15000; i++)
		{
			if ((f*2 / (i*Qt)) >= 1000 && (f*2 / (i*Qt)) <= 20000)
			{
				myfile << f*2 / (i*Qt) << " , " << i << "\n";
			}
			
		}
		myfile.close();
	}
	else
		cout << "Error: can not save the data\n";
	
	if (displayRectify) {
		Mat canvas;
		canvas.create(imageSize_v.height, imageSize_v.width * 2, CV_8UC3);


		for (i = 0; i < nimages; i++)
		{
			for (k = 0; k < 2; k++)
			{
				Mat img = imread(goodImageList_v[i * 2 + k], 0), cimg;

				if (k == 0)
					remap(img, img, rmap_00, rmap_01, INTER_LINEAR);
				if (k == 1)
					remap(img, img, rmap_10, rmap_11, INTER_LINEAR);

				img = img(cv::Rect(57, 82, 1148, 550));

				cvtColor(img, cimg, COLOR_GRAY2BGR, 3);

				Mat canvasPart = canvas(Rect(imageSize_v.width*k, 0, imageSize_v.width, imageSize_v.height));
				resize(cimg, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);

				/*if (useCalibrated)
				{
					Rect vroi(cvRound(validRoi[k].x), cvRound(validRoi[k].y),
						cvRound(validRoi[k].width), cvRound(validRoi[k].height));
					rectangle(canvasPart, vroi, Scalar(0, 0, 255), 3, 8);
				}*/
			}



			int npt = (int)imagePoints_v[0][i].size();
			for (j = 0; j < npt; j++)
			{
				line(canvas, Point2f(imagePoints_v[0][i][j].x, imagePoints_v[0][i][j].y), Point2f(imagePoints_v[1][i][j].x + imageSize_v.width, imagePoints_v[1][i][j].y)
					, Scalar(0, 0, 255), 1, 8);

			}


			if (i == 10 || i == 21 || i == 44) {
				stringstream l_name;
				l_name << "verify_leftright" << i << ".png";
				imwrite(l_name.str(), canvas);
			}

			imshow("verified", canvas);
			//imwrite("verified.png", canvas);
			char c = (char)waitKey(500);
			if (c == 27 || c == 'q' || c == 'Q')
				break;
		}
	}

	if (display3D)
	{
		vector<vector<Point3d> > chessboard3dPoints;
		chessboard3dPoints.resize(nimages);
		// 3d points of chessboard corner
		for (i = 0; i < nimages; i++)
		{
			int npt = (int)imagePoints_v[0][i].size();
			for (j = 0; j < npt; j++) {
				chessboard3dPoints[i].push_back(Point3d((imagePoints_v[0][i][j].x + nCx) / ((imagePoints_v[0][i][j].x - imagePoints_v[1][i][j].x) * Qt),
					(imagePoints_v[0][i][j].y + nCy) / ((imagePoints_v[0][i][j].x - imagePoints_v[1][i][j].x) * Qt),
					f / ((imagePoints_v[0][i][j].x - imagePoints_v[1][i][j].x) * Qt)));
			}
			
		}

		Mat dis_img = imread("v2_left105.png", 0);
		remap(dis_img, dis_img, rmap_00, rmap_01, INTER_LINEAR);
		dis_img = dis_img(cv::Rect(57, 82, 1148, 550));
		viz::Viz3d myWindow("Coordinate Frame");
		myWindow.setBackgroundColor(viz::Color::black());
		myWindow.showWidget("Coordinate Widget", viz::WCoordinateSystem(400));
		//Vec3d cam_pos(0.0, 0.0, 0.0), cam_focal(0.0, 0.0, 0.0), cam_z_dir(0.0, -1.0, 0.0);
		//Affine3d cam_pose = viz::makeCameraPose(cam_pos, cam_focal, cam_z_dir);
		viz::WCameraPosition cam(Cam, dis_img, 300, viz::Color::white());
		myWindow.showWidget("Camera", cam);
		//Affine3d transform = viz::makeTransformToGlobal(Vec3f(0.0f, -1.0f, 0.0f), Vec3f(-1.0f, 0.0f, 0.0f), Vec3f(0.0f, 0.0f, -1.0f));
		
		//horizontal
		viz::WCloud cloud_widget(chessboard3dPoints[44], viz::Color::white());
		const Point3d h_start1(chessboard3dPoints[44][0]);
		const Point3d h_end1(chessboard3dPoints[44][8]);
		viz::WArrow h_arrow1(h_start1, h_end1, 0.001, viz::Color::gray());
		const Point3d h_start2(chessboard3dPoints[44][9]);
		const Point3d h_end2(chessboard3dPoints[44][17]);
		viz::WArrow h_arrow2(h_start2, h_end2, 0.001, viz::Color::gray());
		const Point3d h_start3(chessboard3dPoints[44][18]);
		const Point3d h_end3(chessboard3dPoints[44][26]);
		viz::WArrow h_arrow3(h_start3, h_end3, 0.001, viz::Color::gray());
		const Point3d h_start4(chessboard3dPoints[44][27]);
		const Point3d h_end4(chessboard3dPoints[44][35]);
		viz::WArrow h_arrow4(h_start4, h_end4, 0.001, viz::Color::gray());
		const Point3d h_start5(chessboard3dPoints[44][36]);
		const Point3d h_end5(chessboard3dPoints[44][44]);
		viz::WArrow h_arrow5(h_start5, h_end5, 0.001, viz::Color::gray());
		const Point3d h_start6(chessboard3dPoints[44][45]);
		const Point3d h_end6(chessboard3dPoints[44][53]);
		viz::WArrow h_arrow6(h_start6, h_end6, 0.001, viz::Color::gray());
		//vertical
		const Point3d v_start1(chessboard3dPoints[44][0]);
		const Point3d v_end1(chessboard3dPoints[44][45]);
		viz::WArrow v_arrow1(v_start1, v_end1, 0.001, viz::Color::gray());
		const Point3d v_start2(chessboard3dPoints[44][1]);
		const Point3d v_end2(chessboard3dPoints[44][46]);
		viz::WArrow v_arrow2(v_start2, v_end2, 0.001, viz::Color::gray());
		const Point3d v_start3(chessboard3dPoints[44][2]);
		const Point3d v_end3(chessboard3dPoints[44][47]);
		viz::WArrow v_arrow3(v_start3, v_end3, 0.001, viz::Color::gray());
		const Point3d v_start4(chessboard3dPoints[44][3]);
		const Point3d v_end4(chessboard3dPoints[44][48]);
		viz::WArrow v_arrow4(v_start4, v_end4, 0.001, viz::Color::gray());
		const Point3d v_start5(chessboard3dPoints[44][4]);
		const Point3d v_end5(chessboard3dPoints[44][49]);
		viz::WArrow v_arrow5(v_start5, v_end5, 0.001, viz::Color::gray());
		const Point3d v_start6(chessboard3dPoints[44][5]);
		const Point3d v_end6(chessboard3dPoints[44][50]);
		viz::WArrow v_arrow6(v_start6, v_end6, 0.001, viz::Color::gray());
		const Point3d v_start7(chessboard3dPoints[44][6]);
		const Point3d v_end7(chessboard3dPoints[44][51]);
		viz::WArrow v_arrow7(v_start7, v_end7, 0.001, viz::Color::gray());
		const Point3d v_start8(chessboard3dPoints[44][7]);
		const Point3d v_end8(chessboard3dPoints[44][52]);
		viz::WArrow v_arrow8(v_start8, v_end8, 0.001, viz::Color::gray());
		const Point3d v_start9(chessboard3dPoints[44][8]);
		const Point3d v_end9(chessboard3dPoints[44][53]);
		viz::WArrow v_arrow9(v_start9, v_end9, 0.001, viz::Color::gray());

		myWindow.showWidget("h_arrow1", h_arrow1);
		myWindow.showWidget("h_arrow2", h_arrow2);
		myWindow.showWidget("h_arrow3", h_arrow3);
		myWindow.showWidget("h_arrow4", h_arrow4);
		myWindow.showWidget("h_arrow5", h_arrow5);
		myWindow.showWidget("h_arrow6", h_arrow6);

		myWindow.showWidget("v_arrow1", v_arrow1);
		myWindow.showWidget("v_arrow2", v_arrow2);
		myWindow.showWidget("v_arrow3", v_arrow3);
		myWindow.showWidget("v_arrow4", v_arrow4);
		myWindow.showWidget("v_arrow5", v_arrow5);
		myWindow.showWidget("v_arrow6", v_arrow6);
		myWindow.showWidget("v_arrow7", v_arrow7);
		myWindow.showWidget("v_arrow8", v_arrow8);
		myWindow.showWidget("v_arrow9", v_arrow9);

		myWindow.showWidget("chessboardPoints", cloud_widget);
		
		//myWindow.setViewerPose(cam_pose);
		myWindow.spin();
		
		int shot_count = 0;
		char c = (char)waitKey(500);
		if (c == 'c' || c == 'C')
		{
			myWindow.getScreenshot();
			stringstream l_name;
			l_name << "screen_shot31" << shot_count << ".png";
			myWindow.saveScreenshot(l_name.str());
			shot_count++;
		
		}
		


	}
}


static bool readStringList(const string& filename, vector<string>& l)
{
	l.resize(0);
	FileStorage fs(filename, FileStorage::READ);
	if (!fs.isOpened())
		return false;
	FileNode n = fs.getFirstTopLevelNode();
	if (n.type() != FileNode::SEQ)
		return false;
	FileNodeIterator it = n.begin(), it_end = n.end();
	for (; it != it_end; ++it)
		l.push_back((string)*it);
	return true;
}

int main()
{
	Size boardSize;
	string imagelistfn;
	string imagelistfn_v;
	imagelistfn = "C:/Users/CSK/source/repos/verify_calib/verify_calib/v1/stereo_calib2.xml";
	imagelistfn_v = "C:/Users/CSK/source/repos/verify_calib/verify_calib/v1/verify_calib.xml";
	boardSize.width = 9;
	boardSize.height = 6;
	float squareSize = 100.0;
	
	vector<string> imagelist;
	vector<string> imagelist_v;
	bool ok = readStringList(imagelistfn, imagelist);
	if (!ok || imagelist.empty())
	{
		cout << "can not open " << imagelistfn << " or the string list is empty" << endl;
	}

	Mat rmap00, rmap01, rmap10, rmap11;

	StereoCalib(imagelist, boardSize, squareSize, rmap00, rmap01, rmap10, rmap11, false, true, false);

	bool ok_v = readStringList(imagelistfn_v, imagelist_v);
	if (!ok || imagelist_v.empty())
	{
		cout << "can not open " << imagelistfn_v << " or the string list is empty" << endl;
	}

	VarifyCalib(imagelist_v, boardSize, rmap00, rmap01, rmap10, rmap11, false, false, true);

	return 0;
}