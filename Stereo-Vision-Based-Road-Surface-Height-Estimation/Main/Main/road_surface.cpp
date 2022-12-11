#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include <string>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <math.h> 
#include <fstream>
#include <libsgm.h>
#include "opencv2\viz.hpp"
#include <opencv2/ximgproc/edge_filter.hpp>
#include <vector>
#include <iterator>
#include <ctype.h>
#include <omp.h>
#include <tchar.h>
#include "SerialClass.h"
#include <opencv2/plot.hpp>
#define CVUI_IMPLEMENTATION
#include "cvui.h"

using namespace cv;
using namespace std;

#define WINDOW_NAME "Road Surface Estimation"

// execute fps 
int64 work_begin;
double work_fps;

void workBegin() { work_begin = getTickCount(); }
void workEnd()
{
	int64 d = getTickCount() - work_begin;
	double f = getTickFrequency();
	work_fps = f / d;
}

// show fps on disparity map
string text()
{
	stringstream ss;
	ss << " FPS: " << setiosflags(ios::left)
		<< setprecision(2) << work_fps;
	return ss.str();
}








typedef struct
{
	float roll;
	float pitch;

}IMU_data;

//load calibration parameter
void prepare_remap(Size2i size, int image_resolution, cuda::GpuMat& d_rmap0_left, cuda::GpuMat& d_rmap1_left, cuda::GpuMat& d_rmap0_right, cuda::GpuMat& d_rmap1_right, Mat& Q, int& max_x, int& max_y, int& width, int& height, Mat SR, Mat ST, bool self_calib) {


	Mat cam_left, cam_right, dist_left, dist_right, R_left, R_right, P_left, P_right, rmap0_left, rmap0_right, rmap1_left, rmap1_right, R, T;
	string intrinsic_parameter_file;
	string extrinsic_parameter_file;

	switch (image_resolution)
	{
	case 0:
		//2k
		intrinsic_parameter_file = "intrinsics2k.yml";
		extrinsic_parameter_file = "extrinsics2k.yml";
		break;

	case 1:
		//1080p
		intrinsic_parameter_file = "intrinsics1080.yml";
		extrinsic_parameter_file = "extrinsics1080.yml";
		break;

	case 2:
		//720p
		intrinsic_parameter_file = "intrinsics720.yml";
		extrinsic_parameter_file = "extrinsics720.yml";
		break;
	}
	
	//load camera intrinsics and extrinsics parameter
	
	FileStorage Efs("C:/Users/CSK/source/repos/road_surface_height_estimation/road_surface_height_estimation/" + extrinsic_parameter_file, FileStorage::READ);
	Efs["R"] >> R;
	Efs["T"] >> T;
	

	if (self_calib == true)
		
	{
		double scale;
		scale = -120.0 / ST.at<double>(0, 0);
		R = SR.clone();
		ST *= scale;
		T = ST.clone();
		
	}
	

	cout << R << T << endl;
	
	FileStorage Ifs("C:/Users/CSK/source/repos/road_surface_height_estimation/road_surface_height_estimation/" + intrinsic_parameter_file, FileStorage::READ);
	Ifs["M1"] >> cam_left;
	Ifs["M2"] >> cam_right;
	Ifs["D1"] >> dist_left;
	Ifs["D2"] >> dist_right;

	cout << cam_left << cam_right << endl;

	Rect validRoi[2];
	stereoRectify(cam_left, dist_left, cam_right, dist_right, size, R, T, R_left, R_right, P_left, P_right, Q, CALIB_ZERO_DISPARITY, 1, size, &validRoi[0], &validRoi[1]);

	//valid roi of stereo pair
	max_x = max(validRoi[0].x, validRoi[1].x);
	max_y = max(validRoi[0].y, validRoi[1].y);
	width = min(validRoi[0].x + validRoi[0].width, validRoi[1].x + validRoi[1].width) - max_x;
	height = min(validRoi[0].y + validRoi[0].height, validRoi[1].y + validRoi[1].height) - max_y;
	cout << validRoi[0] << validRoi[1] << endl;

	//reprojection matrix
	Q.at<double>(0, 3) += (double)max_x;
	Q.at<double>(1, 3) += (double)max_y;

	initUndistortRectifyMap(cam_left, dist_left, R_left, P_left, size, CV_32FC1, rmap0_left, rmap1_left);
	initUndistortRectifyMap(cam_right, dist_right, R_right, P_right, size, CV_32FC1, rmap0_right, rmap1_right);

	//upload to gpumat
	d_rmap0_left.upload(rmap0_left);
	d_rmap1_left.upload(rmap1_left);
	d_rmap0_right.upload(rmap0_right);
	d_rmap1_right.upload(rmap1_right);

	cout << P_left << P_right << endl;

}

bool self_calib(cuda::GpuMat& left, cuda::GpuMat& right, Mat left_m, Mat right_m, Mat left_dist, Mat right_dist, Mat& R, Mat& T) {

	//clahe
	Ptr<cuda::CLAHE> clahe = cuda::createCLAHE(4.0, cv::Size(5, 15));
	//clahe->apply(left, left);
	//clahe->apply(right, right);

	//Detect the keypoints and calculate the descriptors
	cuda::GpuMat d_keypoints_1, d_keypoints_2;
	cuda::GpuMat d_descriptors_1, d_descriptors_2;
	vector<KeyPoint> keypoints_1, keypoints_2;
	Mat descriptors_1, descriptors_2;

	//create cuda surf and extract feature points
	cuda::SURF_CUDA surf(500);
	surf(left, cuda::GpuMat(), d_keypoints_1, d_descriptors_1);
	surf(right, cuda::GpuMat(), d_keypoints_2, d_descriptors_2);
	surf.downloadKeypoints(d_keypoints_1, keypoints_1);
	surf.downloadKeypoints(d_keypoints_2, keypoints_2);
	
	//matche feature points 
	vector<vector<DMatch> > knn_matches;
	Ptr< cuda::DescriptorMatcher > matcher = cuda::DescriptorMatcher::createBFMatcher(NORM_L2);
	matcher->knnMatch(d_descriptors_1, d_descriptors_2, knn_matches, 2);

	//filter matchers
	vector<DMatch> good_matches;
	vector<Point2f> left_matched_points;
	vector<Point2f> right_matched_points;
	float ratio = 0.01f;
	bool pointsEnough = true;

	while (pointsEnough)
	{
		for (int i = 0; i < knn_matches.size(); i++)
		{
			int left_idx = knn_matches[i][0].queryIdx;
			int right_idx = knn_matches[i][0].trainIdx;

			if (knn_matches[i][0].distance < ratio*knn_matches[i][1].distance)
			{
				good_matches.push_back(knn_matches[i][0]);
			}
		}

		if (good_matches.size() < 300)
		{
			good_matches.clear();
			ratio += 0.01f;
		}

		else
		{
			pointsEnough = false;
		}

	}
	

	//store good matched points for find fundermental matrix
	for (int i = 0; i < (int)good_matches.size(); i++)
	{
		int left_idx = good_matches[i].queryIdx;
		int right_idx = good_matches[i].trainIdx;
		left_matched_points.push_back(keypoints_1[left_idx].pt);
		right_matched_points.push_back(keypoints_2[right_idx].pt);
	}
	

	//draw good matches
	Mat left_matched, right_matched;
	left.download(left_matched);
	right.download(right_matched);

	//imshow("leftmat", left_matched);
	//imshow("rightmat", right_matched);

	Mat img_matches;
	drawMatches(left_matched, keypoints_1, right_matched, keypoints_2,
		good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	namedWindow("Good Matches", WINDOW_NORMAL);
	resizeWindow("Good Matches", 2400, 800);
	imshow("Good Matches", img_matches);
	//imwrite("matched feature points.png", img_matches);

	//compute fundamevtal matrix
	Mat fundamental_mat;
	Mat inlier;
	fundamental_mat = findFundamentalMat(left_matched_points, right_matched_points, CV_FM_8POINT, 0.1, 0.99, inlier);
	//cout << inlier << endl;

	//quality check
	//epipolar geometry constraint: m2^t*F*m1=0
	double err = 0;
	vector<Vec3f> lines[2];
	int npt = (int)good_matches.size();
	
	//epipolar lines
	computeCorrespondEpilines(left_matched_points, 1, fundamental_mat, lines[0]);
	computeCorrespondEpilines(right_matched_points, 2, fundamental_mat, lines[1]);
	
	for (int i = 0; i < npt; i++)
	{
			double errij = fabs(left_matched_points[i].x*lines[1][i][0] +
				left_matched_points[i].y*lines[1][i][1] + lines[1][i][2]) +
				fabs(right_matched_points[i].x*lines[0][i][0] +
					right_matched_points[i].y*lines[0][i][1] + lines[0][i][2]);
			err += errij;
	}

	err /= npt;
	cout << "average epipolar err = " << err << endl;

	//do it again until epipolar err is less than 0.5
	
	if (err < 0.9)
	{
		//draw epipolar line
		//right
		//for (int i = 0; i < (int)lines[0].size(); i++)
		//{
		//	line(img_matches, Point2f((float)left_matched.cols, -lines[0][i][2] / lines[0][i][1]), Point((float)left_matched.cols + (float)right_matched.cols, -((lines[0][i][2] + lines[0][i][0] * (float)right_matched.cols) / lines[0][i][1])), Scalar(255, 0, 0), 1, 8);
		//}

		////imshow("right Image Epilines", right_matched);
		//for (int i = 0; i < (int)lines[1].size(); i++)
		//{
		//	line(img_matches, Point2f(0, -lines[1][i][2] / lines[1][i][1]), Point(left_matched.cols, -((lines[1][i][2] + lines[1][i][0] * left_matched.cols) / lines[1][i][1])), Scalar(255, 0, 0), 1, 8);
		//}

		//namedWindow("Epilines", WINDOW_NORMAL);
		//resizeWindow("Epilines", 2560, 720);
		//imshow("Epilines", img_matches);

		//essential matrix = m2 * f * m1
		Mat essential_mat;
		int inliers;
		essential_mat = right_m.t() * fundamental_mat * left_m;
		//cout << essential_mat << endl;
		inliers = recoverPose(essential_mat, left_matched_points, right_matched_points, left_m, R, T, 10000.0);
		cout << " R = " << R << "\n" << " T = " << T << " inliers = " << inliers << endl;


		return true;
	}

	else
	{
		return false;
	}
	

}

void sgm_disparity(int disparity_size, cuda::GpuMat left, cuda::GpuMat right, Mat& disparity_map) {

	
	//fixstar sgm
	sgm::LibSGMWrapper sgmw{ disparity_size, 10, 120, 0.99f, true };
	
	//disparity map
	//Mat disparity, scaleDisparityMap, cleft, left_cropped;
	cuda::GpuMat d_disparity;

	//clahe
	//Ptr<cuda::CLAHE> clahe = cuda::createCLAHE(4.0, cv::Size(5, 15));
	//clahe->apply(left, left);
	//clahe->apply(right, right);
	//left.download(cleft);
	//imshow("clahe left", cleft);

	try {
		sgmw.execute(left, right, d_disparity);
		//sgbm->compute(left_cropped, right_cropped, disparity);
		//subpixel
		double scale = 1.0 / 16.0;
		cuda::GpuMat d_subpix_disparity;
		d_disparity.convertTo(d_subpix_disparity, CV_32FC1, scale);
		d_subpix_disparity.download(disparity_map);
		//guided filter
		ximgproc::guidedFilter(disparity_map, disparity_map, disparity_map, 40, 10.0);
	}
	catch (const cv::Exception& e) {
		std::cerr << e.what() << std::endl;
	}


	
	
}


// application reads from the specified serial port and reports the collected data
void imu(bool &frame_stamp, char &key, bool &triger, Mat Q, Mat& reprojection_mat)
{

	Serial* SP = new Serial("\\\\.\\COM4");    // adjust as needed

	if (SP->IsConnected())
		printf("connected\n");

	IMU_data id;
	char incomingData[sizeof(id)];  // don't forget to pre-allocate memory
	int dataLength = sizeof(id);
	char startChar[1];  // don't forget to pre-allocate memory
	char endChar[1];
	int startLength = 1;
	int endLength = 1;
	int readResult = 0;
	//float imu_roll_current, imu_roll_last, imu_pitch_current, imu_pitch_last;
	//bool is_last = true;

	//transformation matrix
	Mat pitch_axis(4, 4, CV_64FC1);
	Mat T(4, 4, CV_64FC1);
	Mat roll_axis(4, 4, CV_64FC1);

	/*ofstream myfile("roll 4.txt");
	if (myfile.is_open())
	{
		myfile << "roll" << " , " << "pitch" << " , "  << "encoder" << " , " << "frame-in" << "\n";
	}
	else
		cout << "Error: can not save the data\n";

	ofstream mefile("roll difference 4.txt");
	if (mefile.is_open())
	{
		mefile << "IMU_roll" << " , " << "IMU_pitch" << " , " "ENCODER" <<  "\n";
	}
	else
		cout << "Error: can not save the data\n";

	ofstream Ifile("roll frame only 4.txt");
	if (Ifile.is_open())
	{
		Ifile << "roll" << " , " << "pitch" << " , " "ENCODER" << "\n";
	}
	else
		cout << "Error: can not save the data\n";*/


	while (SP->IsConnected())
	{

		SP->ReadData(startChar, startLength);
		if (startChar[0] == 'S')
		{
			readResult = SP->ReadData(incomingData, dataLength);
			SP->ReadData(endChar, endLength);
			if (endChar[0] == 'E')
			{
				memcpy(&id, &incomingData, sizeof(id));

				if (frame_stamp == true)
				{
					//myfile << id.roll << " , " << id.pitch << " , "  << id.encoder << " , " << frame_stamp << "\n";
					//Ifile << id.roll << " , " << id.pitch << " , " << id.encoder << " , " << "\n";
					//if (is_last)
					//{
					//	imu_pitch_last = id.pitch;
					//	imu_roll_last = id.roll;
					//	is_last = false;
					//}

					//else
					//{
					//	imu_pitch_current = id.pitch;
					//	imu_roll_current = id.roll;
					//	//mefile << imu_roll_current - imu_roll_last << " , " << imu_pitch_current - imu_pitch_last << " , " << encoder_current - encoder_last << "\n";
					//	is_last = true;
					//}

					cout << "roll: " << id.roll << " , " << "pitch: " << id.pitch << " , "  << "frame-in: " << frame_stamp << endl;
					

					// camera set up (rotation and translation)

					//translation
					double tx = 60.0; //mm
					double ty = 1400 * cos(30.0 * CV_PI / 180); //mm
					double tz = 1400 * cos(60.0 * CV_PI / 180); //mm
					T.at<double>(0, 0) = 1.0;
					T.at<double>(1, 1) = 1.0;
					T.at<double>(2, 2) = 1.0;
					T.at<double>(3, 3) = 1.0;
					T.at<double>(0, 1) = T.at<double>(0, 2) =  T.at<double>(1, 0) = T.at<double>(1, 2) = T.at<double>(2, 0) = T.at<double>(2, 1) = T.at<double>(3, 0) = T.at<double>(3, 1) = T.at<double>(3, 2) = 0.0;
					T.at<double>(0, 3) = -1.0 * tx;
					T.at<double>(1, 3) = -1.0 * ty;
					T.at<double>(2, 3) = -1.0 * tz;

					//pitch (Rx)
					double alpha = 90.0 - 2.8; //degree (off-set)
					pitch_axis.at<double>(0, 0) = 1.0;
					pitch_axis.at<double>(3, 3) = 1.0;
					pitch_axis.at<double>(0, 1) = pitch_axis.at<double>(0, 2) = pitch_axis.at<double>(0, 3) = pitch_axis.at<double>(1, 0) = pitch_axis.at<double>(1, 3) = pitch_axis.at<double>(2, 0) = pitch_axis.at<double>(2, 3) = pitch_axis.at<double>(3, 0) = pitch_axis.at<double>(3, 1) = pitch_axis.at<double>(3, 2) = 0.0;
					pitch_axis.at<double>(1, 1) = cos((alpha + id.pitch) * CV_PI / 180);
					pitch_axis.at<double>(2, 2) = cos((alpha + id.pitch) * CV_PI / 180); 
					pitch_axis.at<double>(1, 2) = sin((alpha + id.pitch) * CV_PI / 180);
					pitch_axis.at<double>(2, 1) = -1 * sin((alpha + id.pitch) * CV_PI / 180);
					

					//roll(Rz)
					roll_axis.at<double>(2, 2) = 1.0;
					roll_axis.at<double>(3, 3) = 1.0;
					roll_axis.at<double>(0, 2) = roll_axis.at<double>(0, 3) = roll_axis.at<double>(1, 2) = roll_axis.at<double>(1, 3) = roll_axis.at<double>(2, 0) = roll_axis.at<double>(2, 1) = roll_axis.at<double>(2, 3) = roll_axis.at<double>(3, 0) = roll_axis.at<double>(3, 1) = roll_axis.at<double>(3, 2) = 0.0;
					roll_axis.at<double>(0, 0) = cos(id.roll * CV_PI / 180);
					roll_axis.at<double>(1, 1) = cos(id.roll * CV_PI / 180);
					roll_axis.at<double>(1, 0) = sin(id.roll * CV_PI / 180);
					roll_axis.at<double>(0, 1) = -1 * sin(id.roll * CV_PI / 180);

					//pitch = id.pitch - 32.8;
					//roll = id.roll;

					//transformation matrix
					reprojection_mat = pitch_axis * roll_axis * T * Q;
					reprojection_mat.convertTo(reprojection_mat, CV_32FC1);
					frame_stamp = false;

				}
				//else
				//{
				//	//myfile << id.roll << " , " << id.pitch << " , " << id.encoder << " , " << frame_stamp << "\n";
				cout << "roll: " << id.roll << " , " << "pitch: " << id.pitch << " , "  << "frame-in: " << frame_stamp << endl;
				//	

				//}

				triger = true;

			}

			/*else
			{
				cout << "FUCK" << endl;
			}*/

		}
		/*else
		{
			cout << "shit" << endl;
		}*/

		if (key == 'q' || key == 'Q') {
			//myfile.close();
			//mefile.close();
			//Ifile.close();
			break;
		}

		//Sleep(1);
	}



}

float median_val(vector<float> inner_points) {
	nth_element(inner_points.begin(), inner_points.begin() + inner_points.size() / 2, inner_points.end());
	return inner_points[inner_points.size() / 2];
}

void estimate_road_surface_height(Mat disparity, Mat reprojection, Mat left_raw, Mat right_raw, cuda::GpuMat left, cuda::GpuMat right, bool ready, Mat& scaleDisparityMap, Mat& left_cropped, Mat& preview_L, Mat& preview_R, Mat& preview_index) {

	//Mat left_cropped;
	left.download(left_cropped);
	//disparity_show = disparity.clone();

	cuda::GpuMat d_filtered_disparity;
	d_filtered_disparity.upload(disparity);


	// normalize result and show color map
	double max_d, min_d;
	//Mat scaleDisparityMap;
	cuda::minMax(d_filtered_disparity, &min_d, &max_d);
	//minMaxIdx(disparity, &min_d, &max_d);
	//cout << "disp min = " << min_d << endl << "disp max = " << max_d << endl;
	convertScaleAbs(disparity, scaleDisparityMap, 256 / (max_d - min_d));
	applyColorMap(scaleDisparityMap, scaleDisparityMap, COLORMAP_HOT);
	

	// reprojection to 3D
	Mat project_3D(left.size(), CV_32FC3);
	cuda::GpuMat d_project_3D(left.size(), CV_32FC3);
	cuda::reprojectImageTo3D(d_filtered_disparity, d_project_3D, reprojection, 3);
	d_project_3D.download(project_3D);

	//roi threshold
	Mat pointcloud_tresh_X, pointcloud_tresh_Y, pointcloud_tresh_Z;
	vector<Mat> channels(3);
	Mat thresholded_X, thresholded_XL, thresholded_XR, thresholded_Y, thresholded_Z, thresholded_XYL, thresholded_XYR, thresholded_XYUL, thresholded_XYUR, thresholded_XYZU, thresholded_XYZ;
	// threshold ROI
	split(project_3D, channels);
	pointcloud_tresh_X = channels[0];
	pointcloud_tresh_Y = channels[1];
	pointcloud_tresh_Z = channels[2];
	inRange(pointcloud_tresh_X, -947.0f, -699.0f, thresholded_XL);
	thresholded_XL.convertTo(thresholded_XL, CV_8UC1);
	inRange(pointcloud_tresh_X, 699.0f, 947.0f, thresholded_XR);
	thresholded_XR.convertTo(thresholded_XR, CV_8UC1);
	//thresholded_X = thresholded_XL + thresholded_XR;
	inRange(pointcloud_tresh_Y, 1000.0f, 5000.0f, thresholded_Y);
	thresholded_Y.convertTo(thresholded_Y, CV_8UC1);
	/*inRange(pointcloud_tresh_Z, -500.0f, 500.0f, thresholded_Z);
	thresholded_Z.convertTo(thresholded_Z, CV_8UC1);*/
	compare(thresholded_XL, thresholded_Y, thresholded_XYUL, CMP_NE);
	compare(thresholded_XR, thresholded_Y, thresholded_XYUR, CMP_NE);
	thresholded_XYL = thresholded_XL + thresholded_Y - thresholded_XYUL;
	thresholded_XYR = thresholded_XR + thresholded_Y - thresholded_XYUR;
	/*compare(thresholded_Z, thresholded_XY, thresholded_XYZU, CMP_NE);
	thresholded_XYZ = thresholded_XY + thresholded_Z - thresholded_XYZU;*/
	//imshow("X", thresholded_X);
	//imshow("Y", thresholded_Y);
	//imshow("Z", thresholded_Z);
	//imshow("threshold XYZ", thresholded_XY);
	//imwrite("X.png", thresholded_X);
	//imwrite("Y.png", thresholded_Y);
	//imwrite("Z.png", thresholded_Z);
	//imwrite("XYZ.png", thresholded_XYZ);
	//cout << thresholded_XY.type() << endl;

	
	left_cropped = left_cropped + thresholded_XYL +thresholded_XYR;
	
	//elevation profile
	Mat pointcloud_L;
	Mat pointcloud_R;
	project_3D.copyTo(pointcloud_L, thresholded_XYL);
	project_3D.copyTo(pointcloud_R, thresholded_XYR);
	//cout << filtered_pointcloud.size() << endl;

	// elevation parameter
	float tile_start = 975.0;
	float tile_end = 5025.0;
	double tile_index = 1000.0;
	int tile_num = (int)(tile_end - tile_start) / 50;
	vector<float> inner_points_L;
	vector<float> inner_points_R;
	/*Mat road_height_mat_L;
	Mat road_height_mat_R;
	Mat preview_index;*/
	int point_count = 0;
	float median;
	float no_point = 0.0;

	//left track
	for (int k = 0; k < tile_num; k++)
	{
		int nCols_L = (int)pointcloud_L.cols;
		int nRows_L = (int)pointcloud_L.rows;

		for ( int j = 0; j < nRows_L; j++){
			for (int i = 0; i < nCols_L; i++){
				
				if (pointcloud_L.at<Vec3f>(j, i)[1] >= tile_start + k * 50.0f && pointcloud_L.at<Vec3f>(j, i)[1] <= tile_start + (k + 1) * 50.0f)
				{
					inner_points_L.push_back(pointcloud_L.at<Vec3f>(j, i)[2]);
					point_count++;
				}
			}
		}

		if (point_count != 0)
		{
			median = median_val(inner_points_L);
			preview_L.push_back(median);
			preview_index.push_back(tile_index);
		}
		else
		{
			preview_L.push_back(no_point);
			preview_index.push_back(tile_index);
		}
		tile_index += 50.0;
		point_count = 0;
		inner_points_L.clear();
	}
	
	// right track
	for (int k = 0; k < tile_num; k++)
	{
		int nCols_R = (int)pointcloud_R.cols;
		int nRows_R = (int)pointcloud_R.rows;

		for (int j = 0; j < nRows_R; j++) {
			for (int i = 0; i < nCols_R; i++) {

				if (pointcloud_R.at<Vec3f>(j, i)[1] >= tile_start + k * 50.0f && pointcloud_R.at<Vec3f>(j, i)[1] <= tile_start + (k + 1) * 50.0f)
				{
					inner_points_R.push_back(pointcloud_R.at<Vec3f>(j, i)[2]);
					point_count++;
				}
			}
		}

		if (point_count != 0)
		{
			median = median_val(inner_points_R);
			preview_R.push_back(median);
		}
		else
		{
			preview_R.push_back(no_point);
		}
		point_count = 0;
		inner_points_R.clear();
	}

	GaussianBlur(preview_L, preview_L, Size(1, 3), 0, 0.5, cv::BORDER_REPLICATE);
	GaussianBlur(preview_R, preview_R, Size(1, 3), 0, 0.5, cv::BORDER_REPLICATE);

	
	
	

	
}

Mat plot_results(Mat preview, Mat elevation) {
	// plot module only acceptes double values
	Mat xd, yd;
	preview.convertTo(xd, CV_64F);
	elevation.convertTo(yd, CV_64F);
	
	//adjust border and margins of the 2 plots to match  together
	double xmin, xmax;
	minMaxIdx(xd, &xmin, &xmax);
	double ymin, ymax;
	minMaxIdx(yd, &ymin, &ymax);
	
	Ptr<plot::Plot2d> plot_preview = plot::Plot2d::create(xd, yd);
	plot_preview->setMinX(xmin);
	plot_preview->setMaxX(xmax);
	plot_preview->setMinY(ymin - 300);
	plot_preview->setMaxY(ymax + 300);
	plot_preview->setPlotLineColor(Scalar(0, 255, 0)); // Green
	plot_preview->setPlotLineWidth(2);
	plot_preview->setNeedPlotLine(false);
	plot_preview->setShowGrid(false);
	plot_preview->setShowText(false);
	plot_preview->setPlotAxisColor(Scalar(0, 0, 255)); // Black (invisible)
	Mat img;
	plot_preview->render(img);
	flip(img, img, 0);
	return img;
}


int main() {

	// capture stereo pair from zed camera
	//open the camera
	VideoCapture cap(1, CAP_DSHOW);
	if (!cap.isOpened()) {
		cout << "camera is not connected" << endl;
		return -1;
	}
	else
	{
		cout << "camera open" << endl;
	}
	

	//video resolution [0 = 2k:15fps, 1 = 1080p:30fps, 2 = 720p:60fps]
	int resolution = 1;
	Size2i image_size;
	int disparity_size;
	

	switch (resolution)
	{
		case 0:
			//2k
			image_size = Size2i(2208, 1242);
			disparity_size = 128;
		break;

		case 1:
			//1080p
			image_size = Size2i(1920, 1080);
			disparity_size = 128;
		break;

		case 2:
			//720p
			image_size = Size2i(1280, 720);
			disparity_size = 64;
		break;

	}
	
	cap.set(CV_CAP_PROP_FRAME_WIDTH, image_size.width * 2);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, image_size.height);
	//cout << cap.get(CV_CAP_PROP_FRAME_WIDTH) << cap.get(CV_CAP_PROP_FRAME_HEIGHT) << endl;
	cap.grab();
	
	//prepare for remap
	cuda::GpuMat d_rmap00, d_rmap01, d_rmap10, d_rmap11;
	Mat Q;
	Mat reprojection_mat;
	Mat disparity_map;
	//stereo pair roi
	int crop_x, crop_y, crop_width, crop_height;
	//self calib
	Mat SR, ST;
	
	prepare_remap(image_size, resolution, d_rmap00, d_rmap01, d_rmap10, d_rmap11, Q, crop_x, crop_y, crop_width, crop_height, SR, ST, false);
	
	//new image principal point
	double C_x = -1 * reprojection_mat.at<double>(0, 3) - crop_x;
	double C_y = -1 * reprojection_mat.at<double>(1, 3) - crop_y;
	cout << crop_x << crop_y << crop_width << crop_height << endl;

	//raw, rectified and cropped matrix
	char key = 'r';
	Mat frame, left_raw, right_raw, left_rectified, right_rectified, left_cropped, right_cropped;
	cuda::GpuMat d_left_raw, d_right_raw, d_left_rectified, d_right_rectified, d_left_cropped, d_right_cropped;


	//self calib success?
	bool success = true;

	// roi pointcloud
	//Mat pointcloud;

	//imu and camera separate into two sections by openmp
	bool frame_stamp = false;
	bool triger = false;
	bool cam_imu_ready = false;

	//image shot count
	int Icount = 0;
	

	//for display
	Mat disparity_show, viewtrack_show, left_elv, right_elv, preview_inx, left_result, right_result;
	
	//double pitch_show, roll_show;
	
	

#pragma omp parallel sections default(none) shared(key) shared(frame_stamp) shared(triger) shared(reprojection_mat) shared(disparity_map) shared(cam_imu_ready)
	{
	#pragma omp section
		imu(frame_stamp, key, triger, Q, reprojection_mat);

	#pragma omp section
		//estimate road surface height
		while (key != 'q' && key != 'Q')
		{
			while (cam_imu_ready == true)
			{	
				workBegin();

				try{

				estimate_road_surface_height(disparity_map, reprojection_mat, left_raw, right_raw, d_left_cropped, d_right_cropped, cam_imu_ready, disparity_show, viewtrack_show, left_elv, right_elv, preview_inx);
	
				}
				catch (const cv::Exception& e) {
					std::cerr << e.what() << std::endl;
				}

				//press 's' to seft calibration
				if (key == 's' || key == 'S' || success == false)
				{
					cout << "self calibration ... " << endl;

					string intrinsic_parameter_file;

					switch (resolution)
					{
					case 0:
						//2k
						intrinsic_parameter_file = "intrinsics2k.yml";
						break;

					case 1:
						//1080p
						intrinsic_parameter_file = "intrinsics1080.yml";
						break;

					case 2:
						//720p
						intrinsic_parameter_file = "intrinsics720.yml";
						break;
					}

					FileStorage Ifs("C:/Users/CSK/source/repos/road_surface_height_estimation/road_surface_height_estimation/" + intrinsic_parameter_file, FileStorage::READ);
					Mat cam_left, cam_right, dist_left, dist_right;
					Ifs["M1"] >> cam_left;
					Ifs["M2"] >> cam_right;
					Ifs["D1"] >> dist_left;
					Ifs["D2"] >> dist_right;

					//cout << cam_left << cam_right << dist_left << dist_right;

					// undistort left and right camera
					Mat left_undistorted, right_undistorted;
					undistort(left_raw, left_undistorted, cam_left, dist_left);
					undistort(right_raw, right_undistorted, cam_right, dist_right);

					cuda::GpuMat d_left_undistorted, d_right_undistorted;
					d_left_undistorted.upload(left_undistorted);
					d_right_undistorted.upload(right_undistorted);
					//convert to greyscale
					cuda::cvtColor(d_left_undistorted, d_left_undistorted, COLOR_BGR2GRAY);
					cuda::cvtColor(d_right_undistorted, d_right_undistorted, COLOR_BGR2GRAY);
					success = self_calib(d_left_undistorted, d_right_undistorted, cam_left, cam_right, dist_left, dist_right, SR, ST);

					if (success)
					{
						prepare_remap(image_size, resolution, d_rmap00, d_rmap01, d_rmap10, d_rmap11, reprojection_mat, crop_x, crop_y, crop_width, crop_height, SR, ST, success);
					}


				}
				
				

				//show disparity
				//putText(disparity_show, text(), Point(10, 35), FONT_HERSHEY_SIMPLEX, 1.5, Scalar::all(255), 2);
				//namedWindow("disparity map", WINDOW_NORMAL);
				//resizeWindow("disparity map", 1280, 720);
				//imshow("disparity map", disparity_show);
				//imwrite("disparity map without self calibration.png", scaleDisparityMap);

				//show track
				//namedWindow("left cropped", WINDOW_NORMAL);
				//resizeWindow("left cropped", 1280, 720);
				//imshow("left cropped", viewtrack_show);

				//if (preview_inx.empty() || left_elv.empty() || right_elv.empty())
				//{
				//	cout << "no data" << endl;
				//}
				//else
				//{
				////show elevation
				////cout << left_elv << endl;
				//	left_result = plot_results(preview_inx, left_elv);
				//	right_result = plot_results(preview_inx, right_elv);
				//	imshow("Left Wheel Path", left_result);
				//	imshow("Right Wheel Path", right_result);
				//}
				//cout << disparity_show.size() << endl;
				//cout << left_result.size() << endl;


				//cvui display
				/*stringstream pp;
				pp << " Pitch: " << setiosflags(ios::left)
					<< setprecision(2) << pitch_show;

				stringstream rr;
				rr << " Roll: " << setiosflags(ios::left)
					<< setprecision(2) << roll_show;*/

				Mat display(1080, 1920, CV_8UC3);
				//cvui::init(WINDOW_NAME);
				namedWindow(WINDOW_NAME, WINDOW_NORMAL);
				resizeWindow(WINDOW_NAME, 1920, 1080);
				//namedWindow(WINDOW_NAME, WINDOW_NORMAL);
				display = Scalar(200, 200, 200);
				resize(disparity_show, disparity_show, Size(), 0.5, 0.5, INTER_AREA);
				cvtColor(viewtrack_show, viewtrack_show, COLOR_GRAY2BGR);
				resize(viewtrack_show, viewtrack_show, Size(), 0.5, 0.5, INTER_AREA);
				cvui::image(display, 997, 60, disparity_show);
				cvui::image(display, 37, 60, viewtrack_show);
				//cvui::window(display, 835, 685, 250, 250, "Camera Pose");
				putText(display, "Left Camera View", Point(330, 545), FONT_HERSHEY_COMPLEX, 1.0, Scalar::all(0), 2);
				putText(display, "Disparity Map", Point(1327, 545), FONT_HERSHEY_COMPLEX, 1.0, Scalar::all(0), 2);
				putText(display, "Left Wheel Path", Point(340, 1020), FONT_HERSHEY_COMPLEX, 1.0, Scalar::all(0), 2);
				putText(display, "Right Wheel Path", Point(1270, 1020), FONT_HERSHEY_COMPLEX, 1.0, Scalar::all(0), 2);
				//putText(display, pp.str(), Point(900, 750), FONT_HERSHEY_COMPLEX, 0.5, Scalar::all(240), 2);
				//putText(display, rr.str(), Point(900, 800), FONT_HERSHEY_COMPLEX, 0.5, Scalar::all(240), 2);
				
				if (preview_inx.empty() || left_elv.empty() || right_elv.empty())
				{
					cout << "no data" << endl;
				}
				else
				{
				//show elevation
				//cout << left_elv << endl;
					left_result = plot_results(preview_inx, left_elv);
					right_result = plot_results(preview_inx, right_elv);
					resize(left_result, left_result, Size(), 1.2, 0.9, INTER_AREA);
					resize(right_result, right_result, Size(), 1.2, 0.9, INTER_AREA);
					cvui::image(display, 137, 630, left_result);
					cvui::image(display, 1063, 630, right_result);
					//imshow("Left Wheel Path", left_result);
					//imshow("Right Wheel Path", right_result);
				}
				cvui:imshow(WINDOW_NAME, display);
				

				


				

				if (key == 'c' || key == 'C') {

					//save files at proper locations if user presses 'c'
					stringstream l_name, r_name, lr_name, rr_name;
					//save files at proper locations if user presses 'c'
					//l_name << "left_model_one2k_" << Icount << ".png";
					//r_name << "right_model_one2k_" << Icount << ".png";
					lr_name << "1080 preview system" << Icount << ".png";
					//rr_name << "1080_right" << Icount << ".png";

					//imwrite(l_name.str(), left_rectified);
					//imwrite(r_name.str(), right_rectified);
					imwrite(lr_name.str(), display);
					//imwrite(rr_name.str(), right_raw);

					cout << "Saved set" << Icount << endl;
					Icount++;
				}

				

				if (key == 'q' || key == 'Q') {
					break;
				}

				preview_inx.release();
				left_elv.release();
				right_elv.release();

				key = (char)waitKey(1);
				workEnd();
			}

		}
		
		
		
	
	#pragma omp section
		while (key != 'q' && key != 'Q')
		{
			while (triger == true && frame_stamp == false)
			{
				
				
				

				// without self_calib
				// Get a new frame from camera
				cap >> frame;
				frame_stamp = true;
				
				// Extract left and right images from side-by-side
				left_raw = frame(Rect(0, 0, frame.cols / 2, frame.rows));
				right_raw = frame(Rect(frame.cols / 2, 0, frame.cols / 2, frame.rows));

				//remap image
				d_left_raw.upload(left_raw);
				d_right_raw.upload(right_raw);


				cuda::remap(d_left_raw, d_left_rectified, d_rmap00, d_rmap01, INTER_LINEAR);
				cuda::remap(d_right_raw, d_right_rectified, d_rmap10, d_rmap11, INTER_LINEAR);


				//cropped
				d_left_cropped = d_left_rectified(Rect(crop_x, crop_y, crop_width, crop_height));
				d_right_cropped = d_right_rectified(Rect(crop_x, crop_y, crop_width, crop_height));

				//convert to greyscale
				cuda::cvtColor(d_left_cropped, d_left_cropped, COLOR_BGR2GRAY);
				cuda::cvtColor(d_right_cropped, d_right_cropped, COLOR_BGR2GRAY);


				//download from gpu
				//d_left_rectified.download(left_rectified);
				//d_right_rectified.download(right_rectified);
				//d_left_cropped.download(left_cropped);
				//d_right_cropped.download(right_cropped);

				//wait for the reprojection matrix
				while (frame_stamp == true && key != 'q' && key != 'Q')
				{

					if (frame_stamp == false)
					{
						break;
					}
					Sleep(1);
				}
				
				sgm_disparity(disparity_size, d_left_cropped, d_right_cropped, disparity_map);

				

				//imshow("s", left_raw);
				cam_imu_ready = true;

				if (key == 'Q' || key == 'q')
				{
					cap.release();
					//cv::destroyAllWindows();
					break;
				}

				//key = (char)waitKey(1);
				

			}	
		}


	}

	
	
	return 0;


}