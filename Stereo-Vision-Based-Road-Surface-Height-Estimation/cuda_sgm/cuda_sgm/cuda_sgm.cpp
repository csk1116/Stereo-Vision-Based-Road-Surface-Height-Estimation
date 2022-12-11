/*
Copyright 2016 Fixstars Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http ://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/version.hpp>
#include <libsgm.h>
#include <opencv2/opencv.hpp>


using namespace cv;
using namespace std;

int64 work_begin;
double work_fps;

void workBegin() { work_begin = getTickCount(); }
void workEnd()
{
	int64 d = getTickCount() - work_begin;
	double f = getTickFrequency();
	work_fps = f / d;
}

string text()
{
	stringstream ss;
	ss << " FPS: " << setiosflags(ios::left)
		<< setprecision(4) << work_fps;
	return ss.str();
}

string type2str(int type) {
	string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');

	return r;
}

#define ASSERT_MSG(expr, msg) \
	if (!(expr)) { \
		std::cerr << msg << std::endl; \
		std::exit(EXIT_FAILURE); \
	} \


void prepare_remap(Size2i size, int image_resolution, cuda::GpuMat& d_rmap0_left, cuda::GpuMat& d_rmap1_left, cuda::GpuMat& d_rmap0_right, cuda::GpuMat& d_rmap1_right, Mat& Q, int& max_x, int& max_y, int& width, int& height) {

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

	//load camera matrix 
	FileStorage Efs("C:/Users/CSK/source/repos/road_surface_height_estimation/road_surface_height_estimation/" + extrinsic_parameter_file, FileStorage::READ);
	Efs["R"] >> R;
	Efs["T"] >> T;
	/*if (image_resolution == 0 || image_resolution == 1)
	{
		Rodrigues(R, R);
	}*/

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

	initUndistortRectifyMap(cam_left, dist_left, R_left, P_left, size, CV_32FC1, rmap0_left, rmap1_left);
	initUndistortRectifyMap(cam_right, dist_right, R_right, P_right, size, CV_32FC1, rmap0_right, rmap1_right);

	//upload to gpumat
	d_rmap0_left.upload(rmap0_left);
	d_rmap1_left.upload(rmap1_left);
	d_rmap0_right.upload(rmap0_right);
	d_rmap1_right.upload(rmap1_right);

	cout << P_left << P_right << endl;

}

static void execute(sgm::LibSGMWrapper& sgmw, const cv::Mat& h_left, const cv::Mat& h_right, cv::Mat& h_disparity) noexcept(false)
{
	cv::cuda::GpuMat d_left, d_right;
	d_left.upload(h_left);
	d_right.upload(h_right);
	/*Mat cleft;
	Ptr<cuda::CLAHE> clahe = cuda::createCLAHE(4.0, cv::Size(5, 15));
	clahe->apply(d_left, d_left);
	clahe->apply(d_right, d_right);
	d_left.download(cleft);
	imshow("CLAHE", cleft);*/

	cv::cuda::GpuMat d_disparity;
	sgmw.execute(d_left, d_right, d_disparity);
	/*Mat origin_disparity;
	d_disparity.download(origin_disparity);
	origin_disparity.convertTo(origin_disparity, CV_32FC1);
	origin_disparity /= 16.0;
	string ty = type2str(origin_disparity.type());
	printf("Matrix: %s %dx%d \n", ty.c_str(), origin_disparity.cols, origin_disparity.rows);
	cout << origin_disparity(cv::Rect(640, 500, 5, 50)) << endl;*/

	// normalize result
	cv::cuda::GpuMat d_normalized_disparity;
	d_disparity.convertTo(d_normalized_disparity, CV_8UC1, 256. / (16*sgmw.getNumDisparities()));
	//d_disparity.convertTo(d_normalized_disparity, CV_8UC1, 256. / (max_d - min_d));
	d_normalized_disparity.download(h_disparity);
}

int main() {

	//int argc, char* argv[]
	/*ASSERT_MSG(argc >= 3, "usage: stereosgm left_img right_img [disp_size]");
	Mat left = cv::imread(argv[1], 0);
	Mat right = cv::imread(argv[2], 0);
	const int disp_size = argc > 3 ? std::atoi(argv[3]) : 128;
	left = left(cv::Rect(57, 82, 1148, 550));
	right = right(cv::Rect(57, 82, 1148, 550));

	ASSERT_MSG(left.size() == right.size() && left.type() == right.type(), "input images must be same size and type.");
	ASSERT_MSG(left.type() == CV_8U || left.type() == CV_16U, "input image format must be CV_8U or CV_16U.");
	ASSERT_MSG(disp_size == 64 || disp_size == 128, "disparity size must be 64 or 128.");*/

	// calibrate and rectify the camera and capture
	//video resolution [2k:15fps 1080p:30fps 720p:60fps wvga:100fps]
	//Size2i image_size = Size2i(1280, 720);

	////open the camera
	//VideoCapture cap(1);
	//if (!cap.isOpened())
	//	cout << "camera is not connected" << endl;
	//else
	//{
	//	cout << "camera open" << endl;
	//}
	////cap.grab();

	//// Set the video resolution (2*Width * Height)
	//cap.set(CV_CAP_PROP_FRAME_WIDTH, image_size.width * 2);
	//cap.set(CV_CAP_PROP_FRAME_HEIGHT, image_size.height);
	//cap.grab();

	//Mat frame, left_raw, right_raw, left_rectified, right_rectified, left_cropped, right_cropped;
	//Mat cameraMatrix_left, distCoeffs_left, R1, P1;
	//Mat cameraMatrix_right, distCoeffs_right, R2, P2;
	//Mat rmap00, rmap01, rmap10, rmap11;
	//Mat Q;



	//cout << "Image resolution:" << image_size << endl;

	////capture image
	//cout << "Press 'c' to capture ..." << endl;

	//int Icount = 0;

	////load camera matrix 
	//FileStorage Efs("C:/Users/CSK/source/repos/cuda_sgm/cuda_sgm/extrinsics.yml", FileStorage::READ);
	//Efs["Q"] >> Q;
	//Efs["R1"] >> R1;
	//Efs["R2"] >> R2;
	//Efs["P1"] >> P1;
	//Efs["P2"] >> P2;
	//FileStorage Ifs("C:/Users/CSK/source/repos/cuda_sgm/cuda_sgm/intrinsics.yml", FileStorage::READ);
	//Ifs["M1"] >> cameraMatrix_left;
	//Ifs["M2"] >> cameraMatrix_right;
	//Ifs["D1"] >> distCoeffs_left;
	//Ifs["D2"] >> distCoeffs_right;

	//double C_x = -1 * Q.at<double>(0, 3) - 57;
	//double C_y = -1 * Q.at<double>(1, 3) - 82;
	//double B_v = 550 * 1 / 4;

	//initUndistortRectifyMap(cameraMatrix_left, distCoeffs_left, R1, P1, image_size, CV_32FC1, rmap00, rmap01);
	//initUndistortRectifyMap(cameraMatrix_right, distCoeffs_right, R2, P2, image_size, CV_32FC1, rmap10, rmap11);

	//cuda::GpuMat rmapx0, rmapy0, rmapx1, rmapy1;
	//cuda::GpuMat d_left_raw, d_right_raw, d_left_rectified, d_right_rectified;

	//rmapx0.upload(rmap00);
	//rmapy0.upload(rmap01);
	//rmapx1.upload(rmap10);
	//rmapy1.upload(rmap11);

	//sgm::LibSGMWrapper sgmw{ disp_size };
	int disp_size = 128;
	sgm::LibSGMWrapper sgmw{ disp_size, 10, 120, 0.999F, true };
	/*Ptr<StereoSGBM> sgbm;
	sgbm = StereoSGBM::create(0, 128, 9, 0, 0, 1, 0, 0, 0, 0);*/
	//sgm::LibSGMWrapper sgmw{ disp_size, 504, 2016, 0.99F, true };
	cv::Mat disparity;

	int key = cv::waitKey();
	//char capture = 'r';
	int mode = 0;

	Mat left_raw = imread("left_model_one2kraw_3.png", 0);
	Mat right_raw = imread("right_model_one2kraw_3.png", 0);

	//imshow("left raw", left_raw);

	int resolution = 0;
	Size2i image_size;
	switch (resolution)
	{
	case 0:
		//2k
		image_size = Size2i(2208, 1242);
		break;

	case 1:
		//1080p
		image_size = Size2i(1920, 1080);
		break;

	case 2:
		//720p
		image_size = Size2i(1280, 720);
		break;

	}

	//prepare for remap
	cuda::GpuMat d_rmap00, d_rmap01, d_rmap10, d_rmap11;
	Mat reprojection_mat;
	int crop_x, crop_y, crop_width, crop_height;
	prepare_remap(image_size, resolution, d_rmap00, d_rmap01, d_rmap10, d_rmap11, reprojection_mat, crop_x, crop_y, crop_width, crop_height);
	//image center
	double C_x = -1 * reprojection_mat.at<double>(0, 3) - crop_x;
	double C_y = -1 * reprojection_mat.at<double>(1, 3) - crop_y;

	Mat  left_rectified, right_rectified, left_cropped, right_cropped;
	cuda::GpuMat d_left_raw, d_right_raw, d_left_rectified, d_right_rectified, d_left_cropped, d_right_cropped;
	//remap image
	d_left_raw.upload(left_raw);
	d_right_raw.upload(right_raw);
	cuda::remap(d_left_raw, d_left_rectified, d_rmap00, d_rmap01, INTER_LINEAR);
	cuda::remap(d_right_raw, d_right_rectified, d_rmap10, d_rmap11, INTER_LINEAR);
	//cropped
	d_left_cropped = d_left_rectified(Rect(crop_x, crop_y, crop_width, crop_height));
	d_right_cropped = d_right_rectified(Rect(crop_x, crop_y, crop_width, crop_height));

	d_left_rectified.download(left_rectified);
	d_right_rectified.download(right_rectified);

	d_left_cropped.download(left_cropped);
	d_right_cropped.download(right_cropped);
	
	/*for (int i = 0; i < 18; i++)
	{
		line(left_cropped, Point2d(C_x, 0), Point2d(C_x, left_cropped.rows), Scalar(0, 0, 255), 1, 8);
		line(left_cropped, Point2d(0, C_y), Point2d(left_cropped.cols, C_y), Scalar(0, 0, 255), 1, 8);
		line(left_cropped, Point2d(0, (i + 1) * 50), Point2d(left_cropped.cols, (i + 1) * 50), Scalar(0, 0, 255), 1, 8);

		line(right_cropped, Point2d(C_x, 0), Point2d(C_x, right_cropped.rows), Scalar(0, 0, 255), 1, 8);
		line(right_cropped, Point2d(0, C_y), Point2d(right_cropped.cols, C_y), Scalar(0, 0, 255), 1, 8);
		line(right_cropped, Point2d(0, (i + 1) * 50), Point2d(right_cropped.cols, (i + 1) * 50), Scalar(0, 0, 255), 1, 8);
	}*/
	

	imshow("left", left_cropped);
	imshow("right", right_cropped);
	imwrite("left_model_one2k_3.png", left_cropped);
	imwrite("right_model_one2k_3.png", right_cropped);

	//imshow("left r", left_rectified);
	//imshow("right r", right_rectified);

	//left_cropped = left_cropped(cv::Rect(57, 82, 1148, 550));
	//right_cropped = right_cropped(cv::Rect(57, 82, 1148, 550));
	

	while (key != 27) {

		//// Get a new frame from camera
		//cap >> frame;
		//// Extract left and right images from side-by-side
		//left_raw = frame(cv::Rect(0, 0, frame.cols / 2, frame.rows));
		//right_raw = frame(cv::Rect(frame.cols / 2, 0, frame.cols / 2, frame.rows));

		//d_left_raw.upload(left_raw);
		//d_right_raw.upload(right_raw);


		//cuda::remap(d_left_raw, d_left_rectified, rmapx0, rmapy0, INTER_LINEAR);
		//cuda::remap(d_right_raw, d_right_rectified, rmapx1, rmapy1, INTER_LINEAR);

		//d_left_rectified.download(left_rectified);
		//d_right_rectified.download(right_rectified);

		//left_cropped = left_rectified.clone();
		//right_cropped = right_rectified.clone();

		//left_cropped = left_cropped(cv::Rect(57, 82, 1148, 550));
		//right_cropped = right_cropped(cv::Rect(57, 82, 1148, 550));
		//
		//cvtColor(left_cropped, left_cropped, COLOR_BGR2GRAY);
		//cvtColor(right_cropped, right_cropped, COLOR_BGR2GRAY);
		
		


			workBegin();
			try {
				execute(sgmw, left_cropped, right_cropped, disparity);
				//sgbm->compute(left_cropped, right_cropped, disparity);
				//disparity.convertTo(disparity, CV_8UC1, 256. / (16 * 128));
				//disparity.convertTo(disparity, CV_32FC1, 1 / 16);
			}
			catch (const cv::Exception& e) {
				std::cerr << e.what() << std::endl;
				if (e.code == cv::Error::GpuNotSupported) {
					return 1;
				}
				else {
					return -1;
				}
			}
			workEnd();

			/*string ty = type2str(disparity.type());
			printf("Matrix: %s %dx%d \n", ty.c_str(), disparity.cols, disparity.rows);
			cout << disparity(cv::Rect(640, 500, 5, 50)) << endl;*/


			// post-process for showing image
			cv::Mat colored;
			cv::applyColorMap(disparity, colored, cv::COLORMAP_HOT);
			//putText(disparity, text(), Point(5, 25), FONT_HERSHEY_SIMPLEX, 1.0, Scalar::all(255));
			putText(colored, text(), Point(5, 25), FONT_HERSHEY_SIMPLEX, 1.0, Scalar::all(255));
			//imshow("image", disparity);
			//imshow("RAW", left);
			namedWindow("DISPARITY", WINDOW_NORMAL);
			//resizeWindow("DISPARITY", 1280, 720);
			imshow("DISPARITY", colored);
		
		

		/*if (key == 's') {
			mode += 1;
			if (mode >= 3) mode = 0;

			switch (mode) {
			case 0:
			{
				cv::setWindowTitle("image", "disparity");
				cv::imshow("image", disparity);
				break;
			}
			case 1:
			{
				cv::setWindowTitle("image", "disparity color");
				cv::imshow("image", colored);
				break;
			}
			case 2:
			{
				cv::setWindowTitle("image", "input");
				cv::imshow("image", left);
				break;
			}
			}
		}*/
		
		key = cv::waitKey(10);
		//capture = (char)waitKey(10);
	}

	return 0;
}
