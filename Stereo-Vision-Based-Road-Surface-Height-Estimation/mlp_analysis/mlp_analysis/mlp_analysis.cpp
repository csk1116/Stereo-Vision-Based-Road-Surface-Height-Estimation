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
#include <string>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include "opencv2\viz.hpp"
#include <math.h> 
#include <fstream>
#include <opencv2/ximgproc/edge_filter.hpp>


using namespace cv;
using namespace std;


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
		<< setprecision(4) << work_fps;
	return ss.str();
}

// check message type 
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

float median_val(vector<float> inner_points) {
	nth_element(inner_points.begin(), inner_points.begin() + inner_points.size()/2, inner_points.end());
	return inner_points[inner_points.size() / 2];
}

void load_camera_parameter(Mat& WQ, Matx33d& Cam, Mat& W, int image_resolution, int& max_x, int& max_y, int& width, int& height) {

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


	//load reprojection matrix Q and camera matrix
	FileStorage fs("C:/Users/CSK/source/repos/mlp_analysis/mlp_analysis/" + extrinsic_parameter_file, FileStorage::READ);
	Mat Q, C;
	Rect validRoi[2];

	fs["validRoi left"] >> validRoi[0];
	fs["validRoi right"] >> validRoi[1];
	//valid roi of stereo pair
	max_x = max(validRoi[0].x, validRoi[1].x);
	max_y = max(validRoi[0].y, validRoi[1].y);
	width = min(validRoi[0].x + validRoi[0].width, validRoi[1].x + validRoi[1].width) - max_x;
	height = min(validRoi[0].y + validRoi[0].height, validRoi[1].y + validRoi[1].height) - max_y;
	//cout << width << height << endl;

	//Matx33d Cam;
	fs["P1"] >> C;
	C.at<double>(0, 2) -= max_x;
	C.at<double>(1, 2) -= max_y;
	C = C(cv::Rect(0, 0, 3, 3));
	C.convertTo(C, CV_64FC1);
	Cam = C;

	fs["Q"] >> Q;
	Q.at<double>(0, 3) += max_x;
	Q.at<double>(1, 3) += max_y;

	//1080
	//double theta = 119.6; //degree
	//double cam_height = 1380; //mm
	//2k
	//double theta = 119.1; //degree
	//double cam_height = 1390; //mm
	//720
	//double theta = 120.1; //degree
	//double cam_height = 1370; //mm
	//road
	double theta = 120.1; //degree
	double cam_height = 1420; //mm
	// camera set up (rotation and translation)
	//Mat W(4, 4, CV_64FC1);
	W.at<double>(0, 0) = 1.0;
	W.at<double>(3, 3) = 1.0;
	W.at<double>(0, 1) = W.at<double>(0, 2) = W.at<double>(0, 3) = W.at<double>(1, 0) = W.at<double>(1, 3) = W.at<double>(2, 0) = W.at<double>(3, 0) = W.at<double>(3, 1) = W.at<double>(3, 2) = 0.0;
	W.at<double>(2, 3) = cam_height;
	W.at<double>(1, 1) = cos(theta* CV_PI / 180); W.at<double>(2, 2) = cos(theta* CV_PI / 180); W.at<double>(1, 2) = sin(theta* CV_PI / 180); W.at<double>(2, 1) = -1 * sin(theta* CV_PI / 180);
	WQ = W * Q;
	WQ.convertTo(WQ, CV_32FC1);

	//cout << " W = " << W << " Q = " << Q << " Cam = " << Cam << endl;
	//Q.convertTo(Q, CV_32F);
	/*cout << Q.rows << Q.cols << Q << Q.type() << Q.isContinuous() << endl;
	cout << W.rows << W.cols << W << W.type() << W.isContinuous() << endl;
	cout << WQ.rows << WQ.cols << WQ << WQ.type() << WQ.isContinuous() << endl;*/

}

void prepare_remap(Size2i size, int image_resolution, cuda::GpuMat& d_rmap0_left, cuda::GpuMat& d_rmap1_left, cuda::GpuMat& d_rmap0_right, cuda::GpuMat& d_rmap1_right) {


	Mat cam_left, cam_right, dist_left, dist_right, R_left, R_right, P_left, P_right, rmap0_left, rmap0_right, rmap1_left, rmap1_right, R, T, Q;
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
	FileStorage Efs("C:/Users/CSK/source/repos/mlp_analysis/mlp_analysis/" + extrinsic_parameter_file, FileStorage::READ);
	Efs["R"] >> R;
	Efs["T"] >> T;

	cout << R << T << endl;

	FileStorage Ifs("C:/Users/CSK/source/repos/mlp_analysis/mlp_analysis/" + intrinsic_parameter_file, FileStorage::READ);
	Ifs["M1"] >> cam_left;
	Ifs["M2"] >> cam_right;
	Ifs["D1"] >> dist_left;
	Ifs["D2"] >> dist_right;

	cout << cam_left << cam_right << endl;

	Rect validRoi[2];
	stereoRectify(cam_left, dist_left, cam_right, dist_right, size, R, T, R_left, R_right, P_left, P_right, Q, CALIB_ZERO_DISPARITY, 1, size, &validRoi[0], &validRoi[1]);

	initUndistortRectifyMap(cam_left, dist_left, R_left, P_left, size, CV_32FC1, rmap0_left, rmap1_left);
	initUndistortRectifyMap(cam_right, dist_right, R_right, P_right, size, CV_32FC1, rmap0_right, rmap1_right);

	//upload to gpumat
	d_rmap0_left.upload(rmap0_left);
	d_rmap1_left.upload(rmap1_left);
	d_rmap0_right.upload(rmap0_right);
	d_rmap1_right.upload(rmap1_right);

	cout << P_left << P_right << endl;

}


void sgm_reconstruct( Mat h_left, Mat h_right, Mat& filtered_pointcloud, int image_resolution, int max_x, int max_y, int width, int height) {

	int disp_size;

	switch (image_resolution)
	{
	case 0:
		//2k
		disp_size = 128;
		break;

	case 1:
		//1080p
		disp_size = 128;
		break;

	case 2:
		//720p
		disp_size = 64;
		break;
	}

	//sgm & clahe
	sgm::LibSGMWrapper sgmw{ disp_size, 10, 120, 0.99f, true };
	//Ptr<cuda::CLAHE> clahe = cuda::createCLAHE(4.0, cv::Size(5, 15));
	//Ptr<StereoSGBM> sgbm = StereoSGBM::create(0, 48, 19, 0, 0);
	//Ptr<cuda::DisparityBilateralFilter> bfilter = cuda::createDisparityBilateralFilter(64, 50, 1);
	/*bfilter->setMaxDiscThreshold(0.9);
	bfilter->setEdgeThreshold(0.9);
	bfilter->setSigmaRange(20.0);*/

	//sgbm->setMode(0);

	//cropped and clahe
	Mat left_cropped, right_cropped, cleft, cright;
	

	//disparity map
	Mat disparity, scaleDisparityMap;
	cuda::GpuMat d_left, d_right, d_disparity, fd_disparity;
	double max_d, min_d;

	left_cropped = h_left(cv::Rect(max_x, max_y, width, height));
	right_cropped = h_right(cv::Rect(max_x, max_y, width, height));

	imshow("raw", left_cropped);
	//imwrite("raw image 2.png", left_cropped);

	d_left.upload(left_cropped);
	d_right.upload(right_cropped);

	//clahe->apply(d_left, d_left);
	//clahe->apply(d_right, d_right);
	//d_left.download(cleft);
	//imshow("clahe left", cleft);

	// reprojection to 3D
	Mat project_3D(left_cropped.size(), CV_32FC3);
	cuda::GpuMat d_project_3D(left_cropped.size(), CV_32FC3);

	//threshold
	Mat pointcloud_tresh_X, pointcloud_tresh_Y, pointcloud_tresh_Z, thresholded_XL, thresholded_XR;
	vector<Mat> channels(3);
	//vector<Mat> pointcloud_channels(3);
	//vector<Mat> pointcloud_merge;
	//Mat pointcloud_tresh_XYZ;
	Mat filter_roi_far, filter_roi_near;
	Mat thresholded_X, thresholded_Y, thresholded_Z, thresholded_XY, thresholded_XYU, thresholded_XYZU, thresholded_XYZ;


	try {

		workBegin();

		sgmw.execute(d_left, d_right, d_disparity);
		//sgbm->compute(left_cropped, right_cropped, disparity);

		workEnd();
	}
	catch (const cv::Exception& e) {
		std::cerr << e.what() << std::endl;
	}

	
	//subpixel
	double scale = 1.0 / 16.0;

	//cuda sgm
	cuda::GpuMat d_filter_disparity, d_filterd_disparity, d_subpix_disparity;
	/*d_disparity.convertTo(d_filter_disparity, CV_16SC1);
	bfilter->apply(d_filter_disparity, d_left, d_filterd_disparity);
	d_filterd_disparity.convertTo(d_subpix_disparity, CV_32FC1, scale);*/
	//d_disparity.upload(disparity);
	d_disparity.convertTo(d_subpix_disparity, CV_32FC1, scale);
	d_subpix_disparity.download(disparity);

	

	
	//Ptr<ximgproc::GuidedFilter> guidedfilter = ximgproc::createGuidedFilter();
	//guidedfilter->filter();
	

	Ptr<cuda::Filter> gaussian = cuda::createGaussianFilter(CV_32FC1, CV_32FC1, Size(5, 5), 10.0, 10.0);
	//gaussian->apply(d_subpix_disparity, d_subpix_disparity);
	/*try {
		cuda::bilateralFilter(d_subpix_disparity, d_subpix_disparity, 11, 10.0, 100.0);
	}
	catch (const cv::Exception& e) {
		std::cerr << e.what() << std::endl;
	}*/
	//d_subpix_disparity.download(disparity);
	//d_subpix_disparity = d_subpix_disparity(Rect(0, 0, 1148, 535));
	//d_subpix_disparity.download(disparity);
	//cout << disparity(Rect(574, 0, 1, 535)) << endl;
	
	//////medianBlur(disparity, disparity, 11);
	
	//Mat iteration;
	//GaussianBlur(disparity, iteration, Size(5, 5), 5, 5, cv::BORDER_ISOLATED);
	//ximgproc::guidedFilter(disparity, disparity, iteration, 10, 3.0);
	//ximgproc::guidedFilter(iteration, disparity, disparity, 10, 3.0);
	//ximgproc::guidedFilter(disparity, disparity, disparity, 20, 20.0);

	ximgproc::guidedFilter(disparity, disparity, disparity, 5, 5.0);
	
	

	//cout << disparity(Rect(574, 0, 1, 534)) << endl;
	d_subpix_disparity.upload(disparity);

	//system("PAUSE");

	//cout << disparity(cv::Rect(640, 500, 50, 5)) << endl;
	//disparity.convertTo(disparity, CV_32FC1, scale);
	//fd_disparity.upload(disparity);
	//check disparity type
	/*string ty = type2str(fd_disparity.type());
	printf("Matrix: %s %dx%d \n", ty.c_str(), fd_disparity.cols, fd_disparity.rows);
	cout << disparity(cv::Rect(640, 500, 5, 50)) << endl;*/

	Mat reprojection_mat, rotation_mat(4, 4, CV_64FC1);
	Matx33d cam_mat;

	load_camera_parameter(reprojection_mat, cam_mat, rotation_mat, image_resolution, max_x, max_y, width, height);
	//cout << reprojection_mat << endl;

	//reproject to 3D
	cuda::reprojectImageTo3D(d_subpix_disparity, d_project_3D, reprojection_mat, 3);
	//cuda::reprojectImageTo3D(fd_disparity, d_project_3D, reprojection_mat, 3);
	/*string ty = type2str(d_project_3D.type());
	printf("Matrix: %s %dx%d \n", ty.c_str(), d_project_3D.cols, d_project_3D.rows);*/
	d_project_3D.download(project_3D);

	// normalize result and show color map
	minMaxIdx(disparity, &min_d, &max_d);
	//cout << "disp min = " << min_d << endl << "disp max = " << max_d << endl;
	convertScaleAbs(disparity, scaleDisparityMap, 256 / (max_d - min_d));
	applyColorMap(scaleDisparityMap, scaleDisparityMap, COLORMAP_HOT);
	//putText(scaleDisparityMap, text(), Point(5, 25), FONT_HERSHEY_SIMPLEX, 1.0, Scalar::all(255));
	imshow("disparity map", scaleDisparityMap);
	imwrite(" filtered disparitymap 0.png", scaleDisparityMap);

	// threshold ROI
	split(project_3D, channels);
	pointcloud_tresh_X = channels[0];
	pointcloud_tresh_Y = channels[1];
	pointcloud_tresh_Z = channels[2];
	/*string ty = type2str(project_3D.type());
	printf("Matrix: %s %dx%d \n", ty.c_str(), project_3D.cols, project_3D.rows);
	cout << pointcloud_tresh_X(cv::Rect(640, 500, 5, 50)) << endl;*/
	inRange(pointcloud_tresh_X, -1200.0f, 1200.0f, thresholded_X);
	thresholded_X.convertTo(thresholded_X, CV_8UC1);
	inRange(pointcloud_tresh_X, -910.0f, -710.0f, thresholded_XL);
	thresholded_XL.convertTo(thresholded_XL, CV_8UC1);
	inRange(pointcloud_tresh_X, 710.0f, 910.0f, thresholded_XR);
	thresholded_XR.convertTo(thresholded_XR, CV_8UC1);
	thresholded_X = thresholded_XL + thresholded_XR;

	inRange(pointcloud_tresh_Y, 800.0f, 6000.0f, thresholded_Y);
	thresholded_Y.convertTo(thresholded_Y, CV_8UC1);
	inRange(pointcloud_tresh_Z, -1000.0f, 500.0f, thresholded_Z);
	thresholded_Z.convertTo(thresholded_Z, CV_8UC1);

	compare(thresholded_X, thresholded_Y, thresholded_XYU, CMP_NE);
	thresholded_XY = thresholded_X + thresholded_Y - thresholded_XYU;
	compare(thresholded_Z, thresholded_XY, thresholded_XYZU, CMP_NE);
	thresholded_XYZ = thresholded_XY + thresholded_Z - thresholded_XYZU;
	//imshow("X", thresholded_X);
	//imshow("Y", thresholded_Y);
	//imshow("Z", thresholded_Z);
	//imshow("threshold XYZ", thresholded_XY);
	//imwrite("X.png", thresholded_X);
	//imwrite("Y.png", thresholded_Y);
	//imwrite("Z.png", thresholded_Z);
	//imwrite("XYZ.png", thresholded_XYZ);

	project_3D.copyTo(filtered_pointcloud, thresholded_XYZ);

	//left_cropped = left_cropped + thresholded_XYZ;
	//imwrite("path.png", left_cropped);
	//project_3D.copyTo(filtered_pointcloud, thresholded_XY);

	//project_3D.copyTo(filtered_pointcloud, noArray());
	
	

}

void save_analysis(int model_type, int image_resolution, int max_x, int max_y, int width, int height) {

	int key = cv::waitKey();

	//tile parameter
	int model_tile_num;
	int tile_num = (6025 - 975) / 50;
	double tile_start = 975.0;
	double tile_end = 6025.0;

	//read from imagelist
	string imagelistfn;
	string model;
	Size2i image_size;

	switch (image_resolution)
	{
	case 0:
		image_size = Size2i(2208, 1242);
		switch (model_type)
		{
		case 0:
			//bump
			imagelistfn = "C:/Users/CSK/source/repos/mlp_analysis/mlp_analysis/mlp-data/2kbump/bump.xml";
			model = "2kbump";
			model_tile_num = 13;
			break;

		case 1:
			//pothole
			imagelistfn = "C:/Users/CSK/source/repos/mlp_analysis/mlp_analysis/mlp-data/2kpothole/pothole.xml";
			model = "2kpothole";
			model_tile_num = 15;
			break;

		case 2:
			//slope
			imagelistfn = "C:/Users/CSK/source/repos/mlp_analysis/mlp_analysis/mlp-data/2kslope/slope.xml";
			model = "2kslope";
			model_tile_num = 10;
			break;

		case 3:
			//plane
			imagelistfn = "C:/Users/CSK/source/repos/mlp_analysis/mlp_analysis/mlp-data/2kplane/plane.xml";
			model = "2kplane";
			model_tile_num = 83;
			break;
		}
		break;

	case 1:
		image_size = Size2i(1920, 1080);
		switch (model_type)
		{
		case 0:
			//bump
			imagelistfn = "C:/Users/CSK/source/repos/mlp_analysis/mlp_analysis/mlp-data/1080bump/bump.xml";
			model = "1080bump";
			model_tile_num = 13;
			break;

		case 1:
			//pothole
			imagelistfn = "C:/Users/CSK/source/repos/mlp_analysis/mlp_analysis/mlp-data/1080pothole/pothole.xml";
			model = "1080pothole";
			model_tile_num = 15;
			break;

		case 2:
			//slope
			imagelistfn = "C:/Users/CSK/source/repos/mlp_analysis/mlp_analysis/mlp-data/1080slope/slope.xml";
			model = "1080slope";
			model_tile_num = 10;
			break;

		case 3:
			//plane
			imagelistfn = "C:/Users/CSK/source/repos/mlp_analysis/mlp_analysis/mlp-data/1080plane/plane.xml";
			model = "1080plane";
			model_tile_num = 83;
			break;
		}
		break;

	case 2:
		switch (model_type)
		{
		case 0:
			//bump
			imagelistfn = "C:/Users/CSK/source/repos/mlp_analysis/mlp_analysis/mlp-data/bump/bump.xml";
			model = "bump";
			model_tile_num = 13;
			break;

		case 1:
			//pothole
			imagelistfn = "C:/Users/CSK/source/repos/mlp_analysis/mlp_analysis/mlp-data/pothole/pothole.xml";
			model = "pothole";
			model_tile_num = 15;
			break;

		case 2:
			//slope
			imagelistfn = "C:/Users/CSK/source/repos/mlp_analysis/mlp_analysis/mlp-data/slope/slope.xml";
			model = "slope";
			model_tile_num = 10;
			break;

		case 3:
			//plane
			imagelistfn = "C:/Users/CSK/source/repos/mlp_analysis/mlp_analysis/mlp-data/plane/plane.xml";
			model = "plane";
			model_tile_num = 83;
			break;
		}
		break;

	}

	

	vector<string> imagelist;
	bool ok = readStringList(imagelistfn, imagelist);
	if (!ok || imagelist.empty())
	{
		cout << "can not open " << imagelistfn << " or the string list is empty" << endl;
	}

	//for data output
	ofstream myfile( model + " wheel track analysis.csv");
	ofstream ifile( model +"merge.csv");
	if (myfile.is_open() && ifile.is_open())
	{
		myfile << " Y " << "," << " M_Z " << "," << " Z_Z " << "," << " K_Z " << "," << " MM_Z " 
			<< "," << " M_num of points " << "," << " Z_num of points " << "," << " K_num of points " 
			<< "," << " ground truth " << "," << " M_error " << "," << " Z_error " << "," << " K_error " << "," << " MM_error"
			<< "\n";

		ifile << " filter result " << "\n";

		if (imagelist.size() % 2 != 0)
		{
			cout << "Error: the image list contains odd (non-even) number of elements\n";
		}

		int nimages = (int)imagelist.size() / 2;
		
		// elevation parameter
		double model_start = 1000.0;
		int point_count = 0;
		//mean
		double elv_per_tile = 0.0; 
		//z score
		double sum_Z = 0.0;
		int count_Z = 0;
		vector<double> mean;
		vector<double> std;
		vector<float> inner_points_z;
		Mat inner_points, hist;
		//kmeans
		vector<Point2f> inner_points_k;
		Mat centers, centers_1;
		vector<int> bestLabels;
		vector<int> bestLabels_1;
		//filter result
		vector<float> road_height;
		Mat road_height_mat;
		//int icount = 0;
		

		for (int i = 0; i < nimages; i++)
		{
			
			const string& filename_left = "C:/Users/CSK/source/repos/mlp_analysis/mlp_analysis/mlp-data/" + model + "/" + imagelist[i * 2];
			const string& filename_right = "C:/Users/CSK/source/repos/mlp_analysis/mlp_analysis/mlp-data/" + model + "/" + imagelist[i * 2 + 1];
			
			Mat left = imread(filename_left, IMREAD_GRAYSCALE);
			Mat right = imread(filename_right, IMREAD_GRAYSCALE);
			
			if (left.empty() || right.empty())
				break;

			else if (left.size() != right.size())
			{
				cout << "The image " << filename_left << " has the size different from the first image size. Skipping the pair\n";
				break;
			}

			//Mat left_rectified, right_rectified;
			cuda::GpuMat d_left_raw, d_right_raw, d_left_rectified, d_right_rectified;
			Mat reprojection_mat, rotation_mat(4, 4, CV_64FC1);
			Matx33d cam_mat;
			//load camera parameter
			load_camera_parameter(reprojection_mat, cam_mat, rotation_mat, image_resolution, max_x, max_y, width, height);

			if (image_resolution == 0 || image_resolution == 1)
			{

				//prepare for remap
				cuda::GpuMat d_rmap00, d_rmap01, d_rmap10, d_rmap11;
				prepare_remap(image_size, image_resolution, d_rmap00, d_rmap01, d_rmap10, d_rmap11);

				//remap
				d_left_raw.upload(left);
				d_right_raw.upload(right);
				cuda::remap(d_left_raw, d_left_rectified, d_rmap00, d_rmap01, INTER_LINEAR);
				cuda::remap(d_right_raw, d_right_rectified, d_rmap10, d_rmap11, INTER_LINEAR);
				d_left_rectified.download(left);
				d_right_rectified.download(right);

			}

			Mat pointcloud;
			sgm_reconstruct(left, right, pointcloud, image_resolution, max_x, max_y, width, height);
			
			//save mean zscore kmeans data
			for (int k = 0; k < tile_num; k++) {

				workBegin();
				int nCols = (int)pointcloud.cols;
				int nRows = (int)pointcloud.rows;

				double tile_center = ((tile_start + k * 50) + (tile_start + (k + 1) * 50)) / 2;
				double ground_truth;
				double model_center;

				//ground truth
				switch (model_type)
				{
				case 0:
					//bump
					model_center = (model_start + model_start + 600) / 2;
					if ( tile_center >= model_start && tile_center <= model_start + 600 )
					{
						ground_truth = sqrt(pow(500.0, 2) - pow(abs(tile_center - model_center), 2)) - 400.0;
					}
					else
					{
						ground_truth = 0.0f;
					}
					
					break;

				case 1:
					//pothole
					model_center = (model_start + model_start + 700) / 2;
					if (tile_center >= model_start && tile_center <= model_start + 700)
					{
						ground_truth = 120.0 - (sqrt(pow(500.0, 2) - pow(abs(tile_center - model_center), 2)) - 400.0);
						if (tile_center == model_start || tile_center == model_start + 700)
						{
							ground_truth = 120.0;
						}
					}
					else
					{
						ground_truth = 0.0f;
					}
					
					
					break;

				case 2:
					//slope
					if (tile_center >= model_start && tile_center <= model_start +450)
					{
						ground_truth = (tile_center - model_start) * 120.0 / 450.0;
					}
					else
					{
						ground_truth = 0.0f;
					}
					break;

				case 3:
					//plane
					ground_truth = 0.0;
					break;
				}

				for (int j = 0; j < nRows; j++) {
					for (int i = 0; i < nCols; i++) {
						if (pointcloud.at<Vec3f>(j, i)[0] >= 0.0f && pointcloud.at<Vec3f>(j, i)[0] <= 245.0f
							&& pointcloud.at<Vec3f>(j, i)[1] >= tile_start + k * 50.0f && pointcloud.at<Vec3f>(j, i)[1] <= tile_start + (k + 1) * 50.0f)
						{
							elv_per_tile += pointcloud.at<Vec3f>(j, i)[2]; // mean
							inner_points_z.push_back(pointcloud.at<Vec3f>(j, i)[2]); //z score
							inner_points_k.push_back(Point2f(pointcloud.at<Vec3f>(j, i)[1], pointcloud.at<Vec3f>(j, i)[2])); //kmeans
							point_count++;
						}
						else
						{
							continue;
						}
					}
				}
				if (point_count != 0) {

					//show histogram of each tile
					inner_points.push_back(inner_points_z); //histogram
					double max_d, min_d;
					float max, min;
					minMaxIdx(inner_points, &min_d, &max_d);
					max = (float)max_d;
					min = (float)min_d;
					float range[] = { min, max };
					const float* histrange = { range };
					int histsize = 10;
					calcHist(&inner_points, 1, 0, Mat(), hist, 1, &histsize, &histrange, true, false);
					
					int hist_w = 512, hist_h = 400;
					int bin_w = cvRound((double)hist_w / histsize);
					Mat histimage(hist_h, hist_w, CV_8UC3, Scalar(245, 245, 245));
					normalize(hist, hist, 0, histimage.rows, NORM_MINMAX, -1, Mat());
					float total = 0.0f;
					float num_count = 0.0f;
					float binsize = (max - min) / histsize;

					for (int i = 1; i < histsize; i++)
					{
						line(histimage, Point(bin_w*(i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
							Point(bin_w*(i), hist_h - cvRound(hist.at<float>(i))),
							Scalar(0, 0, 255), 3, 8, 0);
						total = total + hist.at<float>(i - 1);

					}

					for (int i = 1; i < histsize; i++)
					{
						num_count = num_count + hist.at<float>(i - 1);
						if (num_count >= total/2)
						{
							line(histimage, Point(bin_w*(i - 1), 400),
								Point(bin_w*(i-1), 0),
								Scalar(0, 255, 0), 3, 8, 0);
							break;
						}
					}


					meanStdDev(inner_points_z, mean, std);
					for (int i = 1; i < histsize; i++)
					{
						
						if ( min + binsize*(i-1) <= mean[0] && min + binsize*i > mean[0] )
						{
							line(histimage, Point(bin_w*(i - 1), 400),
								Point(bin_w*(i - 1), 0),
								Scalar(255, 0, 0), 3, 8, 0);
							break;
						}
					}

					
					//stringstream name;
					//name << "hist_" << icount << ".png";
					imshow("Hist", histimage);
					imwrite("hist.png", histimage);
					waitKey();
					//icount++;
					//median value
					float median_z;
					median_z = median_val(inner_points_z);
					road_height.push_back(median_z);
					

					// zscore
					//meanStdDev(inner_points_z, mean, std);
					for (int i = 0; i < inner_points_z.size(); i++)
					{
						if (inner_points_z[i] >= (mean[0] - 1 * std[0]) && inner_points_z[i] <= (mean[0] + 1 * std[0]))
						{
							sum_Z += inner_points_z[i];
							count_Z++;
						}
					}

					// kmeans
					if (point_count >= 2) {
						kmeans(inner_points_k, 2, bestLabels, TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 3, 1e-4), 1, KMEANS_PP_CENTERS, centers);
						kmeans(inner_points_k, 1, bestLabels_1, TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 3, 1e-4), 1, KMEANS_PP_CENTERS, centers_1);
						int indx0 = 0, indx1 = 0;
						double center_different = fabs(centers.at<float>(0, 1) - centers.at<float>(1, 1));
						for (int i = 0; i < bestLabels.size(); i++)
						{

							if (bestLabels[i] == 0)
							{
								indx0++;
							}
							else if (bestLabels[i] == 1)
							{
								indx1++;
							}
						}
						if (center_different >= 50)
						{

							/*myfile << " Y " << "," << " M_Z " << "," << " Z_Z " << "," << " K_Z "
						<< "," << " M_num of points " << "," << " Z_num of points " << "," << " K_num of points "
						<< "," << " bump ground truth " << "," << " M_error " << "," << " Z_error " << "," << " K_error "
						<< "," << " M_rmse " << "," << " Z_rmse " << "," << " K_rmse " << "\n";*/

							if (indx0 >= indx1)
							{
								myfile << tile_center << " , "
									<< double(elv_per_tile / (double)point_count) << " , " << double(sum_Z / (double)count_Z) << " , " << centers.at<float>(0, 1) << " , " << median_z << " , "
									<< point_count << " , " << count_Z << " , " << indx0 << " , "
									<< ground_truth << " , "
									<< ground_truth - (double(elv_per_tile / (double)point_count)) << " , " << ground_truth - double(sum_Z / (double)count_Z) << " , " << ground_truth - centers.at<float>(0, 1) << " , " << ground_truth - median_z << "\n";
							}
							else
							{
								myfile << tile_center << " , "
									<< double(elv_per_tile / (double)point_count) << " , " << double(sum_Z / (double)count_Z) << " , " << centers.at<float>(1, 1) << " , " << median_z << " , "
									<< point_count << " , " << count_Z << " , " << indx1 << " , "
									<< ground_truth << " , "
									<< ground_truth - (double(elv_per_tile / (double)point_count)) << " , " << ground_truth - double(sum_Z / (double)count_Z) << " , " << ground_truth - centers.at<float>(1, 1) << " , " << ground_truth - median_z << "\n";
							}
						}
						else
						{
							myfile << tile_center << " , "
								<< double(elv_per_tile / (double)point_count) << " , " << double(sum_Z / (double)count_Z) << " , " << centers_1.at<float>(0, 1) << " , " << median_z << " , "
								<< point_count << " , " << count_Z << " , " << point_count << " , "
								<< ground_truth << " , "
								<< ground_truth - (double(elv_per_tile / (double)point_count)) << " , " << ground_truth - double(sum_Z / (double)count_Z) << " , " << ground_truth - centers_1.at<float>(0, 1) << " , " << ground_truth - median_z << "\n";
						}

					}

					else
					{
						/*myfile << " Y " << "," << " M_Z " << "," << " Z_Z " << "," << " K_Z "
						<< "," << " M_num of points " << "," << " Z_num of points " << "," << " K_num of points "
						<< "," << " bump ground truth " << "," << " M_error " << "," << " Z_error " << "," << " K_error "
						<< "," << " M_rmse " << "," << " Z_rmse " << "," << " K_rmse " << "\n";*/

						myfile << tile_center << " , " 
							<< double(elv_per_tile / (double)point_count) << " , " << double(sum_Z / (double)count_Z) << " , " << elv_per_tile << " , " << median_z << " , "
							<< point_count << " , " << count_Z << " , " << point_count << " , "
							<< ground_truth << " , " 
							<< ground_truth - (double(elv_per_tile / (double)point_count)) << " , " << ground_truth - double(sum_Z / (double)count_Z) << " , " << ground_truth - elv_per_tile << " , " << ground_truth - median_z << "\n";
					}
					
					

				}
				point_count = 0;
				elv_per_tile = 0;
				count_Z = 0;
				sum_Z = 0;
				inner_points_z.clear();
				inner_points_k.clear();

				
				
				
			}

			//filter result
			//Mat guided_filter = Mat::zeros(Size(3,101), CV_32FC1);
			road_height_mat.push_back(road_height);
			//Mat zero = Mat::zeros(Size(1,road_height_mat.rows), CV_32FC1);
			//Mat guided_filter;
			//hconcat(zero, road_height_mat, guided_filter);
			//hconcat(guided_filter, zero, guided_filter);
			//ximgproc::guidedFilter(guided_filter, guided_filter, guided_filter, 1 , 2.0);
			//cout << guided_filter << endl;

			GaussianBlur(road_height_mat, road_height_mat, Size(1, 3), 0, 0.5, cv::BORDER_REPLICATE);
			for (int w = 0; w < road_height_mat.rows; w++)
			{
				ifile << road_height_mat.at<float>(w,0) << "\n";
			}
			
			//cout << road_height_mat << endl;
			road_height.clear();
			road_height_mat.release();

			model_start += 50;

			//cout << model_start << endl;
			
			
			if (key == 27) break;
			key = cv::waitKey(100);

		}

		

	}
	else
		cout << "Error: can not save the data\n";


	myfile.close();
	ifile.close();
	cv::destroyAllWindows();
	
}

void vtk_show(Mat Wpointcloud, Mat Wleft, Mat Wright, Matx33d Cam, Mat W, int max_x, int max_y, int width, int height) {

	//load_camera_parameter
	//viz show
	Mat R = W(cv::Rect(0, 0, 3, 3));
	Rodrigues(R, R);

	Wleft = Wleft(cv::Rect(max_x, max_y, width, height));
	Wright = Wright(cv::Rect(max_x, max_y, width, height));
	//imwrite("left raw image 4.png", Wleft);
	//Wleft = Wleft(cv::Rect(57, 82, 1148, 550));
	//Wright = Wright(cv::Rect(57, 82, 1148, 550));
	// world coordinate and camera pos
	viz::Viz3d myWindow("Coordinate Frame");
	myWindow.setBackgroundColor(viz::Color::white());
	myWindow.showWidget("Coordinate Widget", viz::WCoordinateSystem(600));
	viz::WCameraPosition camL(Cam, Wleft, 500, viz::Color::white());
	viz::WCameraPosition campL(200);
	viz::WCameraPosition camR(Cam, Wright, 500, viz::Color::gray());
	viz::WCameraPosition campR(200);
	Vec3d cam_posL(0.0, 0.0, 1360);
	Vec3d cam_posR(119.0, 0.0, 1360);
	Affine3d cam_poseL(R, cam_posL);
	Affine3d cam_poseR(R, cam_posR);

	//track_L
	const Point3d planeCenter_L(-810, 3500.0, 0.0);
	const Vec3d planeNormal(0.0, 0.0, 1.0);
	const Vec3d planeY(0.0, 1.0, 0.0);
	const Vec2i planeCell(1, 100);
	const Vec2d planeCellSize(245, 50);
	viz::WGrid wheel_plane_L(planeCenter_L, planeNormal, planeY, planeCell, planeCellSize, viz::Color::maroon());

	//track_R
	const Point3d planeCenter_R(810, 3500.0, 0.0);
	viz::WGrid wheel_plane_R(planeCenter_R, planeNormal, planeY, planeCell, planeCellSize, viz::Color::maroon());

	//plane
	const Point3d planeCenter_p(0.0, 3500.0, 0.0);
	const Vec3d planeNormal_p(0.0, 0.0, 1.0);
	const Vec3d planeY_p(0.0, 1.0, 0.0);
	viz::WPlane plane(planeCenter_p, planeNormal_p, planeY_p, Size2d(2400, 2500), viz::Color::silver());

	//viz pointcloud
	destroyAllWindows();
	viz::WCloud roadCloud(Wpointcloud, Wleft);
	//viz::WCloud roadCloud(project_3D, left_cropped);
	myWindow.showWidget("road cloud", roadCloud);
	//myWindow.showWidget("CameraL", camL, cam_poseL);
	//myWindow.showWidget("CamerapL", campL, cam_poseL);
	//myWindow.showWidget("CameraR", camR, cam_poseR);
	//myWindow.showWidget("CamerapR", campR, cam_poseR);
	//myWindow.showWidget("track_L", wheel_plane_L);
	//myWindow.showWidget("track_R", wheel_plane_R);
	myWindow.showWidget("plane", plane);
	myWindow.spin();

	myWindow.removeAllWidgets();
	myWindow.close();

}


int main() {

	int key = cv::waitKey();
	int resolution = 2;
	Size2i image_size;
	int max_x, max_y, width, height;

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

	
	//save data 0 = bump, 1 = pothole, 2 = slope, 3 = plane 
	//save_analysis(1, resolution, 1, 0, 0, 0);

	

	 //read stereo pair
	//Mat left = imread("C:/Users/CSK/source/repos/mlp_analysis/mlp_analysis/mlp-data/bump/left_model_one_0.png", 0);
	//Mat right = imread("C:/Users/CSK/source/repos/mlp_analysis/mlp_analysis/mlp-data/bump/right_model_one_0.png", 0);
	//Mat left = imread("C:/Users/CSK/source/repos/mlp_analysis/mlp_analysis/mlp-data/2kbump/left_model_one2kraw_0.png", 0);
	//Mat right = imread("C:/Users/CSK/source/repos/mlp_analysis/mlp_analysis/mlp-data/2kbump/right_model_one2kraw_0.png", 0);
	//Mat left = imread("C:/Users/CSK/source/repos/mlp_analysis/mlp_analysis/mlp-data/1080pothole/left_model_two1080raw_1.png", 0);
	//Mat right = imread("C:/Users/CSK/source/repos/mlp_analysis/mlp_analysis/mlp-data/1080pothole/right_model_two1080raw_1.png", 0);
	Mat left = imread("left_road15.png", 0);
	Mat right = imread("right_road15.png", 0);
	if (left.empty()) throw runtime_error("can't open left file");
	if (right.empty()) throw runtime_error("can't open right file");
	

	//Mat left_rectified, right_rectified;
	cuda::GpuMat d_left_raw, d_right_raw, d_left_rectified, d_right_rectified;
	Mat reprojection_mat, rotation_mat(4, 4, CV_64FC1);
	Matx33d cam_mat;
	//load camera parameter
	load_camera_parameter(reprojection_mat, cam_mat, rotation_mat, resolution, max_x, max_y, width, height);

	if (resolution == 0 || resolution == 1)
	{

		//prepare for remap
		cuda::GpuMat d_rmap00, d_rmap01, d_rmap10, d_rmap11;
		prepare_remap(image_size, resolution, d_rmap00, d_rmap01, d_rmap10, d_rmap11);

		//remap
		d_left_raw.upload(left);
		d_right_raw.upload(right);
		cuda::remap(d_left_raw, d_left_rectified, d_rmap00, d_rmap01, INTER_LINEAR);
		cuda::remap(d_right_raw, d_right_rectified, d_rmap10, d_rmap11, INTER_LINEAR);
		d_left_rectified.download(left);
		d_right_rectified.download(right);

	}
	
	// runtime
	ofstream file("libsgm rumtime 3.txt");
	if (file.is_open())
	{
		file << "rumtime" << "\n";
	}

	else
		cout << "Error: can not save the data\n";

	Mat pointcloud;
	while (key != 27) {

	//workBegin();

	sgm_reconstruct(left, right, pointcloud, resolution, max_x, max_y, width, height);

	//workEnd();

	file << work_fps << "\n";
	
	key = cv::waitKey(1);
	if (key == 27) break;
	}

	file.close();

	Mat left_vtk = imread("left_road15.png");
	
	//vtk 3d view
	vtk_show(pointcloud, left_vtk, right, cam_mat, rotation_mat, max_x, max_y, width, height);


	return 0;
	
}
