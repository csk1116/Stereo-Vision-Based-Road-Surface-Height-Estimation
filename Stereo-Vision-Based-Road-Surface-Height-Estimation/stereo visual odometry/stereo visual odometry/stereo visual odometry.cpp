#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <algorithm>
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudafeatures2d.hpp"
#include "opencv2/xfeatures2d/cuda.hpp"
#include "opencv2/opencv_modules.hpp"



using namespace cv;
using namespace std;

//bool points_are_equal(const Point2f& p1, const Point2f& p2) {
//	return((p1.x == p2.x) && (p1.y == p2.y));
//}
//
//bool trainIdx_are_equal(const DMatch& train1, const DMatch& train2) {
//	return((train1.trainIdx == train2.trainIdx) && (train1.distance < train2.distance));
//}


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


double surf_gpu_feature_match(cuda::GpuMat cg_left, cuda::GpuMat pg_left, cuda::GpuMat pg_right) {

	//cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());

	workBegin();
	//Detect the keypoints and calculate the descriptors
	cuda::GpuMat dc_keypoints_1, dp_keypoints_1, dp_keypoints_2;
	cuda::GpuMat dc_descriptors_1, dp_descriptors_1, dp_descriptors_2, dp_descriptors_left;
	vector<KeyPoint> c_keypoints_1, p_keypoints_1, p_keypoints_2, p_keypoints_left, p_keypoints_right;
	Mat p_descriptors_left, p_descriptors_1, c_descriptors_1;

	double pitch;
	double roll;
	

	//Create SURF CUDA
	cuda::SURF_CUDA surf(500, 4, 3, false, 0.01f, false);

	surf(cg_left, cuda::GpuMat(), dc_keypoints_1, dc_descriptors_1);
	surf(pg_left, cuda::GpuMat(), dp_keypoints_1, dp_descriptors_1);
	surf(pg_right, cuda::GpuMat(), dp_keypoints_2, dp_descriptors_2);
	
	

	//-- Step 2: Matching descriptor vectors with a FLANN based matcher
   // Since SURF is a floating-point descriptor NORM_L2 is used
	Ptr< cuda::DescriptorMatcher > matcher = cuda::DescriptorMatcher::createBFMatcher(NORM_L2);
	//Ptr<DescriptorMatcher> matcher_flann = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
	std::vector< std::vector<DMatch> > p_knn_matches, cp_knn_matches;
	//-- Filter matches 
	std::vector<DMatch> p_good_matches, cp_good_matches;


	surf.downloadKeypoints(dp_keypoints_1, p_keypoints_1);
	surf.downloadKeypoints(dp_keypoints_2, p_keypoints_2);
	surf.downloadKeypoints(dc_keypoints_1, c_keypoints_1);

	//cout << "current num k1 = " << c_keypoints_1.size() << " , current num k2 = " << c_keypoints_2.size() << endl;
	//cout << "previous num k1 = " << p_keypoints_1.size() << " , previous num k2 = " << p_keypoints_2.size() << endl;

	

	// 3d to 2d method
	matcher->knnMatch(dp_descriptors_1, dp_descriptors_2, p_knn_matches, 2);

	for (int i = 0; i < p_knn_matches.size(); i++)
	{
		int p_left_idx = p_knn_matches[i][0].queryIdx;
		int p_right_idx = p_knn_matches[i][0].trainIdx;

		if ((p_knn_matches[i][0].distance < 0.8*(p_knn_matches[i][1].distance)) && fabs(p_keypoints_1[p_left_idx].pt.y - p_keypoints_2[p_right_idx].pt.y) <= 1.0 && p_keypoints_1[p_left_idx].pt.y > 0.0)
		{
			p_good_matches.push_back(p_knn_matches[i][0]);
		}
	}

	dp_descriptors_1.download(p_descriptors_1);
	dc_descriptors_1.download(c_descriptors_1);

	for (int i = 0; i < p_good_matches.size(); i++)
	{
		int p_left_idx = p_good_matches[i].queryIdx;
		int p_right_idx = p_good_matches[i].trainIdx;
		p_descriptors_left.push_back(p_descriptors_1.row(p_left_idx));
		p_keypoints_left.push_back(p_keypoints_1[p_left_idx]);
		p_keypoints_right.push_back(p_keypoints_2[p_right_idx]);
	}

	dp_descriptors_left.upload(p_descriptors_left);
	matcher->knnMatch(dp_descriptors_left, dc_descriptors_1, cp_knn_matches, 2);
	
	float ratio = 0.01f;
	bool pointsEnough = true;

	while (pointsEnough)
	{
		for (int i = 0; i < cp_knn_matches.size(); i++)
		{
			if (cp_knn_matches[i][0].distance < ratio*(cp_knn_matches[i][1].distance))
			{
				cp_good_matches.push_back(cp_knn_matches[i][0]);
			}
		}

		//30
		if (cp_good_matches.size() < 30)
		{
			cp_good_matches.clear();
			ratio += 0.01f;
		}

		else
		{
			pointsEnough = false;
		}
	}
	
	
	//cout << "previous match = " << p_good_matches.size() << endl;
	//cout << "previous current match = " << cp_good_matches.size() << endl;

	
		//load projection matrix Q
		FileStorage fs("C:/Users/CSK/source/repos/stereo visual odometry/stereo visual odometry/extrinsics.yml", FileStorage::READ);
		Mat Q;
		fs["Q"] >> Q;
		Q.at<double>(0, 3) += 57;
		Q.at<double>(1, 3) += 82;
		//cout << Q.rows << Q.cols << Q << Q.type() << Q.isContinuous() << endl;

		Mat M;
		//Mat D = Mat::zeros(1, 5, CV_32F);
		fs["P1"] >> M;
		M.at<double>(0, 2) -= 57;
		M.at<double>(1, 2) -= 82;
		M = M(cv::Rect(0, 0, 3, 3));


		vector<Point3d> object_points_previous;
		vector<Point2d> image_points_current;
		double disparity_previous;
		double object_x, object_y, object_z, object_w;
		bool too_far = true;
		double distance = 5000.0;

		while (too_far)
		{
			for (int i = 0; i < cp_good_matches.size(); i++)
			{
				int c_idx = cp_good_matches[i].trainIdx;
				int p_idx = cp_good_matches[i].queryIdx;
				disparity_previous = p_keypoints_left[p_idx].pt.x - p_keypoints_right[p_idx].pt.x;
				//cout << disparity_previous << endl;
				//cout << p_keypoints_left[p_idx].pt.x << "," <<p_keypoints_left[p_idx].pt.y << endl;

				object_w = disparity_previous * Q.at<double>(3, 2);
				object_z = Q.at<double>(2, 3) / object_w;
				object_x = (p_keypoints_left[p_idx].pt.x + Q.at<double>(0, 3)) / object_w;
				object_y = (p_keypoints_left[p_idx].pt.y + Q.at<double>(1, 3)) / object_w;

				if (object_z < distance)
				{
					image_points_current.push_back(c_keypoints_1[c_idx].pt);
					object_points_previous.push_back(Point3d(object_x, object_y, object_z));
				}
				//cout << object_w << endl;
				//cout << " X = " << object_x << " Y = " << object_y << " Z = " << object_z << endl;
			}

			if (image_points_current.size() < 20)
			{
				distance += 500.0;
				image_points_current.clear();
				object_points_previous.clear();
			}

			else
			{
				too_far = false;
				//cout << image_points_current.size() << endl;
			}

		}
		

		Mat R, T;
		
		//cuda::solvePnPRansac();
		solvePnPRansac(object_points_previous, image_points_current, M, noArray(), R, T, false, 100, 0.5f, 0.99, noArray(), SOLVEPNP_EPNP);
		//solvePnP(object_points_previous, image_points_current, M, noArray(), R, T, false, SOLVEPNP_EPNP);
		Rodrigues(R, R);
		//cout << " R = " << R << "\n" << " T = " << T << endl;

		//Rotation Matrix to Euler Angles
		Mat rotationX, rotationY, rotationZ, mR, mQ;
		RQDecomp3x3(R, mR, mQ, rotationX, rotationY, rotationZ);
		//cout << "X= " << rotationX << "\n" << "Y= " << rotationY << "\n" << "Z= " << rotationZ << endl;

		//pitch
		//pitch = asin(rotationX.at<double>(1, 2)) / CV_PI *180;
		//pitch = atan2()
		//cout << "vision pitch= " << pitch << endl;
		roll = asin(rotationZ.at<double>(0, 1)) / CV_PI * 180;
		//cout << "vision roll= " << roll << endl;
	
	workEnd();



	//-- Draw only "good" matches
	Mat c_left, p_left, p_right;
	cg_left.download(c_left);
	pg_left.download(p_left);
	pg_right.download(p_right);

	Mat p_img_matches, cp_img_matches;

	drawMatches(p_left, p_keypoints_1, p_right, p_keypoints_2,
		p_good_matches, p_img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	drawMatches(p_left, p_keypoints_left, c_left, c_keypoints_1,
		cp_good_matches, cp_img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);



	//draw keypoint
	Mat current_keypoints, previous_keypoints;
	drawKeypoints(p_left, p_keypoints_1, previous_keypoints);
	drawKeypoints(c_left, c_keypoints_1, current_keypoints);
	imshow("p1", previous_keypoints);
	imwrite("p1.png", previous_keypoints);
	imshow("c1", current_keypoints);
	imwrite("Current-previous Good Matches.png", cp_img_matches);
	//imwrite("Previous Good Matches.png", p_img_matches);

	//-- Show detected matches
	putText(cp_img_matches, text(), Point(5, 25), FONT_HERSHEY_SIMPLEX, 1.0, Scalar::all(255));
	//imshow("Current Good Matches", c_img_matches);
	//imshow("Previous Good Matches", p_img_matches);
	imshow("Current-previous Good Matches", cp_img_matches);
	
	/*if (fabs(pitch) > 20)
	{
		system("PAUSE");
	}*/

	//return pitch;
	return roll;

	


}

double orb_gpu_feature_match(cuda::GpuMat cg_left, cuda::GpuMat cg_right, cuda::GpuMat pg_left, cuda::GpuMat pg_right) {

	cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());

	workBegin();
	//Detect the keypoints and calculate the descriptors
	cuda::GpuMat dc_keypoints_1, dc_keypoints_2, dp_keypoints_1, dp_keypoints_2, dc_keypoints, dp_keypoints_left, dp_keypoints_right;
	cuda::GpuMat dc_descriptors_1, dc_descriptors_2, dp_descriptors_1, dp_descriptors_2, dc_descriptors, dp_descriptors_left;
	vector<KeyPoint> c_keypoints_1, c_keypoints_2, p_keypoints_1, p_keypoints_2, c_keypoints, p_keypoints_left, p_keypoints_right;
	Mat c_descriptors_1, c_descriptors_2, p_descriptors_1, p_descriptors_2, c_descriptors, p_descriptors_left;

	double pitch;

	//Create SURF CUDA
	Ptr<cuda::ORB> orb = cuda::ORB::create();

	orb->detectAndComputeAsync(cg_left, cuda::GpuMat(), dc_keypoints_1, dc_descriptors_1);
	orb->convert(dc_keypoints_1, c_keypoints_1);
	dc_descriptors_1.convertTo(dc_descriptors_1, CV_32F);

	orb->detectAndComputeAsync(cg_right, cuda::GpuMat(), dc_keypoints_2, dc_descriptors_2);
	orb->convert(dc_keypoints_2, c_keypoints_2);
	dc_descriptors_2.convertTo(dc_descriptors_2, CV_32F);

	orb->detectAndComputeAsync(pg_left, cuda::GpuMat(), dp_keypoints_1, dp_descriptors_1);
	orb->convert(dp_keypoints_1, p_keypoints_1);
	dp_descriptors_1.convertTo(dp_descriptors_1, CV_32F);

	orb->detectAndComputeAsync(pg_right, cuda::GpuMat(), dp_keypoints_2, dp_descriptors_2);
	orb->convert(dp_keypoints_2, p_keypoints_2);
	dp_descriptors_2.convertTo(dp_descriptors_2, CV_32F);

	cout << "current num k1 = " << c_keypoints_1.size() << " , current num k2 = " << c_keypoints_2.size() << endl;
	cout << "previous num k1 = " << p_keypoints_1.size() << " , previous num k2 = " << p_keypoints_2.size() << endl;
	/*Mat c_img_keypoints_1, c_img_keypoints_2;
	drawKeypoints(c_left, c_keypoints_1, c_img_keypoints_1);
	drawKeypoints(c_right, c_keypoints_2, c_img_keypoints_2);
	imshow("k1", img_keypoints_1);
	imwrite("k1.png", img_keypoints_1);
	imshow("k2", img_keypoints_2);
	imwrite("k2.png", img_keypoints_2);*/


	//-- Step 2: Matching descriptor vectors with a FLANN based matcher
   // Since ORB is a floating-point descriptor NORM_L2 is used
	Ptr< cuda::DescriptorMatcher > matcher = cuda::DescriptorMatcher::createBFMatcher(NORM_L2);
	std::vector< std::vector<DMatch> > c_knn_matches, p_knn_matches, cp_knn_matches;

	matcher->knnMatch(dp_descriptors_1, dp_descriptors_2, p_knn_matches, 2);
	dp_descriptors_1.download(p_descriptors_1);

	//-- Filter matches 
	std::vector<DMatch> c_good_matches, p_good_matches, cp_good_matches;

	for (int i = 0; i < p_knn_matches.size(); i++)
	{
		int p_left_idx = p_knn_matches[i][0].queryIdx;
		int p_right_idx = p_knn_matches[i][0].trainIdx;
		if ((p_knn_matches[i][0].distance < 1*(p_knn_matches[i][1].distance)) && fabs(p_keypoints_1[p_left_idx].pt.y - p_keypoints_2[p_right_idx].pt.y) <= 1)
		{
			p_good_matches.push_back(p_knn_matches[i][0]);
		}
	}

	for (int i = 0; i < p_good_matches.size(); i++)
	{
		int p_left_idx = p_good_matches[i].queryIdx;
		int p_right_idx = p_good_matches[i].trainIdx;
		p_descriptors_left.push_back(p_descriptors_1.row(p_left_idx));
		p_keypoints_left.push_back(p_keypoints_1[p_left_idx]);
		p_keypoints_right.push_back(p_keypoints_2[p_right_idx]);
	}

	dp_descriptors_left.upload(p_descriptors_left);

	matcher->knnMatch(dc_descriptors_1, dp_descriptors_left, cp_knn_matches, 2);

	float ratio = 0.1f;
	bool pointsEnough = true;

	while (pointsEnough)
	{
		for (int i = 0; i < cp_knn_matches.size(); i++)
		{
			if (cp_knn_matches[i][0].distance < ratio*(cp_knn_matches[i][1].distance))
			{
				cp_good_matches.push_back(cp_knn_matches[i][0]);
			}
		}

		if (cp_good_matches.size() < 20)
		{
			cp_good_matches.clear();
			ratio += 0.01f;
		}

		else
		{
			pointsEnough = false;
		}
	}

	cout << "previous match = " << p_good_matches.size() << endl;
	cout << "previous current match = " << cp_knn_matches.size() << endl;


	
		//load projection matrix Q
		FileStorage fs("C:/Users/CSK/source/repos/stereo visual odometry/stereo visual odometry/extrinsics.yml", FileStorage::READ);
		Mat Q;
		fs["Q"] >> Q;
		Q.at<double>(0, 3) += 57;
		Q.at<double>(1, 3) += 82;
		//cout << Q.rows << Q.cols << Q << Q.type() << Q.isContinuous() << endl;

		Mat M;
		Mat D = Mat::zeros(1, 5, CV_32F);
		fs["P1"] >> M;
		M.at<double>(0, 2) -= 57;
		M.at<double>(1, 2) -= 82;
		M = M(cv::Rect(0, 0, 3, 3));


		vector<Point3d> object_points_previous;
		vector<Point2d> image_points_current;
		double disparity_previous;
		double object_x, object_y, object_z, object_w;
		for (int i = 0; i < cp_good_matches.size(); i++)
		{
			int c_idx = cp_good_matches[i].queryIdx;
			image_points_current.push_back(c_keypoints_1[c_idx].pt);
			int p_idx = cp_good_matches[i].trainIdx;
			disparity_previous = p_keypoints_left[p_idx].pt.x - p_keypoints_right[p_idx].pt.x;
			object_w = disparity_previous * Q.at<double>(3, 2);
			object_z = Q.at<double>(2, 3) / object_w;
			object_x = (p_keypoints_left[p_idx].pt.x + Q.at<double>(0, 3)) / object_w;
			object_y = (p_keypoints_left[p_idx].pt.y + Q.at<double>(1, 3)) / object_w;
			object_points_previous.push_back(Point3d(object_x, object_y, object_z));

		}

		Mat R, T;
		solvePnPRansac(object_points_previous, image_points_current, M, D, R, T);
		Rodrigues(R, R);
		//cout << " R = " << R << " T = " << T << endl;

		//Rotation Matrix to Euler Angles
		Mat rotationX, rotationY, rotationZ, mR, mQ;
		RQDecomp3x3(R, mR, mQ, rotationX, rotationY, rotationZ);
		//cout << "X= " << rotationX << "\n" << "Y= " << rotationY << "\n" << "Z= " << rotationZ << endl;
		pitch = asin(rotationX.at<double>(1, 2));
		cout << "vision pitch= " << pitch << endl;
	

	workEnd();

	//-- Draw only "good" matches
	Mat c_left, p_left, p_right;
	cg_left.download(c_left);
	pg_left.download(p_left);
	pg_right.download(p_right);

	Mat c_img_matches, p_img_matches, cp_img_matches;
	/*drawMatches(c_left, c_keypoints_1, c_right, c_keypoints_2,
		c_good_matches, c_img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);*/

	drawMatches(p_left, p_keypoints_1, p_right, p_keypoints_2,
		p_good_matches, p_img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	drawMatches(c_left, c_keypoints_1, p_left, p_keypoints_left,
		cp_good_matches, cp_img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	//-- Show detected matches
	putText(cp_img_matches, text(), Point(5, 25), FONT_HERSHEY_SIMPLEX, 1.0, Scalar::all(255));
	//imshow("Current Good Matches", c_img_matches);
	imshow("Previous Good Matches", p_img_matches);
	imshow("Current-previous Good Matches", cp_img_matches);
	
	return pitch;

}

double e_surf_gpu_feature_match(cuda::GpuMat cg_left, cuda::GpuMat pg_left) {

	//cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());

	workBegin();
	//Detect the keypoints and calculate the descriptors
	cuda::GpuMat dc_keypoints_left, dp_keypoints_left;
	cuda::GpuMat dc_descriptors_left, dp_descriptors_left;
	vector<KeyPoint> c_keypoints_left, p_keypoints_left;
	Mat p_descriptors_left, c_descriptors_left;

	double pitch;
	double roll;


	//Create SURF CUDA
	cuda::SURF_CUDA surf(500, 4, 3, false, 0.01f, false);

	surf(cg_left, cuda::GpuMat(), dc_keypoints_left, dc_descriptors_left);
	surf(pg_left, cuda::GpuMat(), dp_keypoints_left, dp_descriptors_left);

	//Mat c_img_keypoints_1, c_img_keypoints_2;
	//drawKeypoints(c_left, c_keypoints_1, c_img_keypoints_1);
	//drawKeypoints(c_right, c_keypoints_2, c_img_keypoints_2);
	//imshow("k1", img_keypoints_1);
	//imwrite("k1.png", img_keypoints_1);
	//imshow("k2", img_keypoints_2);
	//imwrite("k2.png", img_keypoints_2);


	//-- Step 2: Matching descriptor vectors with a FLANN based matcher
   // Since SURF is a floating-point descriptor NORM_L2 is used
	Ptr< cuda::DescriptorMatcher > matcher = cuda::DescriptorMatcher::createBFMatcher(NORM_L2);
	std::vector< std::vector<DMatch> > cp_knn_matches;
	//-- Filter matches 
	std::vector<DMatch> cp_good_matches;


	surf.downloadKeypoints(dp_keypoints_left, p_keypoints_left);
	surf.downloadKeypoints(dc_keypoints_left, c_keypoints_left);
	

	//cout << "current num k1 = " << c_keypoints_1.size() << " , current num k2 = " << c_keypoints_2.size() << endl;
	//cout << "previous num k1 = " << p_keypoints_1.size() << " , previous num k2 = " << p_keypoints_2.size() << endl;

	
	matcher->knnMatch(dp_descriptors_left, dc_descriptors_left, cp_knn_matches, 2);

	float ratio = 0.01f;
	bool pointsEnough = true;

	while (pointsEnough)
	{
		for (int i = 0; i < cp_knn_matches.size(); i++)
		{
			if (cp_knn_matches[i][0].distance < ratio*(cp_knn_matches[i][1].distance))
			{
				cp_good_matches.push_back(cp_knn_matches[i][0]);
			}
		}

		if (cp_good_matches.size() < 30)
		{
			cp_good_matches.clear();
			ratio += 0.01f;
		}

		else
		{
			pointsEnough = false;
		}
	}

	vector<Point2d> gc_keypoints_left, gp_keypoints_left;

	for (int i = 0; i < cp_good_matches.size(); i++)
	{
		int p_left_idx = cp_good_matches[i].queryIdx;
		int c_right_idx = cp_good_matches[i].trainIdx;
		gp_keypoints_left.push_back(p_keypoints_left[p_left_idx].pt);
		gc_keypoints_left.push_back(c_keypoints_left[c_right_idx].pt);
	}

	//cout << "previous match = " << p_good_matches.size() << endl;
	//cout << "previous current match = " << cp_good_matches.size() << endl;
	//cout << "dp_descriptors_left = " << dp_descriptors_left.size() << endl;


		//load projection matrix Q
	FileStorage fs("C:/Users/CSK/source/repos/stereo visual odometry/stereo visual odometry/extrinsics.yml", FileStorage::READ);
	
	Mat M,E;
	//Mat D = Mat::zeros(1, 5, CV_32F);
	fs["P1"] >> M;
	M.at<double>(0, 2) -= 57;
	M.at<double>(1, 2) -= 82;
	M = M(cv::Rect(0, 0, 3, 3));

	E = findEssentialMat(gp_keypoints_left, gc_keypoints_left, M, RANSAC, 0.99f, 0.5);

	//Mat R1, R2, TS;
	//decomposeEssentialMat(E, R1, R2, TS);
	//cout << "R1 = " << R1 << " R2 = " << R2 << endl;
	//cout << fabs(determinant(R1)) << " , " << fabs(determinant(R2)) << endl;

	int inlier;
	Mat R, T;
	inlier = recoverPose(E, gp_keypoints_left, gc_keypoints_left, M, R, T, 10000.0);

	//cout << " R = " << R << "\n" << " T = " << T << " inlier = " << inlier << " points = " << cp_good_matches.size() << endl;

	//Rotation Matrix to Euler Angles
	Mat rotationX, rotationY, rotationZ, mR, mQ;
	RQDecomp3x3(R, mR, mQ, rotationX, rotationY, rotationZ);
	//cout << "X= " << rotationX << "\n" << "Y= " << rotationY << "\n" << "Z= " << rotationZ << endl;

	//pitch
	//pitch = asin(rotationX.at<double>(1, 2)) / CV_PI * 180;
	//cout << "vision pitch= " << pitch << endl;

	//roll
	roll = asin(rotationZ.at<double>(0, 1)) / CV_PI * 180;
	cout << "vision roll= " << roll << endl;
	
	workEnd();



	//-- Draw only "good" matches
	Mat c_left, p_left;
	cg_left.download(c_left);
	pg_left.download(p_left);

	Mat cp_img_matches;

	drawMatches(p_left, p_keypoints_left, c_left, c_keypoints_left,
		cp_good_matches, cp_img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	putText(cp_img_matches, text(), Point(5, 25), FONT_HERSHEY_SIMPLEX, 1.0, Scalar::all(255));	
	imshow("Current-previous Good Matches", cp_img_matches);

	/*if (fabs(pitch) > 20)
	{
		system("PAUSE");
	}*/

	//return pitch;
	return roll;


}

int main()
{
	VideoCapture capLeft("C:/Users/CSK/Desktop/rotation experiment data/roll Left 2.avi");
	VideoCapture capRight("C:/Users/CSK/Desktop/rotation experiment data/roll Right 2.avi");

	double half_total_frames;

	if (capLeft.isOpened() == false || capRight.isOpened() == false)
	{
		cout << "Cannot open the video file" << endl;
		return -1;
	}

	else
	{
		half_total_frames = capLeft.get(CAP_PROP_FRAME_COUNT) / 2;
		double framesRight = capRight.get(CAP_PROP_FRAME_COUNT);
		//cout << "L = " << half_total_frames << " R = " << framesRight << endl;
	}

	char key = 'r';
	Mat c_left, c_right, p_left, p_right;
	double vision_pitch;
	double vision_pitch_sum = 0;
	double vision_roll;
	double vision_roll_sum = 0; 

	/*ofstream myfile("vision roll 4 3d.txt");
	if (myfile.is_open())
	{
		myfile << "visionroll" << "\n";
	}
	else
		cout << "Error: can not save the data\n";

	ofstream mefile("vision roll sum 4 3d.txt");
	if (mefile.is_open())
	{
		mefile << "visionrollsum" << "\n";
		mefile << vision_pitch_sum << "\n";
	}
	else
		cout << "Error: can not save the data\n";*/

	/*ofstream wfile("fps 2d.txt");
	if (wfile.is_open())
	{
		wfile << "workfps" << "\n";
	}
	else
		cout << "Error: can not save the data\n";*/

	
	bool start = true;
	while (1)
	{

		////read calibrated and rectified image pair
		

		if (start == true)
		{
			capLeft >> p_left;
			capRight >> p_right;
		}

		else
		{
			p_left = c_left.clone();
			p_right = c_right.clone();
		}

		capLeft >> c_left;
		capRight >> c_right;

		

		if (c_left.empty() || key == 'q' || key == 'Q')
			break;
		
		

		c_left = c_left(cv::Rect(57, 82, 1148, 550));
		c_right = c_right(cv::Rect(57, 82, 1148, 550));

		if (start == true)
		{
			p_left = p_left(cv::Rect(57, 82, 1148, 550));
			p_right = p_right(cv::Rect(57, 82, 1148, 550));
			start = false;
		}
		

		cuda::GpuMat cg_left, cg_right, pg_left, pg_right;
		cg_left.upload(c_left);
		cg_right.upload(c_right);
		pg_left.upload(p_left);
		pg_right.upload(p_right);

		cuda::cvtColor(cg_left, cg_left, COLOR_BGR2GRAY);
		cuda::cvtColor(cg_right, cg_right, COLOR_BGR2GRAY);
		cuda::cvtColor(pg_left, pg_left, COLOR_BGR2GRAY);
		cuda::cvtColor(pg_right, pg_right, COLOR_BGR2GRAY);

		Ptr<cuda::CLAHE> clahe = cuda::createCLAHE(4.0, cv::Size(5, 15));
		clahe->apply(cg_left, cg_left);
		//clahe->apply(cg_right, cg_right);
		clahe->apply(pg_left, pg_left);
		clahe->apply(pg_right, pg_right);
		/*cg_left.download(c_left);
		cg_right.download(c_right);
		pg_left.download(p_left);
		pg_right.download(p_right);*/

		/*imshow("current left", c_left);
		imshow("current right", c_right);
		imshow("previous left", p_left);
		imshow("previous right", p_right);*/

		try
		{
			//vision_pitch = surf_gpu_feature_match(cg_left, pg_left, pg_right);
			//vision_pitch = e_surf_gpu_feature_match(cg_left, pg_left);
			//vision_pitch_sum += vision_pitch;
			//cout << vision_pitch_sum << endl;
			//vision_roll = surf_gpu_feature_match(cg_left, pg_left, pg_right);
			vision_roll = e_surf_gpu_feature_match(cg_left, pg_left);
			vision_roll_sum += vision_roll;
			cout << vision_roll_sum << endl;
			//vision_pitch = orb_gpu_feature_match(cg_left, cg_right, pg_left, pg_right);
			//myfile << vision_roll << "\n";
			/*if (fabs(vision_pitch) > 5)
			{
				myfile << 0.0 << "\n";
			}
			else myfile << vision_pitch << "\n";*/
			//myfile << vision_pitch << "\n";
			//mefile << vision_pitch_sum << "\n";
			//myfile << vision_roll << "\n";
			//mefile << vision_roll_sum << "\n";
			//cout << work_fps << endl;
			//wfile << work_fps << "\n";
			
		}
		catch (cv::Exception & e)
		{
			cerr << e.msg << endl; // output exception message
		}
	

		

		


	key = (char)waitKey(10);
	
	

	}
	capLeft.release();
	capRight.release();
	//wfile.close();
	//myfile.close();
	//mefile.close();
	//system("PAUSE");
	return 0;
	


}