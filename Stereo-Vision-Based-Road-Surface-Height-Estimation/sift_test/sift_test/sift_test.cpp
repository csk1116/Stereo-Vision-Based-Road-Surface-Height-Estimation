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




void sift_match(Mat c_left, Mat c_right, Mat p_left, Mat p_right) {

	//workBegin();

	//Create SIFT class pointer
	Ptr<Feature2D> f2d = xfeatures2d::SURF::create();
	
	//cv::cuda::SURF_CUDA cuda_surf();

	//Detect the keypoints and calculate the descriptors
	vector<KeyPoint> c_keypoints_1, c_keypoints_2, p_keypoints_1, p_keypoints_2;
	Mat c_descriptors_1, c_descriptors_2, p_descriptors_1, p_descriptors_2;
	f2d->detectAndCompute(c_left, noArray(), c_keypoints_1, c_descriptors_1);
	f2d->detectAndCompute(c_right, noArray(), c_keypoints_2, c_descriptors_2);
	//f2d->detectAndCompute(p_left, noArray(), p_keypoints_1, p_descriptors_1);
	//f2d->detectAndCompute(p_right, noArray(), p_keypoints_2, p_descriptors_2);
	cout << "current num k1 = " << c_keypoints_1.size() << " , current num k2 = " << c_keypoints_2.size() << endl;
	Mat c_img_keypoints_1, c_img_keypoints_2;
	//drawKeypoints(c_left, c_keypoints_1, c_img_keypoints_1);
	//drawKeypoints(c_right, c_keypoints_2, c_img_keypoints_2);
	//imshow("k1", img_keypoints_1);
	//imwrite("k1.png", img_keypoints_1);
	//imshow("k2", img_keypoints_2);
	//imwrite("k2.png", img_keypoints_2);


	//-- Step 2: Matching descriptor vectors with a FLANN based matcher
   // Since SURF is a floating-point descriptor NORM_L2 is used
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
	std::vector< std::vector<DMatch> > knn_matches;
	matcher->knnMatch(c_descriptors_1, c_descriptors_2, knn_matches, 2);

	//-- Filter matches using the Lowe's ratio test
	const float ratio_thresh = 0.35f;
	std::vector<DMatch> good_matches;
	for (int i = 0; i < knn_matches.size(); i++)
	{
		if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
		{
			good_matches.push_back(knn_matches[i][0]);
		}
	}

	workEnd();

	////-- Step 2: Matching descriptor vectors using FLANN matcher
	//FlannBasedMatcher matcher;
	//vector< DMatch > matches;
	//matcher.match(descriptors_1, descriptors_2, matches);

	//double max_dist = 0; double min_dist = 100;
	////-- Quick calculation of max and min distances between keypoints
	//for (int i = 0; i < descriptors_1.rows; i++)
	//{
	//	double dist = matches[i].distance;
	//	if (dist < min_dist) min_dist = dist;
	//	if (dist > max_dist) max_dist = dist;
	//}
	//printf("-- Max dist : %f \n", max_dist);
	//printf("-- Min dist : %f \n", min_dist);

	//
	////-- Draw only "good" matches (i.e. whose distance is (1) less than s*min_dist,
	////-- or a small arbitary value ( 0.02 ) in the event that min_dist is very small and (2) matching point's abs(left keypoint's row - right keypoint's row) <= k )
	////-- PS.- radiusMatch can also be used here.
	//vector< DMatch > good_matches;
	//for (int i = 0; i < descriptors_1.rows; i++)
	//{
	//		if (matches[i].distance <= max(3 * min_dist, 0.02))
	//		{
	//				good_matches.push_back(matches[i]);	
	//		}
	//}
	
	//filter same train
	//good_matches.erase(unique(good_matches.begin(), good_matches.end(), trainIdx_are_equal), good_matches.end());
	


	//-- Draw only "good" matches
	Mat img_matches;
	drawMatches(c_left, c_keypoints_1, c_right, c_keypoints_2,
		good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	//-- Show detected matches
	putText(img_matches, text(), Point(5, 25), FONT_HERSHEY_SIMPLEX, 1.0, Scalar::all(255));
	imshow("Good Matches", img_matches);

	//-- show matching points' locations
	int num_matches = (int)good_matches.size();
	vector<Point2f> matched_points1;
	vector<Point2f> matched_points2;

	for (int i = 0; i < num_matches; i++)
	{
		int idx1 = good_matches[i].queryIdx;
		int idx2 = good_matches[i].trainIdx;
		matched_points1.push_back(c_keypoints_1[idx1].pt);
		matched_points2.push_back(c_keypoints_2[idx2].pt);
	}

    	
	// filter same xy
	
	//matched_points1.erase(unique(matched_points1.begin(), matched_points1.end(), points_are_equal), matched_points1.end());
	//matched_points2.erase(unique(matched_points2.begin(), matched_points2.end(), points_are_equal), matched_points2.end());
	//int a = 0;
	
	for (int i = 0; i < (int)matched_points1.size(); i++)
	{
		cout << "good match" << i << " left " << matched_points1[i] << " right " << matched_points2[i] << "\n";
	}

	
	//-- save matching points' data
	ofstream myfile("fabs of matching points' y coordinate model.txt");
	if (myfile.is_open())
	{
		double err = 0;
		double count = 0;
		myfile << "Point" << " , " << "Lefty" << " , " << "fabs(Lefty-Righty)" << "\n";
		for (int i = 0; i < (int)matched_points1.size(); i++)
		{
			err += fabs(matched_points1[i].y - matched_points2[i].y);
			myfile << i << " , " << matched_points1[i].y << " , " << fabs(matched_points1[i].y - matched_points2[i].y) << "\n";
			count++;
		}

		myfile << "average err" << err/count;

		myfile.close();
	}
	else
		cout << "Error: can not save the data\n";

	//-- save matching image
	imwrite("slam_match0.png", img_matches);

	

}


int main()
{
	
	//timer
	time_t start, end;
	double elapsed;
	start = time(NULL);

	//read calibrated and rectified image pair
	Mat img_left = imread("left_model_one2k_0.png", IMREAD_GRAYSCALE);
	Mat img_right = imread("right_model_one2k_0.png", IMREAD_GRAYSCALE);
	Mat left_cropped, right_cropped;
	Mat left_eq, right_eq;
	Mat re_left, re_right;
	
	//left_cropped = img_left(cv::Rect(57, 82, 1148, 550));
	//right_cropped = img_right(cv::Rect(57, 82, 1148, 550));

	//left_cropped = img_left(cv::Rect(0, 0, 2208, 1242));
	//right_cropped = img_right(cv::Rect(0, 0, 2208, 1242));

	left_cropped = img_left(cv::Rect(0, 0, 1986, 994));
	right_cropped = img_right(cv::Rect(0, 0, 1986, 994));

	cuda::GpuMat d_left, d_right;
	d_left.upload(left_cropped);
	d_right.upload(right_cropped);

	Ptr<cuda::CLAHE> clahe = cuda::createCLAHE(4.0, cv::Size(5, 15));
	clahe->apply(d_left, d_left);
	clahe->apply(d_right, d_right);
	d_left.download(left_cropped);
	d_right.download(right_cropped);

	imshow("left_1", left_cropped);
	imshow("left_2", right_cropped);

	//char key = 'r';
	


	
	//matching by sift
	//while (key != 'q' || key != 'Q')
	//{
	sift_match(left_cropped, right_cropped, re_left, re_right);
	
	//key = (char)waitKey(30);
	waitKey();
	return 0;
	
	//}

	//timer
	end = time(NULL);
	elapsed = difftime(end, start);
	cout << "time elapsed = " << (double)elapsed << endl;

	
}

