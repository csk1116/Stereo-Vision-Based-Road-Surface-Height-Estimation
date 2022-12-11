#include<iostream>
#include<string>
#include<sstream>
#include<opencv2/opencv.hpp>


using namespace std;
using namespace cv;


int main(int argc)
{
	//video resolution [2k:15fps 1080p:30fps 720p:60fps wvga:100fps]
	cv::Size2i image_size = cv::Size2i(1280, 720);

	//chessboard 
	cv::Size2i board_sz = cv::Size2i(9, 6);


	//open the camera
	VideoCapture cap(1);
	if (!cap.isOpened())
		return -1;
	cap.grab();

	// Set the video resolution (2*Width * Height)
	cap.set(CV_CAP_PROP_FRAME_WIDTH, image_size.width * 2);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, image_size.height);
	cap.grab();

	Mat frame, left_raw, right_raw, f_left_raw, f_right_raw;

	char capture = 'r';

	cout << "Image resolution:" << image_size << endl;

	//capture image
	cout << "Press 'c' to capture ..." << endl;

	int Icount = 0;


	while (capture != 'q') {
		// Get a new frame from camera
		cap >> frame;
		// Extract left and right images from side-by-side
		left_raw = frame(cv::Rect(0, 0, frame.cols / 2, frame.rows));
		right_raw = frame(cv::Rect(frame.cols / 2, 0, frame.cols / 2, frame.rows));
		//f_left_raw = left_raw.clone();
		//f_right_raw = right_raw.clone();

		//// find chessboard
		//vector<cv::Point2f> cornersl;
		//vector<cv::Point2f> cornersr;
		//bool foundl = findChessboardCorners(f_left_raw, board_sz, cornersl);
		//bool foundr = findChessboardCorners(f_right_raw, board_sz, cornersr);

		////draw corners
		//if (foundl && foundr == 1) {
		//	drawChessboardCorners(f_left_raw, board_sz, cornersl, foundl);
		//	drawChessboardCorners(f_right_raw, board_sz, cornersr, foundr);
		//}


		// Display images
		//imshow("left RAW", f_left_raw);
		//imshow("right RAW", f_right_raw);

		imshow("left", left_raw);
		imshow("right", right_raw);


		if (capture == 'c') {

			//save files at proper locations if user presses 'c'
			stringstream l_name, r_name, ld_name, rd_name;
			l_name << "v2_left" << setw(2) << setfill('0') << Icount << ".png";
			r_name << "v2_right" << setw(2) << setfill('0') << Icount << ".png";
			//ld_name << "d_left" << setw(2) << setfill('0') << Icount << ".png";
			//rd_name << "d_right" << setw(2) << setfill('0') << Icount << ".png";
			imwrite(l_name.str(), left_raw);
			imwrite(r_name.str(), right_raw);
			//imwrite(ld_name.str(), f_left_raw);
			//imwrite(rd_name.str(), f_right_raw);
			cout << "Saved set" << Icount << endl;
			Icount++;
		}




		capture = waitKey(1);
	}
	cap.release();
	return 0;
}
