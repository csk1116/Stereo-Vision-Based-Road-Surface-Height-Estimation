#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <fstream>
#include <omp.h>
#include <tchar.h>
#include"SerialClass.h"
#include <sstream>

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

typedef struct
{
	float roll;
	float pitch;
	//float accuracy;
	float encoder;

}IMU_data;

// application reads from the specified serial port and reports the collected data
void imu_encoder(bool &frame_stamp , char &capture, bool &triger)
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
	float imu_roll_current, imu_roll_last, imu_pitch_current, imu_pitch_last, encoder_current, encoder_last;
	bool is_last = true;

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


	while (SP->IsConnected() )
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
					if (is_last)
					{
						imu_pitch_last = id.pitch;
						imu_roll_last = id.roll;
						encoder_last = id.encoder;
						is_last = false;
					}

					else
					{
						imu_pitch_current = id.pitch;
						imu_roll_current = id.roll;
						encoder_current = id.encoder;
						//mefile << imu_roll_current - imu_roll_last << " , " << imu_pitch_current - imu_pitch_last << " , " << encoder_current - encoder_last << "\n";
						is_last = true;
					}
					
					cout << "roll: " << id.roll << " , " << "pitch: " << id.pitch << " , " << "encoder: " << id.encoder << " , " << "frame-in: " << frame_stamp << endl;
					frame_stamp = false;
				}
				else
				{
					//myfile << id.roll << " , " << id.pitch << " , " << id.encoder << " , " << frame_stamp << "\n";
					cout << "roll: " << id.roll << " , " << "pitch: " << id.pitch << " , " << "encoder: " << id.encoder << " , " << "frame-in: " << frame_stamp << endl;
					
				}
				
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

		if (capture == 'q' || capture == 'Q') {
			//myfile.close();
			//mefile.close();
			//Ifile.close();
			break;
		}

		//Sleep(1);
	}



}

void video_record(bool &frame_stamp, char &capture, bool &triger) {
	// calibrate and rectify the camera and capture
	//video resolution [2k:15fps 1080p:30fps 720p:60fps wvga:100fps]
	Size2i image_size = Size2i(1280, 720);

	//open the camera
	VideoCapture cap(1);
	if (!cap.isOpened())
		cout << "camera is not connected" << endl;
	else
	{
		cout << "camera open" << endl;
	}
	//cap.grab();

	// Set the video resolution (2*Width * Height)
	cap.set(CV_CAP_PROP_FRAME_WIDTH, image_size.width * 2);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, image_size.height);
	cap.grab();

	Mat frame, left_raw, right_raw, left_rectified, right_rectified, left_cropped, right_cropped;
	Mat cameraMatrix_left, distCoeffs_left, R1, P1;
	Mat cameraMatrix_right, distCoeffs_right, R2, P2;
	Mat rmap00, rmap01, rmap10, rmap11;
	Mat Q;



	cout << "Image resolution:" << image_size << endl;

	//capture image
	cout << "Press 'c' to capture ..." << endl;

	int Icount = 0;

	//load camera matrix 
	FileStorage Efs("C:/Users/CSK/source/repos/experiment_posture/experiment_posture/extrinsics.yml", FileStorage::READ);
	Efs["Q"] >> Q;
	Efs["R1"] >> R1;
	Efs["R2"] >> R2;
	Efs["P1"] >> P1;
	Efs["P2"] >> P2;
	FileStorage Ifs("C:/Users/CSK/source/repos/experiment_posture/experiment_posture/intrinsics.yml", FileStorage::READ);
	Ifs["M1"] >> cameraMatrix_left;
	Ifs["M2"] >> cameraMatrix_right;
	Ifs["D1"] >> distCoeffs_left;
	Ifs["D2"] >> distCoeffs_right;

	double C_x = -1 * Q.at<double>(0, 3) - 57;
	double C_y = -1 * Q.at<double>(1, 3) - 82;
	double B_v = 550 * 1 / 4;

	initUndistortRectifyMap(cameraMatrix_left, distCoeffs_left, R1, P1, image_size, CV_32FC1, rmap00, rmap01);
	initUndistortRectifyMap(cameraMatrix_right, distCoeffs_right, R2, P2, image_size, CV_32FC1, rmap10, rmap11);

	cuda::GpuMat rmapx0, rmapy0, rmapx1, rmapy1;
	cuda::GpuMat d_left_raw, d_right_raw, d_left_rectified, d_right_rectified;

	rmapx0.upload(rmap00);
	rmapy0.upload(rmap01);
	rmapx1.upload(rmap10);
	rmapy1.upload(rmap11);

	//VideoWriter videoLeft("roll left 4.avi", CV_FOURCC('M', 'J', 'P', 'G'), 60, Size(image_size.width, image_size.height));
	//VideoWriter videoRight("roll right 4.avi", CV_FOURCC('M', 'J', 'P', 'G'), 60, Size(image_size.width, image_size.height));


	while (capture != 'q' && capture != 'Q') {

		//capture only
		//triger = true;
		//frame_stamp = false;

		while (triger == true && frame_stamp == false)
		{
			//workBegin();
			
			
			// Get a new frame from camera
			cap >> frame;
			frame_stamp = true; //remember

			//double fps = cap.get(CAP_PROP_FPS);
			//cout << "fps = " << fps << endl;
			// Extract left and right images from side-by-side
			left_raw = frame(cv::Rect(0, 0, frame.cols / 2, frame.rows));
			right_raw = frame(cv::Rect(frame.cols / 2, 0, frame.cols / 2, frame.rows));

			//cout << frame.size() << endl;
			//cout << left_raw.size() << endl;
			//cout << right_raw.size() << endl;
			d_left_raw.upload(left_raw);
			d_right_raw.upload(right_raw);


			cuda::remap(d_left_raw, d_left_rectified, rmapx0, rmapy0, INTER_LINEAR);
			cuda::remap(d_right_raw, d_right_rectified, rmapx1, rmapy1, INTER_LINEAR);

			d_left_rectified.download(left_rectified);
			d_right_rectified.download(right_rectified);

			left_cropped = left_rectified.clone();
			right_cropped = right_rectified.clone();

			left_cropped = left_cropped(cv::Rect(57, 82, 1148, 550));
			right_cropped = right_cropped(cv::Rect(57, 82, 1148, 550));



			line(left_cropped, Point2d(C_x, 0), Point2d(C_x, left_cropped.rows), Scalar(0, 0, 255), 1, 8);
			line(left_cropped, Point2d(0, C_y), Point2d(left_cropped.cols, C_y), Scalar(0, 0, 255), 1, 8);
			line(right_cropped, Point2d(C_x, 0), Point2d(C_x, right_cropped.rows), Scalar(0, 0, 255), 1, 8);
			line(right_cropped, Point2d(0, C_y), Point2d(right_cropped.cols, C_y), Scalar(0, 0, 255), 1, 8);
			line(left_cropped, Point2d(0, B_v), Point2d(left_cropped.cols, B_v), Scalar(0, 255, 0), 1, 8);
			line(right_cropped, Point2d(0, B_v), Point2d(right_cropped.cols, B_v), Scalar(0, 255, 0), 1, 8);

			imshow("left_cropped", left_cropped);
			imshow("right_cropped", right_cropped);

			//imshow("left_rectified", left_rectified);
			//imshow("right_rectified", right_rectified);

			//videoLeft << left_rectified;
			//videoRight << right_rectified;
			//videoLeft.write(left_rectified);
			//videoRight.write(right_rectified);


			if (capture == 'c' || capture == 'C') {

				//save files at proper locations if user presses 'c'
				stringstream l_name, r_name, lr_name, rr_name;
				l_name << "left_model_four_" << Icount << ".png";
				r_name << "right_model_four_" << Icount << ".png";
				//lr_name << "left_road_distance" << Icount << ".png";
				//rr_name << "right_road_distance" << Icount << ".png";

				imwrite(l_name.str(), left_rectified);
				imwrite(r_name.str(), right_rectified);

				if (Icount == 0)
				{
					imwrite("left_road_distance.png", left_cropped);
					imwrite("right_road_distance.png", right_cropped);
				}
				
				cout << "Saved set" << Icount << endl;
				Icount++;
			}

			if (capture == 'r' || capture == 'R')
			{
				Icount = 0;
			}


			if (capture == 27  || capture == 'q' || capture == 'Q')
			{
				break;
			}


			capture = (char)waitKey(1);

			//workEnd();

			//cout << work_fps << endl;

		}
	}
	cap.release();
	//videoLeft.release();
	//videoRight.release();
}

int main()
{
	//waitkey
	char capture = 'r';
	//frame stamp
	bool frame_stamp = false;
	bool triger = false;

#pragma omp parallel sections default(none) shared(capture) shared(frame_stamp) shared(triger)
	{
	#pragma omp section
	imu_encoder(frame_stamp, capture, triger);
	#pragma omp section
	video_record(frame_stamp, capture, triger);
	}

	//video_record(frame_stamp, capture, triger);
	
	return 0;
}
