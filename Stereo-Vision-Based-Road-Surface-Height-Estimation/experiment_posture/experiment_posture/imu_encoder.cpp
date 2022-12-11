#include <stdio.h>
#include <tchar.h>
#include "SerialClass.h"	// Library described above
#include <string>
#include <iostream>
#include <string>
#include <sstream>
//#include <opencv2/opencv.hpp>



using namespace std;
//using namespace cv;


//int64 work_begin;
//double work_fps;
//void workBegin() { work_begin = getTickCount(); }
//void workEnd()
//{
//	int64 d = getTickCount() - work_begin;
//	double f = getTickFrequency();
//	work_fps = f / d;
//}

typedef struct
{
	float roll;
	float pitch;
	float accuracy;
	float encoder;

}IMU_data;


// application reads from the specified serial port and reports the collected data
void imu_encoder(bool &frame_stamp)
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
				cout << "roll: " << id.roll << " , " << "pitch: " << id.pitch << " , " << "accuracy: " << id.accuracy << " , " << "encoder: " << id.encoder << " , " << "frame-in: " << frame_stamp <<endl;
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
	
		
		Sleep(5);
	}
	
}
