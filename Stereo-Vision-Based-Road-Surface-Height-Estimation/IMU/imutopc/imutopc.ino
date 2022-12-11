/*
  Using the BNO080 IMU
  By: Nathan Seidle
  SparkFun Electronics
  Date: December 21st, 2017
  License: This code is public domain but you buy me a beer if you use this and we meet someday (Beerware license).

  Feel like supporting our work? Buy a board from SparkFun!
  https://www.sparkfun.com/products/14586

  This example shows how to output the i/j/k/real parts of the rotation vector.
  https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation

  It takes about 1ms at 400kHz I2C to read a record from the sensor, but we are polling the sensor continually
  between updates from the sensor. Use the interrupt pin on the BNO080 breakout to avoid polling.

  Hardware Connections:
  Attach the Qwiic Shield to your Arduino/Photon/ESP32 or other
  Plug the sensor onto the shield
  Serial.print it out at 9600 baud to serial monitor.
*/

#define BAUD 115200
#include <Wire.h>
#include "SparkFun_BNO080_Arduino_Library.h"
BNO080 myIMU;

// struct for transmitting the data to the computer through USB
struct IMU_data{
  // rotation
  float roll;
  float pitch;
};

// instanciate one struct
IMU_data IMU_data_holder;

// length of the structure
int len_struct = sizeof(IMU_data_holder);

// send the structure giving the IMU state through serial
void send_IMU_struct(){
  Serial.write('S');
  Serial.write((uint8_t *)&IMU_data_holder, len_struct);
  Serial.write('E');
  return;
}


void set_IMU_data(float roll,  float pitch) {
  IMU_data_holder.roll = roll;
  IMU_data_holder.pitch = pitch;
}



void setup(){
  Serial.begin(BAUD);
  Wire.begin();

  if (myIMU.begin() == false)
  {
    Serial.println("BNO080 not detected at default I2C address. Check your jumpers and the hookup guide. Freezing...");
    while (1);
  }

  Wire.setClock(400000); //Increase I2C data rate to 400kHz

  myIMU.enableGameRotationVector(10); //Send data update

}

void loop(){

   //Look for reports from the IMU
  if (myIMU.dataAvailable() == true)
  {
    float quatI = myIMU.getQuatI();
    float quatJ = myIMU.getQuatJ();
    float quatK = myIMU.getQuatK();
    float quatReal = myIMU.getQuatReal();
    //float quatRadianAccuracy = myIMU.getQuatRadianAccuracy();

    //roll (x-axis rotation)
    float sinr_cosp = +2.0 * (quatReal * quatI + quatJ * quatK);
    float cosr_cosp = +1.0 -2.0 * (quatI * quatI + quatJ * quatJ);
    float roll = atan2(sinr_cosp, cosr_cosp)/M_PI*180;

    // pitch (y-axis rotation)
    float sinp = +2.0 * (quatReal * quatJ - quatK * quatI);
    float pitch;
     if (fabs(sinp) >= 1)
       pitch = copysign(M_PI / 2, sinp)/M_PI*180; // use 90 degrees if out of range
     else
       pitch = asin(sinp)/M_PI*180;

    
    // yaw (z-axis rotation)
//    float siny_cosp = +2.0 * (quatReal * quatK + quatI * quatJ);
//    float cosy_cosp = +1.0 - 2.0 * (quatJ * quatJ + quatK * quatK);  
//    float yaw = atan2(siny_cosp, cosy_cosp);

    
    set_IMU_data(roll, pitch);
    //delay(1);
    // transmit the struct
    send_IMU_struct();
   
//    Serial.print("roll");
//    Serial.print(F(","));
//    Serial.print(roll,4);
//    Serial.print(F(","));
//    Serial.print("pitch");
//    Serial.print(F(","));
//    Serial.print(pitch,4);
//    Serial.print(F(","));
//    Serial.print("accuracy");
//    Serial.print(F(","));
//    Serial.print(quatRadianAccuracy,4);
    

 }
}
