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

volatile float counter = 0; //This variable will increase or decrease depending on the rotation of encoder

// struct for transmitting the data to the computer through USB
struct IMU_data{
  // rotation
  float roll;
  float pitch;
  //float accuracy;
  float encoder;
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


void set_IMU_data(float roll,  float pitch, float counter) {
  IMU_data_holder.roll = roll;
  IMU_data_holder.pitch = pitch;
  //IMU_data_holder.accuracy = accuracy;
  IMU_data_holder.encoder = counter;
}



void setup(){
  Serial.begin(BAUD);
  //Serial.begin(9600);
  //Serial.println();
  //Serial.println("BNO080 Read Example");

  Wire.begin();

  if (myIMU.begin() == false)
  {
    Serial.println("BNO080 not detected at default I2C address. Check your jumpers and the hookup guide. Freezing...");
    while (1);
  }

  Wire.setClock(400000); //Increase I2C data rate to 400kHz

  myIMU.enableRotationVector(1); //Send data update every 50ms

//encoder pin
  pinMode(2, INPUT_PULLUP); // internal pullup input pin 2 
  pinMode(3, INPUT_PULLUP); // internal pullup input pin 3
//Setting up interrupt
  //A rising pulse from encodenren activated ai0(). AttachInterrupt 0 is DigitalPin nr 2 on moust Arduino.
  attachInterrupt(0, ai0, RISING);
   
  //B rising pulse from encodenren activated ai1(). AttachInterrupt 1 is DigitalPin nr 3 on moust Arduino.
  attachInterrupt(1, ai1, RISING);

}

void loop(){

   //Look for reports from the IMU
  if (myIMU.dataAvailable() == true)
  {
    float quatI = myIMU.getQuatI();
    float quatJ = myIMU.getQuatJ();
    float quatK = myIMU.getQuatK();
    float quatReal = myIMU.getQuatReal();
    float quatRadianAccuracy = myIMU.getQuatRadianAccuracy();

    //roll (x-axis rotation)
    float sinr_cosp = +2.0 * (quatReal * quatI + quatJ * quatK);
    float cosr_cosp = +1.0 -2.0 * (quatI * quatI + quatJ * quatJ);
    float roll = atan2(sinr_cosp, cosr_cosp);

    // pitch (y-axis rotation)
    float sinp = +2.0 * (quatReal * quatJ - quatK * quatI);
    float pitch;
     if (fabs(sinp) >= 1)
       pitch = copysign(M_PI / 2, sinp); // use 90 degrees if out of range
     else
       pitch = asin(sinp);

    
    // yaw (z-axis rotation)
//    float siny_cosp = +2.0 * (quatReal * quatK + quatI * quatJ);
//    float cosy_cosp = +1.0 - 2.0 * (quatJ * quatJ + quatK * quatK);  
//    float yaw = atan2(siny_cosp, cosy_cosp);

    
    set_IMU_data(roll, pitch, counter);
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

void ai0() {
  // ai0 is activated if DigitalPin nr 2 is going from LOW to HIGH
  // Check pin 3 to determine the direction
  if(digitalRead(3)==LOW)
  {
  counter += M_PI/1000;
  }
  else
  {
  counter -= M_PI/1000 ;
  }
}
  
   
  void ai1(){
  // ai0 is activated if DigitalPin nr 3 is going from LOW to HIGH
  // Check with pin 2 to determine the direction
  if(digitalRead(2)==LOW)
  {
  counter -= M_PI/1000;
  }
  else
  {
  counter += M_PI/1000;
  }
 }
