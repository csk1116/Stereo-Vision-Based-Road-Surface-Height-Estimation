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

int encoderPin1 = 2;
int encoderPin2 = 3;

volatile int lastEncoded = 0;
volatile float encoderValue = 0;

float lastencoderValue = 0;
volatile float resolution = 0.09;

int lastMSB = 0;
int lastLSB = 0;

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


void set_IMU_data(float roll,  float pitch, float encoderValue) {
  IMU_data_holder.roll = roll;
  IMU_data_holder.pitch = pitch;
  //IMU_data_holder.accuracy = accuracy;
  IMU_data_holder.encoder = encoderValue;
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

  myIMU.enableGameRotationVector(1); //Send data update

  pinMode(encoderPin1, INPUT_PULLUP); 
  pinMode(encoderPin2, INPUT_PULLUP);

  digitalWrite(encoderPin1, HIGH); //turn pullup resistor on
  digitalWrite(encoderPin2, HIGH); //turn pullup resistor on

  //call updateEncoder() when any high/low changed seen
  //on interrupt 0 (pin 2), or interrupt 1 (pin 3) 
  attachInterrupt(0, updateEncoder, CHANGE); 
  attachInterrupt(1, updateEncoder, CHANGE);

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

    
    set_IMU_data(roll, pitch, encoderValue);
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

void updateEncoder(){
  int MSB = digitalRead(encoderPin1); //MSB = most significant bit
  int LSB = digitalRead(encoderPin2); //LSB = least significant bit

  int encoded = (MSB << 1) |LSB; //converting the 2 pin value to single number
  int sum  = (lastEncoded << 2) | encoded; //adding it to the previous encoded value

  if(sum == 0b1101 || sum == 0b0100 || sum == 0b0010 || sum == 0b1011) encoderValue += resolution;
  if(sum == 0b1110 || sum == 0b0111 || sum == 0b0001 || sum == 0b1000) encoderValue -= resolution;

  //if(sum == 0b1101 || sum == 0b0100 || sum == 0b0010 || sum == 0b1011) encoderValue += M_PI/2000;
  //if(sum == 0b1110 || sum == 0b0111 || sum == 0b0001 || sum == 0b1000) encoderValue -= M_PI/2000;


  lastEncoded = encoded; //store this value for next time
}
