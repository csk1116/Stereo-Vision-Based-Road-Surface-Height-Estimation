#define BAUD 9600

void setup(){
  Serial.begin(BAUD);
}

void loop(){
  Serial.println("Kate is my treasure");
  delay(500);
}
