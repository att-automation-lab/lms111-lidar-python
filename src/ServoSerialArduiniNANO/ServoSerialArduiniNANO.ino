#include <Servo.h>

Servo myServo;
String inputString = "";
bool stringComplete = false;

void setup() {
  Serial.begin(115200);
  myServo.attach(9);
  myServo.write(90);   // เริ่มกลาง
  inputString.reserve(32);
  Serial.println("Servo ready");
}

void loop() {
  if (stringComplete) {
    int angle = inputString.toInt();

    if (angle < 0) angle = 0;
    if (angle > 180) angle = 180;

    myServo.write(angle);

    Serial.print("Angle set to: ");
    Serial.println(angle);

    inputString = "";
    stringComplete = false;
  }
}

void serialEvent() {
  while (Serial.available()) {
    char inChar = (char)Serial.read();

    if (inChar == '\n') {
      stringComplete = true;
    } else {
      inputString += inChar;
    }
  }
}
