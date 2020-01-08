// Mod the fuck outta this
// Note port # in bottom right for python script

#include<Servo.h>

Servo servoVer; //Vertical Servo
Servo servoHor; //Horizontal Servo

int x;
int y;
int z;
int c;

int light;
int rise;
int count = 1;

bool objectDetected;
bool initialized = false;

int xOld;
int yOld;

int xRef;
int yRef;

int xShift;
int yShift;

int prevX;
int prevY;

int servoX;
int servoY;

int randomSlow;
int randomSlowOld;
int randomFast;

void setup()
{    
  Serial.begin(19200);
  servoVer.attach(10); //Attach Vertical Servo to Pin 10
  servoHor.attach(9); //Attach Horizontal Servo to Pin 9
  servoVer.write(90);
  servoHor.write(90);
  pinMode(11, OUTPUT);
  pinMode(5, OUTPUT);
  pinMode(12, OUTPUT);
  servoX = min(servoX, 140);
  servoX = max(servoX, 40);
  servoY = min(servoY, 170);
  servoY = max(servoY, 10);
}
void loop()
{
  serial(); 
  detect();
}

void detect(){ 
  if(Serial.available() > 0)
   {
   int bytes = Serial.read();
   count = 0;
   objectDetected = false;
   process();
   delay(2);

   }
   else
   {
   count++;
   objectDetected = false;
   process();
   delay(count);
   }
  }
void indicate(){
  digitalWrite(5, HIGH);
  digitalWrite(5, LOW);
  }  
void indicate2(){
  digitalWrite(5, LOW);
  analogWrite(13, HIGH);

  }
  
void process(){
  if(objectDetected == false && count >= 50){   
    idle();   
    indicate2();

}
  if(objectDetected == true && count <= 50){   
    count = 0;
    serial();
  } 
  }

void initialize(){
  tone(8, 1000);
  delay(200);
  tone(8,800);
  delay(80);
  tone(8, 3000);
  delay(100);
  servoX = 90;
  servoY = 90;
  defPos();
  initialized = true;  

  
 }

void checkCenter(){
}

void Pos()
{      
    look();
    indicate();
    bright();
    defPos();
    count = 0;
    detect();
}
void idle(){
  do{
    servoX = random(30,150);   
    servoY = random(50,130); 
    defPos(); 
    randomSlow = random(500,random(2000, randomSlowOld));
    tone(8, randomSlow);
    noTone(8);
    delay(randomSlow/5);
    delay(random(randomSlow/5, randomSlow)); 
    detect();
    randomSlowOld++;
  }while(objectDetected = false);
  
}

void bright(){
  serial2();  
  int light = map(z, 400, 0, 0, 255);
  analogWrite(13, z);
  analogWrite(5, light);        

}


void look(){
  if(initialized = true){
    serialC();
    tone(8, 300);
    rise = rise++;
    tone(8, rise / z++);

    if(c != 1){

      if(y > 280){
        servoY++;
        yOld = servoY;
        } 
      if(y < 220){
        servoY--;
        yOld = servoY;
        
        } 
      if(x > 320){
        servoX++;
        xOld = servoX;    
        digitalWrite(11, LOW);
        digitalWrite(12, HIGH);  //orange
        } //Right Visual Field Active
      if(x < 280){
        servoX--;
        xOld = servoX;

        digitalWrite(11, HIGH); //green
        digitalWrite(12, LOW);
        } //Left Visual Field Active}
      }

      
    if(c == 1){
    tone(8, 3000);
    servoX = xOld;
    servoY = yOld;
    }

   }
  if(initialized = false){
    initialize();
  }
    serial();

 }

 void defPos(){
    servoHor.write(servoX);
    servoVer.write(servoY);
    yOld = servoY;        
    xOld = servoX;
    }

 void defPosClimb(){
  if(yOld > servoY){
    servoY--;
    servoVer.write(servoY);
    yOld = servoY;
    }
  if(yOld < servoY){
    servoY++;
    servoVer.write(servoY);
    yOld = servoY;
    }
  if(xOld > servoX){
    servoX--;
    servoHor.write(servoX);
    xOld = servoX;
    }
  if(xOld < servoX){
    servoX++;
    servoHor.write(servoX);
    xOld = servoX;
    }
    
    }

void serial(){
  if(Serial.available() > 0)
  {
    if(Serial.read() == 'X')
    {
      x = Serial.parseInt();
      if(Serial.read() == 'Y')

      {
        y = Serial.parseInt();
        Pos();
      }
    }
    while(Serial.available() > 0)
    {
      Serial.read();
    }
  } 
}






void serial2(){
  if(Serial.available() > 0)
  {
        if(Serial.read() == 'Z')
        {
          z = Serial.parseInt();
          if(Serial.read() == 'C')
          {
            c = Serial.parseInt();
          }
        }
    while(Serial.available() > 0)
    {
      Serial.read();
    }
  } 
}





void serialC(){
  if(Serial.available() > 0)
  {
       if(Serial.read() == 'C')
       {
         c = Serial.parseInt();
         
        }
    while(Serial.available() > 0)
    {
      Serial.read();
    }
  } 
}
