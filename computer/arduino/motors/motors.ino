unsigned long current_time = 0.0;
unsigned long left_time = 0.0;
unsigned long right_time = 0.0;
//update rates in Hz
unsigned long max_long = 4294967295;
unsigned long left_delay = max_long;
unsigned long right_delay = max_long;
int ledState = LOW;
const int right_dir = 8;
const int right_step = 9;
const int left_dir = 11;
const int left_step = 10;

typedef enum { NONE, GOT_L, GOT_R, GOT_E } states;
states state = NONE;
unsigned long currentValue;

void setup() {
  Serial.begin(57600);
  pinMode(LED_BUILTIN, OUTPUT);
  pinMode(right_dir, OUTPUT);
  pinMode(right_step, OUTPUT);
  pinMode(left_dir, OUTPUT);
  pinMode(left_step, OUTPUT);
  digitalWrite(right_dir, HIGH);
  digitalWrite(left_dir, LOW);
  digitalWrite(right_step, LOW);
  digitalWrite(left_dir, LOW);
  left_time = millis();
  right_time = millis();
  state = NONE;
}

void loop() {
  current_time = millis();
  while (Serial.available())
  {
    processIncomingByte(Serial.read());
  }

  checkLeft();
  checkRight();
}

void handlePreviousState()
{
  switch (state)
  {
    case GOT_L:
      processLeft(currentValue);
      break;
    case GOT_R:
      processRight(currentValue);
      break;
    case GOT_E:
      break;
  }
  currentValue = 0;
}

void processIncomingByte(const byte c)
{
  if (isdigit(c))
  {
    currentValue *= 10;
    currentValue += c - '0';
  }
  else
  {
    handlePreviousState();
    switch (c)
    {
      case 'L':
        state = GOT_L;
        break;
      case 'R':
        state = GOT_R;
        break;
      case 'E':
        state = GOT_E;
        break;
      default:
        state = NONE;
        break;
    }
  }
}

void processLeft(unsigned long value)
{
  if (value == 0){
    left_delay = max_long;
  }
  else {
    left_delay = (1.0 / value) * 1000;
  }
}

void processRight(unsigned long value)
{
  if (value == 0){
    right_delay = max_long;
  }
  else {
    right_delay = (1.0 / value) * 1000;
  }
}

void checkLeft()
{
  if (current_time - left_time > left_delay) {
    digitalWrite(left_step, HIGH);
    delayMicroseconds(50);
    digitalWrite(left_step, LOW);
    left_time = current_time;
  }
}

void checkRight()
{
  if (current_time - right_time > right_delay) {
    digitalWrite(right_step, HIGH);
    delayMicroseconds(50);
    digitalWrite(right_step, LOW);
    right_time = current_time;
  }
}
