from gpiozero import DigitalOutputDevice as GPIO

enable = GPIO(pin=26)

while True:
    enable.on()
