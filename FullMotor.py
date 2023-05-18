import time
import board
import csv
import digitalio
import busio
import adafruit_lis3dh
from adafruit_bme280 import basic as adafruit_bme280

i2c = board.I2C()

bme280 = adafruit_bme280.Adafruit_BME280_I2C(i2c)

bme280.sea_level_pressure = 1013.25

SampleTime = 1
PrevTimeTemp = 0
Samples = 0
timestamp = 1000

Enable = digitalio.DigitalInOut(board.D17)
Direct = digitalio.DigitalInOut(board.D27)
Step = digitalio.DigitalInOut(board.D22)

Enable.direction = digitalio.Direction.OUTPUT
Direct.direction = digitalio.Direction.OUTPUT
Step.direction = digitalio.Direction.OUTPUT


SoundSensor = digitalio.DigitalInOut(board.D26)
SoundSensor.direction = digitalio.Direction.INPUT


Enable.value = True
Direct.value = False
Step.value = False
Enable.value = False

int1 = digitalio.DigitalInOut(board.D16)
lis3dh = adafruit_lis3dh.LIS3DH_I2C(i2c, int1=int1)


field = ['timestamp','xaccel','yaccel','zaccel','temp','press','hum','sound']
filename = "Motor4EdgeImpulseRaspberry.csv"
with open(filename, 'a', newline='') as csvfile:
	csv_writer = csv.writer(csvfile)
	csv_writer.writerow(field)

while True:
	CurrentTime = time.time()

	Step.value = True
	time.sleep(0.002)
	Step.value = False
	time.sleep(0.002)

	if ((CurrentTime - PrevTimeTemp) > SampleTime):
		x, y, z = lis3dh.acceleration
		with open(filename, 'a', newline='') as csvfile:
			csv_writer = csv.writer(csvfile)
			row = [timestamp,x,y,z,bme280.temperature,bme280.pressure,bme280.relative_humidity,Samples]
			csv_writer.writerow(row)
		timestamp += 1000
		Samples = 0
		PrevTimeTemp = CurrentTime
		
	if (SoundSensor.value == False):
		Samples = Samples + 1
