import serial
import time

PORT = "COM9"   # change if needed
BAUD = 9600

ser = serial.Serial(PORT, BAUD, timeout=1)

time.sleep(2)

print("Place finger on sensor...")

while True:

    line = ser.readline().decode().strip()

    if line.startswith("BPM:"):
        parts = line.split(",")

        bpm = parts[0].split(":")[1]
        spo2 = parts[1].split(":")[1]

        print("Average BPM =", bpm)
        print("SpO2 =", spo2, "%")

        break

ser.close()

print("Program finished")