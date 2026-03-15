import serial
import time

PORT = "COM1"
BAUD = 115200

ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)

print("Connected to Arduino on", PORT)

angles = [30, 90, 150, 90]

for angle in angles:
    cmd = f"{angle}\n"
    ser.write(cmd.encode("utf-8"))
    print("Sent:", angle)
    time.sleep(2)

ser.close()
print("Done")