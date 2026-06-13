import serial
import time

ser = serial.Serial(port="/dev/ttyUSB0", baudrate=115200, timeout=1)

def send_at(cmd, delay=1):
    ser.write((cmd + "\r\n").encode())
    time.sleep(delay)
    return ser.read(ser.in_waiting or 1).decode(errors="ignore")

# Check modem
print(send_at("AT"))
print(send_at("AT+CSQ"))  # signal quality

# Set APN
print(send_at('AT+QICSGP=1,1,"internet","","",1'))

# Activate PDP
print(send_at("AT+QIACT=1"))

# Ping test
print(send_at('AT+QPING=1,"8.8.8.8"'))

# Open TCP socket
print(send_at('AT+QIOPEN=1,0,"TCP","google.com",80,0,1'))

ser.close()
