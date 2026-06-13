#!/bin/bash

LOG=/var/log/ec200-init.log
echo "==== EC200 init started: $(date) ====" >> $LOG

# Wait for USB devices to appear
for i in {1..20}; do
  PORT=$(ls /dev/ttyUSB* 2>/dev/null | head -n 1)
  if [ -n "$PORT" ]; then
    break
  fi
  echo "Waiting for ttyUSB... ($i)" >> $LOG
  sleep 1
done

if [ -z "$PORT" ]; then
  echo "❌ No ttyUSB port found" >> $LOG
  exit 1
fi

echo "✅ Using port: $PORT" >> $LOG

# Configure serial port
stty -F "$PORT" 115200 raw -echo

send_at () {
  echo -e "$1\r" > "$PORT"
  sleep 1
  timeout 2 cat "$PORT" >> $LOG
}

send_at "AT"
send_at "AT+CSQ"
send_at "AT+CPIN?"
send_at "AT+QNETDEVCTL=1,1,1"

echo "==== EC200 init finished: $(date) ====" >> $LOG
