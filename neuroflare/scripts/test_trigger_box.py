import serial
import time

# Open the TriggerBox virtual serial port
triggerbox = serial.Serial('COM3', baudrate=115200)

# Send a test trigger value (1–255)
print("Sending trigger 5...")
triggerbox.write(bytes([5]))

# Keep the pulse high for a short time
time.sleep(0.01)

# Reset the trigger line to zero
triggerbox.write(bytes([0]))

# Close the port
triggerbox.close()

print("Done.")
