import winsound
import time

def beep(frequency, duration):
    winsound.Beep(frequency, duration)

# Example: Beep with frequency 1000 Hz for 500 milliseconds
beep(1000, 5000)