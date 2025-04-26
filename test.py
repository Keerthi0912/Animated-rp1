from picamera2 import Picamera2
import time

picam2 = Picamera2()

camera_config = picam2.create_still_configuration(main={"size": (800, 600)})
picam2.configure(camera_config)

while True:
    picam2.start()

    print("")
    input("Press ENTER to take a picture (ctrl-c to quit): ")
    picam2.capture_file("test.jpg")

    picam2.stop()
