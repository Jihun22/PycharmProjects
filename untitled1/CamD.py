import  picamera , time

Camera = picamera.PiCamera()
Camera.framerate = 30
Camera.resolution = (1920,1080)
#Camera.rotation = 180
#Camera.hflip =True
Camera.start_recording('movie.h264',format='h264')
print('Camera Recording Start')

try:
    while True:
        print('frame number :%d' % Camera.frame.index)
        time.sleep(1)
except KeyboardInterrupt:
    print('Camera Recording Stop')
    Camera.stop_recording()