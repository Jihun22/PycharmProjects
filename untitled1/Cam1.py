import picamera , time

Camera = picamera.PiCamera()
Camera.resolution = (1920,1080)
#Camera.rotation =180
#Camera.hflip = true
#Camera.start_recording('movie.h264', format='h264')
print('Camera Start')

try:
    while True:
        input();
        str = time.ctime() + '.jpg'
        Camera.capture(str)
        print(str + ' file created')
except KeyboardInterrupt:
    print('Camera Stop')
    #Camera.stop_recording()