import jetson.utils
import argparse
import numpy
from PIL import Image
import csv
import neuralNet

# parse the command line
parser = argparse.ArgumentParser()

parser.add_argument("--width", type=int, default=840,
                    help="desired width of camera stream (default is 1280 pixels)")
parser.add_argument("--height", type=int, default=840,
                    help="desired height of camera stream (default is 720 pixels)")
parser.add_argument("--camera", type=str, default="0",
                    help="index of the MIPI CSI camera to use (NULL for CSI camera 0), or for VL42 cameras the /dev/video node to use (e.g. /dev/video0).  By default, MIPI CSI camera 0 will be used.")

opt = parser.parse_args()
print(opt)

# create display window
display = jetson.utils.glDisplay()

# create camera device
camera = jetson.utils.gstCamera(opt.width, opt.height, opt.camera)

# open the camera for streaming
camera.Open()

# capture frames until user exits
# final frame before camera closes will be used on the neural net
while display.IsOpen():
    image, width, height = camera.CaptureRGBA(zeroCopy=1)
    display.RenderOnce(image, width, height)
    display.SetTitle("{:s} | {:d}x{:d} | {:.1f} FPS".format(
        "Camera Viewer", width, height, display.GetFPS()))
    # print(image)
    #print("IMAGE TYPE")
    # print(type(image))

    cuda_img = jetson.utils.cudaAllocMapped(
        width=opt.width, height=opt.height, format='rgb32f')
    #array = jetson.utils.cudaToNumpy(image)
    # print(array)

    jetson.utils.saveImageRGBA('number.png', image, width, height)
# close the camera
camera.Close()

oImage = Image.open('number.png')
# oImage.show()

greyscaleImage = oImage.convert('L')
greyscaleImage.save('number(greyscale).png')
# greyscaleImage.show()

resizeImage = greyscaleImage.resize((28, 28))
resizeImage.save('number(28x28).png')
# resizeImage.show()

imageData = resizeImage.load()
imageArray = [0] * 784
count = 0
# print('data')
for x in range(28):
    for y in range(28):
        #imageArray[count] += 255 - imageData[y, x]
        if (255 - imageData[y, x]) < 170:
            imageArray[count] = 0
        else:
            imageArray[count] = 255
        count = count+1

# print(imageArray)

with open('number.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(imageArray)

print("Running Net")
neuralNet.run()
