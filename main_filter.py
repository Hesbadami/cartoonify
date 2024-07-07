import sys
import cv2
import os
import ffmpeg
from pathlib import Path

video_file = sys.argv[1]
filename, _ = os.path.splitext(video_file)

if not os.path.isdir(filename):
        os.mkdir(filename)

ffmpeg.input(video_file, ss = 0, r = 0.5)\
        .filter('fps', fps='1/60')\
        .output(f'{filename}/%d.jpg', start_number=0)\
        .overwrite_output()\
        .run(quiet=True)


root = Path(__file__).parent


def run():

    path = Path(filename)
    for file in sorted([int(f.stem) for f in path.glob('*.jpg')]):
        print('processing {}'.format(filename+'/'+str(file)+'.jpg'))
        originalmage = cv2.imread(filename+'/'+str(file)+'.jpg')
        grayScaleImage = cv2.cvtColor(originalmage, cv2.COLOR_BGR2GRAY)
        smoothGrayScale = cv2.medianBlur(grayScaleImage, 5)

        getEdge = cv2.adaptiveThreshold(smoothGrayScale, 255,
          cv2.ADAPTIVE_THRESH_MEAN_C,
          cv2.THRESH_BINARY, 9, 9)

        cv2.imwrite(filename+'/'+str(file)+'.png', getEdge)

    print('finished processing files, closing app.')



run()
#print("hello_world")
# ffmpeg\
#     .input(f'{filename}/*.png', pattern_type='glob', framerate=1)\
#     .output(f'{filename}/{filename}.mp4')\
#     .run(overwrite_output=True)
input2 = ffmpeg.input(f'{filename}/*.png', pattern_type='glob', framerate=1)
input1 = ffmpeg.input(f'{video_file}').audio
# input2 = ffmpeg.input(f'{filename}/{filename}.mp4')

ffmpeg.concat(input2, input1, v=1, a=1).output(f'{filename}/{filename}.mp4').run(overwrite_output=True)
