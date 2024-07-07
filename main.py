import sys
import os
import ffmpeg
from app.workflow import Workflow
from app.drawing_dataset import DrawingDataset
from app.image_processor import ImageProcessor, tensorflow_model_name, model_path
from app.sketch import SketchGizeh
from pathlib import Path
import random

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

# init objects
dataset = DrawingDataset(str(root / 'downloads/drawing_dataset'), str(root / 'app/label_mapping.jsonl'))
imageprocessor = ImageProcessor(str(model_path),
                                str(root / 'app' / 'object_detection' / 'data' / 'mscoco_label_map.pbtxt'),
                                tensorflow_model_name)
which = random.randint(1, 1000)

def run():

    app = Workflow(dataset, imageprocessor)
    app.setup()

    path = Path(filename)
    for file in sorted([int(f.stem) for f in path.glob('*.jpg')]):
        print('processing {}'.format(filename+'/'+str(file)+'.jpg'))
        app.process(filename+'/'+str(file)+'.jpg', 0.3, 3, random.randint(1, 1000))
        app.save_results()
        app.count += 1
    print('finished processing files, closing app.')
    app.close()

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