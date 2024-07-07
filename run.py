import sys
import os
from app.workflow import Workflow
from app.drawing_dataset import DrawingDataset
from app.image_processor import ImageProcessor, tensorflow_model_name, model_path
from app.sketch import SketchGizeh
from pathlib import Path
import random

video_file = sys.argv[1]

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

    print('processing {}'.format(video_file))
    app.process(video_file, 0.3, 3, random.randint(1, 1000))
    app.save_results()
    app.count += 1
    print('finished processing files, closing app.')
    app.close()

run()