from __future__ import division
import png
import numpy as np
from pathlib import Path
from app.sketch import SketchGizeh
import subprocess
from csv import writer


class Workflow(object):
    """controls execution of app
    """

    def __init__(self, dataset, imageprocessor):
        self._path = Path('')
        self._image_path = Path('')
        self._dataset = dataset
        self._image_processor = imageprocessor
        self._sketcher = None
        self._image = None
        self._annotated_image = None
        self._image_labels = []
        self._boxes = None
        self._classes = None
        self._scores = None
        self.count = 0

    def setup(self):
        self._dataset.setup()
        self._sketcher = SketchGizeh()
        self._sketcher.setup()
        self._image_processor.setup()
        self._path = Path(__file__).parent / '..' / '..' / 'images'
        if not self._path.exists():
            self._path.mkdir()
        self.count = len(list(self._path.glob('image*.jpg')))

    def process(self, image_path, threshold, top_x, which):
        """processes an image. If no path supplied, then capture from camera

        :param top_x: If not none, only the top X results are drawn (overrides threshold)
        :param float threshold: threshold for object detection (0.0 to 1.0)
        :param path: directory to save results to
        :param bool camera_enabled: whether to use raspi camera or not
        :param image_path: image to process, if camera is disabled
        :return:
        """
        try:
            self._image_path = Path(image_path)
            img = self._image_processor.load_image_into_numpy_array(image_path)
            # load a scaled version of the image into memory
            img_scaled = self._image_processor.load_image_into_numpy_array(image_path, scale=300 / max(img.shape))
            self._boxes, self._scores, self._classes, num = self._image_processor.detect(img_scaled)
            # annotate the original image
            self._annotated_image = self._image_processor.annotate_image(img, self._boxes, self._classes, self._scores, threshold=threshold)
            self._sketcher = SketchGizeh()
            self._sketcher.setup(img.shape[1], img.shape[0])
            if top_x:
                sorted_scores = sorted(self._scores.flatten())
                threshold = sorted_scores[-min([top_x, self._scores.size])]
            self._image_labels = self._sketcher.draw_object_recognition_results(np.squeeze(self._boxes),
                                   np.squeeze(self._classes).astype(np.int32),
                                   np.squeeze(self._scores),
                                   self._image_processor.labels,
                                   self._dataset,
                                   threshold,
                                   which)
        except (ValueError, IOError) as e:
            raise e

    def save_results(self, debug=False):
        """save result images as png and list of detected objects as txt
        if debug is true, save a list of all detected objects and their scores

        :return tuple: (path to annotated image, path to cartoon image)
        """
        annotated_path = self._image_path
        cartoon_path = self._image_path.with_name('cartoon' + str(self.count) + '.png')
        labels_path = self._image_path.with_name('labels' + str(self.count) + '.txt')
        with open(str(labels_path), 'w') as f:
            f.writelines(self.image_labels)
        if debug:
            scores_path = self._image_path.with_name('scores' + str(self.count) + '.txt')
            with open(str(scores_path), 'w', newline='') as f:
                fcsv = writer(f)
                fcsv.writerow(map(str, self._scores.flatten()))
        # self._save_3d_numpy_array_as_png(self._annotated_image, annotated_path)
        self._sketcher.save_png(cartoon_path)
        return annotated_path, cartoon_path

    def _save_3d_numpy_array_as_png(self, img, path):
        """saves a NxNx3 8 bit numpy array as a png image

        :param img: N.N.3 numpy array
        :param path: path to save image to, e.g. './img/img.png
        :return:
        """
        if len(img.shape) != 3 or img.dtype is not np.dtype('uint8'):
            raise TypeError('image must be NxNx3 array')
        with open(str(path), 'wb') as f:
            writer = png.Writer(img.shape[1], img.shape[0], greyscale=False, bitdepth=8)
            writer.write(f, np.reshape(img, (-1, img.shape[1] * img.shape[2])))

    def close(self):
        self._image_processor.close()

    @property
    def image_labels(self):
        return self._image_labels
