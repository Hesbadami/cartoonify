import six.moves.urllib as urllib
import struct
from struct import unpack
from pathlib import Path
import jsonlines
import click
from fuzzywuzzy import fuzz
import random
from PIL import Image

class DrawingDataset(object):
    """
    interface to the drawing dataset
    """

    def __init__(self, path_to_drawing_dataset, path_to_label_mapping):
        self._path = Path(path_to_drawing_dataset)
        self._categories_filepath = self._path / 'categories.txt'
        self._category_mapping_filepath = path_to_label_mapping
        self._quickdraw_dataset_url = 'https://storage.googleapis.com/quickdraw_dataset/full/binary/'
        self._categories = []
        self._category_mapping = dict()

    def setup(self):
        try:
            with jsonlines.open(self._category_mapping_filepath, mode='r') as reader:
                self._category_mapping = reader.read()
        except IOError as e:
            print('label_mapping.jsonl not found')
            raise e
        self._categories = self.load_categories(self._path)
        if not self._categories:
            if click.confirm('no drawings available, would you like to download the dataset? '
                             'download will take approx 5gb of space'):
                self.download_recurse(self._quickdraw_dataset_url, self._path)
                self._categories = self.load_categories(self._path)
            else:
                raise ValueError('no drawings available, please download dataset')


    def load_categories(self, path):
        files = Path(path).glob('*')
        categories = [f.stem for f in files]
        return categories

    def _unpack_drawing(self, file_handle):
        """unpack single drawing from google draw dataset binary files
        """
        key_id, = unpack('Q', file_handle.read(8))
        countrycode, = unpack('2s', file_handle.read(2))
        recognized, = unpack('b', file_handle.read(1))
        timestamp, = unpack('I', file_handle.read(4))
        n_strokes, = unpack('H', file_handle.read(2))
        image = []
        for i in range(n_strokes):
            n_points, = unpack('H', file_handle.read(2))
            fmt = str(n_points) + 'B'
            x = unpack(fmt, file_handle.read(n_points))
            y = unpack(fmt, file_handle.read(n_points))
            image.append((x, y))

        return {
            'key_id': key_id,
            'countrycode': countrycode,
            'recognized': recognized,
            'timestamp': timestamp,
            'image': image
        }

    def unpack_drawings(self, path):
        """read all drawings from binary file, and return a generator
        """
        with open(path, 'rb') as f:
            while True:
                try:
                    yield self._unpack_drawing(f)
                except struct.error:
                    break

    def get_drawing(self, name, index):
        """get a drawing by name and index, e.g. 100th 'pelican'
        """
        try:
            if name not in self._categories:
                # try and get the closest matching drawing. If nothing suitable foumd then return a scorpion
                name = self._category_mapping.get(name, 'scorpion')
            if index < 1 or not isinstance(index, int):
                raise ValueError('index must be integer > 0')

            name = max([(fuzz.token_set_ratio(name, j), j) for j in self._categories])[1]
            
            itr = self.unpack_drawings(str(self._path / Path(name).with_suffix('.bin')))
            for i in range(index):
                drawing = next(itr)

            return drawing['image']
        except ValueError as e:
            self.log.exception(e)
            raise e


    def get_drawing_pil(self, name, which):
        """get a drawing by name and index, e.g. 100th 'pelican'
        """
        try:
            cat = max([(fuzz.token_set_ratio(name, j), j) for j in self._categories])[1]
            cat_folder = self._path / Path(cat)
            files = [f.stem for f in cat_folder.glob('*.png')]
            image = Image.open(str(cat_folder / Path(files[which]).with_suffix('.png')))
            return image
        except ValueError as e:
            self.log.exception(e)
            raise e

    @property
    def categories(self):
        return self._categories
