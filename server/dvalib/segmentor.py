import os, logging
from PIL import Image
import numpy as np

if os.environ.get('PYTORCH_MODE',False):
    logging.info("In pytorch mode, not importing TF")
elif os.environ.get('CAFFE_MODE',False):
    pass
else:
    from .crfasrnn.crfrnn_model import get_crfrnn_model_def


class BaseSegmentor(object):

    def __init__(self):
        pass

    def segment(self,path):
        pass

    def load(self):
        pass


class CRFRNNSegmentor(BaseSegmentor):

    _PALETTE = [0, 0, 0,
                128, 0, 0,
                0, 128, 0,
                128, 128, 0,
                0, 0, 128,
                128, 0, 128,
                0, 128, 128,
                128, 128, 128,
                64, 0, 0,
                192, 0, 0,
                64, 128, 0,
                192, 128, 0,
                64, 0, 128,
                192, 0, 128,
                64, 128, 128,
                192, 128, 128,
                0, 64, 0,
                128, 64, 0,
                0, 192, 0,
                128, 192, 0,
                0, 64, 128,
                128, 64, 128,
                0, 192, 128,
                128, 192, 128,
                64, 64, 0,
                192, 64, 0,
                64, 192, 0,
                192, 192, 0]

    def __init__(self,model_path="crfrnn_keras_model.h5",class_index_to_string=None):
        super(CRFRNNSegmentor, self).__init__()
        self.model_path = model_path
        self.class_index_to_string = class_index_to_string
        self.session=None

    def segment(self,path):
        if self.session is None:
            self.load()

        im = Image.open(path)
        im.thumbnail((500, 500), Image.ANTIALIAS)
        path = path.replace(".jpg",".temp.jpg")
        im.save(path, "JPEG")
        output_file = path.replace(".jpg", ".png")
        img_data, img_h, img_w = self.get_preprocessed_image(path)
        probs = self.session.predict(img_data, verbose=False)[0, :, :, :]
        segmentation = self.get_label_image(probs, img_h, img_w)
        segmentation.save(output_file)

    def load(self):
        self.session = get_crfrnn_model_def()
        self.session.load_weights(self.model_path)

    @staticmethod
    def get_preprocessed_image(file_name):
        mean_values = np.array([123.68, 116.779, 103.939], dtype=np.float32)  # RGB mean values
        mean_values = mean_values.reshape(1, 1, 3)
        im = np.array(Image.open(file_name)).astype(np.float32)
        assert im.ndim == 3, "Only RGB images are supported."
        im = im - mean_values
        im = im[:, :, ::-1]
        img_h, img_w, img_c = im.shape
        assert img_c == 3, "Only RGB images are supported."
        if img_h > 500 or img_w > 500:
            raise ValueError("Please resize your images to be not bigger than 500 x 500.")

        pad_h = 500 - img_h
        pad_w = 500 - img_w
        im = np.pad(im, pad_width=((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)
        return im.astype(np.float32).reshape(1, 500, 500, 3), img_h, img_w

    @staticmethod
    def get_label_image(probs, img_h, img_w):
        labels = probs.argmax(axis=2).astype("uint8")[:img_h, :img_w]
        label_im = Image.fromarray(labels, "P")
        label_im.putpalette(CRFRNNSegmentor._PALETTE)
        return label_im