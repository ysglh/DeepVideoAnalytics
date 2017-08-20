import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from .keras_yolo import preprocess_true_boxes, yolo_body, yolo_eval, yolo_head, yolo_loss
import logging
import os
from PIL import Image

DEFAULT_ANCHORS = [(0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434), (7.88282, 3.52778), (9.77052, 9.16828)]


class YOLOTrainer(object):

    def __init__(self,images,boxes,class_names,args,test_mode=False):
        self.images = images
        self.args = args
        self.boxes = boxes
        self.processed_boxes = None
        self.processed_images = None
        self.detectors_mask, self.matching_true_boxes = None, None
        self.class_names = class_names
        self.anchors = np.array(args['anchors'] if 'anchors' in args else DEFAULT_ANCHORS)
        self.validation_split = args['validation_split'] if 'validation_split' in args else 0.1
        self.model_body = None
        self.model = None
        self.phase_1_epochs = args['phase_1_epochs'] if 'phase_1_epochs' in args else 10
        self.phase_2_epochs = args['phase_2_epochs'] if 'phase_2_epochs' in args else 10
        self.root_dir = args['root_dir']
        if test_mode:
            self.create_model(load_pretrained=False,freeze_body=False)
        else:
            self.base_model = args['base_model']
            self.process_data()
            self.get_detector_mask()
            self.create_model()

    def process_data(self):
        orig_sizes = []
        processed_images = []
        boxes = []
        for iindex,ipath in enumerate(self.images):
            im = Image.open(ipath)
            sz = np.expand_dims(np.array([float(im.width), float(im.height)]), axis=0)
            image_array = np.array(im.resize((416, 416), Image.BICUBIC),dtype=np.float)/255.
            if len(image_array.shape) != 3:
                logging.warning("skipping {} contains less than 3 channels".format(ipath))
            else:
                boxes.append(np.array(self.boxes[iindex],dtype=np.uint16).reshape((-1, 5)))
                processed_images.append(image_array)
                orig_sizes.append(sz)
        boxes_extents = [box[:, [2, 1, 4, 3, 0]] for box in boxes]
        boxes_xy = [0.5 * (box[:, 3:5] + box[:, 1:3]) for box in boxes]
        boxes_wh = [box[:, 3:5] - box[:, 1:3] for box in boxes]
        boxes_xy = [boxxy / orig_sizes[i] for i,boxxy in enumerate(boxes_xy)]
        boxes_wh = [boxwh / orig_sizes[i] for i,boxwh in enumerate(boxes_wh)]
        boxes = [np.concatenate((boxes_xy[i], boxes_wh[i], box[:, 0:1]), axis=1) for i, box in enumerate(boxes)]
        max_boxes = 0
        for boxz in boxes:
            if boxz.shape[0] > max_boxes:
                max_boxes = boxz.shape[0]
        for i, boxz in enumerate(boxes):
            if boxz.shape[0] < max_boxes:
                zero_padding = np.zeros( (max_boxes-boxz.shape[0], 5), dtype=np.float32)
                boxes[i] = np.vstack((boxz, zero_padding))
        self.processed_images =  np.array(processed_images)
        self.processed_boxes = np.array(boxes)
        self.get_detector_mask()

    def get_detector_mask(self):
        boxes = self.processed_boxes
        anchors = self.anchors
        detectors_mask = [0 for i in range(len(boxes))]
        matching_true_boxes = [0 for i in range(len(boxes))]
        for i, box in enumerate(boxes):
            detectors_mask[i], matching_true_boxes[i] = preprocess_true_boxes(box, anchors, [416, 416])
        self.detectors_mask = np.array(detectors_mask)
        self.matching_true_boxes = np.array(matching_true_boxes)

    def create_model(self, load_pretrained=True, freeze_body=True):
        anchors, class_names = self.anchors, self.class_names
        detectors_mask_shape = (13, 13, 5, 1)
        matching_boxes_shape = (13, 13, 5, 5)

        # Create model input layers.
        image_input = Input(shape=(416, 416, 3))
        boxes_input = Input(shape=(None, 5))
        detectors_mask_input = Input(shape=detectors_mask_shape)
        matching_boxes_input = Input(shape=matching_boxes_shape)

        # Create model body.
        yolo_model = yolo_body(image_input, len(anchors), len(class_names))
        topless_yolo = Model(yolo_model.input, yolo_model.layers[-2].output)

        if load_pretrained:
            # Save topless yolo:
            topless_yolo_path = os.path.join('{}/'.format(self.root_dir), 'yolo_headless.h5')
            if not os.path.exists(topless_yolo_path):
                yolo_path = self.base_model
                model_body = load_model(yolo_path)
                model_body = Model(model_body.inputs, model_body.layers[-2].output)
                model_body.save_weights(topless_yolo_path)
            topless_yolo.load_weights(topless_yolo_path)

        if freeze_body:
            for layer in topless_yolo.layers:
                layer.trainable = False
        else:
            for layer in topless_yolo.layers[:8]:
                layer.trainable = False

        final_layer = Conv2D(len(anchors)*(5+len(class_names)), (1, 1), activation='linear')(topless_yolo.output)

        self.model_body = Model(image_input, final_layer)

        # Place model loss on CPU to reduce GPU memory usage.
        with tf.device('/cpu:0'):
            # TODO: Replace Lambda with custom Keras layer for loss.
            model_loss = Lambda(
                yolo_loss,
                output_shape=(1, ),
                name='yolo_loss',
                arguments={'anchors': anchors,
                           'num_classes': len(class_names)})([
                               self.model_body.output, boxes_input,
                               detectors_mask_input, matching_boxes_input
                           ])

        self.model = Model([self.model_body.input, boxes_input, detectors_mask_input, matching_boxes_input], model_loss)

    def train(self):
        validation_split = self.validation_split
        image_data = self.processed_images
        class_names = self.class_names
        anchors = self.anchors
        detectors_mask = self.detectors_mask
        matching_true_boxes = self.matching_true_boxes
        boxes = self.processed_boxes
        self.model.compile(optimizer='adam', loss={'yolo_loss': lambda y_true, y_pred: y_pred})
        logging_1 = TensorBoard(log_dir="{}/tensorboard_logs_1".format(self.root_dir))
        logging_2 = TensorBoard(log_dir="{}/tensorboard_logs_2".format(self.root_dir))
        csv_logger_1 = CSVLogger('{}/phase_1.log'.format(self.root_dir))
        csv_logger_2 = CSVLogger('{}/phase_2.log'.format(self.root_dir))
        checkpoint = ModelCheckpoint("{}/phase_2_best.h5".format(self.root_dir), monitor='val_loss',save_weights_only=True, save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')
        self.model.fit([image_data, boxes, detectors_mask, matching_true_boxes],np.zeros(len(image_data)),
                       validation_split=validation_split,batch_size=32,epochs=self.phase_1_epochs,callbacks=[logging_1,csv_logger_1])
        self.model.save_weights('{}/phase_1.h5'.format(self.root_dir))
        self.create_model(load_pretrained=False, freeze_body=False)
        self.model.load_weights('{}/phase_1.h5'.format(self.root_dir))
        self.model.compile(optimizer='adam', loss={'yolo_loss': lambda y_true, y_pred: y_pred})
        self.model.fit([image_data, boxes, detectors_mask, matching_true_boxes],np.zeros(len(image_data)),
                  validation_split=validation_split,batch_size=8,epochs=self.phase_2_epochs,callbacks=[logging_2, checkpoint, early_stopping,csv_logger_2])
        self.model.save_weights('{}/phase_2.h5'.format(self.root_dir))

    def predict(self):
        weights_name = '{}/phase_2_best.h5'.format(self.root_dir)
        self.model_body.load_weights(weights_name)
        yolo_outputs = yolo_head(self.model_body.output, self.anchors, len(self.class_names))
        input_image_shape = K.placeholder(shape=(2,))
        boxes, scores, classes = yolo_eval(yolo_outputs, input_image_shape, score_threshold=0.5, iou_threshold=0)
        sess = K.get_session()
        results = []
        for i_path in self.images:
            im = Image.open(i_path)
            image_data = np.array(im.resize((416, 416), Image.BICUBIC), dtype=np.float) / 255.
            if len(image_data.shape) >= 3:
                image_data = np.expand_dims(image_data, 0)
                feed_dict = {self.model_body.input: image_data,input_image_shape: [im.size[1], im.size[0]], K.learning_phase(): 0}
                out_boxes, out_scores, out_classes = sess.run([boxes, scores, classes],feed_dict=feed_dict)
                for i, c in list(enumerate(out_classes)):
                    box_class = self.class_names[c]
                    box = out_boxes[i]
                    score = out_scores[i]
                    label = '{}'.format(box_class)
                    top, left, bottom, right = box
                    top = max(0, np.floor(top + 0.5).astype('int32'))
                    left = max(0, np.floor(left + 0.5).astype('int32'))
                    bottom = min(im.size[1], np.floor(bottom + 0.5).astype('int32'))
                    right = min(im.size[0], np.floor(right + 0.5).astype('int32'))
                    results.append((i_path,box_class,score,top, left, bottom, right))
            else:
                logging.warning("skipping {} contains less than 3 channels".format(i_path))
        return results

    def load(self):
        weights_name = '{}/phase_2_best.h5'.format(self.root_dir)
        self.model_body.load_weights(weights_name)
        yolo_outputs = yolo_head(self.model_body.output, self.anchors, len(self.class_names))
        self.input_image_shape = K.placeholder(shape=(2,))
        self.tfboxes, self.tfscores, self.tfclasses = yolo_eval(yolo_outputs, self.input_image_shape, score_threshold=0.5, iou_threshold=0)
        self.sess = K.get_session()

    def apply(self,path,min_score):
        im = Image.open(path)
        image_data = np.array(im.resize((416, 416), Image.BICUBIC), dtype=np.float) / 255.
        image_data = np.expand_dims(image_data, 0)
        feed_dict = {self.model_body.input: image_data, self.input_image_shape: [im.size[1], im.size[0]],
                     K.learning_phase(): 0}
        out_boxes, out_scores, out_classes = self.sess.run([self.tfboxes, self.tfscores, self.tfclasses],
                                                           feed_dict=feed_dict)
        results = []
        for i, c in list(enumerate(out_classes)):
            box_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]
            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(im.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(im.size[0], np.floor(right + 0.5).astype('int32'))
            if score > min_score:
                results.append({'x': left,'y':top,'w':right - left,'h':bottom - top,
                                'score': score,'object_name':box_class})
        return results
