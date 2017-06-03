import numpy as np
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from .keras_yolo import preprocess_true_boxes, yolo_body, yolo_eval, yolo_head, yolo_loss
import draw_boxes
import os
from PIL import Image

DEFAULT_ANCHORS = [(0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434), (7.88282, 3.52778), (9.77052, 9.16828)]


class YOLOTrainer(object):

    def __init__(self,images,boxes,class_names,args):
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
        self.process_data()
        self.root_dir = args['root_dir']
        self.base_model = args['base_model'] if 'base_model' in args else "dvalib/yolo/model_data/yolo.h5"
        self.get_detector_mask()
        self.create_model()

    def process_data(self):
        images = [Image.open(i) for i in self.images]
        orig_size = np.array([float(images[0].width), float(images[0].height)])
        orig_size = np.expand_dims(orig_size, axis=0)
        print orig_size
        processed_images = [i.resize((416, 416), Image.BICUBIC) for i in images]
        processed_images = [np.array(image, dtype=np.float) for image in processed_images]
        processed_images = [image/255. for image in processed_images]
        boxes = [np.array(box,dtype=np.uint16).reshape((-1, 5)) for box in self.boxes]
        boxes_extents = [box[:, [2, 1, 4, 3, 0]] for box in boxes]
        boxes_xy = [0.5 * (box[:, 3:5] + box[:, 1:3]) for box in boxes]
        boxes_wh = [box[:, 3:5] - box[:, 1:3] for box in boxes]
        print boxes_wh[0]
        boxes_xy = [boxxy / orig_size for boxxy in boxes_xy]
        boxes_wh = [boxwh / orig_size for boxwh in boxes_wh]
        print "width,height",boxes_wh[0]
        boxes = [np.concatenate((boxes_xy[i], boxes_wh[i], box[:, 0:1]), axis=1) for i, box in enumerate(boxes)]
        max_boxes = 0
        for boxz in boxes:
            if boxz.shape[0] > max_boxes:
                max_boxes = boxz.shape[0]
        print "max_boxes",max_boxes
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
            topless_yolo_path = os.path.join('{}/'.format(self.root_dir), 'yolo_topless.h5')
            if not os.path.exists(topless_yolo_path):
                print("CREATING TOPLESS WEIGHTS FILE")
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
                print layer
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
        logging = TensorBoard()
        checkpoint = ModelCheckpoint("trained_stage_3_best.h5", monitor='val_loss',save_weights_only=True, save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')
        self.model.fit([image_data, boxes, detectors_mask, matching_true_boxes],np.zeros(len(image_data)),
                       validation_split=validation_split,batch_size=32,epochs=20,callbacks=[logging])
        self.model.save_weights('{}/trained_stage_1.h5'.format(self.root_dir))
        self.create_model(load_pretrained=False, freeze_body=False)
        self.model.load_weights('{}/trained_stage_1.h5'.format(self.root_dir))
        self.model.compile(optimizer='adam', loss={'yolo_loss': lambda y_true, y_pred: y_pred})
        self.model.fit([image_data, boxes, detectors_mask, matching_true_boxes],np.zeros(len(image_data)),
                  validation_split=validation_split,batch_size=8,epochs=50,callbacks=[logging, checkpoint, early_stopping])
        self.model.save_weights('{}/trained_stage_3.h5'.format(self.root_dir))

    def draw(self,model_body, class_names, anchors, image_data, image_set='val',save_all=True):
        weights_name = '{}/trained_stage_3_best.h5'.format(self.root_dir)
        out_path = "{}/output_images".format(self.root_dir)
        if image_set == 'train':
            image_data = np.array([np.expand_dims(image, axis=0)
                                   for image in image_data[:int(len(image_data) * .9)]])
        elif image_set == 'val':
            image_data = np.array([np.expand_dims(image, axis=0)
                                   for image in image_data[int(len(image_data) * .9):]])
        elif image_set == 'all':
            image_data = np.array([np.expand_dims(image, axis=0)
                                   for image in image_data])
        else:
            ValueError("draw argument image_set must be 'train', 'val', or 'all'")
        # model.load_weights(weights_name)
        print(image_data.shape)
        model_body.load_weights(weights_name)

        # Create output variables for prediction.
        yolo_outputs = yolo_head(model_body.output, anchors, len(class_names))
        input_image_shape = K.placeholder(shape=(2,))
        boxes, scores, classes = yolo_eval(
            yolo_outputs, input_image_shape, score_threshold=0.5, iou_threshold=0)

        # Run prediction on overfit image.
        sess = K.get_session()  # TODO: Remove dependence on Tensorflow session.

        if not os.path.exists(out_path):
            os.makedirs(out_path)
        for i in range(len(image_data)):
            out_boxes, out_scores, out_classes = sess.run(
                [boxes, scores, classes],
                feed_dict={
                    model_body.input: image_data[i],
                    input_image_shape: [image_data.shape[2], image_data.shape[3]],
                    K.learning_phase(): 0
                })
            print('Found {} boxes for image.'.format(len(out_boxes)))
            print(out_boxes)

            # Plot image with predicted boxes.
            image_with_boxes = draw_boxes.draw_boxes(image_data[i][0], out_boxes, out_classes,
                                                     class_names, out_scores)
            # Save the image:
            if save_all or (len(out_boxes) > 0):
                image = PIL.Image.fromarray(image_with_boxes)
                image.save(os.path.join(out_path, str(i) + '.png'))
            # plt.imshow(image_with_boxes, interpolation='nearest')
            # plt.show()