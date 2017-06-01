# import matplotlib.pyplot as plt
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


class YOLOTrainer(object):

    def __init__(self):
        pass

    def get_detector_mask(self,boxes, anchors):
        '''
        Precompute detectors_mask and matching_true_boxes for training.
        Detectors mask is 1 for each spatial position in the final conv layer and
        anchor that should be active for the given boxes and 0 otherwise.
        Matching true boxes gives the regression targets for the ground truth box
        that caused a detector to be active or 0 otherwise.
        '''
        detectors_mask = [0 for i in range(len(boxes))]
        matching_true_boxes = [0 for i in range(len(boxes))]
        for i, box in enumerate(boxes):
            detectors_mask[i], matching_true_boxes[i] = preprocess_true_boxes(box, anchors, [416, 416])

        return np.array(detectors_mask), np.array(matching_true_boxes)


    def process_data(self,images, boxes):
        images = [PIL.Image.fromarray(i) for i in images]
        orig_size = np.array([float(images[0].width), float(images[0].height)])
        orig_size = np.expand_dims(orig_size, axis=0)
        print orig_size
        processed_images = [i.resize((416, 416), PIL.Image.BICUBIC) for i in images]
        processed_images = [np.array(image, dtype=np.float) for image in processed_images]
        processed_images = [image/255. for image in processed_images]
        boxes = [box.reshape((-1, 5)) for box in boxes]
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
            if boxz.shape[0]  < max_boxes:
                zero_padding = np.zeros( (max_boxes-boxz.shape[0], 5), dtype=np.float32)
                boxes[i] = np.vstack((boxz, zero_padding))
        return np.array(processed_images), np.array(boxes)


    def create_model(self,anchors, class_names, load_pretrained=True, freeze_body=True):
        '''
        returns the body of the model and the model
        load_pretrained: whether or not to load the pretrained model or initialize all weights
        freeze_body: whether or not to freeze all weights except for the last layer's
        model_body: YOLOv2 with new output layer
        model: YOLOv2 with custom loss Lambda layer
        '''
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
            topless_yolo_path = os.path.join('/home/aub3/repos/DeepVideoAnalytics/dvalib/yolo/model_data/', 'yolo_topless.h5')
            if not os.path.exists(topless_yolo_path):
                print("CREATING TOPLESS WEIGHTS FILE")
                yolo_path = '/home/aub3/repos/DeepVideoAnalytics/dvalib/yolo/model_data/yolo.h5'
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

        model_body = Model(image_input, final_layer)

        # Place model loss on CPU to reduce GPU memory usage.
        with tf.device('/cpu:0'):
            # TODO: Replace Lambda with custom Keras layer for loss.
            model_loss = Lambda(
                yolo_loss,
                output_shape=(1, ),
                name='yolo_loss',
                arguments={'anchors': anchors,
                           'num_classes': len(class_names)})([
                               model_body.output, boxes_input,
                               detectors_mask_input, matching_boxes_input
                           ])

        model = Model(
            [model_body.input, boxes_input, detectors_mask_input,
             matching_boxes_input], model_loss)

        return model_body, model

    def train(self, model, class_names, anchors, image_data, boxes, detectors_mask, matching_true_boxes,
              validation_split=0.1):
        '''
        retrain/fine-tune the model

        logs training with tensorboard

        saves training weights in current directory

        best weights according to val_loss is saved as trained_stage_3_best.h5
        '''
        model.compile(
            optimizer='adam', loss={
                'yolo_loss': lambda y_true, y_pred: y_pred
            })  # This is a hack to use the custom loss function in the last layer.

        logging = TensorBoard()
        checkpoint = ModelCheckpoint("trained_stage_3_best.h5", monitor='val_loss',
                                     save_weights_only=True, save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')

        model.fit([image_data, boxes, detectors_mask, matching_true_boxes],
                  np.zeros(len(image_data)),
                  validation_split=validation_split,
                  batch_size=32,
                  epochs=20,
                  callbacks=[logging])
        model.save_weights('trained_stage_1.h5')

        model_body, model = self.create_model(anchors, class_names, load_pretrained=False, freeze_body=False)

        model.load_weights('trained_stage_1.h5')

        model.compile(
            optimizer='adam', loss={
                'yolo_loss': lambda y_true, y_pred: y_pred
            })  # This is a hack to use the custom loss function in the last layer.

        model.fit([image_data, boxes, detectors_mask, matching_true_boxes],
                  np.zeros(len(image_data)),
                  validation_split=0.1,
                  batch_size=8,
                  epochs=50,
                  callbacks=[logging, checkpoint, early_stopping])

        model.save_weights('trained_stage_3.h5')

    def draw(self,model_body, class_names, anchors, image_data, image_set='val',
             weights_name='trained_stage_3_best.h5', out_path="output_images", save_all=True):
        '''
        Draw bounding boxes on image data
        '''
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