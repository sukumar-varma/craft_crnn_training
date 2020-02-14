import zipfile
import datetime
import string
import glob
import math
import os

import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn.model_selection

import keras_ocr

# assert tf.test.is_gpu_available()
data_dir = '.'

alphabet = string.digits + string.ascii_letters + string.punctuation + ' '
recognizer_alphabet = ''.join(sorted(set(alphabet.lower())))
fonts = keras_ocr.data_generation.get_fonts(
    alphabet=alphabet,
    cache_dir=data_dir
)
backgrounds = glob.glob(os.path.join("backs", '*.png'))

text_generator = keras_ocr.data_generation.get_text_generator(alphabet=alphabet, max_string_length=10)
print('The first generated text is:', next(text_generator))

def get_train_val_test_split(arr):
    train, valtest = sklearn.model_selection.train_test_split(arr, train_size=0.8, random_state=42)
    val, test = sklearn.model_selection.train_test_split(valtest, train_size=0.5, random_state=42)
    return train, val, test

background_splits = get_train_val_test_split(backgrounds)
font_splits = get_train_val_test_split(fonts)

image_generators = [
    keras_ocr.data_generation.get_image_generator(
        height=165,
        width=1024,
        text_generator=text_generator,
        font_groups={
            alphabet: current_fonts
        },
        backgrounds=current_backgrounds,
        background_crop_mode = 'letterbox',
        font_size=(60, 120),
        margin=50,
        rotationX=(-0.05, 0.05),
        rotationY=(-0.05, 0.05),
        rotationZ=(-15, 15)
    )  for current_fonts, current_backgrounds in zip(
        font_splits,
        background_splits
    )
]

# Detector CRAFT
detector = keras_ocr.detection.Detector(weights='clovaai_general')
recognizer = keras_ocr.recognition.Recognizer
    alphabet=recognizer_alphabet,
    weights='kurapan',
    include_top=False
)
recognizer.compile()
for layer in recognizer.backbone.layers:
    layer.trainable = False


detector_batch_size = 1
detector_basepath = os.path.join(data_dir, f'detector_{datetime.datetime.now().isoformat()}')
detection_train_generator, detection_val_generator, detection_test_generator = [
    detector.get_batch_generator(
        image_generator=image_generator,
        batch_size=detector_batch_size
    ) for image_generator in image_generators
]
detector.model.fit_generator(
    generator=detection_train_generator,
    steps_per_epoch=math.ceil(len(background_splits[0]) / detector_batch_size),
    epochs=1000,
    workers=0,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(restore_best_weights=True, patience=5),
        tf.keras.callbacks.CSVLogger(f'{detector_basepath}.csv'),
        tf.keras.callbacks.ModelCheckpoint(filepath=f'{detector_basepath}.h5')
    ],
    validation_data=detection_val_generator,
    validation_steps=math.ceil(len(background_splits[1]) / detector_batch_size)
)


'''
Recognition CRNN
'''
# max_length = 10
# recognition_image_generators = [
#     keras_ocr.data_generation.convert_image_generator_to_recognizer_input(
#         image_generator=image_generator,
#         max_string_length=min(recognizer.training_model.input_shape[1][1], max_length),
#         target_width=recognizer.model.input_shape[2],
#         target_height=recognizer.model.input_shape[1],
#         margin=1
#     ) for image_generator in image_generators
# ]

# recognition_batch_size = 8
# recognizer_basepath = os.path.join(data_dir, f'recognizer_{datetime.datetime.now().isoformat()}')
# recognition_train_generator, recognition_val_generator, recogntion_test_generator = [
#     recognizer.get_batch_generator(
#       image_generator=image_generator,
#       batch_size=recognition_batch_size,
#       lowercase=True
#     ) for image_generator in recognition_image_generators
# ]
# recognizer.training_model.fit_generator(
#     generator=recognition_train_generator,
#     epochs=1000,
#     steps_per_epoch=math.ceil(len(background_splits[0]) / recognition_batch_size),
#     callbacks=[
#       tf.keras.callbacks.EarlyStopping(restore_best_weights=True, patience=25),
#       tf.keras.callbacks.CSVLogger(f'{recognizer_basepath}.csv', append=True),
#       tf.keras.callbacks.ModelCheckpoint(filepath=f'{recognizer_basepath}.h5')
#     ],
#     validation_data=recognition_val_generator,
#     validation_steps=math.ceil(len(background_splits[1]) / recognition_batch_size),
#     workers=0
# )
