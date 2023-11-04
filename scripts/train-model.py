"""Gathers all images and preprocesses them further, use SAM to remove water and air and pad to 720x720."""

import pickle
import sys
import numpy as np
import albumentations as A
import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras import optimizers
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import *

def categorical_focal_loss(alpha, gamma=2.):
    """
    model.compile(loss=[categorical_focal_loss(alpha=[[.25, .25, .25]], gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    alpha = np.array(alpha, dtype=np.float32)
    def categorical_focal_loss_fixed(y_true, y_pred):
        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)
        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
        # Compute mean loss in mini_batch
        return K.mean(K.sum(loss, axis=-1))
    return categorical_focal_loss_fixed

class KerasModel:
    def __init__(self, num_classes, input_shape=(256, 256, 3), safe_augmentation=False, **kwargs):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.safe_augmentation = safe_augmentation
        self.config = kwargs
        self.setup()
        seq = [x for x in [self.l_base, self.l_head] if x is not None]
        for i, layer in enumerate(seq):
            if i == 0: x = layer(self.l_input)
            else: x = layer(x)
        self.model = Model(inputs=self.l_input, outputs=x)
        self.generator_train = ImageDataGenerator(
            preprocessing_function=lambda x: self.preprocess_image(x, True)
        )
        self.generator_valid = ImageDataGenerator(
            preprocessing_function=lambda x: self.preprocess_image(x, False)
        )
    
    def fit(self, directory, epochs=5, batch_size=32):
        train_generator = self.generator_train.flow_from_directory(
            f"{directory}/train", batch_size=batch_size,
            target_size=(self.input_shape[0], self.input_shape[1])
        )
        validation_generator = self.generator_valid.flow_from_directory(
            f"{directory}/valid", batch_size=batch_size,
            target_size=(self.input_shape[0], self.input_shape[1]),
        )
        return self.model.fit(train_generator, epochs=epochs, validation_data=validation_generator, verbose=1)
    
    def setup(self):
        self.l_input = Input(shape=self.input_shape)
        self.f_augmentor = self._augmentor()
        self.f_external = self._external()
        self.l_base = self._base()
        self.l_head = self._head()
        
    def _augmentor(self):
        if self.safe_augmentation:
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Sharpen(p=.5, lightness=(1, 1)),
                A.ShiftScaleRotate(rotate_limit=15, p=1),
                A.Resize(self.input_shape[0], self.input_shape[1], always_apply=True)
            ]) 
        else:
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.UnsharpMask(p=0.5),
                A.ZoomBlur(max_factor=1.05, p=.5),
                A.Spatter(intensity=0.1, p=.5),
                A.Sharpen(p=.5),
                A.Defocus(radius=3, p=0.1),
                A.RandomFog(fog_coef_lower=0.0, fog_coef_upper=0.2, p=0.5),
                A.MedianBlur(blur_limit=5, p=0.3),
                A.GaussianBlur(p=0.1),
                A.GaussNoise(p=0.1),
                A.ShiftScaleRotate(rotate_limit=15, p=1),
                A.Resize(self.input_shape[0], self.input_shape[1], always_apply=True)
            ])
    
    def _external(self):
        return None
    
    def preprocess_image(self, image, for_training=True):
        if for_training and self.f_augmentor:
            image = self.f_augmentor(image=image)['image']
        if self.f_external:
            image = self.f_external(image)
        return image
        
    def _base(self):
        return None
    
    def _head(self):
        return None
    
    def compile(self, optimizer="adam", loss=None):
        if not self.model: return
        topk1 = TopKCategoricalAccuracy(k=1, name="topk1")
        topk2 = TopKCategoricalAccuracy(k=2, name="topk2")
        topk3 = TopKCategoricalAccuracy(k=2, name="topk3")
        if loss is None: loss = categorical_focal_loss(alpha=[[1/self.num_classes for i in range(self.num_classes)]], gamma=2)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=[topk1, topk2, topk3])
    
    def summary(self):
        if self.model: self.model.summary()
            
class GreyCNN(KerasModel):
    
    def _base(self):
        model = Sequential()
        model.add(Rescaling(scale=1./255))
        model.add(Lambda(lambda x: x[:, :, :, 0]))
        model.add(Reshape((self.input_shape[0], self.input_shape[1], 1)))
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.4))
        model.add(BatchNormalization())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.4))
        model.add(BatchNormalization())
        model.add(Dense(self.num_classes, activation='softmax'))
        return model
    
class ConvMixer(KerasModel):
    
    def _base(self):
        def activation_block(x):
            x = Activation("gelu")(x)
            return BatchNormalization()(x)
        def conv_stem(x, filters: int, patch_size: int):
            x = Conv2D(filters, kernel_size=patch_size, strides=patch_size)(x)
            return activation_block(x)
        def conv_mixer_block(x, filters: int, kernel_size: int):
            x0 = x
            x = DepthwiseConv2D(kernel_size=kernel_size, padding="same")(x)
            x = Add()([activation_block(x), x0])
            x = Conv2D(filters, kernel_size=1)(x)
            x = activation_block(x)
            return x
        inputs = Input((self.input_shape[0], self.input_shape[1], 3))
        x = Rescaling(scale=1.0 / 255)(inputs)
        x = Lambda(lambda x: x[:, :, :, 0])(x)
        x = Reshape((self.input_shape[0], self.input_shape[1], 1))(x)
        x = conv_stem(x, 256, 2)
        for _ in range(8): x = conv_mixer_block(x, 256, 5)
        x = GlobalAvgPool2D()(x)
        x = Dropout(0.4)(x)
        outputs = Dense(self.num_classes, activation="softmax")(x)
        return Model(inputs, outputs)
    
class ConvNextS(KerasModel):
        
    def _base(self):
        self._backbone = ConvNeXtSmall(include_top=False, pooling='avg', weights='imagenet', input_shape=self.input_shape)
        model = Sequential()
        model.add(self._backbone)
        model.add(Dropout(0.4))
        model.add(Dense(self.num_classes, activation='softmax'))
        model.layers[0].trainable = False
        return model
    
class ConvNextM(KerasModel):
        
    def _base(self):
        self._backbone = ConvNeXtBase(include_top=False, pooling='avg', weights='imagenet', input_shape=self.input_shape)
        model = Sequential()
        model.add(self._backbone)
        model.add(Dropout(0.4))
        model.add(Dense(self.num_classes, activation='softmax'))
        model.layers[0].trainable = False
        return model
    
class ConvNextL(KerasModel):
        
    def _base(self):
        self._backbone = ConvNeXtLarge(include_top=False, pooling='avg', weights='imagenet', input_shape=self.input_shape)
        model = Sequential()
        model.add(self._backbone)
        model.add(Dropout(0.4))
        model.add(Dense(self.num_classes, activation='softmax'))
        model.layers[0].trainable = False
        return model
    
class EfficientNetB0CNN(KerasModel):
        
    def _base(self):
        self._backbone = EfficientNetB0(include_top=False, pooling='avg', weights='imagenet')
        model = Sequential()
        model.add(self._backbone)
        model.add(Dropout(0.4))
        model.add(Dense(self.num_classes, activation='softmax'))
        model.layers[0].trainable = False
        return model
    
class EfficientNet2B0CNN(KerasModel):
        
    def _base(self):
        self._backbone = EfficientNetV2B0(include_top=False, pooling='avg', weights='imagenet')
        model = Sequential()
        model.add(self._backbone)
        model.add(Dropout(0.4))
        model.add(Dense(self.num_classes, activation='softmax'))
        model.layers[0].trainable = False
        return model
    
class EfficientNet2SCNN(KerasModel):
        
    def _base(self):
        self._backbone = EfficientNetV2S(include_top=False, pooling='avg', weights='imagenet')
        model = Sequential()
        model.add(self._backbone)
        model.add(Dropout(0.4))
        model.add(Dense(self.num_classes, activation='softmax'))
        model.layers[0].trainable = False
        return model
    
class EfficientNet2MCNN(KerasModel):
        
    def _base(self):
        self._backbone = EfficientNetV2M(include_top=False, pooling='avg', weights='imagenet')
        model = Sequential()
        model.add(self._backbone)
        model.add(Dropout(0.4))
        model.add(Dense(self.num_classes, activation='softmax'))
        model.layers[0].trainable = False
        return model
    
class ResNet50CNN(KerasModel):
    
    def _external(self):
        return resnet50.preprocess_input
    
    def _base(self):
        self._backbone = ResNet50(include_top=False, pooling='avg', weights='imagenet')
        model = Sequential()
        model.add(self._backbone)
        model.add(Dropout(0.4))
        model.add(Dense(self.num_classes, activation='softmax'))
        model.layers[0].trainable = False
        return model
    
class ResNet50V2CNN(KerasModel):
    
    def _external(self):
        return resnet_v2.preprocess_input
    
    def _base(self):
        self._backbone = ResNet50V2(include_top=False, pooling='avg', weights='imagenet')
        model = Sequential()
        model.add(self._backbone)
        model.add(Dropout(0.4))
        model.add(Dense(self.num_classes, activation='softmax'))
        model.layers[0].trainable = False
        return model

def fit_model(model, focal_loss=True, interleave_external=False, dont_unfreeze=False):
    epochs_lr_unfreeze = [
        (3, 1e-2, False),
        (10, 1e-4, True),
        (30, 1e-6, True),
    ]
    history = []
    suffix = "" if segmented else "-nowhite"
    for epochs, lr, unfreeze in epochs_lr_unfreeze:
        model.compile(
            optimizer=optimizers.Adam(learning_rate=lr),
            loss = None if focal_loss else "categorical_crossentropy"
        )
        if unfreeze and not dont_unfreeze:
            for layer in model.model.layers[1].layers[0].layers:
                if not isinstance(layer, BatchNormalization):
                    layer.trainable = True
        if interleave_external:
            hist = model.fit(
                f"data_prep/external{suffix}/", 
                epochs=epochs//2
            )
            history.append(hist)
            hist = model.fit(
                f"data_prep/royalnavy{suffix}/", 
                epochs=epochs//2
            )
            history.append(hist)
        else:
            hist = model.fit(f"data_prep/royalnavy{suffix}/", epochs=epochs)
            history.append(hist)
    for layer in model.model.layers[1].layers[0].layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False
    

def test_model(model):
    test_generator_nowhite = model.generator_valid.flow_from_directory(
        f"data_prep/royalnavy-nowhite/valid", batch_size=1,
        target_size=(model.input_shape[0], model.input_shape[1]),
        shuffle=False,
    )
    y_predictions_nowhite = model.model.predict(test_generator_nowhite)
    test_generator = model.generator_valid.flow_from_directory(
        f"data_prep/royalnavy/valid", batch_size=1,
        target_size=(model.input_shape[0], model.input_shape[1]),
        shuffle=False,
    )
    y_predictions = model.model.predict(test_generator)
    return test_generator.labels, test_generator_nowhite.labels, y_predictions, y_predictions_nowhite

configs = {
    "basic_grey": (GreyCNN, {"input_shape": (224, 224, 3)}, {"dont_unfreeze": True}),
    "convmixer": (ConvMixer, {"input_shape": (64, 64, 3)}, {"dont_unfreeze": True}),
    "convnexts": (ConvNextS, {"input_shape": (224, 224, 3)}, {}),
    "convnextm": (ConvNextM, {"input_shape": (224, 224, 3)}, {}),
    "convnextl": (ConvNextL, {"input_shape": (224, 224, 3)}, {}),
    "efficientnetb0": (EfficientNetB0CNN, {"input_shape": (224, 224, 3)}, {}),
    "efficientnetv2b0": (EfficientNet2B0CNN, {"input_shape": (224, 224, 3)}, {}),
    "efficientnetv2s": (EfficientNet2SCNN, {"input_shape": (224, 224, 3)}, {}),
    "efficientnetv2m": (EfficientNet2MCNN, {"input_shape": (224, 224, 3)}, {}),
    "resnet50": (ResNet50CNN, {"input_shape": (224, 224, 3)}, {}),
    "resnet50v2": (ResNet50V2CNN, {"input_shape": (224, 224, 3)}, {}),
}

if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    name = sys.argv[1]
    focal = int(sys.argv[2])
    external = int(sys.argv[3])
    segmented = int(sys.argv[4])
    config = configs[name]

    fn = f"{name}-focal={focal}-external={external}-segmented={segmented}"
    model = config[0](num_classes=8, safe_augmentation=segmented, **config[1])
    history = fit_model(model, focal_loss=focal, interleave_external=external, **config[2])
    results = test_model(model)
    model.model.save(f'models/{fn}.keras')
    

    pickle.dump(results, open(f"models/{fn}.results.pickle", "wb"))
    pickle.dump(history, open(f"models/{fn}.history.pickle", "wb"))




