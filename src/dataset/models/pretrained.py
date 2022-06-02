from tensorflow.keras.applications import *

models = [
    ('DenseNet121',DenseNet121(weights=None)),
    ('DenseNet169', DenseNet169(weights=None)),
    ('DenseNet201', DenseNet201(weights=None)),
    ('EfficientNetB0', EfficientNetB0(weights=None)),
    ('EfficientNetB1', EfficientNetB1(weights=None)),
    ('EfficientNetB2', EfficientNetB2(weights=None)),
    ('EfficientNetB3', EfficientNetB3(weights=None)),
    ('EfficientNetB4', EfficientNetB4(weights=None)),
    ('EfficientNetB5', EfficientNetB5(weights=None)),
    ('EfficientNetB6', EfficientNetB6(weights=None)),
    ('EfficientNetB7', EfficientNetB7(weights=None)),
    ('EfficientNetV2B0', EfficientNetV2B0(weights=None)),
    ('EfficientNetV2B1', EfficientNetV2B1(weights=None)),
    ('EfficientNetV2B2', EfficientNetV2B2(weights=None)),
    ('EfficientNetV2B3', EfficientNetV2B3(weights=None)),
    ('EfficientNetV2L', EfficientNetV2L(weights=None)),
    ('EfficientNetV2M', EfficientNetV2M(weights=None)),
    ('EfficientNetV2S', EfficientNetV2S(weights=None)),
    ('InceptionResNetV2', InceptionResNetV2(weights=None)),
    ('InceptionV3', InceptionV3(weights=None)),
    ('MobileNet', MobileNet(weights=None)),
    ('MobileNetV2', MobileNetV2(weights=None)),
    ('NASNetLarge', NASNetLarge(weights=None)),
    ('NASNetMobile', NASNetMobile(weights=None)),
    ('ResNet101', ResNet101(weights=None)),
    ('ResNet101V2', ResNet101V2(weights=None)),
    ('ResNet152', ResNet152(weights=None)),
    ('ResNet152V2', ResNet152V2(weights=None)),
    ('ResNet50', ResNet50(weights=None)),
    ('ResNet50V2', ResNet50V2(weights=None)),
    ('ResNetRS101', ResNetRS101(weights=None)),
    ('ResNetRS152', ResNetRS152(weights=None)),
    ('ResNetRS200', ResNetRS200(weights=None)),
    ('ResNetRS270', ResNetRS270(weights=None)),
    ('ResNetRS350', ResNetRS350(weights=None)),
    ('ResNetRS420', ResNetRS420(weights=None)),
    ('ResNetRS50', ResNetRS50(weights=None)),
    ('VGG16', VGG16(weights=None)),
    ('VGG19', VGG19(weights=None)),
    ('Xception', Xception(weights=None))
]

def get_model(model_index:int):
    model_name, model = models[model_index]
    return model, model_name