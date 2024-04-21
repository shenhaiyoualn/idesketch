
from .generation_backbones import ResnetGenerator, UnetGenerator
from .sr_backbones import MLANet


# __all__ = [
#     'VGG16', 'PlainDecoder', 'SimpleEncoderDecoder','MSRResNet','MLANet',
#     'GLEncoderDecoder', 'GLEncoder', 'GLDecoder', 'GLDilationNeck',
#     'PConvEncoderDecoder', 'PConvEncoder', 'PConvDecoder', 'ResNetEnc',
#     'ResNetDec', 'ResShortcutEnc', 'ResShortcutDec', 'RRDBNet',
#     'DeepFillEncoder', 'HolisticIndexBlock', 'DepthwiseIndexBlock',
#     'ContextualAttentionNeck', 'DeepFillDecoder', 'EDSR',
#     'DeepFillEncoderDecoder', 'EDVRNet', 'IndexedUpsample', 'IndexNetEncoder',
#     'IndexNetDecoder', 'TOFlow', 'ResGCAEncoder', 'ResGCADecoder', 'SRCNN',
#     'UnetGenerator', 'ResnetGenerator'
# ]
__all__ = [
    'UnetGenerator', 'ResnetGenerator'
]
