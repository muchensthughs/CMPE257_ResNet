from .backbone import ResNetBackbone

NUM_CLASSES = 10


def build_model(cfg: dict) -> ResNetBackbone:
    """
    Construct the ResNet model from a parsed YAML config.

    Reads the `model` section (variant + depth) and returns a ready-to-train
    ResNetBackbone. CIFAR-10 fixes the output dimension, so num_classes is
    not configurable here.
    """
    mcfg = cfg['model']
    return ResNetBackbone(
        variant     = mcfg['variant'],
        depth       = mcfg.get('depth', 4),
        num_classes = NUM_CLASSES,
    )
