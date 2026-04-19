from .backbone import ResNetBackbone

NUM_CLASSES = 10


def build_model(cfg: dict) -> ResNetBackbone:
    mcfg = cfg['model']
    return ResNetBackbone(
        variant     = mcfg['variant'],
        depth       = mcfg.get('depth', 4),
        num_classes = NUM_CLASSES,
    )
