__all__ = ["VolumeRenderer", "Scene", "TFManager", "MultiViewExporter"]


def __getattr__(name):
    if name == "VolumeRenderer":
        from .volume_renderer import VolumeRenderer
        return VolumeRenderer
    if name == "Scene":
        from .scene import Scene
        return Scene
    if name == "TFManager":
        from .tf_manager import TFManager
        return TFManager
    if name == "MultiViewExporter":
        from .exporter import MultiViewExporter
        return MultiViewExporter
    raise AttributeError(name)
