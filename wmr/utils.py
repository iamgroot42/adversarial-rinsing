from wmr.compression import CompressionRemoval
from wmr.low_pass import FilterRemoval
# from bbeval.models.pytorch.wrapper import PyTorchModelWrapper

def get_method(method: str):
    mapping = {
        "compression": CompressionRemoval,
        'lowpass': FilterRemoval
    }
    method_cls = mapping.get(method, None)
    if method_cls is None:
        raise ValueError(f"Method {method} not found")
    return method_cls

def get_models_path():
    # TODO: Replace with ENV variable
    return "/home/groot/work/erasing-the-invisible/models"

# class DetectionWrapped(PyTorchModelWrapper):
#     def __init__(self, model):
#         super().__init__(model_config=None)
#         self.model = model
