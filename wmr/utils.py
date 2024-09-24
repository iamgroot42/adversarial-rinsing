from wmr.compression import CompressionRemoval
from wmr.low_pass import FilterRemoval
from wmr.latent_perturbations import VAERemoval
from wmr.augmentation_ensemble import FilterEnsemble
# from bbeval.models.pytorch.wrapper import PyTorchModelWrapper

def get_method(method: str):
    mapping = {
        "compression": CompressionRemoval,
        'lowpass': FilterRemoval,
        "vae": VAERemoval,
        "filters": FilterEnsemble
    }
    method_cls = mapping.get(method, None)
    if method_cls is None:
        raise ValueError(f"Method {method} not found")
    return method_cls


# class DetectionWrapped(PyTorchModelWrapper):
#     def __init__(self, model):
#         super().__init__(model_config=None)
#         self.model = model
