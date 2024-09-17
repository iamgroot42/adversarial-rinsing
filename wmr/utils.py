from wmr.compression import CompressionRemoval

def get_method(method: str):
    mapping = {
        "compressions": CompressionRemoval,
    }
    method_cls = mapping.get(method, None)
    if method_cls is None:
        raise ValueError(f"Method {method} not found")
    return method_cls

def get_models_path():
    # TODO: Replace with ENV variable
    return "/home/groot/work/erasing-the-invisible/models"
