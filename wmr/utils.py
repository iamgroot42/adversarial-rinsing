from wmr.compression import CompressionRemoval

def get_method(method: str):
    mapping = {
        "compressions": CompressionRemoval,
    }
    method_cls = mapping.get(method, None)
    if method_cls is None:
        raise ValueError(f"Method {method} not found")
    return method_cls
