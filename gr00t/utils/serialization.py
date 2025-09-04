import io
import json

import msgpack
import numpy as np

from gr00t.data.dataset import ModalityConfig


def decode_custom_classes(obj):
    if "__ModalityConfig_class__" in obj:
        obj = ModalityConfig(**json.loads(obj["as_json"]))
    if "__ndarray_class__" in obj:
        obj = np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
    return obj


def encode_custom_classes(obj):
    if isinstance(obj, ModalityConfig):
        return {"__ModalityConfig_class__": True, "as_json": obj.model_dump_json()}
    if isinstance(obj, np.ndarray):
        output = io.BytesIO()
        np.save(output, obj, allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": output.getvalue()}
    return obj


# BASE_CLASSES = {'gr00t.data.dataset': ['ModalityConfig'], 'numpy.core.multiarray': ['_reconstruct'], 'numpy': ['ndarray', 'dtype']}


def dump(obj, file):
    file.write(dumps(obj))


def dumps(obj):
    return msgpack.packb(obj, default=encode_custom_classes)


def load(file):
    return msgpack.unpackb(file.read(), object_hook=decode_custom_classes)


def loads(s):
    return msgpack.unpackb(s, object_hook=decode_custom_classes)
