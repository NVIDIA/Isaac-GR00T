"""
ZeroMQ message serializer for gr00t inference bridge.

This module provides serialization utilities compatible with both Python 3.8 and 3.10,
ensuring seamless data exchange between the robot control client and inference server.

The serializer uses msgpack for efficient binary serialization and handles numpy arrays
by converting them to a portable format (.npy bytes).
"""

import io
from typing import Any

try:
    import msgpack
except ImportError:
    raise ImportError("msgpack is required. Install with: pip install msgpack")

try:
    import numpy as np
except ImportError:
    raise ImportError("numpy is required. Install with: pip install numpy")


class MsgSerializer:
    """
    Message serializer for ZeroMQ communication.
    
    Handles serialization/deserialization of dictionaries containing numpy arrays.
    Uses msgpack for general serialization and numpy's .npy format for arrays.
    """
    
    @staticmethod
    def to_bytes(data: Any) -> bytes:
        """
        Serialize data to bytes using msgpack.
        
        Args:
            data: Data to serialize (typically a dict with numpy arrays)
        
        Returns:
            Serialized bytes
        """
        return msgpack.packb(data, default=MsgSerializer.encode_custom_classes)
    
    @staticmethod
    def from_bytes(data: bytes) -> Any:
        """
        Deserialize bytes to Python objects using msgpack.
        
        Args:
            data: Serialized bytes
        
        Returns:
            Deserialized Python object
        """
        return msgpack.unpackb(data, object_hook=MsgSerializer.decode_custom_classes)
    
    @staticmethod
    def decode_custom_classes(obj):
        """
        Decode custom classes from msgpack object.
        
        Handles numpy arrays that were encoded as dictionaries with '__ndarray_class__' key.
        
        Args:
            obj: Object to decode
        
        Returns:
            Decoded object (numpy array if applicable, otherwise original object)
        """
        if not isinstance(obj, dict):
            return obj
        
        # Handle numpy arrays
        if "__ndarray_class__" in obj:
            # Load numpy array from .npy bytes
            npy_bytes = obj["as_npy"]
            return np.load(io.BytesIO(npy_bytes), allow_pickle=False)
        
        return obj
    
    @staticmethod
    def encode_custom_classes(obj):
        """
        Encode custom classes for msgpack serialization.
        
        Handles numpy arrays by converting them to a dictionary with '__ndarray_class__' key
        and storing the array as .npy bytes.
        
        Args:
            obj: Object to encode
        
        Returns:
            Encoded object (dictionary for numpy arrays, otherwise original object)
        """
        # Handle numpy arrays
        if isinstance(obj, np.ndarray):
            # Save numpy array as .npy bytes
            output = io.BytesIO()
            np.save(output, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": output.getvalue()}
        
        return obj

