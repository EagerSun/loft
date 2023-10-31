from io import BytesIO
import base64
import random
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

def original(value):
    try: return value
    except: raise ValueError("The input value is failed to transform: original!")

def string_to_int(value: str):
    try: return int(value)
    except: raise ValueError("The input value is failed to transform: string_to_int!")

def string_to_float(value: str):
    try: return float(value)
    except: raise ValueError("The input value is failed to transform: string_to_float!")

def int_to_string(value: int):
    try: return str(value)
    except: raise ValueError("The input value is failed to transform: int_to_string!")

def float_to_string(value: float):
    try: return str(value)
    except: raise ValueError("The input value is failed to transform: float_to_string!")

def bytes_to_int(value: bytes):
    try: return int(value.decode("utf-8"))
    except: raise ValueError("The input value is failed to transform: bytes_to_int!")

def bytes_to_float(value: bytes):
    try: return float(value.decode("utf-8"))
    except: raise ValueError("The input value is failed to transform: bytes_to_float!")

def int_to_bytes(value: int):
    try: return str(value).decode("utf-8")
    except: raise ValueError("The input value is failed to transform: int_to_bytes!")
def float_to_bytes(value: float):
    try: return str(value).decode("utf-8")
    except: raise ValueError("The input value is failed to transform: float_to_bytes!")

def bytes_to_string(value: bytes):
    try: return value.decode("utf-8")
    except: raise ValueError("The input value is failed to transform: bytes_to_utf8!")

def string_to_bytes(value: str):
    try: return value.encode("utf-8")
    except: raise ValueError("The input value is failed to transform: string_to_bytes!")

def bytes_to_Image(value: bytes):
    try: return Image.open(BytesIO(value)).convert('RGB')
    except: raise ValueError("The input value is failed to transform: bytes_to_Image!")

def bytes_to_base64(value: bytes):
    try: return base64.b64encode(value).decode('ascii')
    except: raise ValueError("The input value is failed to transform: bytes_to_base64!")

def base64_to_Image(value: str):
    try: return Image.open(BytesIO(base64.b64decode(value))).convert('RGB')
    except: raise ValueError("The input value is failed to transform: base64_to_Image!")

def base64_to_bytes(value: str):
    try: return base64.b64decode(value)
    except: raise ValueError("The input value is failed to transform: base64_to_bytes!")

def Image_to_base64(value: object):
    try:
        _buffer = BytesIO() 
        value.convert('RGB').save(_buffer, "JPEG")
        return base64.b64encode(_buffer.getvalue()).decode('ascii')
    except:
        raise ValueError("The input value is failed to transform: Image_to_base64!")

def bytes_to_urlbase64(value: bytes):
    try: return base64.urlsafe_b64encode(value).decode('ascii')
    except: raise ValueError("The input value is failed to transform: bytes_to_urlbase64!")

def urlbase64_to_Image(value: str):
    try: return Image.open(BytesIO(base64.urlsafe_b64decode(value))).convert('RGB')
    except: raise ValueError("The input value is failed to transform: urlbase64_to_Image!")

def urlbase64_to_bytes(value: str):
    try: return base64.urlsafe_b64decode(value)
    except: raise ValueError("The input value is failed to transform: urlbase64_to_bytes!")

def Image_to_urlbase64(value: object):
    try:
        _buffer = BytesIO()
        value.convert('RGB').save(_buffer, "JPEG")
        return base64.urlsafe_b64encode(_buffer.getvalue()).decode('ascii')
    except:
        raise ValueError("The input value is failed to transform: Image_to_urlbase64!")

def Image_to_bytes(value: object):
    try:
        _buffer = BytesIO()
        value.convert('RGB').save(_buffer, "JPEG")
        return _buffer.getvalue()
    except:
        raise ValueError("The input value is failed to transform: Image_to_bytes!")

_d_transform = {
    "original": original,
    "string_to_int": string_to_int,
    "string_to_float": string_to_float,
    "float_to_string": float_to_string,
    "int_to_string": int_to_string,
    "bytes_to_int": bytes_to_int,
    "bytes_to_float": bytes_to_float,
    "int_to_bytes": int_to_bytes,
    "float_to_bytes": float_to_bytes,
    "bytes_to_string": bytes_to_string,
    "string_to_bytes": string_to_bytes,
    "bytes_to_Image": bytes_to_Image,
    "bytes_to_base64": bytes_to_base64,
    "base64_to_Image": base64_to_Image,
    "base64_to_bytes": base64_to_bytes,
    "Image_to_base64": Image_to_base64,
    "bytes_to_urlbase64": bytes_to_urlbase64,
    "urlbase64_to_Image": urlbase64_to_Image,
    "urlbase64_to_bytes": urlbase64_to_bytes,
    "Image_to_urlbase64": Image_to_urlbase64,
    "Image_to_bytes": Image_to_bytes,
    "DROP": "DROP",
}

def return_transform(transform: str):
    return _d_transform.get(transform, original) 
