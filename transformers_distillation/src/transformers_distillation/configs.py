from typing import Optional
import torch

try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None


def no_quant():
    return None

def quant_8():
    if BitsAndBytesConfig is None:
        raise ImportError("BitsAndBytes Not Available. Install 'BitsAndBytes' To Use 8-bit Quantization")
    
    return BitsAndBytesConfig(load_in_8bit = True)

# def quant_16():
#     return BitsAndBytesConfig(load_in_16bit = True)

def quant_4():
    if BitsAndBytesConfig is None:
        raise ImportError("BitsAndBytes Not Available. Install 'BitsAndBytes' To Use 4-bit Quantization")

    return BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_quant_type = "nf4",
        bnb_4bit_use_double_quant = True,
        bnb_4bit_compute_dtype = torch.bfloat16
    )

def custom_quant(**kwargs):
    if BitsAndBytesConfig is None:
        raise ImportError("BitsAndBytes Not Available. Install 'BitsAndBytes' To Use Custom Quantization")
    
    return BitsAndBytesConfig(**kwargs)