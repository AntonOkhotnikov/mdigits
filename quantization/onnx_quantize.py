#!/usr/bin/env python3
"""Dynamic range quantization of fp32 model"""
import sys
import onnx
from onnxruntime.quantization import quantize_dynamic, quantize_qat


def main(model_fp32, model_quant):
    quantized_model = quantize_dynamic(model_fp32, model_quant)
    print('Quantized!')


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Usage: python3 onnx_quantize.py <fp32_model_path>.onnx <int8_model_path>.quant.onnx')
        sys.exit()

    model_fp32_path = sys.argv[1]
    model_quant_path = sys.argv[2]
    main(model_fp32_path, model_quant_path)
