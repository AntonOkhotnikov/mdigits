## Quantization

* Here `.nemo` -> `.onnx` -> `.quant.onnx` transform is used

* Original `.nemo` model could be converted to `.onnx` as follows:
```bash
git clone https://github.com/NVIDIA/NeMo.git
cd NeMo/scripts/export
python3 convasr_to_single_onnx.py --nemo_file <path_to>/QuartzNet5x1Aug.nemo --onnx_file <path_to>/QuartzNet5x1Aug.onnx --model_type asr
```

* `.onnx` -> `.quant.onnx` transform could be performed using script in this folder as follows:
```bash
python3 onnx_quantize.py <fp32_model_path>.onnx <int8_model_path>.quant.onnx>
```

* Both int8 and fp32 models could be evaluated using the scripts in `../evaluation`

* ONNX-quantized and PyTorch fp32 models outputs currently don't match, e.g.: `897*868 vs 89*47`. Quantization still requires further work to match outputs
