
## Model evaluation

* While training the symbol '\*' was used to replace word 'тысяча(-и)'. Thus, at evaluation time the resulting string requires some postprocessing '\*' symbol removal

* Greedy decoder is used to obtain final transcription


## Description

* `artefacts/` - folder containing `.nemo`, `.onnx` and `.quant.onnx` for best performed model

* `infer_nemo.py` - run `.nemo` model inference over 4 testing files with greedy decoding and no text postprocessing

* `infer_onnx.py` - run `.onnx` model inference over 4 testing files with greedy decoding and no text postprocessing

* `run_nemo_evaluation_over_csv_file.py` - runs `.nemo` inference over testing .csv file with greedy decoding and text postprocessing. Please provide full absolute paths to audio files in .csv


## Final solution evaluation
* Final model could be evaluated as follows:
```bash
python3 run_nemo_evaluation_over_csv_file.py artefacts/QuartzNet5x1Aug.nemo <path_to_input>.csv <path_to_output>.csv [batch_size=32]
```
