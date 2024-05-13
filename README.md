Dependencies
- Python 3.6
- Pytorch 1.10.1+cu113
- Tensorflow 1.15: https://www.tensorflow.org/
- bert_large_L-24_H-1024_A-16_I-512: https://github.com/google-research/bert

### Prediction step-by-step:
### Step 1
Use "extract_pro.py" file to randomly select the same number of negative sample files as the positive sample
- *python extract_seq.py*
### Step 2

Use "spilt_seq.py" to segment protein sequences

### Step 3
Run the "bert.bat" in cmd to complete the initial feature extraction

### Step 4
Run "BertLSTM/run_k_kold.py" file to train the model


