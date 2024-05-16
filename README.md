# Sul-BertGRU
Python Requirement:
Python 3.6

Package Requirement:

pytorch 1.10.1+cu113
tensorflow 1.15
cleanlab 0.1.1
scikit-learn 0.24.2
pandas 1.1.5
numpy 1.15.4
scipy 1.5.2
six 1.16.0
tqdm 4.64.1

Bert Config:
Bert-Base(L=12,H=768): https://github.com/google-research/bert

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Steps:

1. Run extract_pro.py to randomly select the same number of negative samples files as the positive.

2. Run spilt_seq.py to segment protein sequences.

3. Run bert.bat in cmd to extract the initial feature of protein sequences.

4. Run CL/run_k_folds.py to obtain best_model.pth.

5. Uncomment the code in CL/load_data.py and then run run_k_folds.py to use Confidence Learning to clean negative samples.

6. Run Sul-BertGRU/run_k_folds_p.py to train the model.

