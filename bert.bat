set /A a = 1
set /A ALLLINE = 16698
set /A RealAllLine = %ALLLINE%-1

echo blasting...
:loop
if %a% == %ALLLINE% goto end
	python extract_features.py -input_file=RLL\left_neg\%a%.seq -output_file=RLL\output_left_neg\%a%.jsonl -vocab_file=Bert_cofig\vocabs.txt -bert_config_file=Bert_cofig\bert_config.json -init_checkpoint=Bert_cofig\bert_model.ckpt -do_lower_case=False -layers=-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12 -max_seq_length=512 -batch_size=64
	python jsonl2csv.py RLL\output_left_neg\%a%.jsonl RLL\output_left_neg_csv\%a%.csv
	
	python extract_features.py -input_file=RLL\left_pos\%a%.seq -output_file=RLL\output_left_pos\%a%.jsonl -vocab_file=Bert_cofig\vocabs.txt -bert_config_file=Bert_cofig\bert_config.json -init_checkpoint=Bert_cofig\bert_model.ckpt -do_lower_case=False -layers=-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12 -max_seq_length=512 -batch_size=64
	python jsonl2csv.py RLL\output_left_pos\%a%.jsonl RLL\output_left_pos_csv\%a%.csv
	
	python extract_features.py -input_file=RLL\right_pos\%a%.seq -output_file=RLL\output_right_pos\%a%.jsonl -vocab_file=Bert_cofig\vocabs.txt -bert_config_file=Bert_cofig\bert_config.json -init_checkpoint=Bert_cofig\bert_model.ckpt -do_lower_case=False -layers=-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12 -max_seq_length=512 -batch_size=64
	python jsonl2csv.py RLL\output_right_pos\%a%.jsonl RLL\output_right_pos_csv\%a%.csv
	
	python extract_features.py -input_file=RLL\right_neg\%a%.seq -output_file=RLL\output_right_neg\%a%.jsonl -vocab_file=Bert_cofig\vocabs.txt -bert_config_file=Bert_cofig\bert_config.json -init_checkpoint=Bert_cofig\bert_model.ckpt -do_lower_case=False -layers=-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12 -max_seq_length=512 -batch_size=64
	python jsonl2csv.py RLL\output_right_neg\%a%.jsonl RLL\output_right_neg_csv\%a%.csv
	
	python extract_features.py -input_file=RLL\pos\%a%.seq -output_file=RLL\output_pos\%a%.jsonl -vocab_file=Bert_cofig\vocabs.txt -bert_config_file=Bert_cofig\bert_config.json -init_checkpoint=Bert_cofig\bert_model.ckpt -do_lower_case=False -layers=-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12 -max_seq_length=512 -batch_size=64
	python jsonl2csv.py RLL\output_pos\%a%.jsonl RLL\output_pos_csv\%a%.csv
	
	python extract_features.py -input_file=RLL\neg\%a%.seq -output_file=RLL\output_neg\%a%.jsonl -vocab_file=Bert_cofig\vocabs.txt -bert_config_file=Bert_cofig\bert_config.json -init_checkpoint=Bert_cofig\bert_model.ckpt -do_lower_case=False -layers=-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12 -max_seq_length=512 -batch_size=64
	python jsonl2csv.py RLL\output_neg\%a%.jsonl RLL\output_neg_csv\%a%.csv
	set /A a = %a% + 1
	goto :loop
:end
echo blasting end successfully

pause