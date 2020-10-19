# QuanTangshi
- An authorship attribution implement in pytorch for Chinese classical poem.

## Requirement ##

	pyorch : 1.0.1
	python : 3.6
	cuda : 8.0/9.0 (support cuda speed up, can chose)

## Usage ##
 
modify the config file, see the Config directory([here](https://github.com/bamtercelboo/pytorch_text_classification/tree/master/Config)) for detail.  

	1、python main.py
	2、python main.py --config_file ./Config/config.cfg --device cuda:0 --train -p


## Model ##

- CNN
- BiLSTM
- CNN+Attention
- BiLSTM+Attention
- Transformer+CNN(cnn1) 

## Data ##

- LD LiBai, DuFu
- WYLL WangBo, YangJiong, LuZhaolin, LuoBinwang
- 12 poet 12 famous poets in Tang Dynasty
