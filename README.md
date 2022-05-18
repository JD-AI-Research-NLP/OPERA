This is code for the NAACL 2022 Paper "OPERA: Operation-Pivoted Discrete Reasoning over Text".

We implement our model based on the code of NumNet and QDGAT.

## Environment
torch==1.7.1<br>
allennlp==2.0.1<br>
transformers==4.1<br>
spacy==2.1.9<br>

## Running
Preprocess data
```
bash scripts/prepare_albert_v2_data.sh
```
Training
```
bash scripts/train_albert_xxlarge_v2.sh
```
Prediction
```
bash scripts/predict_albert_large_v2.sh
```


## Citation
```
@article{zhou2022opera,
  title={OPERA: Operation-Pivoted Discrete Reasoning over Text},
  author={Zhou, Yongwei and Bao, Junwei and Duan, Chaoqun and Sun, Haipeng and Liang, Jiahui and Wang, Yifan and Zhao, Jing and Wu, Youzheng and He, Xiaodong and Zhao, Tiejun},
  journal={arXiv preprint arXiv:2204.14166},
  year={2022}
}
```
