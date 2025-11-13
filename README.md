# Causal Effect Variational Transformer for Public Health Measures and COVID-19 Infection Cluster Analysis

This repository contains the official PyTorch implementation of CIKM 2025 paper:
"Causal Effect Variational Transformer for Public Health Measures and COVID-19 Infection Cluster Analysis".

Please use BibTeX below to cite.
'''
@inproceedings{kang2025causal,
  title={Causal Effect Variational Transformer for Public Health Measures and COVID-19 Infection Cluster Analysis},
  author={Kang, Jinho and Lim, Sungjun and Park, Hojun and Jung, Jiyoung and Jung, Jaehun and Song, Kyungwoo},
  booktitle={Proceedings of the 34th ACM International Conference on Information and Knowledge Management},
  pages={1282--1291},
  year={2025}
}
'''

## Visual Summary
### Overall Framework
![image](https://github.com/user-attachments/assets/6f753cc9-8288-4835-ae4f-9d31529f1463)

To address the limited availability of standardized medical benchmarks, we adopted the following three key strategies:
1) For the first time, we collected real-world COVID-19 infection spread time series cluster data with social distancing across two distinct distributions.
- Provincial COVID-19 Data

![image](https://github.com/user-attachments/assets/ae1a22f7-fc8c-4d34-aac5-bfd56baa0306)

- Municipal COVID-19 Data

![image](https://github.com/user-attachments/assets/3981dff2-5786-42e4-9449-a866de12c3a4)


2) To solve data scarcity, we proposed and utilized a cut-off data augmentation algorithm.

![image](https://github.com/user-attachments/assets/5fccd5c4-2192-4dca-9aa9-f0031b61430f)

4) To effectively utilize the data, we proposed a Causal Effect Variational Transformer (CEVT) to model the causal relationship between core variables.

![image](https://github.com/user-attachments/assets/251e8b29-a490-4b40-bb57-4a783424cc20)

## Abstract
Recent research increasingly integrates causal inference into deep learning models to enhance the explainability and robustness of medical applications. However, data scarcity remains a fundamental challenge due to privacy constraints and the high cost of data collection. This issue, compounded by complex variable dependencies and unobserved latent confounders, hinders the reliable estimation of causal effects. To address these challenges, we collect two real-world COVID-19 infection cluster datasets, including public health measures, from distinct distributions in collaboration with local governments, a medical university, and a hospital. We also propose a cut-off augmentation method that generates diverse feature–label pairs by slicing time-series sequences at different observation windows, effectively simulating partial observations common in real-world settings. We further introduce the Causal Effect Variational Transformer (CEVT), a Transformer-based model that captures temporal structure while accounting for multiple treatments and latent confounders through an iterative conditioning mechanism. In real-world settings, CEVT outperforms baseline models in predicting confirmed cases and cluster durations, and provides more accurate estimates of the causal impacts of public health measures, demonstrating robust performance across datasets collected from distinct distributions. Moreover, the magnitude of causal effects estimated by CEVT is consistent with previous findings examining the relationship between public health measures and infection spread, further supporting the reliability of CEVT.

## Implement Code Example

### Project Tree
```
├── data
│   ├── data_cut_0.csv # <provincial> cut-off 1~5
│   ├── data_cut_1.csv
│   ├── data_cut_2.csv
│   ├── data_cut_3.csv
│   ├── data_cut_4.csv
│   ├── data_cut_5.csv
│   ├── data_final_mod.csv
│   ├── data_mod.ipynb
│   ├── data_task.csv
│   └── municipal # <municipal>
│       └── preprocessed_municipal_98.csv
├── sh
│   ├── provincial.sh
│   └── municipal.sh
├── main.py
├── run_causal.py
├── run_itransformer.py
├── models.py
├── utils.py
└── README.md
```

We used the following Python packages for core development. We experimented with `Python 3.10.14`.
```
pytorch                   2.2.2
pandas                    2.2.1
numpy                     1.26.4
scikit-learn              1.4.1
scipy                     1.11.4
```

To install overall packages and reproduce the best model, please run the code below.
```
pip install -r requirements.txt

bash sh/provincial_best.sh
bash sh/municipal_best.sh

```
