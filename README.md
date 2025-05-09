# Causal Effect Variational Transformer for Public Health Measures and COVID-19 Infection Cluster Analysis

This repository is an official implementation of the paper "Causal Effect Variational Transformer for Public Health Measures and COVID-19 Infection Cluster Analysis" with Pytorch.

## Visual Summary
### Overall Framework
![image](https://github.com/user-attachments/assets/6f753cc9-8288-4835-ae4f-9d31529f1463)

To mitigate the lack of medical data benchmarks, we utilized three approaches: 
1) For the first time, we collected real-world COVID-19 infection spread time series cluster data with social distancing across two distinct distributions.
- Provincial COVID-19 Data
![image](https://github.com/user-attachments/assets/ae1a22f7-fc8c-4d34-aac5-bfd56baa0306)

- Municipal COVID-19 Data
![image](https://github.com/user-attachments/assets/3981dff2-5786-42e4-9449-a866de12c3a4)


2) To solve data scarcity, we proposed and utilized a cut-off data augmentation algorithm.
![image](https://github.com/user-attachments/assets/5fccd5c4-2192-4dca-9aa9-f0031b61430f)

3) To effectively utilize the data, we proposed a Causal Effect Variational Transformer (CEVT) to model the causal relationship between core variables.
![image](https://github.com/user-attachments/assets/251e8b29-a490-4b40-bb57-4a783424cc20)

## Abstract
Recent research focuses on integrating causal inference into deep learning models to enhance the explainability and robustness of medical contexts. However, privacy concerns and the high cost of data collection pose challenges, causing data scarcity and insufficient modeling of medical time series. To address these challenges, we collected two real-world COVID-19 infection cluster datasets from distinct distributions, including social distancing policies, in collaboration with local governments, a medical university, and a hospital. These datasets are the first of their kind and open up new opportunities to analyze the causal relationship between the spread of COVID-19 and public health measures. We also proposed a cut-off augmentation to mitigate data scarcity by augmenting feature-label pairs, which theoretically improves robustness in terms of generalization bounds. Lastly, we introduced the Causal Effect Variational Transformer (CEVT), a Transformer-based causal model that effectively captures time-series features while accounting for latent confounders and multiple treatments. CEVT improves the prediction accuracy of the causal impact of public health measures on COVID-19 case counts and infection cluster duration, outperforming baseline models. Moreover, the magnitude of causal effects estimated by CEVT is consistent with previous findings examining the relationship between public health measures and infection spread, further supporting the reliability of CEVT.

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
