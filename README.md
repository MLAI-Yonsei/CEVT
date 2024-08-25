# Causal Effect Variational Transformer for Social Distancing and COVID-19 Infection Cluster Analysis

This repository is an official implementation of the paper "Causal Effect Variational Transformer for Social Distancing and COVID-19 Infection Cluster Analysis" with Pytorch.

## Visual Summary
### Overall Framework
![image](https://github.com/user-attachments/assets/6f753cc9-8288-4835-ae4f-9d31529f1463)

To mitigate the lack of medical data benchmarks, we utilized three approaches: 
1) For the first time, we collected COVID-19 infection spread time series cluster data with social distancing.
![image](https://github.com/user-attachments/assets/7741cb96-6baa-4512-aef4-a440624012e8)

2) To solve data scarcity, we proposed and utilized a cut-off data augmentation algorithm.
![image](https://github.com/user-attachments/assets/5fccd5c4-2192-4dca-9aa9-f0031b61430f)

3) To effectively utilize the data, we proposed a Causal Effect Variational Transformer (CEVT) to model the causal relationship between core variables.
![image](https://github.com/user-attachments/assets/251e8b29-a490-4b40-bb57-4a783424cc20)

## Abstract
Recent research focuses on integrating causal inference into deep learning models to enhance the explainability and robustness of sensitive data such as medical records. However, due to high privacy concerns and the cost of collecting medical data, there is a lack of benchmark datasets. Additionally, the available datasets are sparse, and prior research lacks sufficient modeling of causal relationships in medical time series. Therefore, 1) we collected real-world datasets and 2) proposed effective data augmentation methods, 3) introduced a Structural Causal Model (SCM) that considers latent confounder. First, to validate the applicability to real-world scenarios, we collected COVID-19 infection cluster data with social distancing measures for the first time. This provides new data that can analyze the relationship between the spread of COVID-19 and social distancing policies. Second, we propose a cut-off augmentation as a solution to data scarcity. The cut-off algorithm augments feature and label pairs, theoretically demonstrating that it enhances the robustness in terms of the generalization bound. Lastly, we introduce the Causal Effect Variational Transformer (CEVT), a Transformer-based causal model that effectively captures time-series features and considers latent confounder and multiple treatments. CEVT enhances the prediction accuracy of the causal impact of social distancing on COVID-19 cases and infection cluster duration outperforming baseline models. It also effectively infers the causal relationship between social distancing and infection spread.

## Implement Code Example

We used the following Python packages for core development. We experimented with `Python 3.10.14`.
```
pytorch                   2.2.2
pandas                    2.2.1
numpy                     1.26.4
scikit-learn              1.4.1
scipy                     1.11.4
```

To install overall packages and reproduce the best model, please run the codes below.
```
pip install -r requirements.txt

bash sh/bestmodel_bash.sh
```
