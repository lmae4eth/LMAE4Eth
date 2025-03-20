# Ethereum Fraud Detection--KGBERT4Eth
This is an implementation of the paper - "LMAE4Eth: Generalizable and Robust Ethereum Fraud Detection by Exploring Transaction Semantics and Masked Graph Embedding"
## Overview
As Ethereum confronts increasingly sophisticated fraud threats, recent research seeks to improve fraud account detection by leveraging advanced pre-trained Transformer or self-supervised graph neural network. However, current Transformer-based methods rely on context-independent, numerical transaction sequences, failing to capture semantic of account transactions. Furthermore, the pervasive homogeneity in Ethereum transaction records renders it challenging to learn discriminative account embeddings. Moreover, current self-supervised graph learning methods primarily learn node representations through graph reconstruction, resulting in suboptimal performance for node-level tasks like fraud account detection, while these methods also encounter scalability challenges.

To tackle these challenges, we propose LMAE4Eth, a multi-view learning framework that fuses transaction semantics, masked graph embedding, and expert knowledge. We first propose a transaction-token contrastive language model (TxCLM) that transforms context-independent numerical transaction records into logically cohesive linguistic representations, and leverages language modeling to learn transaction semantics. To clearly characterize the semantic differences between accounts, we also use a token-aware contrastive learning pre-training objective, which, together with the masked transaction model pre-training objective, learns high-expressive account representations. We then propose a masked account graph autoencoder (MAGAE) using generative self-supervised learning, which achieves superior node-level account detection by focusing on reconstructing account node features rather than graph structure. To enable MAGAE to scale for large-scale training, we propose to integrate layer-neighbor sampling into the graph, which reduces the number of sampled vertices by several times without compromising training quality. Additionally, we initialize the account nodes in the graph with expert-engineered features to inject empirical and statistical knowledge into the model. Finally, using a cross-attention fusion network, we unify the embeddings of TxCLM and MAGAE to leverage the benefits of both. We evaluate our method against 15 baseline approaches on three datasets. Experimental results show that our method outperforms the best baseline by over 10% in F1-score on two of the datasets. Furthermore, we observe from three datasets that the proposed method demonstrates unprecedented generalization ability compared to previous work. 
## Model Designs
![image](framework.png)


## Requirementse

```
- Python (>=3.8.10)
- Pytorch (>2.3.1)
- Numpy (>=1.24.4)
- Pandas (>=1.4.4)
- Transformers (2.0.0)
- Scikit-learn (>=1.1.3)
- dgl (>=2.0.0)
- Gensim (>=4.3.2)
- Scipy (>=1.10.1)
```

## Dataset

We evaluated the performance of the model using two publicly available and newly released datasets. The composition of the dataset is as follows, you can click on the **"Source"** to download them.

| *Dataset*        | *Nodes*      | *Edges*       | *Avg Degree*   |*Phisher* | *Source*  |
| ---------------- | ------------- | -------------- | -------------- |------- |---------- |
| MultiGraph       |  2,973,489    |  13,551,303    |  4.5574        | 1,165  |  XBlock     |
| B4E              |  597,258      |  11,678,901    |  19.5542       | 3,220  |  Google Drive   |
| SPN  |  496,740      |  1831,082      |  1.6730        | 5,619  |    Github       |

## Getting Started 
#### Step1 Create environment and install required packages for KGBERT4ETH.
#### Step2 Download the dataset.
#### Step3 Preprocess the dataset for pretraining and eval.
```sh
cd Data/gen_MultiGraoh_seq
python dataset1.py
 ...
python dataset11.py

cd gen_b4e_seq
python bedataset1.py
 ...
python bedataset6.py

cd Data/gen_dean_role
python deanrole1.py
python deanrole2.py

cd Data
python gen_corpus_bm25.py
```
#### Step4 Load the transaction knowledge graph and Pretrain the KGBERT4Eth
```sh
python pretrain.py
```
#### Step5 Finetune and Evaluation
```sh
python eval_dean_role.py
python eval_phish.py
```

| Parameter         | Description |
|------------------|-------------|
| `task`          | Specifies the task to execute, with options including `direct`, `finetune`, `linear`, and `aft_ft`, determining different training and evaluation strategies. |
| `dev_tsv`       | Defines the path to the dataset file, which contains the input data for training, fine-tuning, or evaluation. |
| `pretrained_path` | Specifies the path to the pretrained model directory, which stores checkpoint files used for initializing model weights before further training or evaluation. |
| `model_path`    | Provides the path to the trained model file, used either for evaluation or as a starting point for continued training in fine-tuning experiments. |
| `batch_size`    | Sets the number of samples processed in each training or evaluation batch, affecting memory usage and training efficiency. |
| `max_length`    | Determines the maximum number of tokens allowed in the BERT input sequence, truncating longer inputs and padding shorter ones. |
| `epochs`        | Defines the total number of training iterations over the dataset, influencing convergence and generalization performance. |
| `learning_rate` | Specifies the step size for updating model parameters during optimization, impacting the speed and stability of training. |
| `num_labels`    | Represents the number of output classes for classification tasks, determining the structure of the final prediction layer. |
| `lossfuc_weight` | Controls the weighting of different classes in the loss function, which can be used to handle class imbalance during training. |





## Main Results

Here only the F1-Score(Percentage) results are displayed; for other metrics, please refer to the paper.
### Phisher detection task: 
| *Model*              | *MultiGraph* | *B4E*     | *SPN*     |
| -------------------- | ------------ | --------- | --------- |
| *DeepWalk*          | 60.11        | 62.91     | 53.18     | 
| *Role2Vec*         | 56.71        | 68.53     | 57.01     | 
| *Trans2Vec*       | 68.50        | 36.62     | 52.11     | 
| *GCN*              | 43.44        | 62.96     | 48.97     | 
| *GAT*               | 41.12        | 61.32     | 60.07     | 
| *GSAGE*            | 33.48        | 52.03     | 52.82     | 
| *DiffPool*         | 60.30        | 53.08     | 51.63     | 
| *U2GNN*            | 59.65        | 60.97     | 54.60     | 
| *Graph2Vec*        | 55.14        | 58.26     | 57.76     | 
| *TSGN*             | 67.39        | 69.35     | 62.07     | 
| *GrabPhisher*      | 78.89        | 67.38     | 73.76     | 
| *GAE*              | 44.93        | 50.06     | 36.85     | 
| *GATE*             | 45.39        | 58.58     | 68.69     | 
| *BERT4ETH*         | 64.20        | 69.05     | 72.98     | 
| *ZipZap*           | 65.25        | 68.48     | 72.31     | 
| ***LMAE4Eth (Ours)*** | **87.60**    | **85.43** | **88.02** |

### Account identity de-anonymization task: 
| *Model*              | *Overall* | *Airdrop Hunter* | *ICO Wallets* | *Mev Bot* | *Synthetix* |
| -------------------- | --------- | ---------------- | ------------- | --------- | ----------- |
| *DeepWalk*          | 66.62     | 78.86            | 14.20         | 51.76     | 68.81       |
| *Role2Vec*          | 57.36     | 72.71            | 2.65          | 32.37     | 57.88       |
| *Trans2Vec*         | 74.91     | 85.49            | 23.57         | 62.50     | 78.75       |
| *GCN*               | 59.81     | 74.60            | 5.68          | 35.97     | 62.03       |
| *GAT*               | 64.09     | 77.68            | 8.90          | 46.35     | 65.29       |
| *GSAGE*             | 55.72     | 72.01            | 4.10          | 29.30     | 54.80       |
| *DiffPool*          | 72.34     | 84.13            | 19.23         | 59.03     | 76.00       |
| *U2GNN*             | 62.73     | 77.80            | 5.85          | 44.56     | 61.56       |
| *Graph2Vec*         | 66.70     | 80.31            | 12.42         | 49.47     | 69.97       |
| *TSGN*              | 79.38     | 89.74            | 36.10         | 67.74     | 79.44       |
| *GrabPhisher*       | 81.07     | 91.93            | 39.71         | 65.97     | 82.03       |
| *GAE*               | 64.26     | 77.65            | 8.40          | 45.94     | 66.50       |
| *GATE*              | 66.73     | 78.98            | 11.70         | 52.24     | 69.57       |
| *BERT4ETH*          | 76.13     | 87.49            | 27.87         | 63.60     | 78.56       |
| *ZipZap*            | 75.91     | 85.41            | 28.88         | 64.75     | 79.65       |
| ***LMAE4Eth (Ours)*** | **90.52** | **98.36**        | **65.66**     | **84.07** | **88.68** |
