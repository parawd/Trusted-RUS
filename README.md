# TMC
在本目录运行：
```sh
python -m pip install -r requirements.txt
python train.py eval --dataset har
python train.py eval --dataset mfeat
python train.py train --dataset har --epochs 5
```
模型及六组损失见 tmc.py。两个数据集都只使用训练样本估算标准化参数。HAR 按受试者划分，Multiple Features 按类别分层。
UCI Multiple Features（Robert Duin）：https://doi.org/10.24432/C5HC70
UCI HAR（Reyes-Ortiz 等）：https://doi.org/10.24432/C54S4K
数据集许可 CC BY 4.0。runs 中是本次重新训练的权重。
