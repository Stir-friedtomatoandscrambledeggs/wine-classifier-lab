# Model Card: Wine 三分类器

## 用途
根据红酒 13 项理化化验值（酒精含量、苹果酸、灰分、镁、总酚、黄酮类、色彩强度、脯氨酸等）预测葡萄品种（class_0 / class_1 / class_2）。

## 非用途
- 不用于真实葡萄酒质检/商业判定
- 不输出绝对结论，概率仅为模型置信度，不构成事实确定性
- 不支持图像、光谱等其他输入模态

## 训练数据
- UCI Wine 数据集，178 样本，13 数值特征，3 类
- 划分：分层抽样 train 60% / val 20% / test 20%，random_state=42
- 预处理：StandardScaler，仅在 train 上拟合，封装在同一 Pipeline

## 性能（测试集仅评一次）
- Accuracy: 0.972
- macro-F1: 0.971 / weighted-F1: 0.972
- macro-Precision: 0.978 / macro-Recall: 0.967
- 混淆矩阵：12/14/9 正确，1 个 class_2 被误判为 class_1

## 算法比较
| 模型 | 5 折 CV macro-F1 |
|---|---|
| LogisticRegression | 0.990 ± 0.020 |
| RandomForest(200) | 0.982 ± 0.023 |

选 LogReg（CV 均值更高、训练更快、模型更小）。

## 局限与风险
- 数据集小（178 样本），不代表真实生产分布
- 输入超出训练数值范围时会显示越界警告，但仍给出预测
- 三类别样本量不均，已用 macro 平均缓解
