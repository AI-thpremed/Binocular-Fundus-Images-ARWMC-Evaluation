# Binocular-Fundus-Images-ARWMC-Evaluation

Prediction of ARWMC using GAP based feature fusion of binocular fundus images.


![image](/images/example1.png)

Based on PyTorch, run train_img_pair.py.

The model code is located in fusion_multi_model.py:

Build_MultiModel_szzyy_pair_onlyimg: Based on the ResNet series with Global Average Pooling (GAP) and Logistic Regression (LR).

Build_MultiModel_ShareBackbone_mlp: Based on the MLP feature fusion scheme.

Build_MultiModel_szzyy_pair_onlyimg_convnext: Based on the ConVNext approach.


