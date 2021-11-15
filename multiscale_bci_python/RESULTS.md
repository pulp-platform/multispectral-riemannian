
## Reference Results directly with original script
```
(msenv2) cavigelli@dev-cav:/scratch/cavigelli/repos/bci-xia/multiscale_bci_python$ python main_riemannian.py
Test data set
Subject 1: 0.9074733096085409
Subject 2: 0.5123674911660777
Subject 3: 0.8095238095238095
Subject 4: 0.7412280701754386
Subject 5: 0.6304347826086957
Subject 6: 0.5627906976744186
Subject 7: 0.8592057761732852
Subject 8: 0.8265682656826568
Subject 9: 0.8181818181818182
Average success rate: 0.7408637800883047
Time elapsed [s] 483.3574357032776
```
These results match the paper. 

## Results With Default/Original Settings

Generated with commit `71250bb65dffe281ccc03db4144e1d88b73ef5fe`

```
--- best model with shared hparams across patients ---
desc: mode: [900-linearSVM], exp: [num_samples: 1000 -- 
 svm -- c: 0.01; loss: hinge; tol: 1e-05], run: [0.01]
Patient 1: 90.75%
Patient 2: 52.65%
Patient 3: 82.05%
Patient 4: 73.25%
Patient 5: 66.67%
Patient 6: 53.02%
Patient 7: 83.03%
Patient 8: 82.66%
Patient 9: 81.44%
Avg. Acc.: 73.95%
Std. dev.: 12.94%
--- best model with per-patient tuned hparams and classifier ---
Patient 1: 90.75%
Patient 2: 56.54%
Patient 3: 82.05%
Patient 4: 75.88%
Patient 5: 68.12%
Patient 6: 53.02%
Patient 7: 83.03%
Patient 8: 84.50%
Patient 9: 81.44%
Avg. Acc.: 75.04%
Std. dev.: 12.32%
```

Conclusions: 
1. linear SVM performs really well; can reproduce paper results
2. varying hparams and classifiers gains 1.1% avg. accuracy 

## Results With Modified Time Windows
Changing the time windows from [2.5, 6] to [2.0, 5] in riemannian_model.py:205 and 80
```
--- best model with shared hparams across patients ---
desc: mode: [900-linearSVM], exp: [num_samples: 1000 -- 
 svm -- c: 0.01; loss: hinge; tol: 1e-05], run: [0.01]
Patient 1: 82.56%
Patient 2: 46.64%
Patient 3: 80.22%
Patient 4: 68.42%
Patient 5: 54.71%
Patient 6: 48.37%
Patient 7: 73.65%
Patient 8: 80.07%
Patient 9: 75.00%
Avg. Acc.: 67.74%
Std. dev.: 13.36%
--- best model with per-patient tuned hparams and classifier ---
Patient 1: 82.56%
Patient 2: 51.24%
Patient 3: 80.22%
Patient 4: 70.18%
Patient 5: 61.96%
Patient 6: 52.09%
Patient 7: 76.90%
Patient 8: 80.81%
Patient 9: 76.89%
Avg. Acc.: 70.32%
Std. dev.: 11.60%
```
Conclusions: 
1. The alternative time windows that have proven effective in other works seem to be not helpful (-4.7% accuracy from best model)

## Results With Modified Start Frequency
Changing the filter's start frequency from 4Hz ot 0.5Hz in filters.py:62 and added safety factor 1.7 on riemannian_multiscale.py:486
```
--- best model with shared hparams across patients ---
desc: mode: [902-scaledLinearSVM], exp: [num_samples: 1000 -- 
 svm -- c: 0.01; loss: hinge; tol: 1e-05], run: [0.01]
Patient 1: 88.97%
Patient 2: 52.30%
Patient 3: 75.82%
Patient 4: 66.67%
Patient 5: 51.81%
Patient 6: 49.30%
Patient 7: 82.31%
Patient 8: 78.23%
Patient 9: 79.17%
Avg. Acc.: 69.40%
Std. dev.: 14.05%
--- best model with per-patient tuned hparams and classifier ---
Patient 1: 88.97%
Patient 2: 52.30%
Patient 3: 79.12%
Patient 4: 68.86%
Patient 5: 54.35%
Patient 6: 53.49%
Patient 7: 82.31%
Patient 8: 79.34%
Patient 9: 80.68%
Avg. Acc.: 71.05%
Std. dev.: 13.41%
```

Conclusions: 
1. Lowering the lower frequency of the bandpasss filters from 4Hz to 0.5Hz found helpful in other works does not provide a positive impact (-4% accuracy)

## Best results
```
--- best model with per-patient tuned hparams and classifier ---
Patient 1: 90.75%
    -- mode: [900-linearSVM], exp: [num_samples: 1000 -- 
 svm -- c: 0.01; loss: hinge; tol: 1e-05], run: [0.01]
Patient 2: 57.60%
    -- mode: [011-actSigmoid], exp: [num_samples: 1000 -- 
solver: adam, lr: constant (init: 0.001), act: logistic], run: [(64, 8, 16)]
Patient 3: 81.32%
    -- mode: [902-scaledLinearSVM], exp: [num_samples: 1000 -- 
 svm -- c: 0.01; loss: hinge; tol: 1e-05], run: [0.01]
Patient 4: 74.12%
    -- mode: [014-oneHiddenLayer], exp: [num_samples: 1000 -- 
solver: adam, lr: constant (init: 0.001), act: relu], run: [(128,)]
Patient 5: 64.13%
    -- mode: [901-rbfSVM], exp: [num_samples: 1000 -- 
 svm -- c: 10.0; gamma: scale; kernel: rbf; deg: 3], run: [10.0]
Patient 6: 56.28%
    -- mode: [900-linearSVM], exp: [num_samples: 1000 -- 
 svm -- c: 0.01; loss: hinge; tol: 1e-05], run: [0.01]
Patient 7: 85.92%
    -- mode: [900-linearSVM], exp: [num_samples: 1000 -- 
 svm -- c: 0.01; loss: hinge; tol: 1e-05], run: [0.01]
Patient 8: 83.39%
    -- mode: [001-baseline], exp: [num_samples: 1000 -- 
solver: adam, lr: adaptive (init: 0.001), act: relu], run: [(128, 8, 128)]
Patient 9: 81.82%
    -- mode: [900-linearSVM], exp: [num_samples: 1000 -- 
 svm -- c: 0.01; loss: hinge; tol: 1e-05], run: [0.01]
Avg. Acc.: 75.04%
Std. dev.: 12.00%
```
Next best options: 
```

```



## Results For Varying Bandwidth

```
Patient 5:
```


## Results With Default/Original Settings but NO output feature quantization

```
--- best model with shared hparams across patients ---
desc: mode: [900-linearSVM], exp: [num_samples: 1000 -- 
 svm -- c: 0.01; loss: hinge; tol: 1e-05], run: [0.01]
Patient 1: 90.75%
Patient 2: 52.30%
Patient 3: 81.68%
Patient 4: 71.93%
Patient 5: 65.94%
Patient 6: 52.56%
Patient 7: 82.67%
Patient 8: 83.03%
Patient 9: 81.44%
Avg. Acc.: 73.59%
Std. dev.: 13.12%
--- best model with per-patient tuned hparams and classifier ---
Patient 1: 90.75%
Patient 2: 56.54%
Patient 3: 82.05%
Patient 4: 75.88%
Patient 5: 67.03%
Patient 6: 52.56%
Patient 7: 82.67%
Patient 8: 84.50%
Patient 9: 81.44%
Avg. Acc.: 74.82%
Std. dev.: 12.46%
```
Conclusions: 
1. Removing the quantization of the feature before classification has no relevant impact (-0.18% accuracy)

## improved export method
```
--- best model with shared hparams across patients ---
desc: mode: [900-linearSVM], exp: [num_samples: 1000 -- 
 svm -- c: 0.01; loss: hinge; tol: 1e-05], run: [0.01]
Patient 1: 90.75%
Patient 2: 51.24%
Patient 3: 80.95%
Patient 4: 73.68%
Patient 5: 63.04%
Patient 6: 56.28%
Patient 7: 85.92%
Patient 8: 82.29%
Patient 9: 81.82%
Avg. Acc.: 74.00%
Std. dev.: 13.14%
--- best model with per-patient tuned hparams and classifier ---
Patient 1: 90.75%
Patient 2: 57.60%
Patient 3: 81.32%
Patient 4: 74.12%
Patient 5: 64.13%
Patient 6: 56.28%
Patient 7: 85.92%
Patient 8: 83.39%
Patient 9: 81.82%
Avg. Acc.: 75.04%
Std. dev.: 12.00%
```

Next best option: 
```
Patient 5: 63.77% with 011-actSigmoid (instead of 64.13% with rbfSVM)
```


Open Questions: 
1. trade-off accuracy vs. compute vs. parameter count
2. Varying filter bandwidths
3. MLP + hinge loss as additional option?
4. data augmentation?
5. ??
6. 

Analyses for the paper: 
1. Include the above findings in text
2. expand table with optimal hparam + method variant
