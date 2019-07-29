

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,f1_score
from imblearn.over_sampling import SMOTE, ADASYN
import pandas as pd
from vae_oversampler import VAEOversampler
```

    Using TensorFlow backend.
    


```python
X,y = make_classification(n_samples=100000,
                         n_features=10,
                         n_informative=10,
                         n_repeated=0,
                         n_redundant=0,
                         n_classes=2,
                         n_clusters_per_class=3,
                         weights = [.97,.03],
                         random_state=42)
```


```python
Xtrain,Xtest,ytrain,ytest = train_test_split(X,y)
```


```python
lr = LR()
lr.fit(Xtrain,ytrain)
print(classification_report(lr.predict(Xtest),ytest))
```

    C:\Users\dyanni\AppData\Local\Continuum\anaconda3\lib\site-packages\sklearn\linear_model\logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    

                  precision    recall  f1-score   support
    
               0       1.00      0.97      0.98     24965
               1       0.02      0.57      0.05        35
    
       micro avg       0.97      0.97      0.97     25000
       macro avg       0.51      0.77      0.51     25000
    weighted avg       1.00      0.97      0.98     25000
    
    


```python
vae_sampler = VAEOversampler(intermediate_dim=15,
                        latent_dim=1,
                        minority_class_id=1,
                        verbose=0,
                        epochs=100,
                        rescale=False)
Xres,yres = vae_sampler.fit_resample(Xtrain,ytrain,
                                validation_data = [Xtest,ytest])
#takes a long time
sampler_smote = SMOTE()
sampler_adasyn = ADASYN()
Xsmote,ysmote = sampler_smote.fit_resample(Xtrain,ytrain)
Xada,yada = sampler_adasyn.fit_resample(Xtrain,ytrain)
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    encoder_input (InputLayer)   (None, 10)                0         
    _________________________________________________________________
    encoder (Model)              [(None, 1), (None, 1), (N 197       
    _________________________________________________________________
    decoder (Model)              (None, 10)                190       
    =================================================================
    Total params: 387
    Trainable params: 387
    Non-trainable params: 0
    _________________________________________________________________
    


```python
dsets = [(Xres,yres),(Xsmote,ysmote),(Xada,yada),(Xtrain,ytrain)]
sampler_names = ['vae_oversampler', 'smote', 'adasyn','no sampling']
f1s=[]
for dset in dsets:
    lr.fit(*dset)
    f1s.append(f1_score(ytest,lr.predict(Xtest)))
(pd.DataFrame(pd.DataFrame(np.array(f1s)).T.values,
                   columns=sampler_names,
                  index=['f1_score']))
```

    C:\Users\dyanni\AppData\Local\Continuum\anaconda3\lib\site-packages\sklearn\linear_model\logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    C:\Users\dyanni\AppData\Local\Continuum\anaconda3\lib\site-packages\sklearn\linear_model\logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    C:\Users\dyanni\AppData\Local\Continuum\anaconda3\lib\site-packages\sklearn\linear_model\logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    C:\Users\dyanni\AppData\Local\Continuum\anaconda3\lib\site-packages\sklearn\linear_model\logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>vae_oversampler</th>
      <th>smote</th>
      <th>adasyn</th>
      <th>no sampling</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>f1_score</th>
      <td>0.141627</td>
      <td>0.139674</td>
      <td>0.12117</td>
      <td>0.046189</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
