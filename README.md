
# Minimizing the interference of pill coatings for near-IR detection of  substandard and falsified pharmaceuticals


```
*Authors: Olatunde Awotunde, (University of Notre Dame) . Email: aawotund@nd.edu (Developed Chemometric Algorthms using python and      
     
              The  Unscrambler software, Integrated all the works in Github repository, compose the manuscript)
  
    : Jiaqi Lu, (University of Notre Dame) . Email: jlu22@nd.edu  (Generated NIR spectrometer spectra used through out this work)
    
    : Jin Cai, (University of Notre Dame) . Email: jcai@nd.edu (Generated NIR spectrometer spectra used through out this work)
    
    : Ornella Joseph , (University of Notre Dame) . Email: ojoseph2@nd.edu   (Generated XRF data of the multi-colored capsules )
    
    : Alyssa Wicks, (Sapfonte Precise Solutions) . Email: awicks@nd.edu  (Generated SEM images and data of the multi-colored capsules from sanning electron microscope-SEM)                   
    : Marya Lieberman, (University of Notre Dame) . Email: mlieberm@nd.edu (Advisor)*

```

A portable Near Infra Red (NIR) spectometer was ecplored in this study to probe lab formulated Isoniazid(IS) as well as Doxycycline(DE) samples housed in broad spectrum of coated capsules including capsules made from vegetable cellulose and gelatin. The capsules are of varying opacity which introduce variations similar to real life scenerios associated different capssule coatings by manufacturers.

Isoniazid and alpha crystalline cellulose were formulated in the lab with active pharmaceuticals ingredients (API)- Isoniazid content (w/w) used in the regression studies. We used this as hypothetical study as most Isoniazid exist in tablet forms rather than in capsules. A real life case study used Doxycyline as Doxycyline exist in capsules.

Binary mixture of Doxycyline hydrate (ALFA AESAR) and alpha-lactose (Sigma) were formulated in the lab with active pharmaceuticals ingredients (API)- Doxycyline hydrate- content used in the regression studies.

See the supplementary information of our work titled "Minimizing Pill Coatings interference in detection of Fake Pharmaceuticals through Data Pre-treatment" for the lab based formulations used in this work


## Import Libraries


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA as sk_pca
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
from pyopls import OPLS
from sklearn.model_selection import cross_val_predict, LeaveOneOut
from sklearn.metrics import r2_score, accuracy_score
```

```python
# Lab Formulated Isonized (IS) were housed in broad range of capsules and probed with NIR spectrometer for regression studies

# Visual inspection of line plot of the raw date from lab formulated Isoniazid in varying capsules when introduced to NIR spectrometer

spectra_1 = pd.read_csv(r'ISCE_data_mg.csv')
target = pd.read_csv(r'isce_conc.csv')

wv = np.arange(900,1700,3.52) #the wavelength range used for this study

spectra_2 = pd.DataFrame(spectra_1)
spectra_3 = spectra_2.values[0:,0:]
spectra_T_n = spectra_3.T

IS_100mg_RAW =spectra_T_n[:,0:300]
IS_200mg_RAW =spectra_T_n[:,301:524]
IS_300mg_RAW =spectra_T_n[:,525:776]
IS_500mg_RAW =spectra_T_n[:,777:1031]


fig, ax = plt.subplots(figsize=(8, 4.7))
loc = ["upper left"]
ax.plot(wv, IS_100mg_RAW, label='IS_100mg',color='r');
ax.plot(wv, IS_200mg_RAW, label='IS_200mg',color='b');
ax.plot(wv, IS_300mg_RAW, label='IS_300mg',color='y');
ax.plot(wv, IS_500mg_RAW, label='IS_500mg',color='g',ls=':',lw=0.5);
plt.xlabel('Wavelength (nm)')
plt.ylabel('Absorbance')
plt.title("Raw Data")


# Fix legend
hand, labl = ax.get_legend_handles_labels()
handout=[]
lablout=[]
for h,l in zip(hand,labl):
       if l not in lablout:
        lablout.append(l)
        handout.append(h)
        fig.legend(handout, lablout, bbox_to_anchor=[0.15, 0.9],loc="upper left");
```


<img width="426" alt="Screen Shot 2022-10-29 at 12 25 21 PM" src="https://user-images.githubusercontent.com/68889345/198842513-2a47a254-75aa-42dd-a750-7a3f11543eab.png">

## Predictive Models(support vector machine regression (SVM-R) & partial Least square regression(PLS-R)) for the raw dataset

```python
#Split the dataset (SNV) to train and test sets

X_n= pd.DataFrame(StandardScaler().fit_transform(spectra_T_n))

x_n = X_n.T
y_n = pd.DataFrame(target)
x_train_a,x_test_a,y_train_a,y_test_a = train_test_split(x_n,y_n,random_state=0,test_size=0.3)
```

```python
#the raw dataset to train and test sets
#support vector machine (SVM) regression 

#SVM_regression - can be optimised
#Train SVM model

regr_n = svm.SVR(kernel='poly',gamma = 0.02, C = 1)
regr_n.fit(x_train_a, y_train_a)

#Test the model

clf_1svr_a = regr_n.predict(x_test_a)

#plot the predicted against actual

plt.scatter(y_test_a,clf_1svr_a)

plt.xlabel("Actual in mg")
plt.ylabel("Predicted (mg)")
```
<img width="331" alt="Screen Shot 2022-10-29 at 12 30 18 PM" src="https://user-images.githubusercontent.com/68889345/198842662-6328c09f-ddde-41b8-8ff1-511e2f6fba81.png">

```python
#determine the correlation co-efficient (R squared) for SVM-R
q_squared_a1 = r2_score(y_test_a, clf_1svr_a) 
q_squared_a1
```
```
0.877922236689431
```

```python

#determine the root mean sqare error for SVM-R

mean_squared_error(y_test_a, clf_1svr_a,squared=False)
```
```
52.2850733315104
```

```python
#the dataset (OSC) to train and test sets
#OPLS of the raw data

spectra = pd.read_csv(r'ISCE_data_mg.csv')
target = pd.read_csv(r'isce_conc.csv')

opls = OPLS(39)
Z = opls.fit_transform(spectra, target)

pls = PLSRegression(1)

#OPLS for Raw Data
y_preda_raw = cross_val_predict(pls, spectra, target, cv=LeaveOneOut())
q_squared = r2_score(target, y_preda_raw)  
```
```python
#determinr the correlation co-efficient(R-squared) for raw spectra
q_squared = r2_score(target, y_preda_raw)
q_squared
```
```
0.6454037067588276
```

```python
#determine the root mean sqare error for raw spectra

mean_squared_error(target, y_preda_raw,squared=False)
```
```
89.8712729356915
```


## Data Pretreatment of the Raw Data

## Standard Normal Variate (SNV) folloewed by Savitzki-Golay(SG) Data Pretreatment tranformations of the raw spectra

```python
# Define Standard Normal Variate

def snv(x):
  
    # Define a new array and populate it with the corrected data  
    output_data = np.zeros_like(x)
    for i in range(x.shape[0]):
 
        # Apply correction
        output_data[i,:] = (x[i,:] - np.mean(x[i,:])) / np.std(x[i,:])
 
    return output_data
```
# Visual inspection of line plot of the SNV traeted data from lab formulated Isoniazid in capsules of varying colors and opacities when introduced to NIR spectrometer

# SNV transformation of the raw data

```python
data_isce = pd.read_csv(r'ISCE_data_mg.csv')
x_isce= data_isce.values[:,:]
spectra_snv = snv(x_isce)

spectra_snv_d = pd.DataFrame(spectra_snv)
spectra_snv_ = spectra_snv_d.values[0:,0:]
spectra_T = spectra_snv_.T

#Line plot 

IS_100mg_SNV =spectra_T[:,0:300]
IS_200mg_SNV =spectra_T[:,301:524]
IS_300mg_SNV =spectra_T[:,525:776]
IS_500mg_SNV =spectra_T[:,777:1031]

    
    
fig, ax = plt.subplots(figsize=(8, 4.7))
loc = ["upper left"]
ax.plot(wv, IS_100mg_SNV, label='IS_100mg',color='r');
ax.plot(wv, IS_200mg_SNV, label='IS_200mg',color='b');
ax.plot(wv, IS_300mg_SNV, label='IS_300mg',color='y');
ax.plot(wv, IS_500mg_SNV, label='IS_500mg',color='g',ls=':',lw=0.5);
plt.xlabel('Wavelength (nm)')
plt.ylabel('Absorbance')
plt.title("SNV Transformed Data for Lab Formulated Isoniazid")


# Fix legend
hand, labl = ax.get_legend_handles_labels()
handout=[]
lablout=[]
for h,l in zip(hand,labl):
       if l not in lablout:
        lablout.append(l)
        handout.append(h)
        fig.legend(handout, lablout, bbox_to_anchor=[0.15, 0.9],loc="upper left");
```


<img width="417" alt="Screen Shot 2022-10-29 at 12 39 48 PM" src="https://user-images.githubusercontent.com/68889345/198843092-e1f14de6-0a13-446c-b27f-9853b037a0e5.png">

```python
#Visual inspection of line plot of the SNV+SG traeted data from lab formulated Isoniazid in varying capsules when introduced to NIR spectrometer

#SG transformation of the SNV transformed data

Xsnv_sg_ = savgol_filter(spectra_snv, 21, polyorder = 2, deriv=2)

spectra_sg = pd.DataFrame(Xsnv_sg_)
spectra_snvsg = spectra_sg.values[0:,0:]
spectra_T_ = spectra_snvsg.T


#Line plot

IS_100mg_SNVSG =spectra_T_[:,0:300]
IS_200mg_SNVSG =spectra_T_[:,301:524]
IS_300mg_SNVSG =spectra_T_[:,525:776]
IS_500mg_SNVSG =spectra_T_[:,777:1031]


fig, ax = plt.subplots(figsize=(8, 4.7))
loc = ["upper left"]
ax.plot(wv, IS_100mg_SNVSG, label='IS_100mg',color='r');
ax.plot(wv, IS_200mg_SNVSG, label='IS_200mg',color='b');
ax.plot(wv, IS_300mg_SNVSG, label='IS_300mg',color='y');
ax.plot(wv, IS_500mg_SNVSG, label='IS_500mg',color='g',ls=':',lw=0.5);
plt.xlabel('Wavelength (nm)')
plt.ylabel('Absorbance')
plt.title("SNV_SG Transformed")


#Fix legend
hand, labl = ax.get_legend_handles_labels()
handout=[]
lablout=[]
for h,l in zip(hand,labl):
       if l not in lablout:
        lablout.append(l)
        handout.append(h)
        fig.legend(handout, lablout, bbox_to_anchor=[0.15, 0.9],loc="upper left");
```

<img width="432" alt="image" src="https://user-images.githubusercontent.com/68889345/198843245-8e513bc7-b087-4fa5-934e-099adce1f66e.png">

## Orthogonal Scattering Correction (0SC) followed by Savitzki-Golay (SG) Data Pretreatment tranformations of the raw spectra

```python
#Visual inspection of line plot of the Orthogonal Signal Correction(OSC) traeted data from lab formulated Isoniazid in capsules of varying colors and opacities when introduced to NIR spectrometer



spectra = pd.read_csv(r'ISCE_data_mg.csv')
target = pd.read_csv(r'isce_conc.csv')

opls = OPLS(39)
Z_osc = opls.fit_transform(spectra, target)


Z_T = Z_osc.T

IS_100mg_OSC =Z_T[:,0:300]
IS_200mg_OSC =Z_T[:,301:524]
IS_300mg_OSC =Z_T[:,525:776]
IS_500mg_OSC =Z_T[:,777:1031]


wv = np.arange(900,1700,3.52)


fig, ax = plt.subplots(figsize=(8, 4.7))
loc = ["upper left"]
ax.plot(wv, IS_100mg_OSC, label='IS_100mg',color='r');
ax.plot(wv, IS_200mg_OSC, label='IS_200mg',color='b');
ax.plot(wv, IS_300mg_OSC, label='IS_300mg',color='y');
ax.plot(wv, IS_500mg_OSC, label='IS_500mg',color='g',ls=':',lw=0.5);
plt.xlabel('Wavelength (nm)')
plt.ylabel('Absorbance')
plt.title("OSC Transformed Data")



#Fix legend
hand, labl = ax.get_legend_handles_labels()
handout=[]
lablout=[]
for h,l in zip(hand,labl):
       if l not in lablout:
        lablout.append(l)
        handout.append(h)
        fig.legend(handout, lablout, bbox_to_anchor=[0.15, 0.9],loc="upper left");
```
<img width="435" alt="image" src="https://user-images.githubusercontent.com/68889345/198843383-934e6f46-9e65-4c55-9562-b3fb6ec7a5f9.png">

```python
#Visual inspection of line plot of the OSC+SG traeted data from lab formulated Isoniazid in capsules of varying colors and opacities when introduced to NIR spectrometer

Xosc_sg = savgol_filter(Z_osc, 21, polyorder = 2, deriv=2)


Xosc_sg_T = Xosc_sg.T

IS_100mg_OSC_SG =Xosc_sg_T[:,0:300]
IS_200mg_OSC_SG =Xosc_sg_T[:,301:524]
IS_300mg_OSC_SG =Xosc_sg_T[:,525:776]
IS_500mg_OSC_SG =Xosc_sg_T[:,777:1031]


wv = np.arange(900,1700,3.52)

#plt.plot(wv,Xsnv_sg_T);


fig, ax = plt.subplots(figsize=(8, 4.7))
loc = ["upper left"]
ax.plot(wv, IS_100mg_OSC_SG, label='IS_100mg',color='r');
ax.plot(wv, IS_200mg_OSC_SG, label='IS_200mg',color='b');
ax.plot(wv, IS_300mg_OSC_SG, label='IS_300mg',color='y');
ax.plot(wv, IS_500mg_OSC_SG, label='IS_500mg',color='g',ls=':',lw=0.5);
plt.xlabel('Wavelength (nm)')
plt.ylabel('Absorbance')
plt.title("OSC-SG Transformed Data")



 #Fix legend
hand, labl = ax.get_legend_handles_labels()
handout=[]
lablout=[]
for h,l in zip(hand,labl):
       if l not in lablout:
        lablout.append(l)
        handout.append(h)
        fig.legend(handout, lablout, bbox_to_anchor=[0.15, 0.9],loc="upper left");
```
<img width="443" alt="image" src="https://user-images.githubusercontent.com/68889345/198843449-35907304-a1bd-46ac-b725-74ae33312327.png">


# Predictive Models (support vector machine regression (SVM-R) & partial Least square regression(PLS-R)) for the SNV dataset

```python

#Split the dataset (SNV) to train and test sets

X_snv_isce= pd.DataFrame(StandardScaler().fit_transform(spectra_T))

x_snv_isce = X_snv_isce.T
y_snv_isce = pd.DataFrame(target)
x_train_b,x_test_b,y_train_b,y_test_b = train_test_split(x_snv_isce,y_snv_isce,random_state=0,test_size=0.3)
```


```python
#the dataset (SNV) to train and test sets
#support vector machine (SVM) regression 

#SVM_regression - can be optimised
#Train SVM model

regr = svm.SVR(kernel='poly',gamma = 0.2, C = 1)
regr.fit(x_train_b, y_train_b)

#Test the model

clf_1svr_b = regr.predict(x_test_b)


#plot the predicted against actual

plt.scatter(y_test_b,clf_1svr_b)

plt.xlabel("Actual (mg)")
plt.ylabel("Predicted (mg)")

```
<img width="330" alt="image" src="https://user-images.githubusercontent.com/68889345/198843875-1dedd409-20b0-45b3-81b4-d257373e4995.png">

```python

#determine the correlation co-efficient (R squared) for SVM-R
q_squared_b = r2_score(y_test_b, clf_1svr_b) 
q_squared_b
```
```
0.9791339879666874
```
```python
#determine the root mean sqare error for SVM-R

mean_squared_error(y_test_b, clf_1svr_b,squared=False)
```
```
21.61619086590866
```

```python
from sklearn.metrics import r2_score

best_r2 = 0
best_ncmop = 0
for n_comp in range(1, 101):
    my_plsr = PLSRegression(n_components=n_comp, scale=True)
    my_plsr.fit(x_train_b, y_train_b)
    preds = my_plsr.predict(x_test_b)
    
    r2 = r2_score(preds, y_test_b)
    if r2 > best_r2:
        best_r2 = r2
        best_ncomp = n_comp

print(best_r2, best_ncomp)
```
```
0.951334809988406 7
```
```python

#the dataset (SNV) to train and test sets
#Partial Least square regression (PLS-R) 

pls_1 = PLSRegression(n_components=7)

#Train PLS-R model

pls_1.fit(x_train_b, y_train_b)

#test PLS-R model
Y_pred_b = pls_1.predict(x_test_b)

#plot the predicted against actual
plt.scatter(y_test_b,Y_pred_b)
plt.xlabel("Actual (mg)")
plt.ylabel("Predicted (mg)")
```

<img width="328" alt="image" src="https://user-images.githubusercontent.com/68889345/198844175-7f43f52b-2030-4f6a-af96-5f6ebc18e0fe.png">

```python
#determine the correlation co-efficient (R squared) 
q_squared_b = r2_score(y_test_b,Y_pred_b) 
q_squared_b
```
```
0.954393382712631
```
```python
#determine the root mean sqare error 

mean_squared_error(y_test_b, Y_pred_b,squared=False)
```
```
31.957542138044204
```

# Predictive Models(support vector machine (SVM) & partial Least square regression(PLS-R)) for the SNV+SG dataset

```python
#Split the dataset (SNV+SG) to train and test sets

X_sgsnv_isce= pd.DataFrame(StandardScaler().fit_transform(spectra_T_))

x_sgsnv_isce = X_sgsnv_isce.T
y_sgsnv_isce = target
x_train_c,x_test_c,y_train_c,y_test_c = train_test_split(x_sgsnv_isce,y_sgsnv_isce,random_state=0,test_size=0.3)
```

```python
#the dataset (SNV+SG) to train and test sets
#support vector machine (SVM) regression 

#SVM_regression - can be optimised
#Traun SVM model

regr_c = svm.SVR(kernel='poly',gamma = 0.2, C = 1)
regr_c.fit(x_train_c, y_train_c)

#Test the model

clf_1svr_c = regr_c.predict(x_test_c)

#plot the predicted against actual

plt.scatter(y_test_c,clf_1svr_c)

plt.xlabel("Actual (mg)")
plt.ylabel("Predicted (mg)")
```
<img width="329" alt="image" src="https://user-images.githubusercontent.com/68889345/198844432-a078f86e-dd25-4266-89b6-6a8d51f17cc5.png">

```python
#determine the correlation co-efficient (R squared) for SNV_SG  
q_squared = r2_score(y_test_c,clf_1svr_c) 
q_squared
```
```
0.9674542981081713
```
```
#determine the root mean sqare error  for SNV_SG 

mean_squared_error(y_test_c,clf_1svr_c,squared=False)
```
```
26.996425259794137
```

```python
#the dataset (SNV+SG) to train and test sets
#Partial Least square regression (PLS-R) 

pls_1_c = PLSRegression(n_components=7)

#Train PLS-R model

pls_1_c.fit(x_train_c, y_train_c)

#test PLS-R model
Y_pred_c = pls_1_c.predict(x_test_c)

#plot the predicted against actual
plt.scatter(y_test_c,Y_pred_c)
plt.xlabel("Actual")
plt.ylabel("Predicted")
```

<img width="329" alt="image" src="https://user-images.githubusercontent.com/68889345/198844608-70fcfc1c-bdbf-4fa9-b7be-df4ae0b0cd5e.png">

```python
#determine the correlation co-efficient (R squared) 
q_squared_c = r2_score(y_test_c, Y_pred_c) 
q_squared_c
```
0.9515379923609378

```python
#determine the root mean sqare error 

mean_squared_error(y_test_c, Y_pred_c,squared=False)
```
```
32.94277200321166
```

# OPLS (Orthogonal projections to latent structures or Orthogonal Partial Least-Squares - OPLS)

```python

#the dataset (OSC) to train and test sets
spectra = pd.read_csv(r'ISCE_data_mg.csv')
target = pd.read_csv(r'isce_conc.csv')



opls = OPLS(39)
Z = opls.fit_transform(spectra, target)

pls = PLSRegression(1)

#OPLS for Raw Data
y_preda_raw = cross_val_predict(pls, spectra, target, cv=LeaveOneOut())
q_squared = r2_score(target, y_preda_raw)  
 


#OPLS for Orthoginal Scatter Corrected Data
processed_y_opls = cross_val_predict(pls, Z, target, cv=LeaveOneOut())
processed_q_squared = r2_score(target, processed_y_opls)  

r2_X = opls.score(spectra) 


plt.figure(1)
pls.fit(Z, target)
df = pd.DataFrame(np.column_stack([pls.x_scores_, opls.T_ortho_[:, 0]]),
                  index=spectra.index, columns=['t', 't_ortho'])    


pos_df = df[0:300]
neg_df = df[301:524]
neg2_df = df[525:776]
neg3_df = df[777:1031]
plt.scatter(pos_df['t'], pos_df['t_ortho'], c='red', label='IS 100 mg')
plt.scatter(neg_df['t'], neg_df['t_ortho'], c='blue', label='IS 200 mg')
plt.scatter(neg2_df['t'], neg2_df['t_ortho'], c='yellow', label='IS 300 mg')
plt.scatter(neg3_df['t'], neg3_df['t_ortho'], c='green', label='IS 500 mg')
plt.title('PLS Scores')
plt.xlabel('Factor 1')
plt.ylabel('Factor 2')
plt.legend(loc='upper right')
plt.show()
```
<img width="332" alt="image" src="https://user-images.githubusercontent.com/68889345/198845172-c7db516d-3cfa-4dc8-88bb-5c5ef9bac31f.png">

```python
#determine the correlation co-efficient(R-squared) for raw spectra
q_squared = r2_score(target, y_preda_raw)
q_squared
```
```
0.6454037067588276
```

```python
#determine the correlation co-efficient(R-squared) for OSC transformed spectra
rocessed_q_squared = r2_score(target, processed_y_opls)
rocessed_q_squared 
```
```
0.9840622184442938
```
```python
#determine the root mean sqare error for raw spectra

mean_squared_error(target, y_preda_raw,squared=False)
```
```
89.8712729356915
```
```python
#determine the root mean sqare error for OSC transformed spectra

mean_squared_error(target, processed_y_opls,squared=False)
```
```
19.053191496099036
```

# DOXYCYCLINE LAB FORMULATION WITH CRYSTALLINE CELLULOSE
```
#Import the raw lab formulated Doxycycline
```
```python
#Import the raw lab formulated Doxycycline
data_dece = pd.read_csv(r'DE_Studies_.csv')
x_dece= data_dece.values[:,3:]

wv = np.arange(900,1700,3.52)
```

```python
#Visual inspection of line plot of the raw date from lab formulated Doxycycline in capsules of varying opacities and colors when introduced to NIR spectrometer


spectra_dece = pd.read_csv(r'DE_Studies_mg_.csv')
target_dece = pd.read_csv(r'DE_Studies_conc_.csv')

spectra_dece = pd.DataFrame(spectra_dece)
spectra_dece = spectra_dece.values[0:,0:]
spectra_dece_T = spectra_dece.T

DE_100mg_RAW =spectra_dece_T[:,0:492]
DE_200mg_RAW =spectra_dece_T[:,493:1000]
DE_300mg_RAW =spectra_dece_T[:,1001:1495]
DE_500mg_RAW =spectra_dece_T[:,1496:2002]


fig, ax = plt.subplots(figsize=(8, 4.7))
loc = ["upper left"]
ax.plot(wv, DE_100mg_RAW, label='DE_100mg',color='r');
ax.plot(wv, DE_200mg_RAW, label='DE_200mg',color='b');
ax.plot(wv, DE_300mg_RAW, label='DE_300mg',color='y');
ax.plot(wv, DE_500mg_RAW, label='DE_500mg',color='g',ls=':',lw=0.5);
plt.xlabel('Wavelength (nm)')
plt.ylabel('Absorbance')
plt.title("Raw Data of Lab Formulated Doxycycline")


#Fix legend
hand, labl = ax.get_legend_handles_labels()
handout=[]
lablout=[]
for h,l in zip(hand,labl):
       if l not in lablout:
        lablout.append(l)
        handout.append(h)
        fig.legend(handout, lablout, bbox_to_anchor=[0.15, 0.9],loc="upper left");
```

<img width="424" alt="image" src="https://user-images.githubusercontent.com/68889345/198846204-d887694c-6cab-4de4-9afd-11b77daaf82a.png">

loc = ["upper left"]

# Predictive Models(support vector machine regression (SVM-R) & partial Least square regression(PLS-R)) for the raw dataset

```python
#Split the dataset (SNV) to train and test sets

X_dece_n= pd.DataFrame(StandardScaler().fit_transform(spectra_dece_T))

x_dece_n = X_dece_n.T
y_dece_n = pd.DataFrame(target_dece)
x_train_dece_a,x_test_dece_a,y_train_dece_a,y_test_dece_a = train_test_split(x_dece_n,y_dece_n,random_state=0,test_size=0.3)
```

```python
#the raw dataset  to train and test sets
#support vector machine (SVM) regression 

#SVM_regression - can be optimised
#Train SVM model

regr_dece_n = svm.SVR(kernel='poly',gamma = 0.02, C = 1)
regr_dece_n.fit(x_train_dece_a, y_train_dece_a)

#Test the model

clf_1svr_dece_a = regr_dece_n.predict(x_test_dece_a)


#plot the predicted against actual

plt.scatter(y_test_dece_a,clf_1svr_dece_a)

plt.xlabel("Actual")
plt.ylabel("Predicted")
```

<img width="329" alt="image" src="https://user-images.githubusercontent.com/68889345/198846391-326d68c7-5318-45ce-984f-59bb0f434f69.png">


```python
#determine the correlation co-efficient(R-squared) for raw spectra
q_squared_234 = r2_score(y_test_dece_a,clf_1svr_dece_a)
q_squared_234
```
```
0.6182724959832242
```
```python
#determine the root mean sqare error 

mean_squared_error(y_test_dece_a,clf_1svr_dece_a,squared=False)
```
```
91.3835486876951
```

```python
#the dataset (OSC) to train and test sets
#OPLS for Raw Data
spectra_dece = pd.read_csv(r'DE_Studies_mg_.csv')
target_dece = pd.read_csv(r'DE_Studies_conc_.csv')



opls = OPLS(39)
Z_dece = opls.fit_transform(spectra_dece, target_dece)

pls_dece = PLSRegression(1)

#OPLS for Raw Data
y_preda_dece_raw = cross_val_predict(pls_dece, spectra_dece, target_dece, cv=LeaveOneOut())
q_squared = r2_score(target_dece, y_preda_dece_raw)  
```
```python
 #determinr the correlation co-efficient(R-squared) for raw spectra
q_squared_234 = r2_score(target_dece, y_preda_dece_raw)
q_squared_234
```
```
0.04444090366450848
```

```python
#determine the root mean square error for the raw spectra

mean_squared_error(target_dece, y_preda_dece_raw,squared=False)
```
```
144.70481315447762
```

```python
#Visual inspection of line plot of the SNV traeted data from lab formulated Doxycycline in capsules of varying opacities and colors when introduced to NIR spectrometer


data_dece_snv = pd.read_csv(r'DE_Studies_mg_.csv')
x_dece_snv= data_dece_snv.values[:,:]
spectra_dece_snv = snv(x_dece_snv)

spectra_dece_snv = pd.DataFrame(spectra_dece_snv)
spectra_dece_snv = spectra_dece_snv.values[0:,0:]
spectra_dece_snv_T = spectra_dece_snv.T


DE_100mg_SNV =spectra_dece_snv_T[:,0:492]
DE_200mg_SNV =spectra_dece_snv_T[:,493:1000]
DE_300mg_SNV =spectra_dece_snv_T[:,1001:1495]
DE_500mg_SNV =spectra_dece_snv_T[:,1496:2002]


    
    
fig, ax = plt.subplots(figsize=(8, 4.7))
loc = ["upper left"]
ax.plot(wv, DE_100mg_SNV, label='DE_100mg',color='r');
ax.plot(wv, DE_200mg_SNV, label='DE_200mg',color='b');
ax.plot(wv, DE_300mg_SNV, label='DE_300mg',color='y');
ax.plot(wv, DE_500mg_SNV, label='DE_500mg',color='g',ls=':',lw=0.5);
plt.xlabel('Wavelength (nm)')
plt.ylabel('Absorbance')
plt.title("SNV Transformed Data for lab formulate Deoxycycline")



#Fix legend
hand, labl = ax.get_legend_handles_labels()
handout=[]
lablout=[]
for h,l in zip(hand,labl):
       if l not in lablout:
        lablout.append(l)
        handout.append(h)
        fig.legend(handout, lablout, bbox_to_anchor=[0.15, 0.9],loc="upper left");
```
<img width="420" alt="image" src="https://user-images.githubusercontent.com/68889345/198846645-eba92af1-a88a-498a-9326-6ca6418e9b1b.png">

```python
#Visual inspection of line plot of the SNV+SG traeted data from lab formulated Deoxycycline in capsules of varying opacities and colors when introduced to NIR spectrometer

Xsnv_sg_dece = savgol_filter(spectra_dece_snv, 21, polyorder = 2, deriv=2)

spectra_snvsg_dece = pd.DataFrame(Xsnv_sg_dece)
spectra_snvsg_dece = spectra_snvsg_dece.values[0:,0:]
spectra_T_dece = spectra_snvsg_dece.T


DE_100mg_SNVSG =spectra_T_dece[:,0:492]
DE_200mg_SNVSG =spectra_T_dece[:,493:1000]
DE_300mg_SNVSG =spectra_T_dece[:,1001:1495]
DE_500mg_SNVSG =spectra_T_dece[:,1496:2002]


fig, ax = plt.subplots(figsize=(8, 4.7))
loc = ["upper left"]
ax.plot(wv, DE_100mg_SNVSG, label='DE_100mg',color='r');
ax.plot(wv, DE_200mg_SNVSG, label='DE_200mg',color='b');
ax.plot(wv, DE_300mg_SNVSG, label='DE_300mg',color='y');
ax.plot(wv, DE_500mg_SNVSG, label='DE_500mg',color='g',ls=':',lw=0.5);
plt.xlabel('Wavelength (nm)')
plt.ylabel('Absorbance')
plt.title("SNV Transformed Data of Lab Formulated Deoxycycline")



#Fix legend
hand, labl = ax.get_legend_handles_labels()
handout=[]
lablout=[]
for h,l in zip(hand,labl):
       if l not in lablout:
        lablout.append(l)
        handout.append(h)
        fig.legend(handout, lablout, bbox_to_anchor=[0.15, 0.9],loc="upper left");
```

<img width="438" alt="image" src="https://user-images.githubusercontent.com/68889345/198846709-a48b9ea2-0bf1-4d29-9954-43ec3db20d03.png">

# Predictive Models(support vector machine regression (SVM-R) & partial Least square regression(PLS-R)) for the SNV+SG dataset

```python
#Visual inspection of line plot of the Orthogonal Signal Correction(OSC) traeted data from lab formulated Isoniazid in varying capsules when introduced to NIR spectrometer



spectra_osc_dece = pd.read_csv(r'DE_Studies_mg_.csv')
target_osc_dece = pd.read_csv(r'DE_Studies_conc_.csv')

opls = OPLS(39)
Z_osc_dece = opls.fit_transform(spectra_osc_dece, target_osc_dece)


Z_T_osc_dece = Z_osc_dece.T

DE_100mg_OSC =Z_T_osc_dece[:,0:492]
DE_200mg_OSC =Z_T_osc_dece[:,493:1000]
DE_300mg_OSC =Z_T_osc_dece[:,1001:1495]
DE_500mg_OSC =Z_T_osc_dece[:,1496:2002]


wv = np.arange(900,1700,3.52)


fig, ax = plt.subplots(figsize=(8, 4.7))
loc = ["upper left"]
ax.plot(wv, DE_100mg_OSC, label='DE_100mg',color='r');
ax.plot(wv, DE_200mg_OSC, label='DE_200mg',color='b');
ax.plot(wv, DE_300mg_OSC, label='DE_300mg',color='y');
ax.plot(wv, DE_500mg_OSC, label='DE_500mg',color='g',ls=':',lw=0.5);
plt.xlabel('Wavelength (nm)')
plt.ylabel('Absorbance')
plt.title("OSC Transformed Data of Lab Formulated Deoxycycline")



#Fix legend
hand, labl = ax.get_legend_handles_labels()
handout=[]
lablout=[]
for h,l in zip(hand,labl):
       if l not in lablout:
        lablout.append(l)
        handout.append(h)
        fig.legend(handout, lablout, bbox_to_anchor=[0.15, 0.9],loc="upper left");
```
<img width="431" alt="image" src="https://user-images.githubusercontent.com/68889345/198846765-7ae23752-b130-4489-8ea4-4d54f82fc30e.png">

<img width="428" alt="image" src="https://user-images.githubusercontent.com/68889345/198846794-e92b7175-4bc8-4f45-9d0d-4eff4e4ffe5a.png">


```python
#Visual inspection of line plot of the OSC+SG traeted data from lab formulated Deoxycycline in varying capsules when introduced to NIR spectrometer

Xosc_sg_dece = savgol_filter(Z_osc_dece, 21, polyorder = 2, deriv=2)


Xosc_sg_T_dece = Xosc_sg_dece.T

DE_100mg_OSC_SG =Xosc_sg_T_dece[:,0:492]
DE_200mg_OSC_SG =Xosc_sg_T_dece[:,493:1000]
DE_300mg_OSC_SG =Xosc_sg_T_dece[:,1001:1495]
DE_500mg_OSC_SG =Xosc_sg_T_dece[:,1496:2002]




wv = np.arange(900,1700,3.52)



fig, ax = plt.subplots(figsize=(8, 4.7))
loc = ["upper left"]
ax.plot(wv, DE_100mg_OSC_SG, label='DE_100mg',color='r');
ax.plot(wv, DE_200mg_OSC_SG, label='DE_200mg',color='b');
ax.plot(wv, DE_300mg_OSC_SG, label='DE_300mg',color='y');
ax.plot(wv, DE_500mg_OSC_SG, label='DE_500mg',color='g',ls=':',lw=0.5);
plt.xlabel('Wavelength (nm)')
plt.ylabel('Absorbance')
plt.title("OSC-SG Transformed Data of Lab Formulated Deoxycycline")



#Fix legend
hand, labl = ax.get_legend_handles_labels()
handout=[]
lablout=[]
for h,l in zip(hand,labl):
       if l not in lablout:
        lablout.append(l)
        handout.append(h)
        fig.legend(handout, lablout, bbox_to_anchor=[0.15, 0.9],loc="upper left");
```

<img width="439" alt="image" src="https://user-images.githubusercontent.com/68889345/198846883-31ca2f38-7ff8-44e5-a894-41cb82a917cb.png">


# Model development and Evaluation for SNV, SNV_SG and OSC (OPLS) Data Pretreatment
# Model development and Evaluation for SNV Data Pretreatment

```python

#Split the dataset (SNV) to train and test sets

X_snv_dece= pd.DataFrame(StandardScaler().fit_transform(spectra_dece_snv_T))

x_snv_dece = X_snv_dece.T
y_snv_dece = target_dece
x_train_dece,x_test_dece,y_train_dece,y_test_dece = train_test_split(x_snv_dece,y_snv_dece,random_state=0,test_size=0.25)
```
```python
#the dataset (SNV) to train SVM model and test sets

#support vector machine regression model 

#SVM_regression - can be optimised
#Train SVM model

regr_dece = svm.SVR(kernel='poly',gamma = 0.90, C = 1)
regr_dece.fit(x_train_dece, y_train_dece)

#Test the model

clf_1_svr_dece = regr_dece.predict(x_test_dece)



#plot the predicted against actual

plt.scatter(y_test_dece,clf_1_svr_dece)

plt.xlabel("Actual")
plt.ylabel("Predicted")
```
<img width="325" alt="image" src="https://user-images.githubusercontent.com/68889345/198847027-c06a49ac-858d-4f61-8a4f-9b14435ff988.png">

```python
#determine the correlation co-efficient (R squared) 
q_squared_dece_a1 = r2_score(y_test_dece,clf_1_svr_dece) 
q_squared_dece_a1
```
```
0.9723081859569798
```
```python
#determine the mean square error 

mean_squared_error(y_test_dece, clf_1_svr_dece,squared=False)
```

```
24.81072814980493
```
```python
from sklearn.metrics import r2_score

best_r2 = 0
best_ncmop = 0
for n_comp in range(1, 101):
    my_plsr = PLSRegression(n_components=n_comp, scale=True)
    my_plsr.fit(x_train_dece, y_train_dece)
    preds = my_plsr.predict(x_test_dece)
    
    r2 = r2_score(preds, y_test_dece)
    if r2 > best_r2:
        best_r2 = r2
        best_ncomp = n_comp

print(best_r2, best_ncomp)
```
```
0.900178387882586 14
```
```python
#the dataset (SNV) to train and test sets
#Partial Least square regression model

pls_1_dece = PLSRegression(n_components=14)

#Train PLS-R model

pls_1_dece.fit(x_train_dece, y_train_dece)

#test PLS-R model
Y_pred_snv = pls_1_dece.predict(x_test_dece)

#plot the predicted against actual
plt.scatter(y_test_dece,Y_pred_snv)
plt.xlabel("Actual")
plt.ylabel("Predicted")
```

<img width="331" alt="image" src="https://user-images.githubusercontent.com/68889345/198847207-d74d04a4-9f5e-4743-9edb-a8f569b66ab4.png">


```python
#determine the correlation co-efficient (R squared) 
q_squared_a1 = r2_score(y_test_dece,Y_pred_snv) 
q_squared_a1
```
```
0.9066173352590577
```

```python
#determine the root mean square error 

mean_squared_error(y_test_dece,Y_pred_snv,squared=False)
```
```
45.56138717586071
```

# Model development and Evaluation for SNV_SG Data Pretreatment

```python
#Split the dataset (SNV_SG) to train and test sets

X_dece_snvsg= pd.DataFrame(StandardScaler().fit_transform(spectra_T_dece))

x_dece_snvsg = X_dece_snvsg.T
y_dece_snvsg = pd.DataFrame(target_dece)
x_train_dece_a,x_test_dece_a,y_train_dece_a,y_test_dece_a = train_test_split(x_dece_snvsg,y_dece_snvsg,random_state=0,test_size=0.3)
```
```python
#the dataset (SNV+SG) to train and test sets
#support vector machine (SVM) regression 

#SVM_regression - can be optimised
#TraIn SVM model

regr_dece_snvsg = svm.SVR(kernel='poly',gamma = 0.02, C = 1)
regr_dece_snvsg.fit(x_train_dece_a, y_train_dece_a)

#Test the model

clf_1svr_dece_snvsg_a = regr_dece_snvsg.predict(x_test_dece_a)


#plot the predicted against actual

plt.scatter(y_test_dece_a,clf_1svr_dece_snvsg_a)

plt.xlabel("Actual")

plt.ylabel("Predicted")
```
<img width="328" alt="image" src="https://user-images.githubusercontent.com/68889345/198847342-d801c768-7c3a-47fb-9bd0-c19b19a1dbcf.png">

```python
#determine the correlation co-efficient (R squared) 
q_squared_dece_snvsg_a1 = r2_score(y_test_dece_a,clf_1svr_dece_snvsg_a) 
q_squared_dece_snvsg_a1
```
```
0.9489649208009794
```

```python
#determine the root mean sqare error 

mean_squared_error(y_test_dece_a,clf_1svr_dece_snvsg_a,squared=False)
```
```
33.413786181474094
```

```python
from sklearn.metrics import r2_score

best_r2 = 0
best_ncmop = 0
for n_comp in range(1, 101):
    my_plsr = PLSRegression(n_components=n_comp, scale=True)
    my_plsr.fit(x_train_dece_a, y_train_dece_a)
    preds = my_plsr.predict(x_test_dece_a)
    
    r2 = r2_score(preds, y_test_dece_a)
    if r2 > best_r2:
        best_r2 = r2
        best_ncomp = n_comp

print(best_r2, best_ncomp)
```
```
0.9041896935681709 13
```
```python
#the dataset (SNV+SG) to train and test sets
#Partial Least square regression (PLS-R) 

pls_1_dece_snvsg_a = PLSRegression(n_components=13)

#Train PLS-R model

pls_1_dece_snvsg_a.fit(x_train_dece_a, y_train_dece_a)

#test PLS-R model
Y_pred_dece_snvsg_a = pls_1_dece_snvsg_a.predict(x_test_dece_a)

#plot the predicted against actual
plt.scatter(y_test_dece_a,Y_pred_dece_snvsg_a)
plt.xlabel("Actual")
plt.ylabel("Predicted")
```
<img width="330" alt="image" src="https://user-images.githubusercontent.com/68889345/198847489-2d1ade7f-ef07-4b1a-af34-64b1d640dcb7.png">

```python
#determine the correlation co-efficient (R squared) 
q_squared_a = r2_score(y_test_dece_a,Y_pred_dece_snvsg_a) 
q_squared_a
```
```
0.9110553627653938
```

```python
#determine the root mean sqare error 

mean_squared_error(y_test_dece_a,Y_pred_dece_snvsg_a,squared=False)
```
```
44.11143404333726
```
# Model development and Evaluation for OSC Data Pretreatment - OPLS

```python
#the dataset (OSC) to train and test sets
spectra_dece = pd.read_csv(r'DE_Studies_mg_.csv')
target_dece = pd.read_csv(r'DE_Studies_conc_.csv')



opls = OPLS(39)
Z_dece = opls.fit_transform(spectra_dece, target_dece)

pls_dece = PLSRegression(1)

#PLS for Raw Data
y_preda_dece_raw = cross_val_predict(pls_dece, spectra_dece, target_dece, cv=LeaveOneOut())
q_squared = r2_score(target_dece, y_preda_dece_raw)  



#OPLS for Orthoginal Scatter Corrected Data
processed_y_opls = cross_val_predict(pls_dece, Z_dece, target_dece, cv=LeaveOneOut())
processed_q_squared = r2_score(target_dece, processed_y_opls) 


plt.figure(1)
pls.fit(Z_dece, target_dece)
df_dece = pd.DataFrame(np.column_stack([pls.x_scores_, opls.T_ortho_[:, 0]]),
                       index=spectra_dece.index, columns=['t', 't_ortho'])    


pos_df_dece = df_dece[90:492]
neg_df_dece = df_dece[493:1000]
neg2_df_dece = df_dece[1001:1495]
neg3_df_dece = df_dece[1496:2002]



plt.scatter(pos_df_dece['t'], pos_df_dece['t_ortho'], c='red', label='DE 100 mg')
plt.scatter(neg_df_dece['t'], neg_df_dece['t_ortho'], c='blue', label='DE 200 mg')
plt.scatter(neg2_df_dece['t'], neg2_df_dece['t_ortho'], c='yellow', label='DE 300 mg')
plt.scatter(neg3_df_dece['t'], neg3_df_dece['t_ortho'], c='green', label='DE 500 mg')
plt.title('PLS Scores for Doxycycline ')
plt.xlabel('t_ortho')
plt.ylabel('t')
plt.legend(loc='upper right')
plt.show()
```
<img width="328" alt="image" src="https://user-images.githubusercontent.com/68889345/198847692-bdc64efe-02b0-481c-9c5e-c75cdc8a2891.png">

```python

 #determinr the correlation co-efficient(R-squared) for raw spectra
q_squared_234 = r2_score(target_dece, y_preda_dece_raw)
q_squared_234
```
```
0.04444090366450848
```

```python
#determinr the correlation co-efficient(R-squared) for OSC transformed spectra
rocessed_q_squared_234 = r2_score(target_dece, processed_y_opls)
rocessed_q_squared_234
```
```
0.9634250396988377
```
```python
#determine the root mean square error for the raw spectra

mean_squared_error(target_dece, y_preda_dece_raw,squared=False)
```
```
144.70481315447762
```
```python

#determine the root mean square error for the OSC transformed spectra

mean_squared_error(target_dece, processed_y_opls,squared=False)
```
```
28.310408304330522
```

