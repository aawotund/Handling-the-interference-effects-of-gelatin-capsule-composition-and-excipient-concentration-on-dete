
# Minimizing the interference of pill coatings for near-IR detection of  substandard and falsified pharmaceuticals




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

