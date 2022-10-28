# Minimizing the interference of pill coatings for near-IR detection of  substandard and falsified pharmaceuticals







A portable Near Infra Red (NIR) spectometer was ecplored in this study to probe lab formulated Isoniazid(IS) as well as Doxycycline(DE) samples housed in broad spectrum of coated capsules including capsules made from vegetable cellulose and gelatin. The capsules are of varying opacity which introduce variations similar to real life scenerios associated different capssule coatings by manufacturers.

Isoniazid and alpha crystalline cellulose were formulated in the lab with active pharmaceuticals ingredients (API)- Isoniazid content (w/w) used in the regression studies. We used this as hypothetical study as most Isoniazid exist in tablet forms rather than in capsules. A real life case study used Doxycyline as Doxycyline exist in capsules.

Binary mixture of Doxycyline hydrate (ALFA AESAR) and alpha-lactose (Sigma) were formulated in the lab with active pharmaceuticals ingredients (API)- Doxycyline hydrate- content used in the regression studies.

See the supplementary information of our work titled "Minimizing Pill Coatings interference in detection of Fake Pharmaceuticals through Data Pre-treatment" for the lab based formulations used in this work


#Import Libraries

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

