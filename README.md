In [7]:
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline 
import seaborn as sns
In [8]:
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
In [9]:
digits = load_digits()
iris = datasets.load_iris()
In [10]:
iris.feature_names
Out[10]:
['sepal length (cm)',
 'sepal width (cm)',
 'petal length (cm)',
 'petal width (cm)']
In [11]:
digits.data.shape
Out[11]:
(1797, 64)
In [12]:
digits.data[10]
Out[12]:
array([ 0.,  0.,  1.,  9., 15., 11.,  0.,  0.,  0.,  0., 11., 16.,  8.,
       14.,  6.,  0.,  0.,  2., 16., 10.,  0.,  9.,  9.,  0.,  0.,  1.,
       16.,  4.,  0.,  8.,  8.,  0.,  0.,  4., 16.,  4.,  0.,  8.,  8.,
        0.,  0.,  1., 16.,  5.,  1., 11.,  3.,  0.,  0.,  0., 12., 12.,
       10., 10.,  0.,  0.,  0.,  0.,  1., 10., 13.,  3.,  0.,  0.])
In [13]:
digits.target_names
Out[13]:
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
In [9]:
digits.target.shape
Out[9]:
(1797,)
In [40]:
digits.target[10]
Out[40]:
0
In [14]:
import matplotlib.pyplot as plt 
plt.gray() 
plt.matshow(digits.images[10])
Out[14]:
<matplotlib.image.AxesImage at 0x13b0ec30>
<Figure size 432x288 with 0 Axes>

In [19]:
X = digits.data
y = digits.target
# Membagi data training dan testing(80:20)
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size= 0.4, random_state=1)
In [20]:
x_train.shape
Out[20]:
(1078, 64)
In [21]:
x_test.shape
Out[21]:
(719, 64)
In [22]:
y_train.shape
Out[22]:
(1078,)
In [23]:
y_test.shape
Out[23]:
(719,)
In [24]:
clf = MLPClassifier(hidden_layer_sizes=(20,20,20),  max_iter=250, alpha=0.0001,activation='logistic',
                     solver='adam', verbose=10,  random_state=21,tol=0.000000001)
In [25]:
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
Iteration 1, loss = 2.36734712
Iteration 2, loss = 2.34817172
Iteration 3, loss = 2.33271425
Iteration 4, loss = 2.32050890
Iteration 5, loss = 2.31091769
Iteration 6, loss = 2.30406517
Iteration 7, loss = 2.29836173
Iteration 8, loss = 2.29344395
Iteration 9, loss = 2.28980259
Iteration 10, loss = 2.28627223
Iteration 11, loss = 2.28263125
Iteration 12, loss = 2.27876695
Iteration 13, loss = 2.27498134
Iteration 14, loss = 2.27064775
Iteration 15, loss = 2.26576479
Iteration 16, loss = 2.26036642
Iteration 17, loss = 2.25430620
Iteration 18, loss = 2.24754610
Iteration 19, loss = 2.23979390
Iteration 20, loss = 2.23100170
Iteration 21, loss = 2.22132593
Iteration 22, loss = 2.21032674
Iteration 23, loss = 2.19838079
Iteration 24, loss = 2.18464374
Iteration 25, loss = 2.16941498
Iteration 26, loss = 2.15348046
Iteration 27, loss = 2.13547414
Iteration 28, loss = 2.11577265
Iteration 29, loss = 2.09502834
Iteration 30, loss = 2.07329350
Iteration 31, loss = 2.05035541
Iteration 32, loss = 2.02699971
Iteration 33, loss = 2.00318848
Iteration 34, loss = 1.97891969
Iteration 35, loss = 1.95499812
Iteration 36, loss = 1.93096305
Iteration 37, loss = 1.90739017
Iteration 38, loss = 1.88393448
Iteration 39, loss = 1.86142610
Iteration 40, loss = 1.83907261
Iteration 41, loss = 1.81747214
Iteration 42, loss = 1.79764459
Iteration 43, loss = 1.77709483
Iteration 44, loss = 1.75708265
Iteration 45, loss = 1.74017474
Iteration 46, loss = 1.72084895
Iteration 47, loss = 1.70378977
Iteration 48, loss = 1.68622787
Iteration 49, loss = 1.66882117
Iteration 50, loss = 1.65212882
Iteration 51, loss = 1.63550412
Iteration 52, loss = 1.61818743
Iteration 53, loss = 1.60208315
Iteration 54, loss = 1.58485072
Iteration 55, loss = 1.56861866
Iteration 56, loss = 1.55132498
Iteration 57, loss = 1.53499339
Iteration 58, loss = 1.51834979
Iteration 59, loss = 1.50096532
Iteration 60, loss = 1.48481352
Iteration 61, loss = 1.46913654
Iteration 62, loss = 1.45199801
Iteration 63, loss = 1.43507232
Iteration 64, loss = 1.41923035
Iteration 65, loss = 1.40349466
Iteration 66, loss = 1.38784639
Iteration 67, loss = 1.37288427
Iteration 68, loss = 1.35732686
Iteration 69, loss = 1.34373374
Iteration 70, loss = 1.32852261
Iteration 71, loss = 1.31507641
Iteration 72, loss = 1.30132733
Iteration 73, loss = 1.28788633
Iteration 74, loss = 1.27450751
Iteration 75, loss = 1.26202316
Iteration 76, loss = 1.25014068
Iteration 77, loss = 1.23813579
Iteration 78, loss = 1.22638735
Iteration 79, loss = 1.21420797
Iteration 80, loss = 1.20268819
Iteration 81, loss = 1.19184658
Iteration 82, loss = 1.18136723
Iteration 83, loss = 1.16944204
Iteration 84, loss = 1.15826283
Iteration 85, loss = 1.14766899
Iteration 86, loss = 1.13677378
Iteration 87, loss = 1.12601891
Iteration 88, loss = 1.11594658
Iteration 89, loss = 1.10597157
Iteration 90, loss = 1.09673901
Iteration 91, loss = 1.08659875
Iteration 92, loss = 1.07733869
Iteration 93, loss = 1.06820385
Iteration 94, loss = 1.05911235
Iteration 95, loss = 1.05001466
Iteration 96, loss = 1.04095850
Iteration 97, loss = 1.03243896
Iteration 98, loss = 1.02371691
Iteration 99, loss = 1.01491994
Iteration 100, loss = 1.00641962
Iteration 101, loss = 0.99753641
Iteration 102, loss = 0.98937448
Iteration 103, loss = 0.98082259
Iteration 104, loss = 0.97234001
Iteration 105, loss = 0.96452374
Iteration 106, loss = 0.95619655
Iteration 107, loss = 0.94749471
Iteration 108, loss = 0.93946981
Iteration 109, loss = 0.93171869
Iteration 110, loss = 0.92352460
Iteration 111, loss = 0.91568614
Iteration 112, loss = 0.90781969
Iteration 113, loss = 0.90045647
Iteration 114, loss = 0.89265188
Iteration 115, loss = 0.88534228
Iteration 116, loss = 0.87764479
Iteration 117, loss = 0.86973801
Iteration 118, loss = 0.86265081
Iteration 119, loss = 0.85541627
Iteration 120, loss = 0.84780316
Iteration 121, loss = 0.84079732
Iteration 122, loss = 0.83369450
Iteration 123, loss = 0.82653739
Iteration 124, loss = 0.81955655
Iteration 125, loss = 0.81255512
Iteration 126, loss = 0.80592202
Iteration 127, loss = 0.79982742
Iteration 128, loss = 0.79288619
Iteration 129, loss = 0.78615932
Iteration 130, loss = 0.77947845
Iteration 131, loss = 0.77341180
Iteration 132, loss = 0.76703008
Iteration 133, loss = 0.76040541
Iteration 134, loss = 0.75435339
Iteration 135, loss = 0.74853416
Iteration 136, loss = 0.74168078
Iteration 137, loss = 0.73612334
Iteration 138, loss = 0.73041563
Iteration 139, loss = 0.72421019
Iteration 140, loss = 0.71827922
Iteration 141, loss = 0.71248624
Iteration 142, loss = 0.70676180
Iteration 143, loss = 0.70107109
Iteration 144, loss = 0.69519668
Iteration 145, loss = 0.68962563
Iteration 146, loss = 0.68441358
Iteration 147, loss = 0.67900416
Iteration 148, loss = 0.67331430
Iteration 149, loss = 0.66813341
Iteration 150, loss = 0.66302262
Iteration 151, loss = 0.65750735
Iteration 152, loss = 0.65225149
Iteration 153, loss = 0.64750873
Iteration 154, loss = 0.64192460
Iteration 155, loss = 0.63718734
Iteration 156, loss = 0.63226594
Iteration 157, loss = 0.62746514
Iteration 158, loss = 0.62255851
Iteration 159, loss = 0.61780144
Iteration 160, loss = 0.61352194
Iteration 161, loss = 0.60845183
Iteration 162, loss = 0.60431094
Iteration 163, loss = 0.59940481
Iteration 164, loss = 0.59469051
Iteration 165, loss = 0.59025338
Iteration 166, loss = 0.58631529
Iteration 167, loss = 0.58136250
Iteration 168, loss = 0.57690825
Iteration 169, loss = 0.57259851
Iteration 170, loss = 0.56861541
Iteration 171, loss = 0.56394426
Iteration 172, loss = 0.55980897
Iteration 173, loss = 0.55584012
Iteration 174, loss = 0.55158150
Iteration 175, loss = 0.54718134
Iteration 176, loss = 0.54354159
Iteration 177, loss = 0.53918404
Iteration 178, loss = 0.53542071
Iteration 179, loss = 0.53135764
Iteration 180, loss = 0.52770663
Iteration 181, loss = 0.52363072
Iteration 182, loss = 0.51969984
Iteration 183, loss = 0.51568783
Iteration 184, loss = 0.51173440
Iteration 185, loss = 0.50799487
Iteration 186, loss = 0.50397132
Iteration 187, loss = 0.50034925
Iteration 188, loss = 0.49633166
Iteration 189, loss = 0.49245908
Iteration 190, loss = 0.48883655
Iteration 191, loss = 0.48488619
Iteration 192, loss = 0.48084663
Iteration 193, loss = 0.47706075
Iteration 194, loss = 0.47290694
Iteration 195, loss = 0.46894005
Iteration 196, loss = 0.46499053
Iteration 197, loss = 0.46067255
Iteration 198, loss = 0.45653702
Iteration 199, loss = 0.45251020
Iteration 200, loss = 0.44817936
Iteration 201, loss = 0.44382334
Iteration 202, loss = 0.43950179
Iteration 203, loss = 0.43482287
Iteration 204, loss = 0.43054633
Iteration 205, loss = 0.42622685
Iteration 206, loss = 0.42118999
Iteration 207, loss = 0.41681970
Iteration 208, loss = 0.41196636
Iteration 209, loss = 0.40735206
Iteration 210, loss = 0.40317218
Iteration 211, loss = 0.39779262
Iteration 212, loss = 0.39271876
Iteration 213, loss = 0.38829396
Iteration 214, loss = 0.38306914
Iteration 215, loss = 0.37797086
Iteration 216, loss = 0.37294365
Iteration 217, loss = 0.36838312
Iteration 218, loss = 0.36366202
Iteration 219, loss = 0.35879714
Iteration 220, loss = 0.35429422
Iteration 221, loss = 0.34962097
Iteration 222, loss = 0.34579051
Iteration 223, loss = 0.34089824
Iteration 224, loss = 0.33653801
Iteration 225, loss = 0.33240085
Iteration 226, loss = 0.32832740
Iteration 227, loss = 0.32454160
Iteration 228, loss = 0.32060355
Iteration 229, loss = 0.31650436
Iteration 230, loss = 0.31297660
Iteration 231, loss = 0.30924760
Iteration 232, loss = 0.30546965
Iteration 233, loss = 0.30226395
Iteration 234, loss = 0.29889334
Iteration 235, loss = 0.29520362
Iteration 236, loss = 0.29166868
Iteration 237, loss = 0.28842900
Iteration 238, loss = 0.28511433
Iteration 239, loss = 0.28210240
Iteration 240, loss = 0.27838691
Iteration 241, loss = 0.27538281
Iteration 242, loss = 0.27243601
Iteration 243, loss = 0.26939424
Iteration 244, loss = 0.26625555
Iteration 245, loss = 0.26273998
Iteration 246, loss = 0.26025299
Iteration 247, loss = 0.25729051
Iteration 248, loss = 0.25442468
Iteration 249, loss = 0.25118925
Iteration 250, loss = 0.24868404
c:\users\dapodik\appdata\local\programs\python\python37-32\lib\site-packages\sklearn\neural_network\multilayer_perceptron.py:562: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (250) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
In [26]:
loss_values = clf.loss_curve_
In [27]:
plt.plot(loss_values)
Out[27]:
[<matplotlib.lines.Line2D at 0x13c25370>]

In [ ]:
