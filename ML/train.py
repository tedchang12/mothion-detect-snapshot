from os import rename
from PIL import Image
import numpy as np
import cv2
import glob
import numpy as np
import pylab as pl
import random
from sklearn.decomposition import RandomizedPCA
from itertools import cycle
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from pprint import pprint
from sklearn.externals import joblib
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
markers = ['+', 'o', '^', 'v', '<', '>', 'D', 'h', 's']
def extractdata():
	#vectorlize()

	jtor = glob.glob("pos/*.png")
	asn = []
	for i in range(len(jtor)):
		img = Image.open(jtor[i])
		arr = np.array(img)
		flat = arr.ravel()
		#vector = np.matrix(flat)
		#print(vector.shape)
		#print(flat.shape)

		#print(vector(1,100))
		asn.append(flat)
	#print (asn)
	y = np.ones(len(jtor,), dtype=np.int)
	#print(y[2])
	np.save('testx',asn)
	np.save('testy',y)
	jtor = glob.glob("neg/*.png")
	tt = []
	count=0
	for i in range(len(jtor)):
		img = Image.open(jtor[i])
		arr = np.array(img)
		flat = arr.ravel()
		#vector = np.matrix(flat)
		#print(vector(1,100))
		if(flat.shape[0]!=14400):
			tt.append(flat)
			asn.append(flat)
			count+=1
			#print(flat.shape)
	tempy = np.zeros((count,), dtype=np.int)
	y = np.append(y,tempy)
	np.save('trainx',asn)
	np.save('trainy',y)
	np.save('falsex',tt)
	np.save('falsey',tempy)
	k = np.load('trainx.npy')
	print(k.shape)
	#y = np.matrix(y)
		#print(y)
def main():
	extractdata()
	#dim()
	SVM()
	#resiz()
	testTrue()
def rname():
	jtor = glob.glob("neg/*.jpg")
	for i in range(len(jtor)):
		rename(jtor[i],"neg/"+str(random.randint(100,1000000))+'.jpg')
def SVM():
	X = np.load('trainx.npy')
	y = np.load('trainy.npy')
	#print(y)
	pca = RandomizedPCA(n_components=2)
	#X_pca = pca.fit_transform(X)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
	svc_2 = SVC(kernel='rbf', C=10, gamma=0.0001).fit(X_train, y_train)
	train_score = svc_2.score(X_train, y_train) 
	test_scoret = svc_2.score(X_test,y_test)
	print(train_score)
	print (test_scoret)
	svc_params = {
    'C': np.logspace(-1, 2, 4),
    'gamma': np.logspace(-4, 0, 5),
	}
	joblib.dump(svc_2, 'theta.pkl')
	n_subsamples = 500
	#X_small_train, y_small_train = X_train[:n_subsamples], y_train[:n_subsamples]

	#gs_svc = GridSearchCV(SVC(), svc_params, cv=10, n_jobs=-1)

	#gs_svc.fit(X_small_train, y_small_train)

	#print gs_svc.best_params_, gs_svc.best_score_ # {'C': 10.0, 'gamma': 0.001} 0.982

def testTrue():
	svc_2 = joblib.load('theta.pkl')
	jtor = glob.glob("qq/*.png")
	asn = []
	for i in range(len(jtor)):
		img = Image.open(jtor[i])
		arr = np.array(img)
		flat = arr.ravel()
		#vector = np.matrix(flat)
		#print(vector(1,100))
		asn.append(flat)
	np.save('qq',asn)
	asn = np.load('qq.npy')
	print(asn.shape)
	#print (asn)
	xtrue =np.load('testx.npy')
	print(xtrue.shape)
	#print(xtrue.shape)
	ytrue = np.load('testy.npy')
	xf =np.load('falsex.npy')
	yf = np.load('falsey.npy')
	temp = svc_2.predict(asn)
	print(temp)

def dim():
	extractdata()
	X = np.load('trainx.npy')
	y = np.load('trainy.npy')
	#pca = RandomizedPCA(n_components=2)
	#X_pca = pca.fit_transform(X)

	#for i, c, m in zip(np.unique(y), cycle(colors), cycle(markers)):
    #	pl.scatter(X_pca[y == i, 0], X_pca[y == i, 1],c=c, marker=m, label=i, alpha=0.5)
    #
	pl.show()
def resiz():
	jtor = glob.glob("neg/*.png")
	for i in range(len(jtor)):
		temp = cv2.imread(jtor[i])
		temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
		re,temp = cv2.threshold(temp,80,255,cv2.THRESH_BINARY)
		t,d,l,r = getrec(temp)
		temp = temp[t:d-1,l:r-1]
		temp = cv2.resize(temp, (120,60), interpolation = cv2.INTER_CUBIC)
		temp = cv2.GaussianBlur(temp,(9,9),0)
		re,temp = cv2.threshold(temp,80,255,cv2.THRESH_BINARY)
		cv2.imwrite(jtor[i],temp)
def getrec(fgmask):
    top,down,left,right =0,0,0,0
    shape = fgmask.shape
    shape = (shape[1],shape[0])
    bol = False
    k=shape[1]
    for i in range(shape[1]):
        for j in range(shape[0]):
            if(fgmask[i][j]==255 ):
                top = i
                bol = True
                break
        if(bol):
            break
    bol = False
    for i in range(shape[1]):
        for j in range(shape[0]):
            if(fgmask[k-i-1][j]==255):
                down = k-i-1
                bol = True
                break
        if(bol):
            break
    bol = False
    for i in range(shape[0]):
        for j in range(shape[1]):
            if(fgmask[j][i]==255):
                left = i
                bol = True
                break
        if(bol):
            break
    bol = False
    for i in range(shape[0]):
        for j in range(shape[1]):
            if(fgmask[j][shape[0]-i-1]==255):
                right = shape[0]-i-1
                bol = True
                break
        if(bol):
            break
    return top,down,left,right
main()

#rname()