
# coding: utf-8

# In[1]:

####################################################################
#IMPORT STATEMENTS
####################################################################

import numpy as np
import csv

from sklearn import neighbors
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split

import numpy as np
from sklearn.preprocessing import Imputer
import scipy.io       # To read matlab files
from scipy import spatial
import pandas as pd   # To read data tables from csv files

####################################################################
#DEFINICIÓN DE FUNCIONES QUE SE USARÁN A LO LARGO DE LA PRÁCTICA 
####################################################################

#OBTENIDO DE: Notebook: The k-nearest neighbors (kNN) regression algorithm

# We start by defining a function that calculates the average square error
def square_error(s, s_est):
    # Squeeze is used to make sure that s and s_est have the appropriate dimensions.
    y = np.mean(np.power((np.squeeze(s) - np.squeeze(s_est)), 2))
    return y

def knn_regression(X1, S1, X2, k):
    """ Compute the k-NN regression estimate for the observations contained in
        the rows of X2, for the training set given by the rows in X1 and the
        components of S1. k is the number of neighbours of the k-NN algorithm
    """
    if X1.ndim == 1:
        X1 = np.asmatrix(X1).T
    if X2.ndim == 1:
        X2 = np.asmatrix(X2).T
    distances = spatial.distance.cdist(X1,X2,'euclidean')
    neighbors = np.argsort(distances, axis=0, kind='quicksort', order=None)
    closest = neighbors[range(k),:]
    
    est_values = np.zeros([X2.shape[0],1])
    for idx in range(X2.shape[0]):
        est_values[idx] = np.mean(S1[closest[:,idx]])
        
    return est_values

####################################################################
#OBTENCIÓN DE DATOS A PARTIR DEL FICHERO CSV
####################################################################

dataB = pd.read_csv('Input/data_train.csv', header=0)
datatest = pd.read_csv('Input/data_test.csv', header=0)

dataTrain = dataB.values[:,1:]

test = datatest.values[:,1:]
data = dataTrain.copy()

print ('data shape: ')
print data.shape
print ('test shape: ')
print test.shape
print ('\n')

####################################################################
#PREPROCESAMIENTO DE LOS DATOS PARA ELIMINAR LOS ERRORES
####################################################################

#Borrar filas con valor S_tr = -999.0 (dichas filas están compuestas en su mayoría por valores erróneos por lo que no aportan nada).
indexB = 0;
for i, x in enumerate(dataTrain):
    if dataTrain[i,-1] == -9.99000000e+02:       
        data = np.delete(data, i-indexB, axis=0)
        indexB += 1
        
print ('Dimension una vez borradas las filas:')
print data.shape

#######################################################################
#Como la tercera columna esta compuesta en su mayoria de valores -999 () la vamos a eliminar directamente.

data_A = data[:,0:2]
data_B = data[:,3:]
data = np.concatenate((data_A, data_B), axis=1)


Xtest_A = test[:,0:2]
Xtest_B = test[:,3:]
Xtest = np.concatenate((Xtest_A, Xtest_B), axis=1)
print ('Dimension una vez borradas las filas y columnas:')
print data.shape
print Xtest.shape
print ('\n')

#######################################################################
#Sustición de los errores restantes. Se han probado distintos métodos.
####################################################################

#OPCIÓN 1: Sustituir los errores por la media de la columna mediante la función 'Imputer' de scikit-learn
#OBTENIDO DE: PÁGINA DE SCIKIT LEARN: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html

#Imputation of missing values TRAIN

imp = Imputer(missing_values=-999, strategy='mean', axis=0)
imp.fit(data)
data = imp.transform(data) 

#Imputation of missing values TEST

imp = Imputer(missing_values=-999, strategy='mean', axis=0)
imp.fit(Xtest)
Xtest = imp.transform(Xtest) 


Xtrain = data[:,:-1]

print('Xtrain shape una vez sustituidos los errores::')
print Xtrain.shape

Ytrain = data[:,-1]
Ytrain = Ytrain[:,np.newaxis] #Para que Y twnga dos dimensiones como X

print('Ytrain shape una vez sustituidos los errores:')
print Ytrain.shape

print('Xtest shape  una vez sustituidos los errores::')
print Xtest.shape

print('\n')

#######################################################################

'''#OPCIÓN 2:Imputation of missing values with KNN

#Para cada columna vamos a sustituir los valores perdidos por el valor predecido por el algoritmo kNN para ello se usará un bucle que vaya columna por columna
#En primer lugar por cada columna debemos ver que filas tienen errores y cuales no
#El X_tr será el valor de todas las columnas (excepto la del bucle) en las filas donde la columna del bucle no tiene errores
#El S_tr será el valor de la columna del bucle (colum) en las correspondientes filas que no tienen error en colum (OJO A LA INVERSION)
#La T serán los valores de las columanas restantes a la estudiada en las filas en la que se ha producido un error en la columna estudiada
#Por último el valor a obtener (y_) será la predicción del valor erróneo de la columna estudiada

#El procedimiento se repetirá para cada una de las columnas

dataKNN = data.copy() #Copiamos los datos que teniamos en una nueva matriz por si la modificamos 

print ('\n\n')

print ('Entramos en el bucle')
for i, column in enumerate(dataKNN[1,:]):

    colum = i
    print ('Columna:')
    print colum
    
    ErrorList = [] #Creamos una lista que luego rellenaremos con los indices de las filas que contienen un error en la columna estudiada
    OkList = [] #Creamos una lista que luego convertiremos en matrix con las filas correctas, de donde sacaremos el X_tr y S_tr para el metodo kNN
    
    ErrorListTest = [] 
    OkListTest = []
    
    #Comprobamos fila por fila, para la columna estudiada, si tiene un error (-999.0) en dicha fila
    for j, row in enumerate(dataKNN):
        #Si lo tiene metemos el indice de la fila en la ErrorList
        if dataKNN[j,colum] == -9.99000000e+02: 
            ErrorList.append(j)
        #Si no copiamos todos los valores de la fila en la OkList
        else:
            OkList.append(row)

    #Hacemos lo mismo para el Xtest
    for n, row in enumerate(Xtest):
        #Si lo tiene metemos el indice de la fila en la ErrorList
        if colum<10: #Como data incluye la columna de S_tr también, tiene más columnas, y por tanto debemos coger solo las 10 primeras en el caso del Xtes
            if Xtest[n,colum] == -9.99000000e+02: 
                ErrorListTest.append(n)
            #Si no copiamos todos los valores de la fila en la OkList
            else:
                OkListTest.append(row)       
            
            
    dataOkList = np.array(OkList) #Convertimos una lista de arrays en matrix
    print('dataOKList shape:')
    print dataOkList.shape
       
    dataOkListTest = np.array(OkListTest) #Convertimos una lista de arrays en matrix
    print('dataOKListTest shape:')
    print dataOkListTest.shape

    print('\n')
        
    #Como en esta matriz (dataOKlist) habrá errores en otras columnas diferentes a la estudiada, hay que hacer una primera aproximacion de estos erroes por su media
    
    imp = Imputer(missing_values=-999, strategy='mean', axis=0)
    imp.fit(dataOkList)
    dataOkList_B = imp.transform(dataOkList) 
     
    #El X_tr y el S_tr cambian en función de la columna cuyos errores se están sustituyendo en la iteración del bucle    
    X_tr_A = dataOkList_B[:,0:colum]
    if (colum<10):
        X_tr_B = dataOkList_B[:,colum+1:]
        X_tr = np.concatenate((X_tr_A, X_tr_B), axis=1)
    else:
        X_tr = X_tr_A
    S_tr = dataOkList_B[:,colum]
    
    
    #Se repite el procedimiento anterior para Xtest
    if colum<10:
        imp_tst = Imputer(missing_values=-999, strategy='mean', axis=0) 
        imp_tst.fit(dataOkListTest)
        dataOkListTest_B = imp_tst.transform(dataOkListTest)
        
        X_tr_A_tst = dataOkListTest_B[:,0:colum]
        if (colum<9):
            X_tr_B_tst = dataOkListTest_B[:,colum+1:]
            X_tr_tst = np.concatenate((X_tr_A_tst, X_tr_B_tst), axis=1)
        else:
            X_tr_tst = X_tr_A_tst

        S_tr_tst = dataOkListTest_B[:,colum]
    
    
    #######VALIDACIÓN CRUZADA###########
    #Como se va a usar un procedimiento de knn hay que obtener mediante validación cruzada el mejor knn
   
    ### This fragment of code runs k-nn with M-fold cross validation

    # Parameters:
    M = 100       # Number of folds for M-cv
    k_max = 15  # Maximum value of the k-nn hyperparameter to explore

    ## M-CV
    # Obtain the indices for the different folds
    n_tr = X_tr.shape[0]
    permutation = np.random.permutation(n_tr)

    # Split the indices in M subsets with (almost) the same size. 
    set_indices = {i: [] for i in range(M)}
    i = 0
    for pos in range(n_tr):
        set_indices[i].append(permutation[pos])
        i = (i+1) % M

    # Obtain the validation errors
    MSE_val = np.zeros((1,k_max))
    for i in range(M):
        val_indices = set_indices[i]

        # Take out the val_indices from the set of indices.
        tr_indices = list(set(permutation) - set(val_indices))

        MSE_val_iter = [square_error(S_tr[val_indices], 
                                     knn_regression(X_tr[tr_indices, :], S_tr[tr_indices], 
                                                    X_tr[val_indices, :], k)) 
                        for k in range(1, k_max+1)]

        MSE_val = MSE_val + np.asarray(MSE_val_iter).T

    MSE_val = MSE_val/M

    # Select the best k based on the validation error
    k_best = np.argmin(MSE_val) + 1
    print('kbest:')
    print k_best

    #VALIDACION CRUZADA PARA LOS DATOS DE TEST
    ### This fragment of code runs k-nn with M-fold cross validation

    ## M-CV
    # Obtain the indices for the different folds
    n_tr_tst = X_tr_tst.shape[0]
    permutation_tst = np.random.permutation(n_tr_tst)

    # Split the indices in M subsets with (almost) the same size. 
    set_indices_tst = {i: [] for i in range(M)}
    i = 0
    for pos in range(n_tr_tst):
        set_indices_tst[i].append(permutation_tst[pos])
        i = (i+1) % M

    # Obtain the validation errors
    MSE_val_tst = np.zeros((1,k_max))
    for i in range(M):
        val_indices_tst = set_indices_tst[i]

        # Take out the val_indices from the set of indices.
        tr_indices_tst = list(set(permutation_tst) - set(val_indices_tst))

        MSE_val_iter_tst = [square_error(S_tr_tst[val_indices_tst], 
                                     knn_regression(X_tr_tst[tr_indices_tst, :], S_tr_tst[tr_indices_tst], 
                                                    X_tr_tst[val_indices_tst, :], k)) 
                        for k in range(1, k_max+1)]

        MSE_val_tst = MSE_val_tst + np.asarray(MSE_val_iter_tst).T

    MSE_val_tst = MSE_val_tst/M

    # Select the best k based on the validation error
    k_best_tst = np.argmin(MSE_val_tst) + 1
    print('kbest tst:')
    print k_best_tst
    ####################################
    
    #Ahora para los indices que eran erroneos y que guardamos en la error list, obtenemos su prediccion knn y sustituimos en la matriz original
    
    #En primer lugar obtenemos una matriz con todas las columnas excepto la estudiada y con las filas marcadas como erróneas
    data_sin_col_A = dataKNN[:,0:colum]
    if (colum < 10):
        data_sin_col_B = dataKNN[:,colum+1:]
        data_sin_col = np.concatenate((data_sin_col_A, data_sin_col_B), axis=1)
    else:
        data_sin_col = data_sin_col_A
          
    data_sin_col_A_tst = Xtest[:,0:colum]
    if (colum < 9):
        data_sin_col_B_tst = Xtest[:,colum+1:]
        data_sin_col_tst = np.concatenate((data_sin_col_A_tst, data_sin_col_B_tst), axis=1)
    else:
        data_sin_col_tst = data_sin_col_A_tst
        
    #AUNQUE ANTES REALIZAMOS UN PRIMER IMPUT DE LA MEDIA PARA DATAOKLIST, AHORA TENEMOS QUE DATA_SIN_COL, QUE LUEGO SE TRANSFORMARÁ EN T, TIENE ERRROES
    #LO MAS EFICIENTE SERIA ARREGLAR TODA LA LISTA ANTES DE DIVIDIRLA PERO YA QUE ESTA HECHO ASI VAMOS A HACER UNA NUEVA SUSTITUCIÓN
    #PARA EL DATA_SIN_COL_TST
    
    imp = Imputer(missing_values=-999, strategy='mean', axis=0)
    imp.fit(data_sin_col)
    data_sin_col = imp.transform(data_sin_col) 
    
    if colum<10:
        imp_tst = Imputer(missing_values=-999, strategy='mean', axis=0) 
        imp_tst.fit(data_sin_col_tst)
        data_sin_col_tst = imp_tst.transform(data_sin_col_tst)
   
    #La matriz T será aquella con los valores de las columnas en las filas donde la columna estudiada tiene un error
    if(len(ErrorList)>0): 
        T = []
        for j in ErrorList:
            T.append(data_sin_col[j,:])
        T_list = np.array(T) 
                   
        print('Matriz T shape:')
        print T_list.shape

        knn = neighbors.KNeighborsRegressor(k_best, weights='distance')
        y_ = knn.fit(X_tr, S_tr).predict(T_list)

        print('y_ shape:')
        print y_.shape
         
        
        #Aquí se sustituyen los valores obtenidos por KNN y guardados en la lista y_ por los valores erróneos de la columna
        n = 0
        for l in ErrorList:
            dataKNN[l,colum] = y_[n]
            n+=1
            
            
    if colum<10:
        if(len(ErrorListTest)>0):    
            T_tst = []
            for n in ErrorListTest:
                T_tst.append(data_sin_col_tst[n,:])
            T_list_tst = np.array(T_tst)

            print('Matriz T_tst shape:')
            print T_list_tst.shape

            knn_tst = neighbors.KNeighborsRegressor(k_best_tst, weights='distance')
            y_tst = knn_tst.fit(X_tr_tst, S_tr_tst).predict(T_list_tst)

            print('y_tst shape:')
            print y_tst.shape
            
            m = 0
            for l_tst in ErrorListTest:
                Xtest[l_tst,colum] = y_tst[m]
                m+=1     

    print('\n')
    
print('\nbucle terminado')'''

####################################################################
#NORMALIZACIÓN DE LOS DATOS
####################################################################

#OBTENIDO DE: Práctica Bayesian and Gaussian Process regression: 3. Bayesian Inference with real data. The stocks dataset.

# Data normalization
mean_x = np.mean(Xtrain,axis=0)
std_x = np.std(Xtrain,axis=0)
Xtrain = (Xtrain - mean_x) / std_x
Xtest = (Xtest - mean_x) / std_x


# Extend input data matrices with a column of 1's
col_1 = np.ones( (Xtrain.shape[0],1) )
Xtrain_e = np.concatenate( (col_1,Xtrain), axis = 1 )

col_1 = np.ones( (Xtest.shape[0],1))
Xtest_e = np.concatenate( (col_1,Xtest), axis = 1 )

#######################################################################
#PREDICCIÓN DE LA CONCENTRACIÓN DE BENZENO MEDIANTE DIFERENTES TÉCNICAS
#######################################################################

'''#LS_REGRESSION
#OBTENIDO DE: PÁGINA DE SCIKIT LEARN: https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.lstsq.html

#Hyperparameter selection
#w_LS, residuals, rank, s = np.linalg.lstsq(Xtrain,Ytrain)
w_LS, residuals, rank, s = np.linalg.lstsq(Xtrain_e,Ytrain)
sigma_0 = np.sqrt(np.mean(w_LS**2))
sigma_eps = np.sqrt(2 * np.mean((Ytrain - Xtrain_e.dot(w_LS))**2))

print sigma_0
print sigma_eps
print w_LS.shape

print ('xtraine:')
print Xtrain_e
print ('\nytrain:')
print Ytrain

Ytest=np.dot(Xtest_e,w_LS)'''

#######################################################################

#REGRESSION KNN
#OBTENIDO DE: Notebook: The k-nearest neighbors (kNN) regression algorithm

##########VALIDACIÓN CRUZADA PARA OBTENER LA MEJOR K#############
### This fragment of code runs k-nn with M-fold cross validation

# Parameters:
M = 200    # Number of folds for M-cv
k_max = 15  # Maximum value of the k-nn hyperparameter to explore

## M-CV
# Obtain the indices for the different folds
n_tr = Xtrain_e.shape[0]
permutation = np.random.permutation(n_tr)

# Split the indices in M subsets with (almost) the same size. 
set_indices = {i: [] for i in range(M)}
i = 0
for pos in range(n_tr):
    set_indices[i].append(permutation[pos])
    i = (i+1) % M

# Obtain the validation errors
MSE_val = np.zeros((1,k_max))
for i in range(M):
    val_indices = set_indices[i]

    # Take out the val_indices from the set of indices.
    tr_indices = list(set(permutation) - set(val_indices))

    MSE_val_iter = [square_error(Ytrain[val_indices], 
                                 knn_regression(Xtrain_e[tr_indices, :], Ytrain[tr_indices], 
                                                Xtrain_e[val_indices, :], k)) 
                    for k in range(1, k_max+1)]

    MSE_val = MSE_val + np.asarray(MSE_val_iter).T

MSE_val = MSE_val/M

# Select the best k based on the validation error
k_best = np.argmin(MSE_val) + 1
print('kbest KNN final:')
print k_best

#############

knn = neighbors.KNeighborsRegressor(k_best, weights='distance') #Uso de la librería neighbors, también se podría haber usado el método manual definido al principio
Ytest = knn.fit(Xtrain_e, Ytrain).predict(Xtest_e)

#######################################################################

'''#REGRESION BAYESIANA
#Hyperparameter selection (ls)
w_LS, residuals, rank, s = np.linalg.lstsq(Xtrain_e,Ytrain)
sigma_0 = np.sqrt(np.mean(w_LS**2))
sigma_eps = np.sqrt(2 * np.mean((Ytrain - Xtrain_e.dot(w_LS))**2))

#Prior distribution parameters
degree=10 #hemos quitado una columna

mean_w = np.zeros((degree+1,))
var_w = (sigma_0**2)* np.eye(degree+1)

#Compute posterior distribution parameters
Sigma_w = np.linalg.inv(np.dot(Xtrain_e.T,Xtrain_e)/(sigma_eps**2) + np.linalg.inv(var_w))
posterior_mean = Sigma_w.dot(Xtrain_e.T).dot(Ytrain)/(sigma_eps**2)
posterior_mean = np.array(posterior_mean).flatten()
print posterior_mean.shape


Ytest = np.dot(Xtest_e,posterior_mean)
'''

#######################################################################
#APROXIMACIÓN DEL ERROR MEDIANTE EL USO DE XTRAIN COMO XTEST
#######################################################################

'''
#LS
Ytest_comprobar=np.dot(Xtrain_e,w_LS)
MSE_tst = square_error(Ytrain, Ytest_comprobar) 

print ('Error cuadrático con datos de train')
print MSE_tst
print ('\n')
'''
#KNN-->No tiene sentido, va a ser siempre cero
Ytest_comprobar = knn.fit(Xtrain_e, Ytrain).predict(Xtrain_e)
MSE_tst = square_error(Ytrain, Ytest_comprobar) #Se usa el error cuadrático para ver la diferencia entre el ytest real y el calculado

print ('Error cuadrático con datos de train')
print MSE_tst
print ('\n')

'''
#BAYESSIANA
Ytest_comprobar = np.dot(Xtrain_e,posterior_mean)
MSE_tst = square_error(Ytrain, Ytest_comprobar)

print ('Error cuadrático con datos de train')
print MSE_tst
print ('\n')
'''

#######################################################################
#ADAPTACIÓN DEL YTEST OBTENIDO AL FORMATO CSV DE SALIDA
#######################################################################

Ytest = Ytest.squeeze() #quita los parentesis

print ('Ytest final shape:')
print Ytest.shape
print ('Predicción benzeno:')
print Ytest
print ('\n')

csv_file_object = csv.writer(open('outputfile.csv', 'wb')) 
csv_file_object.writerow(['id','Prediction'])
for index, y_aux in enumerate(Ytest):      # Run through each row in the csv file
     csv_file_object.writerow([index,y_aux])    
        
print('PROGRAMA FINALIZADO')


# In[ ]:



