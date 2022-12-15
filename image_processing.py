import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import matplotlib.cm as cm
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, plot_confusion_matrix


dataset = '/home/iialab3/mVtec/carpet/preprocessing/test'

def imageProcessing(file_list, classTitle, filtersize):
    scoreList = []
    global dataset
    kernelsize = 9
    for filename in file_list:
        img = cv2.imread(dataset+classTitle+filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  
        kernel = np.ones((filtersize,filtersize), np.float32) / (filtersize*filtersize)
        dst = cv2.filter2D(img, -1, kernel)
       
        dst = 1 - dst
       
   
        kernel2 = np.ones((kernelsize, kernelsize), np.uint8)
        dst = cv2.dilate(dst, kernel2, iterations = 3)
        kernel3 = np.ones((kernelsize, kernelsize), np.uint8)
        dst = cv2.erode(dst, kernel2, iterations = 2)

        dst[dst<200] = 0
        dst = dst[10:246, 10:246]

        scoreList.append(dst.sum())

        # plt.subplot(131),plt.imshow(img,cmap=cm.gray),plt.title('Original')
        # plt.xticks([]), plt.yticks([])
        # plt.subplot(132),plt.imshow(dst,cmap=cm.gray),plt.title('Averaging')
        # plt.xticks([]), plt.yticks([])
        # plt.subplot(133),plt.imshow(dst,cmap=cm.gray),plt.title('Laplacian')
        # plt.xticks([]), plt.yticks([])
        # plt.show()

    return scoreList

img = cv2.imread('/home/iialab3/mVtec/carpet/preprocessing/test/normal')


for filtersize in [5]:
    normal_file_list = os.listdir(dataset+'/normal/')
    normal_file_list.sort()

    abnormal_file_list = os.listdir(dataset+'/abnormal/')
    abnormal_file_list.sort()


    AbnormalList = imageProcessing(abnormal_file_list, "/abnormal/",filtersize)

    NormalList = imageProcessing(normal_file_list, "/normal/",filtersize)

    x_data = NormalList+AbnormalList

    x_data = (x_data - min(x_data)) / (max(x_data) - min(x_data) + 0.000001)

    normal_true = [0 for i in range(len(normal_file_list))]
    abnormal_true = [1 for i in range(len(abnormal_file_list))]
    
    fpr, tpr, threshold = roc_curve(normal_true+abnormal_true, x_data)
    optimal_index = np.argmax(tpr - fpr) 
    optimal_threshold = threshold[optimal_index]
    print(optimal_threshold)
    sumTrue = normal_true + abnormal_true
    sumPred= [0 if i<optimal_threshold else 1 for i in x_data]

    cnf_matrix = confusion_matrix(sumTrue, sumPred)

    print("AUC= %0.3f"%( roc_auc_score(sumTrue, sumPred)))
    print(classification_report(sumTrue, sumPred,digits=3))
    print(cnf_matrix)

    
      