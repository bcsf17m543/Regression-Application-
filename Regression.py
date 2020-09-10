import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math as mt
trn_data = pd.read_csv('RegressoinTraining.csv')
trn_data2 = pd.read_csv('RegressoinTesting.csv')
trn_data.head()
x = np.array(trn_data['sales'])
y = np.array(trn_data['profit'])
x2 = np.array(trn_data2['sales'])
y2 = np.array(trn_data2['profit'])

# plt.show()
S1 = x.size  # m Linear Regression--------------------
S2 = np.sum(x)
S3 = np.sum(x*x)
S4 = np.sum(y)
S5 = np.sum(y*x)
A = np.array([[S1, S2], [S2, S3]])
B = np.array([[S4, ], [S5, ]])
A_ = np.linalg.inv(A)
C = np.dot(A_, B)
arr = np.ones(x2.size, dtype=int)
tst_x = np.array((arr, x2), order='F').T
#Ynew = np.array(C[0][0] + C[1][0]*x2)
tst_y = np.dot(tst_x, C)

MSE1 = np.square(np.subtract(tst_y, y2)).mean()
plt.plot(x2, y2)  # actual graph
# # plt.plot(x2, Ynew)  # line of regression predicted values on my own logic
plt.plot(x2, tst_y)  # Line of regression--------------------
# Quadratic regression---------------------------------------
X_cube = np.sum(x*x*x)
X_quad = np.sum(x*x*x*x)
Y_Xsq = np.sum(y*x*x)
A_Quad = np.array([[S1, S2, S3], [S2, S3, X_cube], [S3, X_cube, X_quad]])
B_Quad = np.array([[S4, ], [S5, ], [Y_Xsq]])
A_Quad_ = np.linalg.inv(A_Quad)
C_Quad = np.dot(A_Quad_, B_Quad)
tst_y_quad = np.array(C_Quad[0][0] + C_Quad[1][0]*x2 + C_Quad[2][0]*x2*x2)
MSE2 = np.square(np.subtract(tst_y_quad, y2)).mean()
# Line of Regression (Quadratic Regression)-----------------
plt.plot(x2, tst_y_quad)
# plt.show()
X_5 = np.sum(x*x*x*x*x)  # Cubic Regression------------------
X_6 = np.sum(x*x*x*x*x)
Y_X3 = np.sum(y*x*x*x)
A_Cube = np.array([[S1, S2, S3, X_cube], [S2, S3, X_cube, X_quad], [
                  S3, X_cube, X_quad, X_5], [X_cube, X_quad, X_5, X_6]])
B_Cube = np.array([[S4, ], [S5, ], [Y_Xsq], [Y_X3, ]])
A_Cube_ = np.linalg.inv(A_Cube)
C_Cube = np.dot(A_Cube_, B_Cube)
tst_y_cube = np.array(C_Cube[0][0] + C_Cube[1][0]
                      * x2 + C_Cube[2][0]*x2*x2+C_Cube[3][0]*x2*x2*x2)
MSE3 = np.square(np.subtract(tst_y_cube, y2)).mean()
# -----Line of Regression (Cubic Regression)-----------
plt.plot(x2, tst_y_cube)
plt.show()
print(MSE1, MSE2, MSE3)
