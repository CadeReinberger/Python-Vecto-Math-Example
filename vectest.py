import numpy as np

rot_mat = [[0, -1], [-1, 0]] #rotates 90 degrees clockwise
i_hat = np.array([1, 0])
j_hat = np.array([0, 1])

print('Matrix Vector Product')
print(rot_mat @ i_hat)
print('\n')

print('Linear Combination')
print(2 * i_hat + j_hat)
print('\n')

print('Dot Product')
print(np.inner(i_hat, j_hat)) #also np.dot
print('\n')

print('Cross Product')
print(np.cross(i_hat, j_hat))
print('\n')

print('Outer Product')
print(np.outer(i_hat, j_hat))
print('\n')

print('Angle')
print(np.rad2deg(np.arctan2(*np.array([[0, 1], [1, 0]]) @ j_hat))) #matrix mult because numpys convention is flipped
print('\n')

print('Angle Between')
print(np.rad2deg(np.arccos(np.dot(i_hat, j_hat) / (np.linalg.norm(i_hat) * np.linalg.norm(j_hat)))))
print('\n')

print('Projection')
print(j_hat * np.dot(i_hat, j_hat) / np.dot(i_hat, i_hat)) #projection of i onto j
print('\n')

#For 3d manipulation. Most things work the same, just to show scalar triple product
i_hat = np.array([1, 0, 0])
j_hat = np.array([0, 1, 0])
k_hat = np.array([0, 0, 1])

print('Scalar Triple Product')
print(np.dot(np.cross(i_hat, j_hat), k_hat)) #Recall bc of RHR order matters here. Signum is signum of permutation from RHR config


#OUTPUT OF RUN
"""
Matrix Vector Product
[ 0 -1]


Linear Combination
[2 1]


Dot Product
0


Cross Product
1


Outer Product
[[0 1]
 [0 0]]


Angle
90.0


Angle Between
90.0


Projection
[0. 0.]


Scalar Triple Product
1
"""
