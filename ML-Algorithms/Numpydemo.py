import numpy as np
list=np.array([1,2,3,4,5,True])
list1=np.array([6,7,8,9,10,False])
print((list[1]))
print(list[-2])
print(list+list1)
print(sum(list))
# Check if a value exists in the array in numpy
print(3 in list)
print(np.where(list==2))
print(list[1:4]) #slicing
print("-----")
#2d array using numpy 3 rows 4 columns
arr2d=np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print(arr2d)

#numpy 2d methods
print("2d array methods")
list3=np.resize(arr2d,(4,3))
print(list3)
list2=np.reshape(arr2d,(4,3))
print(list2)
print(arr2d.shape)
print(arr2d.ndim) 
print(arr2d.size)  
print(arr2d.dtype) 
print(arr2d.itemsize)
print(arr2d.flatten()) 
print(arr2d.sum(axis=0)) 
print(arr2d.sum(axis=1)) 
