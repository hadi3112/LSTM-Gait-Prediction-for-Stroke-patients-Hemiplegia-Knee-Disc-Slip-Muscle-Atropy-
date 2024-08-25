import scipy.io
import matplotlib.pyplot as plt

# Load MAT file
mat = scipy.io.loadmat('AFIRM_DATA/Final_Resultants/final_resultant25.mat')

# Print keys to understand the structure of your .mat file
print(mat.keys())

data = mat['final_resultant']  
print(len(data[0]))

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(data[0])  #  adjusted since data is multi-dimensional
plt.title('Plot from .mat file')
plt.xlabel('Samples') 
plt.ylabel('MPU resultant angles')  
plt.show()