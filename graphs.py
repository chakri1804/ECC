import numpy as np
import matplotlib.pyplot as plt

pr_432 = np.load('pr_432.npy')
pr_743 = np.load('pr_743.npy')
pr_944 = np.load('pr_944.npy')

SNR_arr_432 = np.load('SNR_arr_432.npy')
SNR_arr_743 = np.load('SNR_arr_743.npy')
SNR_arr_944 = np.load('SNR_arr_944.npy')

# plt.plot(SNR_arr_432 + 5*np.log10(8.0/9),np.log10(pr_432),label='(4,3,2)')
plt.plot(SNR_arr_944,np.log10(pr_944),label='(9,4,4)')
plt.plot(SNR_arr_743 + 5*np.log10(7.0/9),np.log10(pr_743),label='(7,4,3)')
plt.xlabel('SNR')
plt.ylabel('semilog Pr. Err')
plt.legend()
plt.show()
