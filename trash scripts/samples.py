import numpy as np

samples = []
n = 4
k = 5
N = 100000
Pe = 0.022

np.random.seed(42)

A_T = np.array([[1,1,1,1],[1,1,1,0],[1,1,0,1],[1,0,1,1],[0,1,1,1]])
H = np.hstack((A_T,np.identity(k)))

for i in range(N):
    samples.append(np.random.binomial(1,0.5,n))

samples = np.array(samples)
samples = samples.reshape(N,n)
parity = np.zeros((N,k))
parity[:,0] = samples[:,0]+samples[:,1]+samples[:,2]+samples[:,3]
parity[:,1] = samples[:,0]+samples[:,1]+samples[:,2]
parity[:,2] = samples[:,0]+samples[:,1]+samples[:,3]
parity[:,3] = samples[:,0]+samples[:,2]+samples[:,3]
parity[:,4] = samples[:,1]+samples[:,2]+samples[:,3]

parity = parity%2
Code = np.hstack((samples, parity))
noise = np.random.binomial(1,Pe,Code.shape)

Noised_code = Code+noise
Noised_code = Noised_code%2

# print(Noised_code)
# print('Noised_code')

Syndrome = np.matmul(Noised_code,H.T)
Syndrome = Syndrome%2
# print(Syndrome)
# print('Syndrome')

lookup = np.array([[0,0,0,0,1],
[0,0,0,1,0],
[0,0,1,0,0],
[0,1,0,0,0],
[1,0,0,0,0],
[1,0,1,1,1],
[1,1,0,1,1],
[1,1,1,0,1],
[1,1,1,1,0],
[0,0,0,1,1],
[0,0,1,0,1],
[0,1,0,0,1],
[1,0,0,0,1],
[1,0,1,1,0],
[1,1,0,1,0],
[1,1,1,0,0],
[1,1,1,1,1],
[0,0,1,1,0],
[0,1,0,1,0],
[1,0,0,1,0],
[1,0,1,0,1],
[1,1,0,0,1],
[0,1,1,0,0],
[1,0,1,0,0],
[1,0,0,1,1],
[1,1,0,0,0],
[0,0,1,1,1],
[0,1,0,1,1],
[0,1,1,0,1],
[0,1,1,1,0],
[0,0,0,0,0],
[0,1,1,1,1]])

lookup_error = np.array([[0,0,0,0,0,0,0,0,1],
[0,0,0,0,0,0,0,1,0],
[0,0,0,0,0,0,1,0,0],
[0,0,0,0,0,1,0,0,0],
[0,0,0,0,1,0,0,0,0],
[0,0,0,1,0,0,0,0,0],
[0,0,1,0,0,0,0,0,0],
[0,1,0,0,0,0,0,0,0],
[1,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,1,1],
[0,0,0,0,0,0,1,0,1],
[0,0,0,0,0,1,0,0,1],
[0,0,0,0,1,0,0,0,1],
[0,0,0,1,0,0,0,0,1],
[0,0,1,0,0,0,0,0,1],
[0,1,0,0,0,0,0,0,1],
[1,0,0,0,0,0,0,0,1],
[0,0,0,0,0,0,1,1,0],
[0,0,0,0,0,1,0,1,0],
[0,0,0,0,1,0,0,1,0],
[0,0,0,1,0,0,0,1,0],
[0,0,1,0,0,0,0,1,0],
[0,0,0,0,0,1,1,0,0],
[0,0,0,0,1,0,1,0,0],
[0,0,0,1,0,0,1,0,0],
[0,0,0,0,1,1,0,0,0],
[0,0,0,1,1,0,0,0,0],
[0,0,1,0,1,0,0,0,0],
[0,1,0,0,1,0,0,0,0],
[1,0,0,0,1,0,0,0,0],
[0,0,0,0,0,0,0,0,0],
[1,0,0,0,1,0,0,0,1]])

# print(Noised_code[0])
### Decoding part
for i in range(N):
    for j in range(len(lookup)):
        if(np.array_equal(Syndrome[i],lookup[j])):
            Noised_code[i] += lookup_error[j]

Noised_code = Noised_code%2

# print(Code)
# print('Code')
# print(Noised_code)
# print('Corrected Code')
# print(noise)
# print('Noise generated')

###### Probability of Error Calculation

Err = (Code+Noised_code)%2
#print(Err)
Err1 = np.sum(Err, axis = 1)
#print(Err1)
Err2 = np.count_nonzero(Err1)
pr_err = Err2/N
# if(np.sum(Err))
print(pr_err)
