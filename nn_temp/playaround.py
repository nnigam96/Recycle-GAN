from dataset import *
import matplotlib.pyplot as plt
#mnist = Hindi_Digits()
mnist = Custom_MNIST()

triplet,label = mnist[6821]
plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(triplet[0].detach().numpy(), cmap='gray')
plt.xlabel('MNIST')
plt.subplot(1, 3, 2)
plt.imshow(triplet[1].detach().numpy(), cmap='gray')
plt.xlabel('Fake HINDI')
plt.subplot(1, 3, 3)
plt.imshow(triplet[2].detach().numpy(), cmap='gray')
plt.xlabel('EncA1')
plt.savefig('temp1.png')

 
print('break')