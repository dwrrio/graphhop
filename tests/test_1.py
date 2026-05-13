# Test 5 * 5 patterns with different changes. 

import sys
import os
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hopfield_memory import *

def pattern_to_image(pattern, size=5):
    return pattern.reshape(size, size)


pattern1 = np.array([
    1, -1,  1, -1,  1,
   -1,  1, -1,  1, -1,
    1, -1,  1, -1,  1,
   -1,  1, -1,  1, -1,
    1, -1,  1, -1,  1
    ])

pattern2 = np.array([
    1,  1,  1,  1,  1,
    1, -1, -1, -1,  1,
    1, -1, -1, -1,  1,
    1, -1, -1, -1,  1,
    1,  1,  1,  1,  1
    ])


net = HopfieldMemory(25)
net.train([pattern1, pattern2])

corrupted = pattern2.copy()

corrupted[12] =- corrupted[12]
corrupted[13] =- corrupted[13]
corrupted[20] =- corrupted[20]


recovered = net.recall(corrupted)

fig, axes = plt.subplots(1, 3, figsize=(9, 3))
axes[0].imshow(pattern_to_image(pattern2), cmap='viridis', vmin=-1, vmax=1)
axes[0].set_title('Original Pattern')
axes[1].imshow(pattern_to_image(corrupted), cmap='viridis', vmin=-1, vmax=1)
axes[1].set_title('Corrupted Input')
axes[2].imshow(pattern_to_image(recovered), cmap='viridis', vmin=-1, vmax=1)
axes[2].set_title('Recovered')

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.show()


## -------

corrupted = pattern1.copy()

corrupted[1] =- corrupted[1]
corrupted[4] =- corrupted[4]
corrupted[24] =- corrupted[24]

recovered = net.recall(corrupted)

fig, axes = plt.subplots(1, 3, figsize=(9, 3))
axes[0].imshow(pattern_to_image(pattern1), cmap='viridis', vmin=-1, vmax=1)
axes[0].set_title('Original Pattern')
axes[1].imshow(pattern_to_image(corrupted), cmap='viridis', vmin=-1, vmax=1)
axes[1].set_title('Corrupted Input')
axes[2].imshow(pattern_to_image(recovered), cmap='viridis', vmin=-1, vmax=1)
axes[2].set_title('Recovered')

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.show()

# print(f"Energy of original: {net.energy(pattern2):.2f}")
# print(f"Energy of corrupted: {net.energy(corrupted):.2f}")
# print(f"Energy of recovered: {net.energy(recovered):.2f}")
