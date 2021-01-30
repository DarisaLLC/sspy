import logging


import numpy as np
import matplotlib.pyplot as plt
from similarity_fns import information_variation, squared_ncv_cv2, squared_ncv_np, root_nmse, unit_ncv, squared_nsd_cv2
from sequence_similarity import selfsimilarity
logging.basicConfig(filename='ut.log', format='%(asctime)s %(message)s', level=logging.INFO)

sphere = np.random.rand(9,9,9)
sphere = sphere * 10
sphere = sphere.astype(np.uint8)

bloc = [3,3]
bsize = [4,3]
sphere = sphere
bump = np.random.rand(9)
bump = bump * 100
bump = bump.astype(np.uint8)

for ii in range(bsize[0]):
    for jj in range(bsize[1]):
        noise = np.random.rand(9)
        noise = noise * 10
        sphere[:,bloc[0]+ii,bloc[1]+jj] = bump + noise

result = selfsimilarity(sphere, False, -1, 81, True, 1, -1,squared_nsd_cv2)



print(sphere)
print(result)
entropy = result[1]
hist, bins = np.histogram(entropy, bins=81, range=(0, 1))
plt.plot(hist)
plt.show()

