import numpy as np
import torch
import matplotlib.pyplot as plt

import sys
import os


path=sys.path[0]

# path_save=os.path.join(path,"Prb_1_state")
path_save=path


xx=np.arange(-1,1.1,0.1)


y_jsd=np.load(os.path.join(path_save,"JSD_axis.npy"))
y_wd=np.load(os.path.join(path_save,"WD_axis.npy"))

assert y_jsd.shape==xx.shape

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.plot(xx, y_jsd)
plt.title('Estimated JSD')
plt.xlabel('$\phi$')
plt.subplot(1,2,2)
plt.plot(xx, y_wd,color="orange")
plt.xlabel('$\phi$')
plt.title('Estimated WD')
plt.savefig("JSD_WD.jpg")



plt.figure(figsize=(10,10))
plt.plot(xx, y_jsd)
plt.plot(xx, y_wd,color="orange")
plt.title('Estimated JSD and WD')
plt.xlabel('$\phi$')
plt.legend(["JSD","WD"])
plt.savefig("JSD_WD_together.jpg")

print("make plot done")



