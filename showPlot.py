import pylab
from mpl_toolkits.mplot3d.axes3d import Axes3D
from FeaGeneration import ReadOff
import numpy as np
def show_plot(Path):
    #Feature,Label=GetMainVariable(path)
    temp = ReadOff(Path)
    temp = temp[np.arange(start=0, stop=temp.shape[0], step=1), :]
    X = temp[:, 0]
    Y = temp[:, 1]
    Z = temp[:, 2]
    fig = pylab.figure()
    ax = Axes3D(fig)

    ax.plot(X, Y, Z, label='helix')
    pylab.legend()
    pylab.title(Path)
    pylab.show()
    return()
