import numpy as np
from numpy.linalg import inv, norm
import matplotlib.pyplot as plt

normal = np.array([0.0, 1.0, 0.0])
tangent = np.array([1.0, 0.0, 0.0])
binormal = np.array([0.0, 0.0, 1.0])

def chi_plus(x):
    return 1.0 if x >= 0.0 else 0.0

def ellipsoid_ndf_std(ax: float, ay: float, m):
    a = np.ndarray(shape=(3, 3), buffer=np.array([[ax, 0.0, 0.0], [0.0, ay, 0.0], [0.0, 0.0, 1.0]]))
    detA = ax * ay
    anV = np.matmul(a, normal)
    athV = np.matmul(inv(a.transpose()), m)
    
    num = 1.0 if m.dot(normal) >= 0.0 else 0.0
    denom = np.pi * detA * norm(anV) * (athV.dot(athV) ** 2)

    return num / denom

def ndf_std(ax: float, ay: float, m):
    th = tangent.dot(m)
    bh = binormal.dot(m)
    nh = normal.dot(m)

    a2 = ax * ay
    v = np.array([ay * th, ax * bh, a2 * nh])
    v2 = v.dot(v)
    w2 = a2 / v2

    return a2 * w2 * w2 / np.pi

# たしかFrostbiteのドキュメントから引っ張ってきたやつのはず
def frostbite_ndf(ax: float, ay: float, m):
    a = ax * ay
    nh2 = abs(normal.dot(m)) ** 2.0
    w = a / (nh2 * a * a - nh2 + 1)

    return w * w / np.pi

def ndf_ggx_aniso(ax: float, ay: float, m):
    th2 = abs(tangent.dot(m)) ** 2.0
    bh2 = abs(binormal.dot(m)) ** 2.0
    nm2 = abs(normal.dot(m)) ** 2.0
    denom1 = (th2 / (ax ** 2.0)) + (bh2 / (ay ** 2.0)) + nm2;
    denom = np.pi * ax * ay * (denom1 ** 2.0);

    return 1.0 / denom

def mkhvec(th: float):
    return np.array([np.sin(th * 0.5 * np.pi), np.cos(th * 0.5 * np.pi), 0.0])

th = np.arange(-1.0, 1.0, 0.01)
m = np.vectorize(mkhvec, otypes=[np.ndarray])(th)

for a in np.arange(0.2, 1.0, 0.1):
    plt.plot(th, np.vectorize(lambda x: ndf_ggx_aniso(a, a, x))(m))
plt.show()
