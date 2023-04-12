import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, sqrt, pi

eps = 1e-6


def get_affine(a, b, c, alpha, beta, gamma):
    xa = a
    xb = b * cos(gamma)
    yb = b * sin(gamma)
    xc = c * cos(beta)
    yc = c * (cos(alpha) - cos(beta) * cos(gamma)) / sin(gamma)
    zc = sqrt(c ** 2 - xc ** 2 - yc ** 2)
    return np.array([[xa, 0, 0],
                     [xb, yb, 0],
                     [xc, yc, zc]])


def get_tijkl(m, n, p, a, b, c):
    tijkl = np.meshgrid(np.arange(m) * a, np.arange(n) * b, np.arange(p) * c)
    return np.stack(tijkl, axis=3)


def line(ax, p1, p2, *args, **kwargs):
    ax.plot(*(np.vstack((p1, p2)).T), *args, **kwargs)


def periodize(coos, tijkl):
    coos = [coo + tijkl for coo in coos]
    # this 3 lines below are very tricky...
    coos = np.stack(coos, axis=4)
    coos = coos.transpose(0, 1, 2, 4, 3)
    coos = coos.reshape(-1, 3)
    return coos


def reduce_coordinate(coos, a, b, c):
    def func(xs, k):
        xs[xs <= 0] += k
        return xs

    return np.array([func(coos[:, i], [a, b, c][i]) for i in range(3)]).T


def remove_duplications(my_list: list[np.ndarray]):
    # convert each array to a tuple and use set() to remove duplicates
    unique_tuples = set(tuple(x) for x in my_list)

    # convert the tuples back to np.ndarray
    return [np.array(x) for x in unique_tuples]


def find_nearest_origin(coos, a, b, c):
    coos = reduce_coordinate(coos, a, b, c)
    coos -= np.array([a, b, c]) * 2
    tijkl = get_tijkl(3, 3, 3, a, b, c)
    coos = periodize(coos, tijkl)
    coos = remove_duplications(list(coos))
    dist = np.array(list(map(np.linalg.norm, coos)))
    dist[dist < eps] = np.inf
    min_dist = min(dist)
    res = []
    for c, d in zip(coos, dist):
        if abs(d - min_dist) < eps:
            res.append(c)
    return res


def get_rotation_hkl(hkl):
    Q = schmidt(hkl)  # Q is orthogonal and transforms (1,0,0) to (h,k,l)
    # but we need to transform (h,k,l) to (1,0,0) (yoz plane)
    return np.linalg.inv(Q)


def dist2periodic_color(z, z0):
    # z0 is the interplanar spacing
    arg = z - np.floor(z / z0)
    return arg


def schmidt(vec):
    A = np.eye(3)
    A[:, 0] = vec
    Q, R = np.linalg.qr(A)
    return Q


class Atom:
    def __init__(self, x, y, z, color=np.random.rand(3)):
        self.coo = np.array([x, y, z])
        self.color = color  # unused now


class Link:
    def __init__(self, p1, p2):
        self.s = p1
        self.e = p2


class CrystalCell:
    def __init__(self, a, b, c, alpha, beta, gamma):
        self.a = a
        self.b = b
        self.c = c
        self.affine = get_affine(a, b, c, alpha, beta, gamma)
        self.atoms = []
        self.links = []

    def add_atoms(self, *atoms):
        for a in atoms:
            a.coo = a.coo @ self.affine
        self.atoms.extend(atoms)

    def add_links(self, *links):
        self.links.extend(links)

    def get_nearest_links(self, atom):
        center = atom.coo
        pts = np.array(list(map(lambda x: x.coo, self.atoms))) - center
        nearest = find_nearest_origin(pts, self.a, self.b, self.c)
        print("配位数：", len(nearest))
        nearest += center
        links = [Link(nst, center) for nst in nearest]
        self.add_links(*links)

    def draw(self, xperiod=3, yperiod=3, zperiod=3):
        Tijkl = get_tijkl(xperiod, yperiod, zperiod, self.a, self.b, self.c)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        """
        draw atoms
        """
        coos = list(map(lambda x: x.coo, self.atoms))
        coos = periodize(coos, Tijkl)
        ax.scatter(coos[:, 0], coos[:, 1], coos[:, 2], s=100)

        """
        draw links
        """
        ss = list(map(lambda x: x.s, self.links))
        ss = periodize(ss, Tijkl)
        es = list(map(lambda x: x.e, self.links))
        es = periodize(es, Tijkl)
        for i in range(len(ss)):
            line(ax, ss[i], es[i], color=(0.5, 0.5, 0.5), alpha=0.2)

    def draw_hkl_plane(self, h, k, l, xperiod=3, yperiod=3, zperiod=3):
        # calculate 3d coordinates
        Tijkl = get_tijkl(xperiod, yperiod, zperiod, self.a, self.b, self.c)
        coos = list(map(lambda x: x.coo, self.atoms))
        coos = periodize(coos, Tijkl)
        # project to 2d
        nvec = np.array([h, k, l], dtype=float)
        nvec /= np.linalg.norm(nvec)
        coos = coos @ get_rotation_hkl(nvec).T
        # calculate the interplanar spacing
        # do not use abs because points may not be symmetric by the plane
        abs_height = coos[:, 0] - min(coos[:, 0])
        abs_height_no_zero = abs_height.copy()
        abs_height_no_zero[abs_height_no_zero < eps] = np.inf
        pred_d = min(abs_height_no_zero)
        pred_x = np.round(abs_height / pred_d)
        # fit y=ax
        d = np.dot(pred_x, abs_height) / np.dot(pred_x, pred_x)
        print("晶面间距：", d)
        plt.scatter(coos[:, 1], coos[:, 2], c=dist2periodic_color(abs_height, d*4),
                    vmin=0, vmax=1)  # vmin,vmax fix the scaling of color
        plt.axis('equal')


if __name__ == "__main__":
    cry = CrystalCell(1, 1, 1, pi / 2, pi / 2, pi / 2)
    a = Atom(0, 0, 0)
    b = Atom(1 / 2, 1 / 2, 0)
    c = Atom(1 / 2, 0, 1 / 2)
    d = Atom(0, 1 / 2, 1 / 2)
    cry.add_atoms(a, b, c, d)
    cry.get_nearest_links(a)
    cry.get_nearest_links(b)
    cry.get_nearest_links(c)
    cry.get_nearest_links(d)
    cry.draw_hkl_plane(1, 1, 0, 5, 5, 5)
    # cry.draw()
    plt.show()
