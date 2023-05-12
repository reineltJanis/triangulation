import numpy as np
import cv2 as cv
import scipy
import matplotlib.pyplot as plt
from sympy.solvers import solve
from sympy import Symbol
from sympy import Poly


class HZTriangulation:
    # Fundamental matrix
    F = None
    # Left Image
    __imgL = None
    # Right Image
    __imgR = None
    # points left
    __pts1 = []
    # corresponding points right
    __pts2 = []
    # camera matrix 1 (projection matrix)
    __K = None
    # camera matrix 2 (projection matrix)
    __Kp = None
    # current x
    x = None
    # current xp
    xp = None
    M = None

    def __init__(self, imageL, imageR, pts1, pts2, K1, K2):
        assert pts1.shape[1] == 2
        assert pts2.shape[1] == 2
        self.__imgL = imageL
        self.__imgR = imageR
        self.__pts1 = pts1
        self.__pts2 = pts2
        self.__K = K1
        self.__Kp = K2
        self.__computeInitialFundamental()
        assert self.F.shape == (3, 3)

    def __computeInitialFundamental(self):
        p1 = self.__as_homogeneous(self.__pts1)
        p2 = self.__as_homogeneous(self.__pts2)
        F, mask = cv.findFundamentalMat(p1, p2, cv.FM_RANSAC, 3)
        self.F = F
        M, mask = cv.findHomography(p1, p2, cv.FM_RANSAC, 5.0)
        self.M = M
        return F, mask

    def __as_homogeneous(self, arr):
        ones = np.ones((len(arr), 1))
        return np.concatenate((arr, ones), axis=1)

    def getT(self):
        T = np.matrix([[1, 0, -self.x[0]], [0, 1, -self.x[1]], [0, 0, 1]])
        Tp = np.matrix([[1, 0, -self.xp[0]], [0, 1, -self.xp[1]], [0, 0, 1]])
        return T, Tp

    def constructR(self, e):
        return np.matrix([[e[0, 0], e[1, 0], 0], [-e[1, 0], e[0, 0], 0], [0, 0, 1]])

    def __get_g_t(self):
        t = Symbol("t")
        g_t = t*((self.a*t+self.b)**2+self.fp**2*(self.c*t+self.d)**2)**2-(self.a *
                                                                           self.d-self.b*self.c)*(1+self.f**2*t**2)**2*(self.a*t+self.b)*(self.c*t+self.d)
        return g_t, t

    def evaluate_cost(self, t):
        s_t = t**2 / (1+self.f**2*t**2) + (self.c*t+self.d)**2 / \
            ((self.a*t+self.b)**2 + self.fp**2*(self.c*t+self.d)**2)
        return s_t

    def solve_gt(self):
        g_t, t = self.__get_g_t()
        return solve(g_t, t)

    def solve_roots(self):
        g_t, t = self.__get_g_t()
        coeffs = Poly(g_t, t).all_coeffs()
        numpy = np.roots(coeffs)
        opencv = cv.solvePoly(np.array(coeffs, np.float64))[
            1][:, :, 0].flatten()
        # print(numpy, opencv)
        return numpy

    def closest(self, line):
        l = line[0]
        u = line[1]
        v = line[2]
        res = np.array([[-l*v, -u*v, l**2+u**2]]).T
        return res

    def single_point_step(self, index):
        self.x = self.__as_homogeneous(self.__pts1)[index]
        self.xp = self.__as_homogeneous(self.__pts2)[index]
        # (i)
        self.T, self.Tp = self.getT()
        # print("T")
        # print(self.T)
        # print("Tp")
        # print(self.Tp)
        # (ii)
        self.Fp = self.Tp.I.T.dot(self.F).dot(self.T.I)
        # print("Fp")
        # print(Fp)

        # (iii)
        # e, ep = get_es(Fp)
        # e = normalize(e)
        # ep = normalize(ep)

        # right epipole
        self.e = self.compute_epipole(self.Fp)
        self.e = self.normalize_epipole(self.e)

        # left epipole
        self.ep = self.compute_epipole(self.Fp.T)
        self.ep = self.normalize_epipole(self.ep)
        # print("epipoles\n",e,"\n\n",ep)

        # (iv)
        self.R = self.constructR(self.e)
        self.Rp = self.constructR(self.ep)
        # print(R,'\n\n',Rp)
        # print("\nChecks:")
        # print(self.R.dot(self.e))
        # print(self.Rp.dot(self.ep))

        # (v)
        self.Fpp = self.Rp * self.Fp * self.R.T
        # print(Fpp)

        # (vi)
        self.f = self.e[2, 0]
        self.fp = self.ep[2, 0]
        self.a = self.Fpp[1, 1]
        self.b = self.Fpp[1, 2]
        self.c = self.Fpp[2, 1]
        self.d = self.Fpp[2, 2]
        # print("f = {self.f}, f' = {self.fp}, a = {self.a}, b = {self.b}, c = {self.c}, d = {self.d}".format(**locals()))

        # (vii)
        self.roots = self.solve_roots().real
        assert len(self.roots) == 6
        # print('')
        # print(roots)

        # (viii)
        self.costs = np.array([self.evaluate_cost(t) for t in self.roots])
        # self.costs = np.array([np.Infinity if np.iscomplex(t) else self.evaluate_cost(t) for t in self.roots])
        self.t_asym = 1/self.f**2+self.c**2/(self.a**2+self.fp**2*self.c**2)
        self.t_min = self.roots[np.argmin(self.costs)]
        # print(t_min)
        # print(costs)

        # (ix)
        self.l = np.array([self.t_min*self.f, 1, -self.t_min])
        # print("l",l)
        # self.lp = np.array([-self.fp*(self.c*self.t_min + self.d), self.a*self.t_min+self.b,self.c*self.t_min+self.d])
        self.lp = matrix_to_array(self.Fpp.dot(np.array([0, self.t_min, 1])))
        # print(self.lp)
        # print("lp",lp)
        self.hx = self.closest(self.l)
        self.hxp = self.closest(self.lp)
        # print("hx",hx)
        # print("hxp",hxp)

        # (x)
        self.hx2 = self.T.I.dot(self.R.T).dot(self.hx)
        # print(hx2)
        self.hxp2 = self.Tp.I.dot(self.Rp.T).dot(self.hxp)
        # print(hxp2)

        self.X = self.dlt(self.hx2, self.hxp2)
        return self.X

    def dlt(self, x, xp):
        K = self.__K
        Kp = self.__Kp
        A = np.array([
            x[0, 0] * K[2].T - K[0].T,
            x[1, 0] * K[2].T - K[1].T,
            xp[0, 0] * Kp[2].T - Kp[0].T,
            xp[1, 0] * Kp[2].T - Kp[1].T
        ])
        # print(A)
        u, s, vh = np.linalg.svd(A, full_matrices=True)
        h = vh.T[:, -1]
        #print("h: ", h)
        return h/h[-1]

    def triangulate(self, verbose=False):
        points = []
        for i in range(0, len(self.__pts1)):
            X = self.single_point_step(i)
            points.append(X)
            if verbose:
                print(
                    '{i:2} => x:\t{self.x},\txp:\t{self.xp},\tX: {X}'.format(**locals()))
        return points

    def triangulateOpenCv(self):
        try:
            pts1 = self.__pts1.T
            pts2 = self.__pts2.T
            cv_res = cv.triangulatePoints(self.__K, self.__Kp, pts1, pts2).T
            for i in range(0, cv_res.shape[0]):
                # Normalize homogeneous points
                cv_res[i] = cv_res[i]/cv_res[i, -1]
        except:
            print('error')
        return cv_res

    # copied from other source: https://github.com/alyssaq/3Dreconstruction/blob/master/structure.py
    def compute_epipole(self, F):
        """ Computes the (right) epipole from a
            fundamental matrix F.
            (Use with F.T for left epipole.)
        """
        # return null space of F (Fx=0)
        U, S, V = np.linalg.svd(F)
        e = V[-1]
        return e.T

    def normalize_epipole(self, e):
        factor = 1/np.sqrt(e[0, 0]**2 + e[1, 0]**2)
        # print(factor)
        return e.dot(factor)

    def __drawMarkers(self, img, points):
        tmp = img.copy()
        for i in range(0, len(points)):
            point = np.int64(points[i])
            tmp = cv.drawMarker(tmp, point, (i/8*64 % 256, i/4*64 % 256, i*64 % 256),
                                markerType=cv.MARKER_STAR, markerSize=20, thickness=5, line_type=cv.LINE_AA)
        return tmp

    def __printableImage(self):
        out = cv.hconcat([
            self.__drawMarkers(self.__imgL, self.__pts1),
            self.__drawMarkers(self.__imgR, self.__pts2)
        ])
        offsetRx = self.__imgL.shape[1]
        for i, pointL in enumerate(self.__pts1):
            pointR = self.__pts2[i]
            pointR[0] += offsetRx
            cv.line(out, np.int64(pointL), np.int64(pointR),
                    (i/8*64 % 256, i/4*64 % 256, i*64 % 256), 5)

        return out

    def print(self):
        out = self.__printableImage()
        out = cv.cvtColor(out, cv.COLOR_BGR2RGB)
        plt.imshow(out)

    def save_with_markers(self, path):
        out = self.__printableImage()
        cv.imwrite(path, out)


def matrix_to_array(mat):
    return np.squeeze(np.asarray(mat))

# def find_homography(points_source, points_target):
#     A  = construct_A(points_source, points_target)
#     u, s, vh = np.linalg.svd(A, full_matrices=True)

#     # Solution to H is the last column of V, or last row of V transpose
#     homography = vh[-1].reshape((3,3))
#     return homography/homography[2,2]

# def construct_A(points_source, points_target):
#     assert points_source.shape == points_target.shape, "Shape does not match"
#     num_points = points_source.shape[0]

#     matrices = []
#     for i in range(num_points):
#         partial_A = construct_A_partial(points_source[i], points_target[i])
#         matrices.append(partial_A)
#     return np.concatenate(matrices, axis=0)

# def construct_A_partial(point_source, point_target):
#     x, y, z = point_source[0], point_source[1], 1
#     x_t, y_t, z_t = point_target[0], point_target[1], 1

#     A_partial = np.array([
#         [0, 0, 0, -z_t*x, -z_t*y, -z_t*z, y_t*x, y_t*y, y_t*z],
#         [z_t*x, z_t*y, z_t*z, 0, 0, 0, -x_t*x, -x_t*y, -x_t*z]
#     ])
#     return A_partial
