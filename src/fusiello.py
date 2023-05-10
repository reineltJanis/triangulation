import scipy
import numpy as np
import cv2 as cv
import glob


class Fusiello:
    def __init__(self, Mp1: np.ndarray, Mp2: np.ndarray, verbose=False) -> None:
        self.__fusiello(Mp1, Mp2, verbose)

    def factorize(P: np.ndarray):
        M = P[:, 0:3]

        out = cv.decomposeProjectionMatrix(P)
        K, R, T = out[0], out[1], out[2]
        K = K / K[2, 2]
        print("Tcv", T)
        T = R.dot(P[:, 3])
        print("Tf", T)

        C = np.dot(-scipy.linalg.inv(M), P[:, 3])
        print("Sanity (P)", P)
        print("Sanity (K)", K.dot(np.concatenate((R, T.reshape(-1, 1)), axis=1)))
        print("C", C)
        return (K, R, C)

    def __fusiello(self, Mpl: np.ndarray, Mpr: np.ndarray, verbose=False):
        Kl, Rl, Cl = Fusiello.factorize(Mpl)
        Kr, Rr, Cr = Fusiello.factorize(Mpr)
        if verbose:
            print(Kl)
            print(Rl)
            print(Cl)
        x = Cl-Cr
        y = np.array(x)
        y[0] = -x[1]
        y[1] = x[0]
        z = np.cross(x, y)
        if verbose:
            print("x", x)
            print("y", y)
        x = x/np.linalg.norm(x)
        y = y/np.linalg.norm(y)
        z = z/np.linalg.norm(z)
        Rs = np.array([x, y, z])
        tls = -Rs.dot(Cl).reshape(-1, 1)
        trs = -Rs.dot(Cr).reshape(-1, 1)
        if verbose:
            print("Rs", Rs)
            print(tls)
            print(trs)
        Ks = (Kl+Kr)/2
        Ks[0, 1] = 0
        Mls = Ks.dot(np.concatenate((Rs, tls), axis=1))
        Mrs = Ks.dot(np.concatenate((Rs, trs), axis=1))
        if verbose:
            print("Mrs", Mrs)
            print("Mls", Mls)
        self.TFusl = Ks.dot(Rs).dot(np.linalg.inv(Kl.dot(Rl)))
        self.TFusr = Ks.dot(Rs).dot(np.linalg.inv(Kr.dot(Rr)))
        if verbose:
            print("TFusl", self.TFusl)
            print("TFusr", self.TFusr)
        return self.TFusl, self.TFusr
