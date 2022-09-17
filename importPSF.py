import tkinter
from tkinter import ttk
from tkinter import filedialog
import scipy.io as sio
import numpy as np


def psfmat(rpsf, rpsf2):
    # will change back to user input after testing
    # root = tkinter.Tk()
    # root.title('Tkinter Open File Dialog')
    # root.resizable(False, False)
    # root.geometry('300x150')
    #
    # PSFloc = filedialog.askopenfilename(
    #     title='Choose a .mat point spread function file',
    #     initialdir="/Users/Nano/Documents/ctr")  # change this later
    # dataPSF = sio.loadmat(PSFloc, appendmat=False)
    # open_button = ttk.Button(
    #     root,
    #     text='Open a File',
    # )
    #
    # open_button.pack(expand=True)
    dataPSF = sio.loadmat('PSFSi50kV_170_v2')
    psfa = dataPSF['psf']
    if 'version' not in psfa.dtype.names:
        version = 1
    else:
        version = psfa['version']

    if version == 1:
        eta = psfa['eta']
        alpha = psfa['alpha']
        beta = psfa['beta']
        descr = psfa['descr']
        psfFSE = (1/(1 + eta)) * (1/(np.pi * alpha**2) * np.exp(-rpsf / alpha**2))
        psfBSE = (1/(1 + eta)) * (eta/(np.pi * beta**2) * np.exp(-rpsf / beta**2))

        psfn = psfFSE/np.sum(psfFSE) + eta * psfBSE / np.sum(psfBSE)
    elif version == 2:
        descr = psfa['descr']
        params = psfa['params']
        alpha = 10 ** params[0, 0][0, 0] * 1E-3
        beta = 10 ** params[0, 0][0, 1] * 1E-3
        gamma = 10 ** params[0, 0][0, 2] * 1E-3

        r = params[0, 0][0, 3]
        nu = params[0, 0][0, 4]
        eta = r - nu
        psfFSE = (1 / (np.pi * (1 + eta + nu))) * ((1 / alpha ** 2) * np.exp(-(rpsf2 / alpha ** 2)))
        psfBSE = (1 / (np.pi * (1 + eta + nu))) * ((eta / beta ** 2) * np.exp(-(rpsf2 / beta ** 2)) +
                                                   (nu/(24 * gamma**2)) * np.exp(-(np.sqrt(rpsf/gamma))))

        psfn = (psfFSE/np.sum(psfFSE)) + ((eta + nu) * psfBSE / np.sum(psfBSE))

    return psfn
