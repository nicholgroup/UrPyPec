import math
import time

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from scipy import fft
from shapely import geometry, validation
from skimage import draw

# For picking a dxf file
import ezdxf
import tkinter
from tkinter import ttk
from tkinter import filedialog


def importDXF():
    # will redo user inputs after testing
    # root = tkinter.Tk()
    # root.title('Tkinter Open File Dialog')
    # root.resizable(False, False)
    # root.geometry('300x150')
    #
    # dxfFileLocation = filedialog.askopenfilename(
    #     title='Choose a DXF file',
    #     initialdir="/Users/Nano/Documents/urPyPec")  # change this later
    # dataDXF = ezdxf.readfile(dxfFileLocation)
    # open_button = ttk.Button(
    #     root,
    #     text='Open a File',
    # )
    #
    # open_button.pack(expand=True)
    dataDXF = ezdxf.readfile('2D.dxf')
    return dataDXF


def polyfromdxf(dx=0.01, targetpoints=5e7, autores=True, maxiter=6, dvalsnum=15, fracsize=2, fracnum=4, maxxter=10,
                lessvertex=True, dxinfo=importDXF()):
    startime = time.time()
    dvals = np.linspace(1, 2, dvalsnum)  # setting up colormap
    if dvalsnum == 15:
        ctab = (
            np.array((0, 0, 175)), np.array((0, 0, 255)), np.array((0, 63, 255)), np.array((0, 127, 255)),
            np.array((0, 191, 255)),
            np.array((15, 255, 239)), np.array((79, 255, 175)), np.array((143, 255, 111)), np.array((207, 255, 47)),
            np.array((255, 223, 0)), np.array((255, 159, 0)), np.array((255, 95, 0)), np.array((255, 31, 0)),
            np.array((207, 0, 0)),
            np.array((143, 0, 0)))
    else:
        dc = 256 / dvalsnum  # ignore for now

    # commenting out for benchmarkurpec
    # dxinfo = importFile.importDXF()
    # Set up to extract polyline data
    lwpolys = dxinfo.query("LWPOLYLINE")
    # Getting vertices
    # Vertex object is a generator of tuples
    # One way of expanding it was to a structure array
    # Polygon validation checks can be done and return numpy array
    maxdval = np.max(dvals)

    structlist = []
    for polyline in lwpolys:
        strct2add = np.fromiter(polyline.vertices(), dtype=[('x', 'f'), ('y', 'f')])
        structlist += [strct2add]
    pgonlist = []
    for struct in structlist:
        poly = geometry.Polygon([[vertex["x"], vertex["y"]] for vertex in struct])
        pgonlist += [poly]
    allgons = geometry.MultiPolygon(pgonlist)
    str(validation.make_valid(allgons))
    # Shapely validation function

    fig, axs = plt.subplots()
    axs.set_aspect('equal', 'datalim')
    xlist = []
    ylist = []
    xarraylist = []
    yarraylist = []
    for p in allgons.geoms:
        xs, ys = p.exterior.xy
        xlist += xs
        ylist += ys
        xarraylist += [np.around(xs, 4)]
        yarraylist += [np.around(ys, 4)]
        axs.fill(xs, ys, alpha=0.5, fc='k', ec='none')

    plt.show()
    # xlist = np.around(np.array(xlist), 5)
    # ylist = np.around(np.array(ylist), 5)
    # Rounding here gives undesired results

    repete = 0

    # Padding array and getting array shape for mask creation
    # Taken from almost exactly from urpec
    # The following should function exactly as the similar code from urpec without needing to repeat this code chunk
    # It was necessary to round the values of the vectors to mimic MATLAB behavior
    # because the arbitrary floating point accuracy compounded to give very different results for the same code
    while repete < 2:
        minXunpad = min(xlist)
        minYunpad = min(ylist)
        maxXunpad = max(xlist)
        maxYunpad = max(ylist)
        padPoints = math.ceil(5 / dx)
        padSize = padPoints * dx
        maxX = maxXunpad + padSize
        minX = minXunpad - padSize
        maxY = maxYunpad + padSize
        minY = minYunpad - padSize
        xnum = math.ceil((maxX - minX) / dx)
        ynum = math.ceil((maxY - minY) / dx)
        xvec = np.around(np.linspace(minX, maxX, xnum), 4)
        yvec = np.around(np.linspace(minY, maxY, ynum), 4)
        lenx = len(xvec)
        leny = len(yvec)
        totpoints = lenx * leny
        if autores and ((totpoints < .8 * targetpoints) or (totpoints > 1.2 * targetpoints)) and (repete == 0):
            expand = math.ceil(math.log2(math.sqrt(totpoints / targetpoints)))
            dx *= 2 ** expand
            repete += 1
        else:
            repete += 2
    # Make sure there is an odd number of points
    # Addx is addy of original urpec and vice versa
    addx = 0
    if lenx % 2 != 1:
        addx = 1
        lastx = xvec[-1]
        lastx += dx
        xvec = np.append(xvec, lastx)
        lenx = len(xvec)
    addy = 0
    if leny % 2 != 1:
        addy = 1
        lasty = yvec[-1]
        lasty += dx
        yvec = np.append(yvec, lasty)
        leny = len(yvec)

    # This may not actually be necessary and only works for dx powers of 10
    # def transformaxis(xbt, ybt):
    #     xtform = (xbt - xvec[0]) / dx
    #     ytform = (ybt - yvec[0]) / dx
    #     return xtform, ytform

    # taken from stack overflow:
    def get_closest(array, values):
        # get insert positions
        idxs = np.searchsorted(array, values, side="left")

        # find indexes where previous index is closer
        prev_idx_is_less = ((idxs == len(array)) | (np.fabs(values - array[np.maximum(idxs - 1, 0)]) < np.fabs(
            values - array[np.minimum(idxs, len(array) - 1)])))
        idxs[prev_idx_is_less] -= 1

        return idxs

    shape = (leny, lenx)  # mimic meshgrid shape
    # The same behavior from MATLAB Meshgrid should be possible to achieve by changing a few variable names
    # Mask creation could not be done on a grid because of the image shape parameter, so skiimage
    # ignored negative values and gave undesired results
    # Here, I translated the coordinates to the shape coordinates(indices of coordinates on grid),
    # and I should be able to shift them back.
    polysbin = np.zeros(shape)
    # Saving slyce objects for fracturing and dose assignment
    bboxslist = []
    rminlist = []
    rmaxlist = []
    cminlist = []
    cmaxlist = []
    for exv, wyv in zip(xarraylist, yarraylist):
        c = get_closest(xvec, exv)  # rows and columns are reversed from x/y
        r = get_closest(yvec, wyv)
        # the previous comments may provide one method to determine the perimeter/coordinates of the fractured shapes
        # r, c = transformaxis(exv, wyv)
        rr, cc = draw.polygon(r, c, shape)
        rmin = np.amin(rr)
        rmax = np.amax(rr) + 1  # +1 to include endpoints/stop values
        cmin = np.amin(cc)
        cmax = np.amax(cc) + 1
        rminlist += [rmin]
        rmaxlist += [rmax]
        cminlist += [cmin]
        cmaxlist += [cmax]
        bboxslist += [((slice(rmin, rmax, 1)), (slice(cmin, cmax, 1)))]
        polysbin[rr, cc] = 1

    # PSF variables before import
    # The following have the same effect as the min/max/meshgrid
    # functions in the original urpec, but are computationally cheaper
    # Meshgrid with sparse set to true is equivalent
    minSize = min((xvec[-1] - xvec[0]), (yvec[-1] - yvec[0]))
    psfRange = round(min(minSize, 20))
    npsf = round(psfRange / dx)
    xpsf, ypsf = np.ogrid[-npsf:npsf + 1, -npsf:npsf + 1]  # +1 to include endpoints
    xpsf = dx * xpsf
    ypsf = dx * ypsf
    rpsf2 = xpsf ** 2 + ypsf ** 2
    rpsf = np.sqrt(rpsf2)

    # import .mat psf
    import importPSF
    psfn = importPSF.psfmat(rpsf, rpsf2)
    psfn = psfn / np.sum(psfn)

    # Numpy pad function cannot take division in a pad width arguement even if that returns an integer
    # because it returns it as a float
    # Ypadovertwo has to be an integer (which it already should be, but this just makes sure)
    # The order of (padovertwo,) and (0,) tuples changes depending on the dimension you wish to pad
    ypad = (np.shape(polysbin)[0] - np.shape(psfn)[0])
    ypadovertwo = int(ypad / 2)
    if ypad >= 0:
        psfn = np.pad(psfn, [(ypadovertwo,), (0,)], mode='constant', constant_values=0)  # 0 is default constant value
        padpoints1 = padPoints
        ydisp = yvec
    elif ypad < 0:
        polysbin = np.pad(polysbin, [(-ypadovertwo,), (0,)], mode='constant', constant_values=0)  # value may be ommited
        padpoints1 = padPoints - ypad / 2
        ydisp = np.concatenate(
            ((np.arange(-ypad / 2 - 1) + ypad / 2) * dx + yvec[0], yvec, np.arange(1, -ypad / 2) * dx + yvec[-1]))
    padpoints1 = round(padpoints1)

    xpad = (np.shape(polysbin)[1] - np.shape(psfn)[1])
    xpadovertwo = int(xpad / 2)
    if xpad > 0:
        psfn = np.pad(psfn, [(0,), (xpadovertwo,)], mode='constant', constant_values=0)
        padpoints2 = padPoints
        xdisp = xvec
    elif xpad < 0:
        polysbin = np.pad(polysbin, [(0,), (-xpadovertwo,)], mode='constant', constant_values=0)
        padpoints2 = padPoints - xpad / 2
        xdisp = np.concatenate(
            ((np.arange(-xpad / 2 - 1) + xpad / 2) * dx + xvec[0], xvec, np.arange(1, -xpad / 2) * dx + xvec[-1]))
    padpoints2 = round(padpoints2)

    # Polysbin needs to be transposed in figures because plots need to be reflected over y=x
    # this is likely because x and y vectors are flipped in matrix creation compared to urpec
    # extent arguement maps image coordinates to shape coordinates
    fig, axis = plt.subplots()
    img = axis.imshow(polysbin.T, vmin=0, vmax=maxdval, aspect='auto', cmap='jet', origin='lower',
                      extent=[xdisp[0], xdisp[-1], ydisp[0], ydisp[-1]])
    fig.colorbar(img)
    plt.title('CAD pattern')

    plt.show()

    shaapee = polysbin
    doseNew = polysbin.copy()
    meanDose = 0

    for i in range(maxiter):
        doseActual = fft.ifft2(fft.fft2(doseNew) * fft.fft2(psfn))
        doseActual = fft.fftshift(doseActual).real
        doseActual[1:, 1:] = doseActual[:-1, :-1]
        doseShape = doseActual * shaapee
        meanDose = np.nanmean(doseShape) / np.mean(shaapee)

        fig, axes = plt.subplots(1, 2)
        img2 = axes[1].imshow(doseActual.T, vmin=0, vmax=maxdval, aspect='auto', cmap='jet', origin='lower',
                              extent=[xdisp[0], xdisp[-1], ydisp[0], ydisp[-1]])
        fig.colorbar(img2)
        axes[1].set_title('Actual dose. Iteration ' + str(i + 1))

        doseNew += 1.2 * (shaapee - doseShape)
        img1 = axes[0].imshow(doseNew.T, vmin=0, vmax=maxdval, aspect='auto', cmap='jet', origin='lower',
                              extent=[xdisp[0], xdisp[-1], ydisp[0], ydisp[-1]])
        axes[0].set_title('Programmed dose. Iteration ' + str(i + 1))

        plt.show()

    if meanDose < .98:
        print('Deconvolution not converged. Consider increasing maxiter.')

    # Should mimic urpec closely.
    # 0 values in doseNew set to nan
    # SMN is a sub-array of doseNew corresponding to the bounding box slices created before
    # If the mean is greater than dDose, SMN will be fractured and new indices are returned
    # Along with the bounding box slice list are lists of the minimum and maximum x/y coordinates of the box
    # the coordinates are still shape coordinates/indices for the time being
    # If the array's dose variation is too great, the min/max values are used to create vectors which
    # will become the new min/max values and form the new slice objects which will be iterated over
    # The number of sub-iterations is maxxter
    # The iterations only proceed if the array has a minimum value and can be split by fracnum in at least one axis

    ddose = dvals[1] - dvals[0]

    doseNew[doseNew == 0] = np.nan
    doseNew[doseNew < 0] = np.nan

    def dosecheck(slyce, rminn, rmaxx, cminn, cmaxx, doselistforpoly, doseadjustedPolylist, arrayforplot):
        # SMN = np.empty(shape)
        SMN = doseNew[slyce]
        Shapeslice = shaapee[slyce]
        # newidxs = np.nonzero(Shapeslice)
        # rminew = rminn + (np.amin(newidxs[1]))
        # rmaxnew = rminn + (np.amax(newidxs[1]))
        # cminew = cminn + (np.amin(newidxs[0]))
        # cmaxnew = cminn + (np.amax(newidxs[0]))
        if ((np.nanmax(SMN) - np.nanmin(SMN)) < ddose) or (
                (((rmaxx - rminn) / fracnum) < fracsize) and (((cmaxx - cminn) / fracnum) < fracsize)):
            doselistforpoly, doseadjustedPolylist, dosage = dosingfn(SMN, Shapeslice, rminn, cminn,
                                                                     doseadjustedPolylist, doselistforpoly)
            arrayforplot[slyce] = dosage
        else:
            fractureandcheck(SMN, Shapeslice, rmaxx, rminn, cmaxx, cminn, doselistforpoly, doseadjustedPolylist)
        return slicestoadjust, rminlstnest, rmaxlstnest, cminlstnest, cmaxlstnest, doselistforpoly, doseadjustedPolylist, arrayforplot

    def fractureandcheck(SMN, Shapeslice, rmaxx, rminn, cmaxx, cminn, doselistforpoly, doseadjustedPolylist):
        canfracx = (np.floor(((rmaxx - rminn) / fracnum)) > fracsize)
        canfracy = (np.floor(((cmaxx - cminn) / fracnum)) > fracsize)
        xdiff = np.nanmax(np.nanmax(SMN, 1) - np.nanmin(SMN, 1))
        shouldfracx = (xdiff > ddose)
        ydiff = np.nanmax(np.nanmax(SMN, 0) - np.nanmin(SMN, 0))
        shouldfracy = (ydiff > ddose)
        xdivnum = canfracx * shouldfracx * (fracnum - 1) + 2  # +2 because extra element is needed to partition
        ydivnum = canfracy * shouldfracy * (fracnum - 1) + 2  # vector must at least 2 elements
        rdivec = np.int_(np.around(np.linspace(rminn, rmaxx, xdivnum)))  # indices must be converted to int type
        cdivec = np.int_(np.around(np.linspace(cminn, cmaxx, ydivnum)))
        rdivec[-1] += 1  # adding one to include endpoint
        cdivec[-1] += 1  # middle points don't need + 1 additions because the min of the next slice will include it
        # following slices are created from each permutation of new x and y maxs/mins
        # new min/max vals are determined by pairs of elements in vectors
        origsum = np.sum(shaapee[slyce])
        sumcheck = 0
        sleycelist = []
        rminsub = []
        cminsub = []
        rmaxsub = []
        cmaxsub = []
        for rindx in range(len(rdivec) - 1):
            for cindx in range(len(cdivec) - 1):
                sleyce = ((slice(rdivec[rindx], rdivec[rindx + 1], 1)),
                          (slice(cdivec[cindx], cdivec[cindx + 1], 1)))
                # nonzind = shapetocheck.nonzero()
                shapetocheck = shaapee[sleyce]
                if np.sum(shapetocheck) == 0:
                    continue
                sumcheck += np.sum(shapetocheck)
                sleycelist += [sleyce]
                rminsub += [rdivec[rindx]]
                cminsub += [cdivec[cindx]]
                rmaxsub += [rdivec[rindx + 1]]
                cmaxsub += [cdivec[cindx + 1]]
        if sumcheck != origsum:
            doselistforpoly, doseadjustedPolylist, dosage = dosingfn(SMN, Shapeslice, rminn, cminn,
                                                                     doseadjustedPolylist, doselistforpoly)
            arrayforplot[slyce] = dosage  # keep the original because areas unequal
        else:
            for sleyce, rsubmin, csubmin, rsubmax, csubmax in zip(sleycelist, rminsub, cminsub, rmaxsub, cmaxsub):
                # if np.sum(shapetocheck) < (fracsize ** 2) or len(np.unique(nonzind[0])) <= 1 or len(
                #         np.unique(nonzind[1])) <= 1:
                #     continue
                if itr == maxxter:
                    doselistforpoly, doseadjustedPolylist, dosage = dosingfn(doseNew[sleyce], shaapee[sleyce],
                                                                             rsubmin, csubmin,
                                                                             doseadjustedPolylist, doselistforpoly)
                    arrayforplot[sleyce] = dosage
                else:
                    slicestoadjust.append(sleyce)  # need to append or return won't work
                    rminlstnest.append(rsubmin)
                    cminlstnest.append(csubmin)
                    rmaxlstnest.append(rsubmax)
                    cmaxlstnest.append(csubmax)
        return doseadjustedPolylist, doselistforpoly, rminlstnest, cminlstnest, rmaxlstnest, cmaxlstnest, arrayforplot

    def dosingfn(SMN, Shapeslice, rmin, cmin, doseadjustedPolylist, doselistforpoly):
        dst = cv.findContours(np.uint8(Shapeslice), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # binary array necessary
        contours = dst[0]
        dosage = np.argmin(np.absolute(dvals - np.nanmean(SMN)))
        for contour in contours:
            contourray = np.empty((contour.shape[0], 2), dtype=np.int_)
            contourray[:, 0] = contour[:, 0][:, 0] + cmin  # switched !!!!contours is array of vertices
            contourray[:, 1] = contour[:, 0][:, 1] + rmin  # shift coordinates back to original shape indices
            doseadjustedPolylist += [contourray]
            doselistforpoly += [dosage]
        return doselistforpoly, doseadjustedPolylist, dosage

    arrayforplot = np.empty(shaapee.shape)
    doselistforpoly = []
    doseadjustedPolylist = []
    # np.uint8(shaapee)

    # the following is the actual code that uses the functions above
    for slyce, rminn, cminn, rmaxx, cmaxx in zip(bboxslist, rminlist, cminlist, rmaxlist, cmaxlist):
        itr = 1
        slicestoadjust = []  # redefine inner lists for each bounding box
        rminlstnest = []
        cminlstnest = []
        rmaxlstnest = []
        cmaxlstnest = []
        dosecheck(slyce, rminn, rmaxx, cminn, cmaxx, doselistforpoly, doseadjustedPolylist, arrayforplot)
        while slicestoadjust and (itr <= maxxter):
            slicelen = len(slicestoadjust)  # if slicestoadjust has elements, iterate through the initial length of it
            # after iterating over initial list length, itr++, take previous slices out of list and repeat
            for subindex in range(slicelen):
                dosecheck(slicestoadjust[subindex], rminlstnest[subindex], rmaxlstnest[subindex], cminlstnest[subindex],
                          cmaxlstnest[subindex], doselistforpoly, doseadjustedPolylist, arrayforplot)
            slicestoadjust = slicestoadjust[slicelen:]
            rminlstnest = rminlstnest[slicelen:]
            cminlstnest = cminlstnest[slicelen:]
            rmaxlstnest = rmaxlstnest[slicelen:]
            cmaxlstnest = cmaxlstnest[slicelen:]
            itr += 1

    def divideandtriangulate(doseadjustedPolylist, doselistforpoly):
        squareortriangle = []
        fracturedose = []
        for polygon, dose in zip(doseadjustedPolylist, doselistforpoly):
            if np.shape(polygon)[1] <= 4:
                squareortriangle += [polygon]
                fracturedose += [dose]
            else:
                pass

    arrayforplot *= shaapee
    figfinal, axis = plt.subplots()
    img = axis.imshow(arrayforplot.T, vmin=0, vmax=maxdval, aspect='auto', cmap='jet', origin='lower',
                      extent=[xdisp[0], xdisp[-1], ydisp[0], ydisp[-1]])
    figfinal.colorbar(img)
    plt.title('Dose Adjusted Pattern')

    plt.show()

    dxfnew = ezdxf.new(units=13)  # MICRONS
    msp = dxfnew.modelspace()

    # for debugging
    # fig, axs = plt.subplots()
    # axs.set_aspect('auto')
    # ACIcolor = [ezdxf.colors.rgb2int(ACI) for ACI in ctab]

    # Create layers with different colors and assign polygons to those layers
    for layer, color in enumerate(ctab, 1):
        currentlayer = dxfnew.layers.add(name=str(layer))
        currentlayer.rgb = tuple(color)
        currentlayer.transparency = .5
    for poly, doseindex in zip(doseadjustedPolylist, doselistforpoly):
        # dose = ctab[doseindex]  # / 256
        msp.add_lwpolyline(tuple(map(tuple, poly)), format='xy', close=True, dxfattribs={'layer': str(doseindex)})
        # axs.fill(poly[:, 0], poly[:, 1], alpha=0.5, fc=dose, ec='none')
    # plt.show()
    dxfnew.saveas('testingoutput.dxf')
    timetaken = time.time() - startime
    print(str(timetaken))
