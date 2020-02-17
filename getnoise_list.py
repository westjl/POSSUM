#!/usr/bin/env python

import astropy.units as u
import numpy as np
from spectral_cube import SpectralCube
from tqdm import tqdm, trange


def myfit(x, y, fn):
    # Find the width of a Gaussian distribution by computing the second moment of
    # the data (y) on a given axis (x)

    w = np.sqrt(abs(sum(x**2*y)/sum(y)))

    # Or something like this. Probably a Gauss plus a power-law with a lower cutoff is necessary
    #func = lambda x, a, b: np.exp(0.5*x**2/a**2) + x**(-1.*b)

    #[popt, pvar] = curve_fit(func, x, y)
    # Need a newer version of scipy for this to work...

    #a = popt[0]
    #b = popt[1]
    if fn != '':
        import matplotlib.pyplot as plt
        plt.ioff()
        plt.semilogy(x, y, '+', label="Data")
        #plt.semilogy(x, np.exp(-0.5*x**2/a**2) + x**(-1.*b), label="Noise fit")
        plt.semilogy(x, np.exp(-0.5*x**2/w**2), label="Noise fit")
        plt.title('Normalized pixel distribution')
        plt.ylabel('Rel. Num. of Pixels')
        plt.xlabel('Pixel brightness (Jy/beam)')
        plt.legend(loc=3)
        plt.savefig(fn)
        # plt.show()
        plt.close()
    return w


def calcnoise(plane):
    """Get noise in plane from cube.
    """
    imsize = plane.shape
    assert len(imsize) == 2
    nx = imsize[-1]
    ny = imsize[-2]
    Id = plane[ny//3:2*ny//3, nx//3:2*nx//3].flatten()
    if len(Id[np.isnan(Id)]) == len(Id):
        return -1.
    else:
        rms = np.std(Id)
        mval = np.mean(Id)
        Id = Id[np.logical_and(Id < mval+3.*rms, Id > mval-3.*rms)]
        # print mval,rms,len(Id)

        #hrange = (-1,1)
        # , range=hrange) # 0 = values, 1 = left bin edges
        Ih = np.histogram(Id, bins=100)
        if max(Ih[0]) == 0.:
            return -1.
        Ix = Ih[1][:-1] + 0.5*(Ih[1][1] - Ih[1][0])
        Iv = Ih[0]/float(max(Ih[0]))
        Inoise = myfit(Ix, Iv, '')
        return Inoise.value


def getcube(filename):
    """Read FITS file as SpectralCube

    Masks out 0Jy/beam pixels

    """
    cube = SpectralCube.read(filename)
    mask = ~(cube == 0*u.jansky/u.beam)
    cube = cube.with_mask(mask)
    return cube


def getbadchans(qcube, ucube, cliplev=5):
    """Find deviated channels
    """
    assert len(ucube.spectral_axis) == len(qcube.spectral_axis)
    qnoisevals = []
    for plane in tqdm(qcube, desc='Checking Q'):
        qnoisevals.append(calcnoise(plane))
    qnoisevals = np.array(qnoisevals)

    unoisevals = []
    for plane in tqdm(ucube, desc='Checking U'):
        unoisevals.append(calcnoise(plane))
    unoisevals = np.array(unoisevals)

    qmeannoise = np.median(qnoisevals[abs(qnoisevals) < 1.])
    qstdnoise = np.std(qnoisevals[abs(qnoisevals) < 1.])
    print('Q median, std:', qmeannoise, qstdnoise)
    umeannoise = np.median(unoisevals[abs(unoisevals) < 1.])
    ustdnoise = np.std(unoisevals[abs(unoisevals) < 1.])
    print('U median, std:', umeannoise, ustdnoise)
    qbadones = np.logical_or(qnoisevals > (
        qmeannoise+cliplev*qstdnoise), qnoisevals == -1.)
    ubadones = np.logical_or(unoisevals > (
        umeannoise+cliplev*ustdnoise), unoisevals == -1.)
    print(sum(np.asarray(qbadones, dtype=int)),
          'of', len(qcube.spectral_axis), 'are bad (Q)')
    print(sum(np.asarray(ubadones, dtype=int)),
          'of', len(ucube.spectral_axis), 'are bad (U)')
    totalbad = np.logical_or(qbadones, ubadones)
    print(sum(np.asarray(totalbad, dtype=int)), 'of',
          len(qcube.spectral_axis), 'are bad in Q -or- U')
    return totalbad


def blankchans(qcube, ucube, totalbad, blank=False):
    """Mask out bad chans
    """
    chans = np.array([i for i, chan in enumerate(qcube.spectral_axis)])
    badchans = chans[totalbad]
    badfreqs = qcube.spectral_axis[totalbad]
    if not blank:
        print('Nothing will be blanked, but these are the channels/frequencies that would be with the -b option activated:')
    print(f'Bad channels are {badchans}')
    print(f'Bad frequencies are {badfreqs}')
    totalgood = [not bad for bad in totalbad]
    q_msk = qcube.mask_channels(totalgood)
    u_msk = ucube.mask_channels(totalgood)
    if not args.blank:
        print('Nothing in the above list was blanked, use -b to take that action')
    return q_msk, u_msk


def writefits(qcube, ucube, clargs):
    """Write output to disk
    """
    outfile = clargs.qfitslist.replace('.fits', '.blanked.fits')
    print(f'Writing to {outfile}')
    qcube.write(outfile, format='fits', overwrite=True)
    outfile = clargs.ufitslist.replace('.fits', '.blanked.fits')
    print(f'Writing to {outfile}')
    ucube.write(outfile, format='fits', overwrite=True)

def main(clargs):
    qcube = getcube(clargs.qfitslist)
    ucube = getcube(clargs.ufitslist)

    totalbad = getbadchans(qcube, ucube, cliplev=clargs.cliplev)
    # print fitslist[np.asarray(badones,dtype=int)]

    q_msk, u_msk = blankchans(qcube, ucube, totalbad, blank=clargs.blank)

    if clargs.iterate is not None:
        print(f'Iterating {clargs.iterate} additional time(s)...')
        for i in range(clargs.iterate):
            totalbad = getbadchans(q_msk, u_msk, cliplev=clargs.cliplev)
            q_msk, u_msk = blankchans(q_msk, u_msk, totalbad, blank=clargs.blank)
    
    if clargs.blank:
        writefits(q_msk, u_msk, clargs)

    if clargs.file is not None:
        prin(f'Saving bad files to {clargs.file}')
        np.savetxt(clargs.file, totalbad)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('qfitslist', help='Wildcard list of Q fits files')
    ap.add_argument('ufitslist', help='Wildcard list of U fits files')
    ap.add_argument(
        '--blank',
        '-b',
        help='Blank bad channel maps? [default False]',
        default=False,
        action='store_true'
    )
    ap.add_argument(
        '--cliplev',
        '-c',
        help='Clip level in sigma, make this number lower to be more aggressive [default 5]',
        default=5.,
        type=float
    )

    ap.add_argument(
        '--iterate',
        '-i',
        help='Iterate flagging check N additional times [None -- one pass only]',
        default=None,
        type=int
    )

    ap.add_argument(
        '--file',
        '-f',
        help='Filename to write bad channel indices to file [None --  do not write]',
        default=None,
        type=str
    )

    args = ap.parse_args()

    main(args)
