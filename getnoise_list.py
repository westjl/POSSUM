#!/usr/bin/env python
import os
import glob
import sys
from astropy.io import fits
import astropy.units as u
import numpy as np
from spectral_cube import SpectralCube
from tqdm import tqdm, tnrange
import schwimmbad
from functools import partial


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


def calcnoise(args):
    """Get noise in plane from cube.
    """
    cube, i = args
    plane = cube[i]
    imsize = plane.shape
    print('imsize is', imsize)
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
        return Inoise


def getcube(filename):
    """Read FITS file as SpectralCube

    Masks out 0Jy/beam pixels

    """
    cube = SpectralCube.read(filename)
    mask = ~(cube == 0*u.jansky/u.beam)
    cube = cube.with_mask(mask)
    return cube

def main(args):
    pool = schwimmbad.choose_pool(mpi=args.mpi, processes=args.n_cores)
    if args.mpi:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
    
    print(f"Using pool: {pool.__class__.__name__}")

    qcube = getcube(args.qfitslist)
    ucube = getcube(args.ufitslist)
    assert len(ucube.spectral_axis) == len(qcube.spectral_axis)

    inputs = [i for i in range(len(ucube.spectral_axis))]
    qnoisevals = np.array(list(pool.map(calcnoise, zip(args.n_cores * [qcube], inputs))))

    
    unoisevals = np.array(list(pool.map(calcnoise, inputs)))
    
    qmeannoise = np.median(qnoisevals[abs(qnoisevals) < 1.])
    qstdnoise = np.std(qnoisevals[abs(qnoisevals) < 1.])
    print('Q median, std:', qmeannoise, qstdnoise)
    umeannoise = np.median(unoisevals[abs(unoisevals) < 1.])
    ustdnoise = np.std(unoisevals[abs(unoisevals) < 1.])
    print('U median, std:', umeannoise, ustdnoise)
    qbadones = np.logical_or(qnoisevals > (
        qmeannoise+args.cliplev*qstdnoise), qnoisevals == -1.)
    ubadones = np.logical_or(unoisevals > (
        umeannoise+args.cliplev*ustdnoise), unoisevals == -1.)
    print(sum(np.asarray(qbadones, dtype=int)),
          'of', len(qcube.spectral_axis), 'are bad (Q)')
    print(sum(np.asarray(ubadones, dtype=int)),
          'of', len(ucube.spectral_axis), 'are bad (U)')
    totalbad = np.logical_or(qbadones, ubadones)
    print(sum(np.asarray(totalbad, dtype=int)), 'of',
          len(qcube.spectral_axis), 'are bad in Q -or- U')
    # print fitslist[np.asarray(badones,dtype=int)]
    if not args.delete:
        print('Nothing will be deleted, but these are the files that would be with the -d option activated:')
    for i, f in enumerate(qfitslist):
        if totalbad[i]:
            print(qfitslist[i], ufitslist[i])
            if args.delete:
                os.remove(qfitslist[i])
                os.remove(ufitslist[i])
    if not args.delete:
        print('Nothing in the above list was deleted, use -d to take that action')


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('qfitslist', help='Wildcard list of Q fits files')
    ap.add_argument('ufitslist', help='Wildcard list of U fits files')
    ap.add_argument(
        '--delete',
        '-d',
        help='Delete bad channel maps? [default False]',
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
    group = ap.add_mutually_exclusive_group()

    group.add_argument(
        "--ncores",
        dest="n_cores",
        default=1,
        type=int, help="Number of processes (uses multiprocessing)."
    )
    group.add_argument(
        "--mpi",
        dest="mpi",
        default=False,
        action="store_true",
        help="Run with MPI."
    )

    args = ap.parse_args()

    main(args)
