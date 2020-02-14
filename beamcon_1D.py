#!/usr/bin/env python
import os
import numpy as np
import scipy.signal
from astropy import units as u
from astropy.io import fits
from radio_beam import Beam, Beams
from glob import glob
import au2
import functools
print = functools.partial(print, flush=True)

#############################################
#### ADAPTED FROM SCRIPT BY T. VERNSTROM ####
#############################################


def getbeam(datadict, new_beam, verbose=False):
    """Get beam info
    """
    if verbose:
        print(f"Current beam is", datadict['oldbeam'])

    conbm = new_beam.deconvolve(datadict['oldbeam'])
    fac, amp, outbmaj, outbmin, outbpa = au2.gauss_factor(
        [
            conbm.major.to(u.arcsec).value,
            conbm.minor.to(u.arcsec).value,
            conbm.pa.to(u.deg).value
        ],
        beamOrig=[
            datadict['oldbeam'].major.to(u.arcsec).value,
            datadict['oldbeam'].minor.to(u.arcsec).value,
            datadict['oldbeam'].pa.to(u.deg).value
        ],
        dx1=datadict['dx'].to(u.arcsec).value,
        dy1=datadict['dy'].to(u.arcsec).value
    )

    return conbm, fac


def getimdata(cubenm, verbose=False):
    """Get fits image data
    """
    if verbose:
        print(f'Getting image data from {cubenm}')
    with fits.open(cubenm, memmap=True, mode='denywrite') as hdu:

        dxas = hdu[0].header['CDELT1']*-1*u.deg
        dyas = hdu[0].header['CDELT2']*u.deg

        nx, ny = hdu[0].data[0, 0, :,
                             :].shape[0], hdu[0].data[0, 0, :, :].shape[1]

        old_beam = Beam.from_fits_header(
            hdu[0].header
        )

        datadict = {
            'image': hdu[0].data[0, 0, :, :],
            'header': hdu[0].header,
            'oldbeam': old_beam,
            'nx': nx,
            'ny': ny,
            'dx': dxas,
            'dy': dxas
        }
    return datadict


def smooth(datadict, verbose=False):
    """Do the smoothing
    """
    # using Beams package
    if verbose:
        print(f'Smoothing so beam is', datadict["final_beam"])
        print(f'Using convolving beam', datadict["conbeam"])
    pix_scale = datadict['dy']

    gauss_kern = datadict["conbeam"].as_kernel(pix_scale)

    conbm1 = gauss_kern.array/gauss_kern.array.max()

    newim = scipy.signal.convolve(datadict['image'].astype('f8'), conbm1, mode='same')

    newim *= datadict["sfactor"]
    return newim


def savefile(datadict, filename, outdir='.', verbose=False):
    """Save file to disk
    """
    outfile = f'{outdir}/{filename}'
    if verbose:
        print(f'Saving to {outfile}')
    header = datadict['header']
    beam = datadict['final_beam']
    header['BMIN'] = beam.minor.to(u.arcsec).value
    header['BMAJ'] = beam.minor.to(u.arcsec).value
    header['BPA'] = beam.pa.to(u.deg).value
    fits.writeto(outfile, datadict['newimage'], header=header, overwrite=True)


def worker(args):
    file, outdir, new_beam, clargs, verbose = args
    if verbose:
        print(f'Working on {file}')


    outfile = os.path.basename(file)
    outfile = file.replace('.fits', '.sm.fits')
    if clargs.prefix is not None:
        outfile =  clargs.prefix + outfile
    datadict = getimdata(file)

    conbeam, sfactor = getbeam(
        datadict,
        new_beam,
        verbose=verbose
    )

    datadict.update(
        {
            "conbeam": conbeam,
            "final_beam": new_beam,
            "sfactor": sfactor
        }
    )

    newim = smooth(datadict, verbose=verbose)

    datadict.update(
        {
            "newimage": newim,
        }
    )

    savefile(datadict, outfile, outdir, verbose=verbose)


def getmaxbeam(files, verbose=False):
    """Get largest beam
    """
    beams = []
    for file in files:
        header = fits.getheader(file, memmap=True)
        beam = Beam.from_fits_header(header)
        beams.append(beam)

    beams = Beams(
        [beam.major.value for beam in beams]*u.deg,
        [beam.minor.value for beam in beams]*u.deg,
        [beam.pa.value for beam in beams]*u.deg
    )

    return beams.largest_beam()


def main(pool, args, verbose=False):
    """Main script
    """

    # Fix up outdir
    outdir = args.outdir
    if outdir is not None:
        if outdir[-1] == '/':
            outdir = outdir[:-1]
    else:
        outdir = '.'

    # Get file list
    files = glob(args.infile)
    if files == []:
        raise Exception('No files found!')

    # Parse args
    bmaj = args.bmaj
    bmin = args.bmin
    bpa = args.bpa

    # Find largest bmax
    if bmaj is None or bmin is None:
        big_beam = getmaxbeam(files, verbose=verbose)

    # Set to largest
    if bmaj is None:
        bmaj = big_beam.major.to(u.arcsec).round()
    else:
        bmaj *= u.arcsec
    if bmin is None:
        bmin = big_beam.major.to(u.arcsec).round()
    else:
        bmin *= u.arcsec
    if bpa is None:
        bpa = 0*u.deg

    new_beam = Beam(
        bmaj,
        bmin,
        bpa
    )
    if verbose:
        print(f'Final beam is', new_beam)

    inputs = [[file, outdir, new_beam, args, verbose]
              for i, file in enumerate(files)]

    output = list(pool.map(worker, inputs))
    pool.close()

    if verbose:
        print('Done!')

def cli():
    """Command-line interface
    """
    import argparse

    # Help string to be shown using the -h option
    descStr = """
    Smooth a field of 2D images to a common resolution.

    Names of output files are 'infile'.sm.fits

    """

    # Parse the command line options
    parser = argparse.ArgumentParser(description=descStr,
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        'infile',
        metavar='infile',
        type=str,
        help='Input FITS image to smooth (can be a wildcard) - beam info must be in header.')

    parser.add_argument(
        '-p',
        '--prefix',
        dest='prefix',
        type=str,
        default=None,
        help='Add prefix to output filenames.')

    parser.add_argument(
        '-o',
        '--outdir',
        dest='outdir',
        type=str,
        default=None,
        help='Output directory of smoothed FITS image(s) [./].')

    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true",
                        help="verbose output [False].")

    parser.add_argument(
        "--bmaj",
        dest="bmaj",
        type=float,
        default=None,
        help="BMAJ to convolve to [max BMAJ from given image(s)].")

    parser.add_argument(
        "--bmin",
        dest="bmin",
        type=float,
        default=None,
        help="BMIN to convolve to [max BMAJ from given image(s)].")

    parser.add_argument(
        "--bpa",
        dest="bpa",
        type=float,
        default=None,
        help="BPA to convolve to [0].")

    group = parser.add_mutually_exclusive_group()

    group.add_argument("--ncores", dest="n_cores", default=1,
                       type=int, help="Number of processes (uses multiprocessing).")
    group.add_argument("--mpi", dest="mpi", default=False,
                       action="store_true", help="Run with MPI.")

    args = parser.parse_args()
    try:
        import schwimmbad
        pool = schwimmbad.choose_pool(mpi=args.mpi, processes=args.n_cores)
    except ModuleNotFoundError:
        import multiprocessing as mp
        pool = mp.Pool(processes=args.n_cores)
        if args.mpi or args.n_cores == 1:
            raise Exception('Please use Schwimmbad!')

    verbose = args.verbose

    main(pool, args, verbose=verbose)


if __name__ == "__main__":
    cli()
