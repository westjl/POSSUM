#!/usr/bin/env python

from astropy.io import fits
import argparse
import glob
import numpy as np

def main(args):

	for infits in glob.glob(args.infits):

		#Load in the fits files
		outfits = infits.split('.fits')[0]+'.3ax.fits'
		print '\n'+infits,'->',outfits
		h=fits.open(infits)

		#Mod header first; modding the data first doesn't play well with astropy's auto-detect of fits parameters
		header = h[0].header
		print('Read-in done. Data cube has shape:')

		#Re-assign the fits header axes specs
		print('\nRe-assigning header info to correct axes...')
		header.remove('NAXIS3',ignore_missing=True)
		header.rename_keyword('NAXIS4','NAXIS3')
		header.remove('CTYPE3',ignore_missing=True)
		header.rename_keyword('CTYPE4','CTYPE3')
		header.remove('CRVAL3',ignore_missing=True)
		header.rename_keyword('CRVAL4','CRVAL3')
		header.remove('CDELT3',ignore_missing=True)
		header.rename_keyword('CDELT4','CDELT3')
		header.remove('CRPIX3',ignore_missing=True)
		header.rename_keyword('CRPIX4','CRPIX3')
		header.remove('CUNIT3',ignore_missing=True)
		header.rename_keyword('CUNIT4','CUNIT3')
		header['NAXIS']=3
		
		#Update the header
		print('Updating the header...')
		h[0].header = header

		#Now mod the data:
		print('Killing the useless Stokes axis...')
		d = np.squeeze(h[0].data)
		h[0].data = d

		#Write the fits file
		print('Writing to new fits file...')
		h.writeto(outfits)
		print('Done.\n')

ap = argparse.ArgumentParser()
ap.add_argument('infits',help='Input fits file')
args = ap.parse_args()
main(args)

