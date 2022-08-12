from os.path import abspath

from matplotlib.cm import get_cmap

# Constants
G_GRAV = 4.3e-9  # Mpc (km/s)^2 / M_sun
TO_GYR = (3_086. / 3.1536) / 0.672  # Convert Mpc s / km / h to Gyr
PARTMASS = 7.754657e+10  # M_sun / h
RHOCRIT = 2.77536627e+11  #
COSMO = {'flat': True, 'H0': 67.2, 'Om0': 0.3, 'Ob0': 0.049, 'sigma8': 0.81,
         'ns': 0.95}
RHOM = RHOCRIT * COSMO["Om0"]

RSOFT = 0.015  # Softening length in Mpc/h
BOXSIZE = 1_000  # Mpc / h
MEMBSIZE = int(10 * 1000**3)  # 10.0 GB file

# Paths
SDD = abspath('/spiff/edgarmsc/halo_model/')
SRC = abspath('/spiff/edgarmsc/simulations/susmita_sim/')

# Mass bins edges (log10)
MBINEDGES = [13.40, 13.55, 13.70, 13.85, 14.00, 14.15, 14.30, 14.45, 14.65,
             15.00]
MBINSTRS = ['13.40-13.55', '13.55-13.70', '13.70-13.85', '13.85-14.00',
            '14.00-14.15', '14.15-14.30', '14.30-14.45', '14.45-14.65',
            '14.65-15.00']

# Formatting (bash output)
BULLET = '\u25CF'

class COLS:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

OKGOOD = f'{COLS.OKGREEN}{BULLET}{COLS.ENDC} '
FAIL = f'{COLS.FAIL}{BULLET}{COLS.ENDC} '

# Nice color scale
CMAP = get_cmap('nipy_spectral')
