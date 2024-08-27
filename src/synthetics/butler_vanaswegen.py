import numpy as np


def Butler_VanAswegen_1993(m, r):
    gmm = {}

    # Butler and Van Aswegen (1993),
    gmm['f_lt100'] = lambda m, r: 1e-3 * \
        10 ** (3.343 + 0.775 * m - 1.423 * np.log10(r))  # R<100m in [m/s]
    gmm['f_gt100'] = lambda m, r: 1e-3 * \
        10 ** (3.488 + 0.780 * m - 1.489 * np.log10(r))  # R>100m in [m/s]

    # if r < 100:
    #     pgv = gmm['f_lt100'](m, r)
    # else:
    #     pgv = gmm['f_gt100'](m, r)

    pgv = gmm['f_gt100'](m, r)

    gmm['ylab'] = 'PGV [m/s]'
    gmm['unit'] = '[m/s]'

    return pgv, gmm
