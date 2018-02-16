from os.path import expanduser, join

# The last '' ensures that the paths have a trailing slash (os independent)
rootdir = expanduser(join('~', 'Data', 'OpenSense', 'Shared', 'UFP_Delivery_Lautenschlager', 'matlab', ''))
datadir = join(rootdir, 'data', 'seasonal_maps', '')
filtereddatadir = join(datadir, 'filt', '')
hadatadir = join(datadir, 'ha', '')
extdatadir = join(datadir, 'ext', '')
modeldatadir = join(datadir, 'model', '')
landusedir = join(rootdir, 'landuse_data', '')
rdir = join(rootdir, '..', 'R', '')
lurdata = join(datadir, 'lur', '')
bayesiandata = join(datadir, 'bayes', '')
autosklearn = join(datadir, "autosklearn", '')
featuresel = join(datadir, 'featureselection', '')

apicdir = expanduser(join('~', 'Data', 'APIC', ''))
