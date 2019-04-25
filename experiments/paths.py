from os.path import expanduser, join

# The last '' ensures that the paths have a trailing slash (os independent)
rootdir = expanduser(join('~', 'work', 'P2Map', 'mapmodels', 'data', 'OpenSense', ''))
#rootdir = expanduser(join('~', 'data', 'UFP_Delivery_Lautenschlager', 'matlab', ''))

datadir = join(rootdir, 'seasonal_maps', '')
filtereddatadir = join(datadir, 'filt', '')
hadatadir = join(datadir, 'ha', '')
extdatadir = join(datadir, 'ext', '')
modeldatadir = join(datadir, 'model', '')
landusedir = join(rootdir, 'landuse_data', '')
lurdata = join(datadir, 'lur', '')
autosklearn = join(datadir, "autosklearn", '')
featuresel = join(datadir, 'featureselection', '')

osmdir = join('OSM-data','')

apicdir = expanduser(join('~', 'Data', 'APIC', ''))
