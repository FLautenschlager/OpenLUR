from OSM_featureExtraction.database_generation_utils import *
import time


class DBGenerator:

    @staticmethod
    def create_db(dbname, latmin, latmax, lonmin, lonmax, osmfile=None, rebuild=False):
        if not rebuild:
            if not check_db_exists(dbname):
                rebuild = True

        if rebuild:
            downloadtime = time.time()
            if not osmfile:
                osmfile = download_bbox("/tmp/" + dbname + ".osm", latmin - 0.1, latmax + 0.1,
                                        lonmin - 0.1, lonmax + 0.1)
            downloadtime = time.time() - downloadtime

            croploadtime = time.time()
            load_db(osmfile, dbname)
            croploadtime = time.time() - croploadtime
        else:
            downloadtime = 0
            croploadtime = 0

        print("Times needed:")
        print("Download: {}s \ncropping and loading: {}s.".format(downloadtime, croploadtime))
