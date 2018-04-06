#!/usr/bin/python3

from utils import paths
import urllib.request as r
import sys
import time
import os.path

def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                    (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()

def download(outfile):
    r.urlretrieve('https://ftp5.gwdg.de/pub/misc/openstreetmap/planet.openstreetmap.org/pbf/planet-latest.osm.pbf', outfile, reporthook)

def main():
    outfile = paths.osmdir + 'planet-latest.osm.pbf'

    if os.path.isfile(outfile):
        print('File {} already exists.'.format(outfile))
        ans = input('Use it? (Y/n')
        if ans=='n':
            download()
        elif (ans=='Y') || (ans==''):
            sys.exit(0)
        else:
            print('please give valid answer')
            sys.exit(1)
    else:
        download()

if __name__=='__main__':
    main()


