# Code for land-use and similar models
OpenSense Data is in /scratch/lautenschlager/OpenSense/UFP_dezipped

## Instructions for LUR_osm

In order to run the feature extraction of this model, you need to have a postGIS database with the osm data of your study area. In the respecting table, indroduce a new column, geog, where you convert the way-column to a geography. This reduces the computation time of the feature extraction a lot.

## Instructions for HeatMapBCC

* [HeatMapBCC](https://github.com/edwinrobots/HeatMapBCC)
* [pyIBCC](https://github.com/edwinrobots/pyIBCC)

clone both directories to the 'HeatMapBcc' subdirectory