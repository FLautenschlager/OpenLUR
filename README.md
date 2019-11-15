OpenLUR is a off-the-shelf solution for globally available land use regression for e.g. pollution prediction.

# Requirements

- python3 with requirements from requirements.txt
- docker (recommended)
- docker-compose (recommended)

# Usage

## Feature extraction from OpenStreetMap

First start docker-container for the PostGIS database: 
        docker-compose up -d
Alternatively you have to have a postgres database with postgis extension.

The next steps are based upon the application scenario: 
You can extract features either for a grid 
        python3 osm_feature_generation.py map <databasename in lowercase (e.g. city name)> <minimum latitude> <maximum latitude> <minimum longitude> <maximum longitude>
or a file with latitude and longitude values:
        python3 osm_feature_generation.py file <databasename in lowercase (e.g. city name)> <file (csv-file with lat and lon columns)> (-v <value to keep in the output file, optional>)

Both will output a csv file (filename indicated in the command line output) containing lat, lon, value (if specified) and the land usage features.

## Recreation of paper experiments

(will be available upon published paper)
