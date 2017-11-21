import math

def clean_data(data, IND_AVG_DATA=3, IND_GEO_ACC=4):

    data = data[data[:, IND_GEO_ACC] < 3, :]  # Remove inaccurate data
    data = data[data[:, IND_AVG_DATA] < math.pow(10, 5), :]  # Remove outliers
    data = data[data[:, IND_AVG_DATA] != 0, :]  # Remove 0-values

    return data
