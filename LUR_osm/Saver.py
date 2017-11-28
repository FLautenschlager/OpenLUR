import MySQLdb
import datetime


def saveToDb(dataset, feat, preproc, regressor, iterations, r2, rmse, mae):
	db = MySQLdb.connect(host='supergirl',
	                     user='lautenschlager',
	                     passwd='Arschloch9!',
	                     db='lautenschlager_db')

	cur = db.cursor()

	add_row = ("INSERT INTO lur_osm"
	           "(timestamp, data, features, preprocessing, regressor, cv_iterations, r_squared, rmse, mae)"
	           "VALUES (%S, %S, %S, %S, %S, %S, %S, %S, %S)")

	values = []
	values.append(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
	values.append(dataset)
	values.append(feat)
	if preproc:
		values.append("polynomial")
	else:
		values.append("none")
	values.append(regressor)
	values.append(iterations)
	values.append(r2)
	values.append(rmse)
	values.append(mae)

	cur.execute(add_row, values)

	print(values)