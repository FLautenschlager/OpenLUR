import MySQLdb
import datetime
import json


def saveToDb(dataset, feat, preproc, regressor, iterations, r2, rmse, mae):
	cre = json.load(open('../mysqlCredentials.json', 'r'))

	db = MySQLdb.connect(host=cre['host'],
	                     user=cre['user'],
	                     passwd=cre['passwd'],
	                     db=cre['db'])

	cur = db.cursor()

	add_row = ("INSERT INTO lur_osm"
	           " (timestamp, data, features, preprocessing, regressor, cv_iterations, r_squared, rmse, mae)"
	           " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)")

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
	db.commit()
	cur.execute("SELECT COUNT(*) FROM lur_osm")
	print(cur.fetchone())
	cur.close()
	db.close()
	print(values)
