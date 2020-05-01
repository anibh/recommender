import pyspark
import json
import numpy as np
from datetime import datetime
import sys

start_time = datetime.now()
sc = pyspark.SparkContext('local[*]')
sc.setLogLevel('WARN')

rdd = sc.textFile(sys.argv[1])
list = sys.argv[2]

def mapReviews(x):
	try:
		overall = x['overall']
	except KeyError:
		return (('', ''), (0.0, ''))
		
	try:
		asin = x['asin']
	except KeyError:
		return (('', ''), (0.0, ''))
		
	try:
		reviewerID = x['reviewerID']
	except KeyError:
		return (('', ''), (0.0, ''))
		
	return ((asin, reviewerID), (overall, x['unixReviewTime']))
	
def oneReview(x):
	temp = 10000000000
	for (overall, time) in x[1]:
		if time < temp: 
			temp = time
			item = overall
	
	return (x[0], item)
	
def filter1(x):
	if len(x[1]) < 25: return False
	else: return True
	
def map(x):
	list= []
	for k in x[1]:
		list.append((k[0], (x[0], k[1])))
	return list
	
def filter2(x):
	if len(x[1]) < 5: return False
	else: return True
	
def mapper(x):
	list = []
	for k in x[1]:
		list.append(((x[0], k[0]), k[1]))
	return list
	
def vector(x):
	dict = userDict
	for (k, v) in x[1]:
		dict[k] = v
	return (x[0], dict)
	
def simMap(x):
	list = []
	for item in result:
		list.append((x, item))
	return list
	
def simCal(x):
	ssx = 0
	ssy = 0
	dot = 0
	for user in users.value:
		if(x[0][1][user]): ssx += x[0][1][user]**2
		if(x[1][1][user]): ssy += x[1][1][user]**2
		if(x[0][1][user] and x[1][1][user]): dot += x[0][1][user]*x[1][1][user]
	cosSim = dot/(np.sqrt(ssx) * np.sqrt(ssy))
	
	return (x[1][0], (x[0][0], cosSim, x[0][1]))
	
def uncommon(x):
	if len(x[0][1].keys() & x[1][1].keys()) < 2: return False
	else: return True
	
def neighbor(x):
	neighbors = []
	for (k, v, d) in x[1]: 
		neighbors.append([k, v, d])
	neighbors.sort(key=lambda tup: tup[1])
	if len(neighbors) > 50: neighbors = neighbors[:50]
	return (x[0], neighbors)
	
def recommend(x):
	for user in users.value:
		if(not x[1][1][user]):
			n = 0
			d = 0
			k = 0
			for row in x[1][0]:
				if(row[2][user]):
					n += row[1] * row[2][user]
					d += row[1]
					k += 1
			if k >= 2: x[1][1][user] = n / d
	return (x[0], x[1][1])

rdd1 = rdd.map(json.loads)\
			.map(mapReviews)\
			.filter(lambda x: x[0][0] != '' and x[0][1] != '' )\
			.groupByKey()\
			.map(oneReview)			
rdd1 = rdd1.map(lambda x: (x[0][0], (x[0][1], x[1])))\
			.groupByKey()\
			.filter(filter1)

rdd1.persist()
# print(rdd1.take(5))
softwares = sc.broadcast(rdd1.map(lambda x: x[0])\
			.collect())

rdd1 = rdd1.flatMap(map)\
			.groupByKey()\
			.filter(filter2)
			
rdd1.persist()

users = sc.broadcast(rdd1.map(lambda x: x[0])\
					.collect())


userDict = {k:'' for k in users.value}

rdd1 = rdd1.flatMap(mapper)
rdd.unpersist()
rdd1.persist()

meanRDD = rdd1.map(lambda x: (x[0][1], (x[1], 1)))\
							.reduceByKey(lambda a, b: (a[0]+b[0], a[1]+b[1]))\
							.mapValues(lambda x: (x[0]/x[1]))
rdd2 = rdd1.map(lambda x: (x[0][1], (x[0][0], x[1]))).join(meanRDD)\
			.map(lambda x: (x[0], (x[1][0][0], x[1][0][1] - x[1][1])))\
			.groupByKey().map(vector)
			
rdd1.unpersist()
rdd2.persist()
result = rdd2.filter(lambda x: x[0] in list).collect()
rdd3 = rdd2.flatMap(simMap)\
			.filter(uncommon)\
			.map(simCal)\
			.filter(lambda x: x[1][1] > 0)\
			.groupByKey()\
			.map(neighbor).join(sc.parallelize(result)).map(recommend)

rdd2.unpersist()
print(rdd3.collect())
end_time = datetime.now()
print("Execution Time : ", end_time - start_time)