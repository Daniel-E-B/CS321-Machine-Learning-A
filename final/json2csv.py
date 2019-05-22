import csv
import json
import os

important_things = ["title", "description", "summary", "score", "price", "developer", "genre", "genreId", "minInstalls"]

c = csv.writer(open("appdata.csv", "w"))
c.writerow(important_things)

for filename in os.listdir(os.fsencode("./appdata/")):
    try:
        j = json.loads(open("./appdata/"+filename.decode(), "r").read())
        c.writerow([
           j["title"],
            j["description"],
            j["summary"],
            j["score"],
            j["price"],
            j["developer"],
            j["genre"],
            j["genreId"],
            j["minInstalls"]
       ])
    except:
        print(filename.decode() + " failed")