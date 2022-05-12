#!/bin/sh
# FOR DEPLOYMENT, Port=80, for development port=8000
env FLASK_APP=start.py flask run --port=80


# CONVERT DB TO CSV
# >sqlite3 annotations.db
# sqlite> .headers on
# sqlite> .mode csv
# sqlite> .output data.csv
# sqlite> SELECT * FROM annotation;
# sqlite> .quit
