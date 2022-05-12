# Annotation Tool for breast cancer histopathology reports

## Install Dependencies:

apt-get update && apt-get install -y python3 \ python3-pip


pip3 install --upgrade pip


pip3 install -r requirements.txt


## Requirements:
	
	1) The input database to be loaded must be stored as data/mammo_path_enc_database.xlsx and must have the following properties (fields)
		1) EMPI_ENC : Unique id 
		2) ACC_NBR_ENC : Unique id
		3) PART_DESIGNATOR : Partial Descriptor - Used to better display
		4) PART_FINAL_TEXT : Text to be annotated - This is either Part A, B, or C 

	2) Database -  The annoations will be stored in a sqlite database file (annotations.db) inside of folder data/
		1) Open terminal and type: sqlite3 data/annotations.db
		2) Create users table: CREATE TABLE user ( email, password, authenticated,qc );
		3) Create annotation table: CREATE TABLE annotation ( id INTEGER PRIMARY KEY, identifier, annotation,user );
		4) Insert new user: insert into user values('admin', 'admin',1,0);
		5) .quit

## How to Run:

1) First, start the application using ./start.sh
2) Each annotation will be saved on the table annotation

## Export Annotations to CSV
1) First, open the databse: sqlite3 data/annotations.db  
2) Export by running: 
	* .headers on
	* .mode csv
	* .output data/data.csv
	* select id, identifier, annotation,user from annotation;
	* .quit

