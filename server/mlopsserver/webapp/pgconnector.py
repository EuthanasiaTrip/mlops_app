import psycopg2
import json
import uuid
import os

class DBConnector:
    def __init__(self):
        dirpath = os.path.dirname(os.path.realpath(__file__))
        connection_strings = dirpath + '/connectionstrings.json'
        with open(connection_strings) as f:
            self.connection_json = json.load(f)
    
    def execute_SQL(self, sql):    
        with psycopg2.connect(
            host=self.connection_json['Host'],
            database=self.connection_json['Database'],
            user=self.connection_json['User'],
            password=self.connection_json['Password']
        ) as conn:
            with conn.cursor() as curs:
                curs.execute(sql)
    
    def selectCols(self, colNames=None, tableName="PATIENTS_DATA_ALL", colsNotNull=[], limit=0, isDead=None):  
        if colNames:      
            formattedCols = "\", \"".join(colNames)
            sqlText = "select \"" + formattedCols + f"\" from \"{tableName}\""
        else:
            sqlText = f"select * from \"{tableName}\""
        if colsNotNull:
            colsNotNullFormatted = "\" is not null and \"".join(colsNotNull)
            sqlText += " where \"" + colsNotNullFormatted + "\" is not null "
            if isDead != None:
                sqlText += " and \"IsDead\" = " + str(isDead)
        if limit > 0 :
            sqlText += "limit " + str(limit)    
        # print(sqlText)  

        result = []
        with psycopg2.connect(
            host=self.connection_json['Host'],
            database=self.connection_json['Database'],
            user=self.connection_json['User'],
            password=self.connection_json['Password']
        ) as conn:
            with conn.cursor() as curs:
                curs.execute(sqlText)
                result = curs.fetchall()

        return result
    
    def validateKey(self, key):
        sqlText = f"select 1 from \"api_keys\" where \"key\" = '{key}' and \"active\" = 'true'"
        result = 0
        print(self.connection_json)
        with psycopg2.connect(
            host=self.connection_json['Host'],
            database=self.connection_json['Database'],
            user=self.connection_json['User'],
            password=self.connection_json['Password']
        ) as conn:
            with conn.cursor() as curs:
                curs.execute(sqlText)
                fetch = curs.fetchone()
                if fetch:
                    result = fetch[0]

        return result
    
    # obvious risk of sql injection
    def insert_new_data(self, input_data):
        cols = ', '.join([f"\"{key}\"" for key in input_data.keys()])
        sqlText = f"insert into \"PATIENTS_DATA_NEW\"(\"Id\", {cols}) values("
        sqlText += f"'{uuid.uuid4()}',"
        for key, value in input_data.items():
            if key == 'NumberIB' or key == 'NumberCard':
                value = f"'{value}'"
            if not value:
                value = "null"
            if key != list(input_data)[-1]:
                sqlText += f"{value},"
            else:
                sqlText += f"{value})"
        sqlText += " RETURNING \"Id\";"

        result = ''
        with psycopg2.connect(
            host=self.connection_json['Host'],
            database=self.connection_json['Database'],
            user=self.connection_json['User'],
            password=self.connection_json['Password']
        ) as conn:
            with conn.cursor() as curs:
                curs.execute(sqlText)
                result = curs.fetchone()[0]

        return result