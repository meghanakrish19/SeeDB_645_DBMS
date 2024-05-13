import psycopg2
from scipy.stats import entropy
import pandas as pd

class DB_Functions:
    def connect_database( self , db_name , user ):
        '''
        Connects to the given database
        Returns a dict:
        conn       : db connection object
        count      : (int)
        '''
        conn = psycopg2.connect( database = db_name , user = user , password = "Seafoam@6613" ,
        host = "localhost" ,  port = "5432" )
        return( conn )

    def fetch_data( self , conn , query , return_df = False ):
        if return_df == True:
            return pd.read_sql_query( query , conn , params = None )
        cur = conn.cursor()
        cur.execute( query )
        rows = cur.fetchall()
        return rows
    
    

