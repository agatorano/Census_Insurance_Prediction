import pandas as pd
import psycopg2


def insert_by_chunk():

    try:
        conn = psycopg2.connect("dbname='holder' user='holder' host='ip' password='psswd'")
    except:
        print "I am unable to connect to the database"

    cur = conn.cursor()
    data = pd.DataFrame()
    for chunk in pd.read_csv('ss13pusa.csv',chunksize=1000,header=0):
        if i ==5:
            break
        data = pd.concat([data,chunk])
        i+=1



def main():
    insert_by_chunk()


if __name__ == '__main__':
    main()
