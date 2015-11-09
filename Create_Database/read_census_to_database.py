import pandas as pd
import psycopg2
import clean_data


def insert_into_table(create_data_table=False):

    count = 0
    column_list = []
    for chunk in pd.read_csv('../ss13pusa.csv', chunksize=5000, header=0):

        chunk = clean_data.clean_chunk(chunk)

        if count == 0 and create_data_table is True:
            create_table(chunk.columns)
            column_list += [col for col in chunk.columns if col not in column_list]

        for i in range(len(chunk)):
            column_list = insert_chunk_into_table(chunk, i, column_list)

        column_list += [col for col in chunk.columns if col not in column_list]

        if count == 50:
            break

        count += 1


def get_connection():

    try:
        conn = psycopg2.connect("dbname='dummy' user='dummy' host='/tmp/' password='password'")
    except:
        print "I am unable to connect to the database"

    return conn


def create_table(columns):

    conn = get_connection()
    cur = conn.cursor()

    sql = "CREATE TABLE test (%s);" % ','.join('%s FLOAT' % col for col in columns)

    cur.execute(sql)
    conn.commit()

    cur.close()
    conn.close()


def insert_chunk_into_table(chunk, i, column_list):
    '''
    if the column hasn't been seen yet it is added to the table
    insert all data into the updated table
    '''

    conn = get_connection()
    cur = conn.cursor()

    for col in chunk.columns:
        if col not in column_list:
            add_column_to_table(col)
            column_list.append(col)

    sql = "INSERT INTO test (%s)" % ','.join("%s" % col for col in chunk.columns)
    sql = sql + " VALUES (%s);" % ','.join('%s' % col for col in chunk.iloc[i])
    cur.execute(sql)

    conn.commit()
    cur.close()
    conn.close()

    return column_list


def add_column_to_table(column):
    '''
    Add dummy columns to table
    '''
    conn = get_connection()
    cur = conn.cursor()

    sql = 'ALTER TABLE test ADD COLUMN %s INTEGER;' % column
    cur.execute(sql)

    conn.commit()
    cur.close()
    conn.close()


def main():
    insert_into_table(create_data_table=True)


if __name__ == '__main__':
    main()
