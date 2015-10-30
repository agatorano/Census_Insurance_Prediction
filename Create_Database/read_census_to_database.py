import pandas as pd
import csv
import psycopg2

bad_col = ["PRIVCOV", "PUBCOV", "HINS1", "HINS2", "HINS3", "HINS4", "HINS5",
           "HINS6", "HINS7", "WKHP", "PUMA", "RAC3P", "POWSP", "POBP", "OCCP",
           "LANP", "INDP", "FOD1P", "FOD2P", "ANC2P", "ANC1P", "CITWP",
           "RT", "SPORDER", "ADJINC", "INTP", "JWRIP", "JWTR", "MARHYP",
           "NWRE", "RELP", "ANC", "YOEP", "DRIVESP", "JWAP", "JWDP",
           "MIGPUMA", "MIGSP", "NAICSP", "POWPUMA", "SFN", "SFR", "SERIALNO"
           "SOCP", "FAGEP", "FANCP", "FCITP", "FCITWP", "FCOWP",
           "FPUBCOVP", "FPOWSP", "FPRIVCOVP", "FHINS1P", "FHINS2P", "FHINS3C",
           "FHINS3P", "FHINS4C", "FHINS4P", "FHINS5C", "FHINS5P", "FHINS6P",
           "FDDRSP", "FDEARP", "FDEYEP", "FDISP", "FDOUTP", "FHINS7P"
           "FDPHYP", "FDRATP", "FDRATXP", "FDREMP", "FENGP",
           "FESRP", "FFERP", "FFODP", "FGCLP", "FGCMP", "FGCRP",
           "FHISP", "FINDP", "FINTP", "FJWDP", "FJWMNP", "FJWRIP",
           "FJWTRP", "FLANP", "FLANXP", "FMARP", "FMARHDP", "FMARHMP",
           "FMARHTP", "FMARHWP", "FMARHYP", "FMIGP", "FMIGSP",
           "FMILPP", "FMILSP", "FOCCP", "FOIP", "FPAP", "FPERNP",
           "FPINCP", "FPOBP", "FRACP", "FRELP", "FRETP", "FSCHGP",
           "FSCHLP", "FSCHP", "FSEMP", "FSEXP", "FSSIP", "FSSP",
           "FWAGP", "FWKHP", "FWKLP", "FWKWP", "FWRKP", "FYOEP",
           "pwgtp1", "pwgtp2", "pwgtp3", "pwgtp4", "pwgtp5",
           "pwgtp6", "pwgtp7", "pwgtp8", "pwgtp9", "pwgtp10",
           "pwgtp11", "pwgtp12", "pwgtp13", "pwgtp14", "pwgtp15",
           "pwgtp16", "pwgtp17", "pwgtp18", "pwgtp19", "pwgtp20",
           "pwgtp21", "pwgtp22", "pwgtp23", "pwgtp24", "pwgtp25",
           "pwgtp26", "pwgtp27", "pwgtp28", "pwgtp29", "pwgtp30",
           "pwgtp31", "pwgtp32", "pwgtp33", "pwgtp34", "pwgtp35",
           "pwgtp36", "pwgtp37", "pwgtp38", "pwgtp39", "pwgtp40",
           "pwgtp41", "pwgtp42", "pwgtp43", "pwgtp44", "pwgtp45",
           "pwgtp46", "pwgtp47", "pwgtp48", "pwgtp49", "pwgtp50",
           "pwgtp51", "pwgtp52", "pwgtp53", "pwgtp54", "pwgtp55",
           "pwgtp56", "pwgtp57", "pwgtp58", "pwgtp59", "pwgtp60",
           "pwgtp61", "pwgtp62", "pwgtp63", "pwgtp64", "pwgtp65",
           "pwgtp66", "pwgtp67", "pwgtp68", "pwgtp69", "pwgtp70",
           "pwgtp71", "pwgtp72", "pwgtp73", "pwgtp74", "pwgtp75",
           "pwgtp76", "pwgtp77", "pwgtp78", "pwgtp79", "pwgtp80"]


num_col = ["AGEP", "PWGTP", "RETP", "SEMP", "SSIP", "SSP",
           "WAGP", "PERNP", "PINCP", "POVPIP", "OIP", "PAP"]


health_col = ["HIVOC"]


def insert_into_table(create_data_table=False):

    count = 0
    column_list = []
    for chunk in pd.read_csv('ss13pusa.csv', chunksize=5000, header=0):

        chunk = clean_chunk(chunk)

        if count == 0 and create_data_table is True:
            create_table(chunk.columns)
            column_list += [col for col in chunk.columns if col not in column_list]

        for i in range(len(chunk)):
            column_list = insert_chunk_into_table(chunk, i, column_list)

        column_list += [col for col in chunk.columns if col not in column_list]

        if count == 1:
            break

        count += 1


def get_connection():

    try:
        conn = psycopg2.connect("dbname='agatorano' user='agatorano' host='/tmp/' password='529382Ag'")
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


def clean_chunk(chunk):
    '''
    run data cleaning functions
    '''
    chunk = chunk.drop(bad_col, axis=1)
    chunk = chunk[chunk.AGEP > 18]
    chunk = create_dummy_columns(chunk)
    chunk = standardize_num_columns(chunk)
    chunk = chunk.dropna(axis=0)

    return chunk


def standardize_num_columns(chunk):
    '''
    standardize all numerical columns to their z_score
    '''
    for col in num_col:
        chunk[col] = (chunk[col] - chunk[col].mean())/chunk[col].std()

    return chunk


def create_dummy_columns(chunk):
    '''
    create dummy variables for the categorical columns
    '''
    cat_col = [col for col in chunk.columns if col not in num_col]
    pre = {col: col for col in cat_col}

    chunk[cat_col] = chunk[cat_col].applymap(lambda x: str(int(x)) if x >= 0 else str(x))

    dum_df = pd.get_dummies(chunk[cat_col], pre)
    dum_col = list(dum_df.columns)

    chunk[dum_col] = dum_df
    chunk.drop(cat_col, 1, inplace=True)

    return chunk


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
