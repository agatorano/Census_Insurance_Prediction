import pandas as pd
import csv
import psycopg2

bad_col = ["WKHP", "PUMA", "RAC3P", "POWSP", "POBP", "OCCP",
           "LANP", "INDP", "FOD1P", "FOD2P", "ANC2P", "ANC1P", "CITWP",
           "RT", "SPORDER", "ADJINC", "INTP", "JWRIP", "JWTR", "MARHYP",
           "NWRE", "RELP", "ANC", "YOEP", "DRIVESP", "JWAP", "JWDP",
           "MIGPUMA", "MIGSP", "NAICSP", "POWPUMA", "SFN", "SFR",
           "SOCP", "FAGEP", "FANCP", "FCITP", "FCITWP", "FCOWP",
           "FDDRSP", "FDEARP", "FDEYEP", "FDISP", "FDOUTP",
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

id_col = ["SERIALNO"]

num_col = ["AGEP", "PWGTP", "RETP", "SEMP", "SSIP", "SSP",
           "WAGP", "PERNP", "PINCP", "POVPIP", "OIP", "PAP"]

flag_col = ["FPOWSP", "FPRIVCOVP", "FHINS1P", "FHINS2P", "FHINS3C",
            "FHINS3P", "FHINS4C", "FHINS4P", "FHINS5C", "FHINS5P",
            "FHINS6P", "FHINS7P"]

health_col = ["HIVOC", "PRIVCOV", "PUBCOV", "HINS1", "HINS2",
              "HINS3", "HINS4", "HINS5", "HINS6", "HINS7"]


def create_table():

    '''
    create the postgres table we will inser the table into
    '''

    try:
        conn = psycopg2.connect("dbname='holder' user='holder' host='ip' password='psswd'")
    except:
        print "I am unable to connect to the database"

    cur = conn.cursor()

    with open("../ss13pusa.csv", "rb") as f:
        reader = csv.reader(f)
        columns = reader.next()

    sql = "CREATE TABLE test (%s);" % ','.join('%s INTEGER' % col for col in columns)

    cur.execute(sql)
    conn.commit()

    cur.close()
    conn.close()


def insert_by_chunk():
    pass


def clean_chunk(chunk):
    '''
    run data cleaning functions
    '''
    chunk = filter_data(chunk)
    chunk = create_dummy_columns(chunk)
    chunk = standardize_num_columns(chunk)

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
    pre = {col: col for col in chunk[cat_col].columns}

    chunk[cat_col] = chunk[cat_col].applymap(lambda x: str(x) if x >= 0 else x)

    dum_col = list(pd.get_dummies(chunk, pre, dummy_na=True).columns)
    chunk[dum_col] = pd.get_dummies(chunk, pre, dummy_na=True)
    chunk.drop(cat_col, 1, inplace=True)

    return chunk


def filter_data(chunk):
    '''
    remove columns that are difficult to learn from or are not appropriate
    remove people under the age of 18, who are required to have their parents
    insurance plans
    '''
    chunk = chunk.drop(bad_col+flag_col+id_col, axis=1)
    chunk = chunk[chunk.AGEP > 18]

    return chunk


def main():
    insert_by_chunk()


if __name__ == '__main__':
    main()
