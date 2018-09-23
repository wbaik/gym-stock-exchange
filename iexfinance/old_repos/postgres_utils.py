import pandas as pd
import psycopg2
from sqlalchemy import create_engine


class PostgresPandas:

    def __init__(self):
        self.pwd = self._get_password()
        self.conn, self.engine = self.get_conn_and_engine()

    def __del__(self):
        self.conn.close()

    def _get_password(self):
        return open('./pwd', 'r').readline().strip()

    def run_query(self, query):
        return pd.read_sql_query(query, con=self.engine)

    def create_tables(self, *tickers):

        def lower_cased_columns(df):
            lowered = [i.lower() for i in df.columns]
            df.set_axis(lowered, axis=1, inplace=True)
            return df

        for ticker in tickers:
            try:
                ticker = ticker.lower()
                path = 'Data/Stocks/{}.us.txt'.format(ticker)
                ticker_result = pd.read_csv(path)
                ticker_result = lower_cased_columns(ticker_result)
                ticker_result.to_sql(ticker, self.engine, if_exists='fail')
            except ValueError:
                print('*** Database for the ticker: {} already exist! ***'.format(ticker))
            except FileExistsError:
                print('*** File for the ticker: {} does not exist! ***'.format(ticker))
            except:
                print('*** Unkown Error in processing {}'.format(ticker))

    def get_tables(self, *tickers):
        ret = {}
        for ticker in tickers:
            try:
                ticker = ticker.lower()
                ret[ticker] = pd.read_sql_table(ticker, self.engine)
            except:
                print('*** Exception occurred for {} ***'.format(ticker))
        return ret

    def get_connection(self):
        try:
            conn = psycopg2.connect(
                'dbname=stock_data password={}'.format(self.pwd))
            conn.autocommit = True
            return conn

        except ConnectionError:
            print('*** Unable to connect to the database ***')

        except:
            print('*** Unknown Error ***')

    def get_engine(self, conn):
        try:
            engine = create_engine(
                'postgresql://wbaik:{}@localhost:5432/stock_data'.format(self.pwd))
            return engine

        except ValueError:
            conn.close()

    def get_conn_and_engine(self):
        conn = self.get_connection()
        engine = self.get_engine(conn)

        return conn, engine


