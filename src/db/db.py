import sqlite3 as sql 
import datetime as dt


def db_creation():
    conn = sql.connect('db/user_history.db')
    cur = conn.cursor()

    cur.executescript('''

        CREATE TABLE IF NOT EXISTS AlgorithmParams(
            id INTEGER PRIMARY KEY,
            hypothesis TEXT NOT NULL,
            cost_function TEXT NOT NULL,
            reqularization TEXT NOT NULL,
            scalling_function TEXT NOT NULL,
            reg_coef REAL NOT NULL,
            learning_rate REAL NOT NULL,
            eps REAL NOT NULL,
            max_num_itter INT NOT NULL,
            weights TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS UserHistory(
            id INTEGER PRIMARY KEY,
            algorithm_name TEXT NOT NULL,
            execution_time TEXT NOT NULL,
            ins_date timestamp NOT NULL,
            param_id INT NOT NULL,
            FOREIGN KEY (param_id)
                REFERENCES AlgorithmParams(param_id)
        );

    ''')

    conn.commit()
    conn.close()

def db_clean():
    conn = sql.connect('db/user_history.db')
    cur = conn.cursor()

    cur.executescript(''' 
            DELETE FROM UserHistory;
            DELETE FROM AlgorithmParams;
    ''')

    conn.commit()
    conn.close()


def db_insert(h, properties, excecution_time, choice):
    conn = sql.connect('db/user_history.db')
    cur = conn.cursor()
    
    algorithm_param = (choice.hypothesis, choice.cost_function , choice.regularization , choice.scaler , properties.reg_coef, 
               properties.alpha, properties.eps, properties.max_num_itter , ', '.join(str(x) for x in h.weight))

    script_1 = '''
        INSERT INTO AlgorithmParams
        (hypothesis, cost_function, reqularization, scalling_function, reg_coef, learning_rate, eps, max_num_itter, weights)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
    '''
    cur.execute(script_1, algorithm_param)
    last_ins_id = cur.lastrowid

    user_history = (choice.modification, excecution_time, dt.datetime.now(tz=None), last_ins_id)

    script_2 = '''
        INSERT INTO UserHistory (algorithm_name, execution_time, ins_date, param_id)
        VALUES(?, ?, ?, ?);
    '''
    cur.execute(script_2, user_history)

    conn.commit()
    conn.close()

def db_select():
    conn = sql.connect('db/user_history.db')
    cur = conn.cursor()

    cur.execute('''SELECT uh.algorithm_name, uh.execution_time, uh.ins_date, alparam.hypothesis, alparam.cost_function,
                    alparam.reqularization, alparam.scalling_function, alparam.reg_coef, alparam.learning_rate, 
                    alparam.eps, alparam.max_num_itter, alparam.weights
                    FROM AlgorithmParams as alparam, UserHistory as uh
                    WHERE alparam.id == uh.param_id;''')
    rows = cur.fetchall()

    conn.commit()
    conn.close()

    return rows
