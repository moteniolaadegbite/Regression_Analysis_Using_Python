from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String
from sqlalchemy.exc import SQLAlchemyError

try:
    engine = create_engine('sqlite:///test.db', echo = False)
    meta = MetaData()

    students = Table(
    'Personal_info', meta, 
    Column('PID', Integer, primary_key = True), 
    Column('name', String), 
    Column('lastname', String),
    )

    Courses = Table(
    'Courses', meta, 
    Column('CID', Integer, primary_key = True), 
    Column('Department', String), 
    Column('Course', String),
    )


    meta.create_all(engine)

    print("Database created with tables")
except SQLAlchemyError as error:
    print(error)

