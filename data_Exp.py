# Importing required libraries
import os
os.getcwd()
from os import environ
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import csv
from bokeh.plotting import figure, output_file, show
from bokeh.plotting import figure
from bokeh.io import output_file, show
from sqlalchemy import create_engine, table, Integer, String, Text, DateTime, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, mapper
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

# Create database if not exist and get a connection to it
DB_engine = create_engine('sqlite:///LR.db', echo=True)
DB_connection = DB_engine.connect()
Session = sessionmaker(bind=DB_engine, autocommit=True, autoflush=True)
conn = sqlite3.connect('LR.db') 
cursor = conn.cursor()

#creating a class function: Functions are created within a class so as the objects can reused within the program which also allows easy debugging.
class fileManager(object):

    def readCSV(self,filename): # function used to read all the csv files
        #creating a try and catch error function on a read csv files
        try:
            _CSV = pd.read_csv(filename)
            return _CSV
        except:
            print("invalid file name")

    def openFile(self,path):
        #read_data=[]
        try:
            with open(path, 'r') as data:
                read_data = list(csv.reader(data,delimiter=","))
                read_data_list=np.array(read_data[1:],dtype=float)
            return read_data_list

        except:
            print("Unable to read file")

    def writeFile(self,path,list,col1,col2,col3,col4):
        try:
            myFile = open(path, 'w')
            writer = csv.writer(myFile)
            writer.writerow([col1, col2, col3,col4])
            for data_list in list:
                writer.writerow(data_list)
            myFile.close()

        except:#user defined exception handling error
            print("Unable to create file")
    
"""
Creating a class that holds methods to plot the 
graphs
"""
class plot_graph(object):

    #constructor of class plot_graph
    def __init__(self,path,ouput,x,y,xlabel,ylabel,Title):
        self.path = path
        self.output = ouput
        self.x = x
        self.y = y
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.Title = Title
    # method plot graph
    def plotgraph(self):
        try:
            df= pd.read_csv(self.path)
            x= df[self.x]
            y= df[self.y]
            output_file(self.output)
            fig1= figure(x_axis_label=self.xlabel,y_axis_label=self.ylabel,title=self.Title)
            fig1.circle(x,y)
            show(fig1)
        #system exception handling for debugging the actual error    
        except Exception as e:
            print(e)
 
# creating a main function
def main():
# Load data into base_exp.db
    fm=fileManager()
    train_df =fm.readCSV('train.csv')
    table_name = 'train_data'
    train_df.to_sql(table_name, DB_engine, if_exists='replace', index=False, chunksize=500)
    train_df.info()
    # Loading ideal dataset
    ideal_df = fm.readCSV('ideal.csv')
    table_name = 'ideal_data'
    ideal_df.to_sql(table_name, DB_engine, if_exists='replace', index=False, chunksize=500)
    #ideal_df.info()
    # Loading test dataset
   
#choosing the ideal functions
#declaration of global variables
    chosen_ideal=[]
    sorted_data=[]
    
    
# calculating the maximum and minimum of the deviation square
    cursor.execute('SELECT * from train_data')
    train_data = np.array(cursor.fetchall())
   
    cursor.execute('SELECT * from ideal_data')
    ideal_data = np.array(cursor.fetchall())
   
    for i in range (1,len(train_data[0,:]),1):
        sum_sqr_dev_list= []
        abs_dev_list=[]
        #if i!="x":
            #train_data=np.array(train_df.get(i)) #converting the columns of the train data to an array  
        for p in range(1,len(ideal_data[0,:]),1): # reading the number of columns in the ideal function
           # if p!="x":
                #ideal_data=np.array(ideal_df.get(p))# converting the selected column into an array
                # deviation=np.subtract(train_data,ideal_data)
                deviation=np.subtract(train_data[:,i],ideal_data[:,p]) # calculating the deviation
                max_abs_dev = np.amax(np.abs(deviation)) # calculating the maximum value of the deviation
                sqr_dev=np.square(deviation) # squaring the deviation
                sum_sqr_dev=np.sum(sqr_dev) # summing the deviation
                sum_sqr_dev_list.append(sum_sqr_dev) # adding the sum deviation to the deviation sum list
                abs_dev_list.append(max_abs_dev) # adding the maximum deviation to the maximum deviation list
        chosen_ideal_index=sum_sqr_dev_list.index(min(sum_sqr_dev_list)) # getting the index of the minimum sum deviation list
        
        # calculating the maximum absolute value of the chosen ideal index
        max_abs_chosen_ideal_index= abs_dev_list[chosen_ideal_index]
        chosen_ideal.append((chosen_ideal_index+1,max_abs_chosen_ideal_index))
        #print(chosen_ideal)
    
    read_data=fm.openFile("test.csv")
    read_data = read_data[np.argsort(read_data[:,0])] 
    
    for sd in range(len(read_data[:,0])):
        x_column_test=read_data[sd,0]
        y_column_test=read_data[sd,1]
        if x_column_test in ideal_data[:,0]:
            for ycol in range(4):
                y_column_ideal_index_data=np.where(ideal_data[:,0]==x_column_test)
                y_ideal_selected_value=ideal_data[y_column_ideal_index_data,chosen_ideal[ycol][0]]
                if abs(y_column_test-y_ideal_selected_value)<=np.sqrt(2)*chosen_ideal[ycol][1]:
                    sorted_data.append([x_column_test,y_column_test,chosen_ideal[ycol][0],float(abs(y_column_test-y_ideal_selected_value))])
    print(sorted_data)

 # calling the writeFile function in the class fileManager class to write the sorted data into a file
    fm.writeFile("D:\OLR\sorted_data.csv",sorted_data,'x_test', 'y_test', 'chosen_ideal_index','abs_dev')

 # reading the sorted csv file and writing it to the database
    chosenideal_df = fm.readCSV("D:\OLR\sorted_data.csv")
    table_name = 'sorted_data'
    chosenideal_df.to_sql(table_name, DB_engine, if_exists='replace', index=False, chunksize=500)
    chosenideal_df.info()
    
    #visualisation
# creating instance of the class
# plotting the train data set from y1-y4.
    graph1= plot_graph('train.csv',"TrainDataY1.html","x","y1","x-axis","y-axis","TrainSetY1")
    graph1.plotgraph()
    graph2= plot_graph('train.csv',"TrainDataY2.html","x","y2","x-axis","y-axis","TrainSetY2")
    graph2.plotgraph()
    graph3= plot_graph('train.csv',"TrainDataY3.html","x","y3","x-axis","y-axis","TrainSetY3")
    graph3.plotgraph()
    graph4= plot_graph('train.csv',"TrainDataY4.html","x","y4","x-axis","y-axis","TrainSetY4")
    graph4.plotgraph()
# plotting the test graph
    graphtest= plot_graph('test.csv',"TestData.html","x","y","x-axis","y-axis","TestSet")
    graphtest.plotgraph()
# plotting the chosen_ideal_functions
    graphchosen= plot_graph('sorted_data.csv',"ChosenData.html","x_test","y_test","x-axis","y-axis","ChosenSet_Yset")
    graphchosen.plotgraph()
    graphchosenii= plot_graph('sorted_data.csv',"ChosenData.html","x_test","chosen_ideal_index","x-axis","y-axis","ChosenSet_chosen_ideal_index")
    graphchosenii.plotgraph()
    graphchosenabsdev= plot_graph('sorted_data.csv',"ChosenData.html","x_test","abs_dev","x-axis","y-axis","ChosenSet_abs_dev")
    graphchosenabsdev.plotgraph()
    #print(graph1)
    #print(read_data_list)
    #for data_list in read_data:
       # print(data_list)

if __name__ == "__main__":
    main()


