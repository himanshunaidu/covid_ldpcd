#Utility methods for extraction of features from MySQL database
#import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import mysql.connector

from config import mydb
mycursor = mydb.cursor()


def stringToIntList(dense_result):
    #print(dense_result)
    dense = dense_result.split(' ')
    dense_map = map(float, dense)
    dense_list = list(dense_map)
    return dense_list

def tupleToList(tuple_list):
    class_list = []
    for (id, image_name, dis_class) in tuple_list:
        class_list.append(dis_class)
    return class_list

def extractFeatures(db, cursor, table):
    sql = f'Select id, image_name, image_ldp_path, class,'\
        f'CONVERT(dense USING utf8), CONVERT(dense_ldp USING utf8),'\
        f'CONVERT(dense_1 USING utf8), CONVERT(dense_ldp_1 USING utf8)'\
        f'FROM {table} order by id'
    
    cursor.execute(sql)
    myresult = cursor.fetchall()
    return myresult

def features(db, cursor, table):
    result = extractFeatures(db, cursor, table)
    print(len(result))
    #4, 5, 6, 7, 8, 9 are dense and dense_ldp features alternatively
    dense_info, dense_features = [], []
    for i in range(len(result)):
        #Create a placeholder each for each result row in dense_info and dense_features arrays,
        #then append info into the dense_info,
        dense_tuple = (result[i][0], result[i][1], result[i][3])
        #then go over the results and get the features, and insert into a tuple
        dense_feature = []
        for j in range(4, 8):
            dense_feature.extend(stringToIntList(result[i][j]))
        #Each tuple contains the id, image_path, class and feature array
        dense_info.append(dense_tuple)
        dense_features.append(dense_feature)
    
    return dense_info, dense_features 

if __name__ == '__main__':
    print(mydb)

    dense_train_info, dense_train_features = features(mydb, mycursor, 'train_coro_features4')
    print(dense_train_info[0], len(dense_train_features[0]))
    #dense_test_info, dense_test_features = features(mydb, mycursor, 'test_coro_features4')
    #print(dense_test_info[0], len(dense_test_features[0]))
    #dense_test_class = tupleToList(dense_test_info)
    #print(len(dense_test_class))