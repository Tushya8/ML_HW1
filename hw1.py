import numpy as np
import pandas as pd
eps = np.finfo(float).eps
from numpy import log2 as log
import pprint

#group members: Tushya Gautam

def input_formatting(col_names):

  #------------------------alternate way of reading the input-------------------------
  input_list = []
  with open("dt_data.txt", 'r') as input_file_obj:
  	#input_list.append(input_file_obj.readline()[1:-2])
  	input_file_obj.readline()
  	line = input_file_obj.readline()
  	count = 0
  	while line:
  		line = input_file_obj.readline()[4:-2]
  		words = line.split(", ")
  		input_list.append([])
  		for word in words:
  			input_list[count].append(word)
  		count=count+1
  		#input_list.append([line[3:-2]])
  	input_list.pop()

  df = pd.DataFrame(input_list, columns = col_names)
  print(df)
  return df
#------------------------------------------------------------------------------------

def target_var_entropy(df):
    label = df.keys()[-1]   #retrieving the label column
    entropy = 0
    values = df[label].unique() #finding the number of unique label values
    for value in values:
        fraction = df[label].value_counts()[value]/len(df[label])
        entropy += -fraction*np.log2(fraction)
    return entropy
  
  
def find_entropy_attribute(df,attribute):
  label = df.keys()[-1]   #retrieving the label column
  target_variables = df[label].unique()  #This gives all 'Yes' and 'No'
  variables = df[attribute].unique()    #This gives different features in that attribute (like 'Hot','Cold' in Temperature)
  entropy2 = 0
  for variable in variables:
      entropy = 0
      for target_variable in target_variables:
          num = len(df[attribute][df[attribute]==variable][df[label] ==target_variable])
          den = len(df[attribute][df[attribute]==variable])
          fraction = num/(den+eps)
          entropy += -fraction*log(fraction+eps)
      fraction2 = den/len(df)
      entropy2 += -fraction2*entropy
  return abs(entropy2)


def find_winner_attribute(df):
    Entropy_att = []
    IG = []
    for key in df.keys()[:-1]:
#         Entropy_att.append(find_entropy_attribute(df,key))
        #calculate information gain for each attribute
        IG.append(target_var_entropy(df)-find_entropy_attribute(df,key))
        #print("Attribute: ", key)
        #print("IG: ", target_var_entropy(df)-find_entropy_attribute(df,key))
    return df.keys()[:-1][np.argmax(IG)]
  
  
def get_subtable(df, node, value):
  return df[df[node] == value].reset_index(drop=True)


def buildTree(df, depth, tree=None): 

    #This function is called recursively to build the tree
    #We stop once we reach a pure class or the max depth
    Class = df.keys()[-1]   
    
    #Here we build our decision tree

    #Get attribute with maximum information gain
    node = find_winner_attribute(df)

    #Get distinct values of the attribute. For example, High, Moderate and Low for Occupied attribute
    attValues = np.unique(df[node])

    print(attValues)
    
    print("Winner: ", node)

    if tree is None:                    
        tree={}
        tree[node] = {}

    for value in attValues:
        
        subtable = get_subtable(df,node,value)
        clValue,counts = np.unique(subtable['Enjoy'],return_counts=True)                        
        
        if len(counts)==1 or depth > 1:#Checking purity of subset
            tree[node][value] = clValue[0]                                                    
        else:        
            tree[node][value] = buildTree(subtable, depth+1) #Calling the function recursively 
                   
    return tree

def predict(row,tree):
    #This function is used to predict for any input variable 
    
    #Recursively we go through the tree that we built earlier

    for nodes in tree.keys():        
        
        value = row[nodes]
        print("Attribute picked: ", value)
        tree = tree[nodes][value]
        prediction = 0
            
        if type(tree) is dict:
            prediction = predict(row, tree)
        else:
            prediction = tree
            break;                            
        
    return prediction

#Take the input, format it, and build the tree
col_names = ['Occupied', 'Price', 'Music', 'Location', 'VIP', 'Favorite Beer', 'Enjoy']
df = input_formatting(col_names)
tree = buildTree(df, 0)
pprint.pprint(tree)

#Predict the label for the new row of data
predict_data = ['Moderate', 'Cheap', 'Loud', 'City-Center', 'No', 'No']
col_names_pred = col_names[0:6]
predict_df = pd.DataFrame([predict_data], columns = col_names_pred)
print(predict_df)
new_row = predict_df.iloc[0]
print(new_row)


prediction = predict(new_row, tree)
print(prediction)