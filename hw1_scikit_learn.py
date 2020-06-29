import numpy as np 
import pandas as pd 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
from sklearn import preprocessing
      
#group members: Tushya Gautam

def preprocess_data(df):
    #preprocess the data to handle columns not in numeric format

    print(df)

    le = preprocessing.LabelEncoder()
    df['Occupied']=le.fit_transform(df.Occupied)
    df['Price']=le.fit_transform(df.Price)
    df['Music']=le.fit_transform(df.Music)
    df['Location']=le.fit_transform(df.Location)
    df['VIP']=le.fit_transform(df.VIP)
    df['Favorite Beer']=le.fit_transform(df['Favorite Beer'])
    df['Enjoy']=le.fit_transform(df.Enjoy)
    #print(df)

    return df

# Function to perform training with entropy. 
def train_using_entropy(X_train, X_test, y_train): 
  
    # Decision tree classifier with entropy with max depth set to 3
    tree_model = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth = 3, min_samples_leaf = 5) 
  
    # Performing training 
    tree_model.fit(X_train, y_train) 
    return tree_model 
  
  
# Function to make predictions 
def prediction(X_test, tree_obj): 
  
    y_pred = tree_obj.predict(X_test) 
    return y_pred 
      
# Function to calculate accuracy 
def cal_accuracy(y_test, y_pred): 
      
    print("Confusion Matrix: ", 
        confusion_matrix(y_test, y_pred)) 
      
    print ("Accuracy : ", 
    accuracy_score(y_test,y_pred)*100) 
      
    print("Report : ", 
    classification_report(y_test, y_pred)) 
  
# Driver code 
def main(): 
      

    #taking in the input
    col_names = ['Occupied', 'Price', 'Music', 'Location', 'VIP', 'Favorite Beer', 'Enjoy']
    #df = pd.read_csv("dt_data.csv", skiprows = 1, names = col_names, skipinitialspace = True)

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

    #print(input_list)

    df = pd.DataFrame(input_list, columns = col_names)

    #Take the new row of data and add it to the dataframe so we can predict its label later
    predict_data = ['Moderate', 'Cheap', 'Loud', 'City-Center', 'No', 'No']
    col_names_pred = col_names[0:6]
    predict_df = pd.DataFrame([predict_data], columns = col_names_pred)

    df = df.append(predict_df, ignore_index = True, sort = False)
    df = preprocess_data(df)

    #X contains the attribute variables and Y contains the label
    X = df.values[:, 0:5]
    Y = df.values[:, 6]

    #splitting the model into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.03, random_state = 100, shuffle = False) 
    #print(X_test)
    #print(y_test)

    #calling the function where we train the model using entropy
    tree_model = train_using_entropy(X_train, X_test, y_train) 
      
    #print("Results Using Entropy:") 
    # Prediction for the new row using entropy 
    y_pred_entropy = prediction(X_test, tree_model) 

    #Getting back the result in numeric form and converting to the desired format
    print("Result: ")
    print(predict_data)
    if(y_pred_entropy[0]==2):
        print "Yes"
    else:
        print "No"
    #cal_accuracy(y_test, y_pred_entropy) 
      
      
# Main function 
if __name__=="__main__": 
    main() 