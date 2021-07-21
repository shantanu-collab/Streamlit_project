from os import write
from altair.vegalite.v4.schema.core import Axis
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from streamlit.proto.Selectbox_pb2 import Selectbox
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Machine Learning Deployment")

img = Image.open(r"C:\Users\Shantanu\Desktop\robo.jpg")
st.image(img,use_column_width=True)


def main():
    activities = ['EDA','visualisation','model','About Creator']
    option = st.sidebar.selectbox('Select option',activities)
    if option == 'EDA':
        st.subheader("Exploratory Data analysis")
        data = st.file_uploader("Upload dataset",type=['csv','xlsx','txt','json'])
        st.success("Data successfully loaded")
        if data is not None:
            df  = pd.read_csv(data)
            st.dataframe(df.head(50))

            if st.checkbox("Display shape"):
                st.write(df.shape)
            if st.checkbox("Display columns"):
                st.write(df.columns)
            if st.checkbox("select multiple columns"):
                selected_columns = st.multiselect("select preferred solumns",df.columns)
                df1 = df[selected_columns]
                st.dataframe(df1)
            if st.checkbox("Display summary"):
                st.write(df1.describe().T)

            if st.checkbox("Display null values"):
                st.write(df.isnull().sum())
            if st.checkbox("Data types"):
                st.write(df.dtypes)

            if st.checkbox("correlation matrix"):
                st.write(df.corr())
                
    elif option== 'visualisation':
        st.subheader("Data Visualisation")
        data = st.file_uploader("Upload dataset",type=['csv','xlsx','txt','json'])
        st.success("Data successfully loaded")
        if data is not None:
            df  = pd.read_csv(data)
            st.dataframe(df.head(50))

            if st.checkbox("Select Multiple columns to plot"):
                selected_columns = st.multiselect("select your preferred column", df.columns)
                df1 =df[selected_columns]
                st.dataframe(df1)
            if st.checkbox("Display Heatmap"):
                st.write(sns.heatmap(df1.corr(),vmax=1,cmap = "viridis",annot=True))
                st.pyplot()
            if st.checkbox("Display pairplot"):
                st.write(sns.pairplot(df1,diag_kind="kde" ))
                st.pyplot()
            if st.checkbox("Display Pie Chart"):
                selected_column = st.selectbox("Select the column",df.columns)
                count = df[selected_column].value_counts()
                plt.pie(count,labels=count.index,autopct="%1.1f%%")
                st.pyplot()

    elif option == 'model':
        st.subheader("Model Building")
        data = st.file_uploader("Upload dataset",type=['csv','xlsx','txt','json'])
        st.success("Data successfully loaded")
        if data is not None:
            df  = pd.read_csv(data)
            st.dataframe(df.head(50))
        
        if st.checkbox("Select multiple columns"):
            new_data = st.multiselect("Select your preferred columns", df.columns)
            df1 = df[new_data]
            st.dataframe(df1)
            
            X = df1.iloc[:,0:-1]
            Y = df1.iloc[:,-1]
        
        seed = st.sidebar.slider('Seed',1,200)
        classifier_name = st.sidebar.selectbox("Select your preferred classifier",("KNN","SVM","Logistic regression","Naive Bayes","Decision Tree"))

        def add_parameter(name_of_clf):
            param = {}
            if name_of_clf == "SVM":
                C = st.sidebar.slider("C",0.01,15.0)
                param["C"]= C
                return param
            if name_of_clf == "KNN":
                K = st.sidebar.slider("K",1,15)
                param["K"] = K
                return param

        params = add_parameter(classifier_name)

        def get_classifier(name_of_clf,params):
            clf = None
            if name_of_clf == "SVM":
                clf = SVC(C = params["C"])
            elif name_of_clf == "KNN":
                clf = KNeighborsClassifier(n_neighbors=params["K"])
            elif name_of_clf == "Logistic regression":
                clf = LogisticRegression()
            elif name_of_clf == "Decision Tree":
                clf = DecisionTreeClassifier()
            elif name_of_clf == "Naive Bayes":
                clf = GaussianNB()
            else:
                st.warning("Select your choice of algorithm")
            return clf
            

        clf = get_classifier(classifier_name,params)
        x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=seed)
        clf.fit(x_train,y_train)
        y_pred = clf.predict(x_test)
        st.write("The predicted values are",y_pred)
        st.write("The accuracy of the model is",clf.score(x_test,y_test))
    
    elif option == "About Creator":
        st.balloons()
        st.title("Hello! My name is Shantanu, this is an interactive web page where anyone can import a dataset and see all the data analytics and ML model implementation result on the fly.")







    

   




if __name__ == "__main__":
    main()


  
