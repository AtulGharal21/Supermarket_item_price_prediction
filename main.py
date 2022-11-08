from flask import Flask,request,render_template
import json
import pickle
import numpy as np
with open ("Model.pickle","rb") as f:
    model=pickle.load(f)
with open("column_list.json","r")as f:
    features_names=json.load(f)

app=Flask(__name__)

@app.route("/")
def welcome():
    return render_template("index.html")

@app.route("/prediction",methods=["POST"])
def prediction():
    data=request.form
    
    Item_Weight=float(data["Item_Weight"])
    Item_Fat_Content=int(data["Item_Fat_Content"])
    Item_Visibility=float(data["Item_Visibility"])
    Item_MRP=float(data["Item_MRP"])
    Outlet_Size=float(data["Outlet_Size"])
    Outlet_Location_Type=int(data["Outlet_Location_Type"])
    Outlet_Type=int(data["Outlet_Location_Type"])
    Outlet_age=int(data["Outlet_age"])
    Outlet_Identifier=data["Outlet_Identifier"]
    Item_Type=data["Item_Type"]
    Item_Identifier=data["Item_Identifier"]
    
    arr=np.zeros(len(features_names))
    arr[0]=Item_Weight
    arr[1]=Item_Fat_Content
    arr[2]=Item_Visibility
    arr[3]=Item_MRP
    arr[4]=Outlet_Size
    arr[5]=Outlet_Location_Type
    arr[6]=Outlet_Type
    arr[7]=Outlet_age
    
    text="Outlet_Identifier_"+Outlet_Identifier
    text_index=features_names.index(text)
    arr[text_index]=1
    text1="Item_Type_"+Item_Type
    text1_index=features_names.index(text1)
    arr[text1_index]=1
    text2="Item_Identifier_"+Item_Identifier
    text2_index=features_names.index(text2)
    arr[text2_index]=1
    
    # input_data=[Item_Weight,Item_Fat_Content,Item_Visibility,Item_MRP,Outlet_Size,Outlet_Location_Type,Outlet_Type,Outlet_age,Outlet_Identifier,Item_Type,Item_Identifier]
    pred=model.predict([arr])
    return render_template("index.html",result=pred)

if __name__=="__main__":    
    app.run(host="0.0.0.0",port=8080)