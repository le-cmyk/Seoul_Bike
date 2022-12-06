# Library 
from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error
from sklearn import tree
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
import plotly.express as px
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
import plotly.graph_objects as go
# Library 

##Insertion des données et modification des colonnes##
df = pd.read_csv("SeoulBikeData.csv", encoding="ISO_8859-1")
quant96  = df["Rented Bike Count"].quantile(0.96)
df = df[(df["Rented Bike Count"] < quant96)]
#preparation
df['Date_Format'] = pd.to_datetime(df.Date, format = "%d/%m/%Y")
df["Year"] = df["Date_Format"].dt.year
df['Months'] = df['Date_Format'].dt.month 
df['Holiday'] = df['Holiday'].replace({'No Holiday': 0, 'Holiday':1})
df['Functioning Day'] = df['Functioning Day'].replace({'No': 0, 'Yes':1})
df['Seasons'] = df['Seasons'].replace({'Winter': 0, 'Spring':1, 'Summer':2, 'Autumn':3})
df['test'] = df["Year"] + df["Months"]
del df['Date_Format']
del df['Date']
##Insertion des données et modification des colonnes##
 
## Preprocessing Data for models ##

target=df["Rented Bike Count"].copy()
data = df.drop(columns=["Rented Bike Count"])

data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=42)

categorical_columns_selector = selector(dtype_include=object)
categorical_columns = categorical_columns_selector(data)
categorical_preprocessor = OrdinalEncoder(handle_unknown="use_encoded_value",
                                          unknown_value=-1)

preprocessor = ColumnTransformer([
    ('cat_preprocessor', categorical_preprocessor, categorical_columns)],
    remainder='passthrough', sparse_threshold=0)
## Preprocessing Data for models ##

def index1(request):
    template = loader.get_template('template1.html')
    context = {                    
        }
    return HttpResponse(template.render(context,request))

def acc_model(model):
    mod = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier",
     model)])

    mod.fit(data_train, target_train)
    pred=mod.predict(data_test)
    return mean_absolute_error(target_test, pred)

def train_model(model):
    mod = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier",
     model)])
    mod.fit(data_train, target_train)
    pred=mod.predict(data_test)
    return mod


def HistgradientBoosting(request):
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",
         HistGradientBoostingRegressor(random_state=42, max_leaf_nodes=9, learning_rate=0.4))])
    model
    model.fit(data_train, target_train)
    pred_linsvc = model.predict(data_test)
    a = mean_absolute_error(target_test, pred_linsvc)
    return HttpResponse(a)

def BayesianRidgeModel(request):
    BayesianRidge_acc=acc_model(linear_model.BayesianRidge())
    return HttpResponse(BayesianRidge_acc)

def DecisionTreeRegressorModel(request):
    DecisionTreeRegressor_acc=acc_model(tree.DecisionTreeRegressor(max_depth=1))
    return HttpResponse(DecisionTreeRegressor_acc)

def Pieplot(request):
    df2 = df.groupby(by=["Months"] )['Rented Bike Count'].mean().reset_index(name='nbbikeloué')
    fig = px.pie(df2, values='nbbikeloué', names= "Months", title='Months')
    plot2_html = fig.to_html(full_html =False , default_height = 500, default_width = 700)  
    return plot2_html

def Barplot(request):
    dfg = df.groupby(['Hour'])['Rented Bike Count'].mean()
    fig = px.bar(x=dfg.index, y=dfg)
    plot2_html = fig.to_html(full_html =False , default_height = 500, default_width = 700)
    return plot2_html

def Matricecorrelation(request):
    fig = px.imshow(df.corr(), text_auto=True)
    plot2_html = fig.to_html(full_html =False , default_height = 500, default_width = 700)
    return plot2_html

def GradientBoosting(request):
    model = GradientBoostingRegressor(learning_rate = 0.01, max_depth=7,n_estimators=1000,subsample=0.6)
    trainmodel = train_model(model)
    pred = trainmodel.predict(data_test)
    targ = target_test[0:100]
    y= pred[0:100]
    x_ax = range(len(targ))
    fig1 = px.scatter(x= x_ax,y= targ,title="a").update_traces(marker=dict(color='red'))
    fig2 = px.line(x= x_ax,y= y,title = "a").update_traces(marker=dict(color='yellow'))
    fig3 = go.Figure(data=fig2.data+ fig1.data , layout=go.Layout(
        title=go.layout.Title(text="Prediction of the Gradient Boosting Model (blue) and present values (rouge)")))
    plot2_html = fig3.to_html(full_html =False , default_height = 500, default_width = 700)
    return plot2_html

def BayesianRidge(request):
    model = linear_model.BayesianRidge(alpha_1 = 1.e-7,lambda_1 = 1.e-5, n_iter=100)
    trainmodel = train_model(model)
    pred = trainmodel.predict(data_test)
    targ = target_test[0:100]
    y= pred[0:100]
    x_ax = range(len(targ))
    
    fig1 = px.scatter(x= x_ax,y= targ,title="a").update_traces(marker=dict(color='red'))
    fig2 = px.line(x= x_ax,y= y,title = "a").update_traces(marker=dict(color='yellow'))
    fig3 = go.Figure(data=fig2.data+ fig1.data, layout=go.Layout(
        title=go.layout.Title(text="Prediction of the Bayesian Ridge Model (blue) and present values (rouge)") ))
    plot2_html = fig3.to_html(full_html =False , default_height = 500, default_width = 700)
    return plot2_html

def RandomForest(request):
    model = RandomForestRegressor(max_depth=150,max_features=5,min_samples_leaf=1,min_samples_split=2,n_estimators=2000)
    trainmodel = train_model(model)
    pred = trainmodel.predict(data_test)
    targ = target_test[0:100]
    y= pred[0:100]
    x_ax = range(len(targ))
    fig1 = px.scatter(x= x_ax,y= targ,title="a").update_traces(marker=dict(color='red'))
    fig2 = px.line(x= x_ax,y= y,title = "a").update_traces(marker=dict(color='yellow'))
    fig3 = go.Figure(data=fig2.data+ fig1.data, layout=go.Layout(
        title=go.layout.Title(text="Prediction of the Random Forest Model (blue) and present values (rouge)")
    ))
    plot2_html = fig3.to_html(full_html =False , default_height = 500, default_width = 700)
    return plot2_html


def index(request):
    template = loader.get_template('template1.html')
    if(request.GET['model'] == "HistgradientBoosting"):
        a = HistgradientBoosting(request)
    if(request.GET['model'] == "BayesianRidgeModel"):
        a = BayesianRidgeModel(request)
    if(request.GET['model'] == "DecisionTreeRegressorModel"):
        a = DecisionTreeRegressorModel(request)
    context = {
            'Accuracy' : a
        }
    return HttpResponse(template.render(context,request))

def indexGraph(request):
    template = loader.get_template('template1.html')
    if(request.GET['model'] == "PiePlot"):
        plot2_html = Pieplot(request)
    if(request.GET['model'] == "BarPlot"):
        plot2_html = Barplot(request)
    if(request.GET['model'] == "Matricecorrelation"):
        plot2_html = Matricecorrelation(request)
    context = {  
            'plot2_html':plot2_html,           
        }
    return HttpResponse(template.render(context,request))
    
def indexGraphModel(request):
    template = loader.get_template('template1.html')
    if(request.GET['model'] == "GradientBoosting"):
        plot3_html = GradientBoosting(request)
    if(request.GET['model'] == "BayesianRidge"):
        plot3_html = BayesianRidge(request)
    if(request.GET['model'] == "RandomForestRegressor"):
        plot3_html = RandomForest(request)
    context = {  
            'plot3_html':plot3_html,           
        }
    return HttpResponse(template.render(context,request))

   






