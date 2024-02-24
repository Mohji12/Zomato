import streamlit as st
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as pl
from datetime import date
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn

custom_css = """
<style>
.custom-title {
    font-family:Cambria, Cochin, Georgia, Times, 'Times New Roman', serif;;
    color: #ff6347; /* tomato */
    font-size: 40px;
    text-align: center;

}
</style>
"""
# Inject custom CSS with st.markdown()
st.markdown(custom_css, unsafe_allow_html=True)

# Use the custom CSS class in your title
st.markdown('<span class="custom-title">ZOMATO STOCK PRICE APP</span>', unsafe_allow_html=True)
st.subheader("This App is created stock price of zomato  using Yahoo Finance API")

st.image(image="Zomato_FS_1-1.jpg")

st.sidebar.header("Select the parameters from below")

start_date = st.sidebar.date_input("Start Date",date(2021,8,22))
end_date = st.sidebar.date_input("End Date",date.today())
stock = "ZOMATO.NS"


# Fetch data from user input using yfinance library
data = yf.download(stock,start=start_date,end=end_date)
# Adding Date column as Index
data.insert(0,"Date",data.index,True)
data.reset_index(drop = True,inplace = True)
st.write("Data From",start_date,"to",end_date)
st.dataframe(data)

# Plot data

st.header("Data Visualisation")
st.subheader("Plot of the graph")
fig = pl.line(data,x = "Date",y = data.columns,title="Closing Price of the Stock")
st.plotly_chart(fig)

# Add a select box to select column from data

column = st.selectbox("Select the column to be used for prediction",data.columns[1:])

data = data[['Date',column]]
st.write("Selected Data")
st.write(data)

# lets Decompose the data
st.header("Decomposition of the data")
decomp = seasonal_decompose(data[column],model='additive',period=12)
st.write(decomp.plot())

# Make same plot in plotly

st.plotly_chart(pl.line(x = data["Date"],y = decomp.trend,title="Trend",width=1200,height=400,labels={'x':'Date','y':'price'}).update_traces(line_color = "Blue"))
st.plotly_chart(pl.line(x = data["Date"],y = decomp.seasonal,title="Trend",width=1200,height=400,labels={'x':'Date','y':'price'}).update_traces(line_color = "red"))
st.plotly_chart(pl.line(x = data["Date"],y = decomp.resid,title="Trend",width=1200,height=400,labels={'x':'Date','y':'price'}).update_traces(line_color = "green"))

# Let's Run the model
# User input for three parameters of the model and seasonal order

st.write("Predicting Value")

data1 = yf.download(stock,start=start_date,end=end_date)
data1.reset_index(inplace=True)
 
closed_prices = data1["Close"]

seq_len = 15

mm = MinMaxScaler()
scaled_price = mm.fit_transform(np.array(closed_prices)[... , None]).squeeze()

X = []
y = []

for i in range(len(scaled_price) - seq_len):
    X.append(scaled_price[i : i + seq_len])
    y.append(scaled_price[i + seq_len])

X = np.array(X)[... , None]
y = np.array(y)[... , None]
    
train_x = torch.from_numpy(X[:int(0.8 * X.shape[0])]).float()
train_y = torch.from_numpy(y[:int(0.8 * X.shape[0])]).float()
test_x = torch.from_numpy(X[int(0.8 * X.shape[0]):]).float()
test_y = torch.from_numpy(y[int(0.8 * X.shape[0]):]).float()

class Model(nn.Module):
    def __init__(self , input_size , hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size , hidden_size , batch_first = True)
        self.fc = nn.Linear(hidden_size , 1)
    def forward(self , x):
        output , (hidden , cell) = self.lstm(x)
        return self.fc(hidden[-1 , :])
model = Model(1 , 64)

optimizer = torch.optim.Adam(model.parameters() , lr = 0.001)
loss_fn = nn.MSELoss()

num_epochs = 100

for epoch in range(num_epochs):
    output = model(train_x)
    loss = loss_fn(output , train_y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0 and epoch != 0:
        print(epoch , "epoch loss" , loss.detach().numpy())

model.eval()
with torch.no_grad():
    output = model(test_x)

pred = mm.inverse_transform(output.numpy())
real = mm.inverse_transform(test_y.numpy())

st.plotly_chart(pl.line(x = pred.squeeze(),title="Trend",width=1200,height=400).update_traces(line_color = "Blue"))
st.plotly_chart(pl.line(x = real.squeeze(),title="Trend",width=1200,height=400).update_traces(line_color = "red"))

plt.show()