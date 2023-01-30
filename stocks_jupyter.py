#!/usr/bin/env python
# coding: utf-8

# In[575]:


import pandas as pd
import yfinance as yf
import datetime
import matplotlib.pyplot as plt
import numpy as np

class IterativeBase():
    def __init__(self, symbol, start, end, amount,use_spread=True):
        self.symbol = symbol
        self.start = start
        self.end =end
        self.initial_balance = amount
        self.current_balance = amount
        self.use_spread=use_spread
        self.units=0
        self.trades=0
        self.position=-1
        self.get_data()
        
    def get_data(self):
        df = pd.read_csv("NZDJPY1.csv",index_col=False,sep='\t')
        df.columns=['time','open','high','low','price','volume']
        df = df.set_index('time')
        del df["open"], df["low"],df["high"],df["volume"]

        df["spread"]=0.000
        df["returns"] = np.log(df.price/df.price.shift(1))
        self.data = df
    
    def get_df():
        return this.data
        
    def plot_data(self, cols = None):
        if cols is None:
            cols="price"
            self.data[cols].plot(figsize = (12, 8), title = self.symbol)

    def get_values(self, bar):
        date = str(self.data.index[bar])
        price = round(self.data.price.iloc[bar], 5) 
        spread = round(self.data.spread.iloc[bar], 5) 
        return date, price, spread
    
    def print_current_balance(self, bar):
        date, price, spread = self.get_values(bar)
        print("{} | Current Balance: {}".format(date, round(self.current_balance, 2)))
        
    def buy_instrument (self, bar, units=None, amount = None):
        self.position=-1*self.position
        date, price, spread = self.get_values(bar)
        if self.use_spread:
            price+=spread/2
        if amount is not None: # use units if units are passed, otherwise calculate units 
            units= int(amount / price)
        self.current_balance -= units * price # reduce cash balance by "purchase price"

        self.units += units 
        self.trades += 1 
        print("{} |  Buying {} for {}".format(date, units, round (price, 5)))
        
    def sell_instrument (self, bar, units = None, amount = None):
        self.position=-1*self.position
        date, price, spread = self.get_values(bar)
        if self.use_spread:
            price+=spread/2
        if amount is not None: # use units if units are passed, otherwise calculate units 
            units =int(amount / price)
        self.current_balance += units * price # increases cash balance by "purchase price"
        self.units -=units
        self.trades += 1
        print("{} |. Selling {} for {}".format(date, units, round (price, 5)))

    def print_current_position_value(self, bar): 
        date, price, spread = self.get_values(bar) 
        cpv = self.units * price
        print("{} |  Current Position Value = {}".format(date, round (cpv, 2)))
    
    def print_current_nav (self, bar):
        date, price, spread = self.get_values(bar)
        nav = self.current_balance + self.units * price
        print("{} |  Net Asset Value = {}".format(date, round(nav, 2)))
        
    def close_pos(self, bar):
        date, price, spread = self.get_values(bar)
        print(75*"-")
        print("{} | +++ CLOSING FINAL POSITION +++".format(date))
        self.current_balance += self.units * price # closing final position (works with short and 
        self.current_balance -= (abs (self.units) *spread/2*self.use_spread)
        print("{} |  closing position of {} for {}".format(date, self.units, price))
        self.units = 0 # setting position to neutral
        self.trades += 1
        perf =(self.current_balance - self.initial_balance) / self.initial_balance * 100 
        self.print_current_balance(bar)
        print("{} | net performance (%) = {}".format(date, round (perf, 2))) 
        print("{} |  number of trades executed = {}".format(date, self.trades)) 
        print (75*"-")


# In[576]:


from sklearn.linear_model import LogisticRegression

class BackTest:
    def __init__(self, base):
        self.base=base
    def check_ai(self,part,SMA_SHORT=500,SMA_LONG=2000):
        df=self.base.data.copy(deep=True)
        df["direction"]=np.sign(df.returns)
        df["SMA_SHORT"]=df["price"].rolling(SMA_SHORT).mean()
        df["SMA_LONG"]=df["price"].rolling(SMA_LONG).mean()
        df.dropna(inplace=True)
        print(df)

        lags=5
        cols=[]
        for lag in range(1,lags+1):
            col="lag{}".format(lag)
            df[col]=df.returns.shift(lag)
            cols.append(col)
        df.dropna(inplace=True)
        
        
        for i in range(1):
            i=496
            df1=df.iloc[:50000-100+i*100].copy(deep=True)
            df2=df.iloc[:50000-100+i*100].copy(deep=True)

#             df2=df.iloc[50000-100+i*100:50000-100+(i+1)*100].copy(deep=True)

            lm=LogisticRegression(C=1e6,max_iter=100000,multi_class="ovr")

            lm.fit(df1[cols],df1.direction)
            
            df2["pred"]=lm.predict(df2[cols])
            
            
            for j in range(len(df2["pred"])-1):
                if df2["pred"][j+1] == 1 and self.base.position==-1 :
                    self.base.buy_instrument(j,  amount = self.base.current_balance)
                elif df2["pred"][j] != 1 and self.base.position==1 :
                    self.base.sell_instrument(j,  units = self.base.units)
                    
#         df2["strategy"]=df2.pred*df2.returns
#         df2["creturns"]=df2["returns"].cumsum().apply(np.exp)
#         df2["cstrategy"]=df2.strategy.cumsum().apply(np.exp)
#         df2.dropna(inplace=True)
#         df2[["cstrategy","creturns"]].plot(figsize=(12,8))
        self.base.close_pos(-1)
#         self.base.plot_data()
    def check_sma(self,SMA_LONG,SMA_SHORT):
        df=self.base.data.copy(deep=True)
        df["SMA_SHORT"]=df["price"].rolling(SMA_SHORT).mean()
        df["SMA_LONG"]=df["price"].rolling(SMA_LONG).mean()
        df.loc[:,["SMA_LONG","SMA_SHORT","price"]].plot(figsize=(12,8))


# In[803]:


import collections
import itertools

def count(seq):
    # count the number of items in a sequence.
    return sum(1 for _ in seq)

def count_things(input_string, what):
    grouped = itertools.groupby(input_string)
    counts = [count(subseq) for (key, subseq) in grouped if key == what]
    return collections.Counter(counts)


# In[660]:


x= IterativeBase("AUDCAD","2022-01-01","2022-12-01",100000,True)
df=x.data
SMA=2
SMA_LONG=20
dev=2
df["delta"]=df.price-df.price.shift(1)
df["SMA_SHORT"]=df["price"].rolling(SMA).mean()
df["SMA_LONG"]=df["price"].rolling(SMA_LONG).mean()

df["lower"]=df["SMA_SHORT"]-df["price"].rolling(SMA).std()*dev
df["upper"]=df["SMA_SHORT"]+df["price"].rolling(SMA).std()*dev
df["width"]=df.upper-df.lower
print(df)
df.dropna(inplace=True)

df["direction"]=np.sign(df.delta)
print(df)

df.dropna(inplace=True)

lags=5
cols=[]
for lag in range(1,lags+1):
    col="lag{}".format(lag)
    df[col]=df.delta.shift(lag)
    cols.append(col)
df.dropna(inplace=True)

# df[:1000].loc[:,["price","upper","lower","SMA_LONG","SMA_SHORT"]].plot(figsize=(18,12))

df1=df.iloc[:1500].copy(deep=True)
df2=df.iloc[1500:].copy(deep=True)


lm=LogisticRegression(C=1e6,max_iter=100000,multi_class="ovr")

lm.fit(df1[cols],df1.direction)
ll=''
df2["pred"]=lm.predict(df2[cols])
for t in range(1):
    k=20
    devr=[]
    banks=[]
    rate=0

    for i in range(0,1):
        profit=0
        back_to_game=True
        bank=1000
        to_put=10
        max_put=[0,0]
        fails=1
        total_lost=0
        ll=''
        for j in range(1,len(df["direction"]),1):
            if(df["SMA_LONG"][i+j-1]-df["SMA_LONG"][i+j-2]<=0.003):
                continue
#             if  (not ( abs(df["SMA_LONG"][i+j]-df["SMA_LONG"][i+j-10])/10<0.01)):
#                 continue
#     #             if(to_put/2>bank):
#     #                 continue
#             if to_put>max_put[0]:
#                 max_put[0]=to_put
#                 max_put[1]=i

            if 1==df["direction"][i+j] and not back_to_game:
                fails=1
                back_to_game =True
                to_put=1
                total_lost=0

                continue
            if  1==df["direction"][i+j] and back_to_game:
                total_lost=0
                fails=1
                bank+=to_put*0.92
                to_put=10
            else:
                if fails>3:
                    fails=1
                    back_to_game=False
                    bank-=to_put
                    to_put=1
                    total_lost=0
                    continue
                total_lost+=to_put
                bank-=to_put
                to_put=(1+total_lost)*1/0.92
                fails+=1

        
            profit=bank-1000
            if profit>0:
                rate+=1
            else:
                rate-=1
            banks+=[profit]
#             print(profit,max_put,i,(df["SMA_LONG"][i+j]-df["SMA_LONG"][i+j-10])/10,rate)
    print(k," : ",sum(banks))
    
temp=[]
# for j in range(1,len(banks[:5000])):
#     temp+=[sum(banks[:j])]
    
plt.plot(banks[1200:2200], color='magenta', marker='o',mfc='pink' ) #plot the data
plt.xticks(range(0,len(banks[1200:2200])+1, 1)) #set the tick frequency on x-axis

plt.ylabel('data') #set the label for y axis
plt.xlabel('index') #set the label for x-axis
plt.title("Plotting a list") #set the title of the graph
plt.show() #display the graph


# In[613]:


one_counts = count_things(ll, '0')
print(one_counts)


# In[801]:


x= IterativeBase("AUDCAD","2022-01-01","2022-12-01",100000,True)
df=x.data
SMA=2
SMA_LONG=200
dev=2
df["delta"]=df.price-df.price.shift(1)
df["SMA_SHORT"]=df["price"].rolling(SMA).mean()
df["SMA_LONG"]=df["price"].rolling(SMA_LONG).mean()

df["lower"]=df["SMA_SHORT"]-df["price"].rolling(SMA).std()*dev
df["upper"]=df["SMA_SHORT"]+df["price"].rolling(SMA).std()*dev
df["width"]=df.upper-df.lower
print(df)
df.dropna(inplace=True)

df["direction"]=np.sign(df.delta)
print(df)

df.dropna(inplace=True)

lags=5
cols=[]
for lag in range(1,lags+1):
    col="lag{}".format(lag)
    df[col]=df.delta.shift(lag)
    cols.append(col)
df.dropna(inplace=True)

# df[:1000].loc[:,["price","upper","lower","SMA_LONG","SMA_SHORT"]].plot(figsize=(18,12))

df1=df.iloc[:1500].copy(deep=True)
df2=df.iloc[1500:].copy(deep=True)


lm=LogisticRegression(C=1e6,max_iter=100000,multi_class="ovr")

lm.fit(df1[cols],df1.direction)
ll=''
df2["pred"]=lm.predict(df2[cols])
for t in range(1):
    k=20
    devr=[]
    banks=[]
    rate=0

    for i in range(0,1):
        profit=0
        back_to_game=True
        bank=1000
        to_put=10
        max_put=[0,0]
        fails=1
        total_lost=0
        ll=''
        for j in range(1,len(df["direction"])-15,1):

            if df["direction"][i+j]==df["direction"][i+j+1] :
                ll+='1'
            if df["direction"][i+j]!=df["direction"][i+j+1]  :
                ll+='0'



# In[714]:


def RSI(data, window=14, adjust=False):
    delta = data['price'].diff(1).dropna()
    loss = delta.copy()
    gains = delta.copy()

    gains[gains < 0] = 0
    loss[loss > 0] = 0

    gain_ewm = gains.ewm(com=window - 1, adjust=adjust).mean()
    loss_ewm = abs(loss.ewm(com=window - 1, adjust=adjust).mean())

    RS = gain_ewm / loss_ewm
    RSI = 100 - 100 / (1 + RS)

    return RSI


# In[ ]:





# In[ ]:





# In[ ]:





# In[758]:


import talib


# In[810]:


df["rsi"]=talib.RSI(df['price'], timeperiod=30)
df
ll=''
i=3
while i<len(df["rsi"])-1:
    if(df["rsi"][i-2]>70 and df["rsi"][i-2]-df["rsi"][i-3]<0):
        while df["direction"][i]!=-1:
            ll+='1'
            i+=1
        ll+='0'
            
        ll+='2'
        
    i+=1
one_counts = count_things(ll, '1')
print(one_counts)
one_counts = count_things(ll, '0')
print(one_counts)


# In[813]:


i=3
ll=''

while i<len(df["rsi"])-1:
    if(df["rsi"][i-2]<30 and df["rsi"][i-2]-df["rsi"][i-3]>0):
        while df["direction"][i]!=1:
            ll+='1'
            i+=1
        ll+='0'
            
        ll+='2'
        
    i+=1
one_counts = count_things(ll, '1')
print(one_counts)
one_counts = count_things(ll, '0')
print(one_counts)


# In[814]:


i=3
ll=''

while i<len(df["direction"])-1:
    if(df["direction"][i]==1):
        ll+='1'
    else:
        ll+='0'
        
    i+=1
one_counts = count_things(ll, '1')
print(one_counts)
one_counts = count_things(ll, '0')
print(one_counts)


# In[819]:


i=3
ll=''
import random

while i<len(df["direction"])-1:
    if(random.randrange(0,2,1)==1):
        ll+='1'
    else:
        ll+='0'
        
    i+=1
one_counts = count_things(ll, '1')
print(one_counts)
one_counts = count_things(ll, '0')
print(one_counts)


# In[853]:


ll=''
i=12
count1=0
count2=0
check=False
delta=12
while i<len(df["direction"])-1:
    if df["direction"][i]==0:
        i+=1
        continue
    for j in range(1,delta):
        if df["direction"][i]==0:
            i+=1
            continue
        if df["direction"][i-j]!=df["direction"][i-j-1]:
            check =True
        else:
            check=False
            break
    if check:
        if df["direction"][i-1]!=df["direction"][i]:
            count1+=1
        else:
            count2+=1
        
    i+=1
print(count1,count2)


# In[ ]:




