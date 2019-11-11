
# Rishabh Agrawal
# Assignment 3

# Libraries


```python
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import math
from sklearn.utils import shuffle
import sklearn.linear_model as Lm
from sklearn.linear_model import LinearRegression as Lr
from sklearn.metrics import mean_squared_error
from math import sqrt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import pycountry_convert as pc
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
```

# Problem 1


```python
p1 = pd.read_table("Files/Sdf16_1a.txt")
```

    C:\Users\risha\Anaconda3\lib\site-packages\IPython\core\interactiveshell.py:3020: DtypeWarning: Columns (0,3) have mixed types. Specify dtype option on import or set low_memory=False.
      interactivity=interactivity, compiler=compiler, result=result)
    


```python
stateFund = p1.groupby("STNAME").sum()
stateFund['perStud'] = stateFund['TFEDREV']/stateFund['V33']
sF = pd.DataFrame(stateFund['TFEDREV']).sort_values(by=['TFEDREV'])
sT = pd.DataFrame(stateFund['V33']).sort_values(by=['V33'])
perStud = pd.DataFrame(stateFund['perStud']).sort_values(by=['perStud'])
```


```python
newsF = sF.tail(10)
plt.figure(figsize=(10,10))
plt.barh(newsF.index, newsF['TFEDREV'])
plt.title("Top 10 states taking the Most Federal Funding")
plt.ylabel("States")
plt.xlabel("Total Federal Revenue")
```




    Text(0.5,0,'Total Federal Revenue')




![png](/Images/output_6_1.png)



```python
newsF = sF.head(10)
plt.figure(figsize=(10,10))
plt.barh(newsF.index, newsF['TFEDREV'])
plt.title("Top 10 states taking the Least Federal Funding")
plt.ylabel("States")
plt.xlabel("Total Federal Revenue")
```




    Text(0.5,0,'Total Federal Revenue')




![png](/Images/output_7_1.png)


We can see that the state taking the most federal funding is California, followed by Texas and New York.
The states taking the least funding are Vermont, Wyoming, Delaware etc.


```python
pS = perStud.tail(10)
plt.figure(figsize=(10,10))
plt.barh(pS.index, pS['perStud'])
plt.title("Top 10 states taking the Most Federal Funding Per Student")
plt.ylabel("States")
plt.xlabel("Federal Revenue per Student")
```




    Text(0.5,0,'Federal Revenue per Student')




![png](/Images/output_9_1.png)


The states spending the most federal funding per student is District of Columbia followed by Alaska and Louisiana.

# Problem 2


```python
p2 = p1.groupby("NAME").sum()
```


```python
plt.figure(figsize=(15, 6))
plt.plot(p2['TOTALREV'],p2['TOTALEXP'])
plt.title("Revenue vs Expenditure")
plt.ylabel("Expenditure")
plt.xlabel("Revenue")
```




    Text(0.5,0,'Revenue')




![png](/Images/output_13_1.png)



```python
plt.figure(figsize=(15, 6))
plt.scatter(p2['TOTALREV'],p2['TOTALEXP'])
plt.title("Revenue vs Expenditure")
plt.ylabel("Expenditure")
plt.xlabel("Revenue")
```




    Text(0.5,0,'Revenue')




![png](/Images/output_14_1.png)



```python
debtStu = p1.groupby("STNAME").sum()
debtStu['Total Debt per Student'] = (debtStu['TOTALEXP'] - debtStu['TOTALREV']) / debtStu['V33']
dS = pd.DataFrame(debtStu['Total Debt per Student']).sort_values(by=['Total Debt per Student'])
```

As we can see from the line plot and the scatter plot that as the revenue increases the expenditure increases. We also see that they have almost a linear relationship with respect to each other.

### This states have the most debt per Student


```python
dS1 = dS.tail(10)
plt.figure(figsize=(10,10))
plt.barh(dS1.index, dS1['Total Debt per Student'])
plt.title("Top 10 states with most debt Per Student")
plt.ylabel("States")
plt.xlabel("Debt per Student")
```




    Text(0.5,0,'Debt per Student')




![png](/Images/output_18_1.png)



```python
dS
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Total Debt per Student</th>
    </tr>
    <tr>
      <th>STNAME</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Hawaii</th>
      <td>-1634.248194</td>
    </tr>
    <tr>
      <th>Indiana</th>
      <td>-1091.025064</td>
    </tr>
    <tr>
      <th>Connecticut</th>
      <td>-1082.443045</td>
    </tr>
    <tr>
      <th>Michigan</th>
      <td>-942.548776</td>
    </tr>
    <tr>
      <th>California</th>
      <td>-627.333637</td>
    </tr>
    <tr>
      <th>Idaho</th>
      <td>-579.701662</td>
    </tr>
    <tr>
      <th>Massachusetts</th>
      <td>-520.190411</td>
    </tr>
    <tr>
      <th>Maine</th>
      <td>-492.184424</td>
    </tr>
    <tr>
      <th>Maryland</th>
      <td>-474.226382</td>
    </tr>
    <tr>
      <th>Arizona</th>
      <td>-453.122665</td>
    </tr>
    <tr>
      <th>Vermont</th>
      <td>-350.886141</td>
    </tr>
    <tr>
      <th>Colorado</th>
      <td>-267.990426</td>
    </tr>
    <tr>
      <th>Mississippi</th>
      <td>-254.627900</td>
    </tr>
    <tr>
      <th>New Jersey</th>
      <td>-246.922882</td>
    </tr>
    <tr>
      <th>Ohio</th>
      <td>-237.485049</td>
    </tr>
    <tr>
      <th>Louisiana</th>
      <td>-231.922705</td>
    </tr>
    <tr>
      <th>Missouri</th>
      <td>-212.893230</td>
    </tr>
    <tr>
      <th>Georgia</th>
      <td>-144.574809</td>
    </tr>
    <tr>
      <th>New Hampshire</th>
      <td>-143.136468</td>
    </tr>
    <tr>
      <th>Nevada</th>
      <td>-139.944859</td>
    </tr>
    <tr>
      <th>New Mexico</th>
      <td>-121.621139</td>
    </tr>
    <tr>
      <th>Wyoming</th>
      <td>-110.232396</td>
    </tr>
    <tr>
      <th>West Virginia</th>
      <td>-90.161335</td>
    </tr>
    <tr>
      <th>Rhode Island</th>
      <td>-83.154673</td>
    </tr>
    <tr>
      <th>Pennsylvania</th>
      <td>-72.857382</td>
    </tr>
    <tr>
      <th>South Carolina</th>
      <td>-67.537904</td>
    </tr>
    <tr>
      <th>Florida</th>
      <td>-64.637156</td>
    </tr>
    <tr>
      <th>Utah</th>
      <td>-56.406512</td>
    </tr>
    <tr>
      <th>Delaware</th>
      <td>-38.794126</td>
    </tr>
    <tr>
      <th>Arkansas</th>
      <td>55.510848</td>
    </tr>
    <tr>
      <th>Kentucky</th>
      <td>58.774265</td>
    </tr>
    <tr>
      <th>Tennessee</th>
      <td>94.398330</td>
    </tr>
    <tr>
      <th>Wisconsin</th>
      <td>103.665563</td>
    </tr>
    <tr>
      <th>Iowa</th>
      <td>104.204397</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>110.359253</td>
    </tr>
    <tr>
      <th>Oklahoma</th>
      <td>112.218429</td>
    </tr>
    <tr>
      <th>Illinois</th>
      <td>113.742436</td>
    </tr>
    <tr>
      <th>Kansas</th>
      <td>131.640193</td>
    </tr>
    <tr>
      <th>Oregon</th>
      <td>145.571553</td>
    </tr>
    <tr>
      <th>South Dakota</th>
      <td>150.175716</td>
    </tr>
    <tr>
      <th>Virginia</th>
      <td>185.631094</td>
    </tr>
    <tr>
      <th>Washington</th>
      <td>265.801263</td>
    </tr>
    <tr>
      <th>Alabama</th>
      <td>346.284944</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>504.916474</td>
    </tr>
    <tr>
      <th>North Carolina</th>
      <td>506.761787</td>
    </tr>
    <tr>
      <th>Montana</th>
      <td>543.625086</td>
    </tr>
    <tr>
      <th>Minnesota</th>
      <td>746.329846</td>
    </tr>
    <tr>
      <th>Nebraska</th>
      <td>764.693409</td>
    </tr>
    <tr>
      <th>Alaska</th>
      <td>968.643614</td>
    </tr>
    <tr>
      <th>District of Columbia</th>
      <td>1285.117970</td>
    </tr>
    <tr>
      <th>North Dakota</th>
      <td>1611.360914</td>
    </tr>
  </tbody>
</table>
</div>



North Dakota, District of Columbia and Alaska have most debt per student.

# Problem 3


```python
p3 = pd.read_csv("Files/math-achievement-lea-sy2015-16.csv")
```


```python
def removeBlur(df):
    df = np.array(df)
    ans = []
    maxV = 100
    minV = 0
    for i in df:
        if ('GE' in i):
            a = i.split("GE")[1]
            a = int(a)
            ans.append((a+maxV)/2)
        elif ('GT' in i):
            a = i.split("GT")[1]
            a = int(a)
            ans.append((a+1+maxV)/2)
        elif ('LE' in i):
            a = i.split("LE")[1]
            a = int(a)
            ans.append((a+minV)/2)
        elif ('LT' in i):
            a = i.split("LT")[1]
            a = int(a)
            ans.append((a-1+minV)/2)
        elif ('-' in i):
            a = i.split("-")
            ans.append((int(a[0]) + int(a[1])) / 2)
        elif (i.isdigit()):
            ans.append(int(i))
    return ans    
```


```python
p31 = removeBlur(p3['ALL_MTH00PCTPROF_1516'])
```


```python
plt.figure(figsize=(10,10))
plt.hist(p31,  bins=20, edgecolor="k")
plt.title("Distribution of Schools for Students scoring above a particular proficient")
plt.xlabel("Percentage of students in the school that scored at or above proficient")
plt.ylabel("Count of Schools")
```




    Text(0,0.5,'Count of Schools')




![png](/Images/output_25_1.png)


Here i used a function that takes a column as a input and returns an unblurred version of it. 
Here they have mainly used 4-5 ways to blur the values. 
They have LT, GT, LE, GE, range(-), or no values.
I took average of all the range to replace the blurred values inplace.
Like for example if its given 'GE50', which means greater than equal to 50, so i replaced it with (50+100)/2 i.e. 75. 
Similarly for '20-60', i replaced it by (20+60)/2 i.e. 40.
And for LT30, I replaced it by (0+29)/2 i.e. 14.5.
I feel doing this is fair because we dont know what value in that range it may actually be and taking the average or mean is the best way to go about it to keep the error as low as possible statistically.

I then visualized it using histogram with 20 bins of width 5. As we see there is high frequence from 30-60 range, with frequence being especially low in the 90-100 range. 

# Problem 4


```python
totalFed = p1['TFEDREV'].sum()
print("Total Federal Budget = ", totalFed)
print("Amount to cut i.e. 15% of Total Budget = ", 0.15 * totalFed)
print("Total amount left for the new Budget = ", 0.85 * totalFed)
```

    Total Federal Budget =  55602739138
    Amount to cut i.e. 15% of Total Budget =  8340410870.7
    Total amount left for the new Budget =  47262328267.299995
    


```python
def makeEqualBudget(df, cutPercent, total):
    amountCut = (cutPercent/100) * total
    ans = [0]*len(df)
    for i in range(len(df)):
        if (amountCut == 0):
            break
        if (df['AmountLeft'][i] > 0):
            ans[i] = min(df['TFEDREV'][i], amountCut)
            amountCut -= ans[i]
    for i in range(len(df)):
        if (amountCut == 0):
            break
        if (df['AmountLeft'][i] < 0):
            ans[i] = min(0.15*df['TFEDREV'][i], amountCut)
            amountCut -= ans[i]
    return amountCut, ans
            
```


```python
p4 = p1
p4['AmountLeft'] = p1['TOTALREV'] - p1['TOTALEXP']
```


```python
amountLeft, p4['Federal Amount Cut'] = makeEqualBudget(p4, 15, totalFed)
```


```python
p41 = p4[['LEAID','NAME', 'STNAME','Federal Amount Cut']].sort_values('Federal Amount Cut', ascending=False)
p41.head(25)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LEAID</th>
      <th>NAME</th>
      <th>STNAME</th>
      <th>Federal Amount Cut</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1904</th>
      <td>622710</td>
      <td>Los Angeles Unified</td>
      <td>California</td>
      <td>1.091400e+09</td>
    </tr>
    <tr>
      <th>2150</th>
      <td>0634320</td>
      <td>San Diego Unified</td>
      <td>California</td>
      <td>1.514270e+08</td>
    </tr>
    <tr>
      <th>1735</th>
      <td>614550</td>
      <td>Fresno Unified</td>
      <td>California</td>
      <td>1.207710e+08</td>
    </tr>
    <tr>
      <th>1900</th>
      <td>622500</td>
      <td>Long Beach Unified</td>
      <td>California</td>
      <td>1.103370e+08</td>
    </tr>
    <tr>
      <th>2174</th>
      <td>0635310</td>
      <td>Santa Ana Unified</td>
      <td>California</td>
      <td>8.306200e+07</td>
    </tr>
    <tr>
      <th>2372</th>
      <td>0691030</td>
      <td>San Diego County Office of Education</td>
      <td>California</td>
      <td>7.794200e+07</td>
    </tr>
    <tr>
      <th>2139</th>
      <td>0633840</td>
      <td>Sacramento City Unified</td>
      <td>California</td>
      <td>7.673300e+07</td>
    </tr>
    <tr>
      <th>2377</th>
      <td>0691035</td>
      <td>Santa Clara County Office of Education</td>
      <td>California</td>
      <td>7.442700e+07</td>
    </tr>
    <tr>
      <th>2147</th>
      <td>0634170</td>
      <td>San Bernardino City Unified</td>
      <td>California</td>
      <td>7.231800e+07</td>
    </tr>
    <tr>
      <th>2371</th>
      <td>0691029</td>
      <td>San Bernardino County Office of Education</td>
      <td>California</td>
      <td>6.961200e+07</td>
    </tr>
    <tr>
      <th>2368</th>
      <td>0691026</td>
      <td>Riverside County Office of Education</td>
      <td>California</td>
      <td>6.744100e+07</td>
    </tr>
    <tr>
      <th>918</th>
      <td>408800</td>
      <td>Tucson Unified District</td>
      <td>Arizona</td>
      <td>6.741600e+07</td>
    </tr>
    <tr>
      <th>2019</th>
      <td>628050</td>
      <td>Oakland Unified</td>
      <td>California</td>
      <td>6.512500e+07</td>
    </tr>
    <tr>
      <th>831</th>
      <td>404970</td>
      <td>Mesa Unified District</td>
      <td>Arizona</td>
      <td>6.474200e+07</td>
    </tr>
    <tr>
      <th>2090</th>
      <td>0631320</td>
      <td>Pomona Unified</td>
      <td>California</td>
      <td>6.313100e+07</td>
    </tr>
    <tr>
      <th>2355</th>
      <td>0691012</td>
      <td>Kern County Office of Education</td>
      <td>California</td>
      <td>5.983100e+07</td>
    </tr>
    <tr>
      <th>1693</th>
      <td>612330</td>
      <td>Elk Grove Unified</td>
      <td>California</td>
      <td>5.835700e+07</td>
    </tr>
    <tr>
      <th>2240</th>
      <td>0638010</td>
      <td>Stockton Unified</td>
      <td>California</td>
      <td>5.482200e+07</td>
    </tr>
    <tr>
      <th>2048</th>
      <td>0629580</td>
      <td>Palmdale Elementary</td>
      <td>California</td>
      <td>5.081000e+07</td>
    </tr>
    <tr>
      <th>2156</th>
      <td>0634620</td>
      <td>San Juan Unified</td>
      <td>California</td>
      <td>4.715700e+07</td>
    </tr>
    <tr>
      <th>2125</th>
      <td>0633150</td>
      <td>Riverside Unified</td>
      <td>California</td>
      <td>4.588100e+07</td>
    </tr>
    <tr>
      <th>1509</th>
      <td>603630</td>
      <td>Bakersfield City</td>
      <td>California</td>
      <td>4.505600e+07</td>
    </tr>
    <tr>
      <th>1720</th>
      <td>613920</td>
      <td>Fontana Unified</td>
      <td>California</td>
      <td>4.450500e+07</td>
    </tr>
    <tr>
      <th>1965</th>
      <td>625470</td>
      <td>Montebello Unified</td>
      <td>California</td>
      <td>4.398200e+07</td>
    </tr>
    <tr>
      <th>68</th>
      <td>100390</td>
      <td>Birmingham City</td>
      <td>Alabama</td>
      <td>4.326200e+07</td>
    </tr>
  </tbody>
</table>
</div>



# Problem 5

For selecting which schools budget to cut what I did initially is found out which of those had excess funds left from their revenue. What I mean to say is the schools which have some amount left over even after all the total expenditures they do. So we cut the amount left over from their federal budget first from all the schools. Then if there is still amount left then we go on to cut 15% from all the schools equally because that seems like the logical and fair thing to do at this point. This way no school will have a large change in their federal revenue and it would be fair to every school.
But what I found out is the quota of 15% budget cut was met just by taking amount from the schools which had excess amount left. So in short no schools with already less budget had amount cut from them. So this seems like a win - win situation to me, because we only took budget from schools having excess thus not affecting those already struggling.
