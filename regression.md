---
layout: wide_default
---

## Part 1: EDA


```python
import pandas as pd
from statsmodels.formula.api import ols as sm_ols
import numpy as np
import seaborn as sns
from statsmodels.iolib.summary2 import summary_col 
import matplotlib.pyplot as plt
```


```python
housing = pd.read_csv('input_data2/housing_train.csv')
housing
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
      <th>parcel</th>
      <th>v_MS_SubClass</th>
      <th>v_MS_Zoning</th>
      <th>v_Lot_Frontage</th>
      <th>v_Lot_Area</th>
      <th>v_Street</th>
      <th>v_Alley</th>
      <th>v_Lot_Shape</th>
      <th>v_Land_Contour</th>
      <th>v_Utilities</th>
      <th>...</th>
      <th>v_Pool_Area</th>
      <th>v_Pool_QC</th>
      <th>v_Fence</th>
      <th>v_Misc_Feature</th>
      <th>v_Misc_Val</th>
      <th>v_Mo_Sold</th>
      <th>v_Yr_Sold</th>
      <th>v_Sale_Type</th>
      <th>v_Sale_Condition</th>
      <th>v_SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1056_528110080</td>
      <td>20</td>
      <td>RL</td>
      <td>107.0</td>
      <td>13891</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>1</td>
      <td>2008</td>
      <td>New</td>
      <td>Partial</td>
      <td>372402</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1055_528108150</td>
      <td>20</td>
      <td>RL</td>
      <td>98.0</td>
      <td>12704</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>1</td>
      <td>2008</td>
      <td>New</td>
      <td>Partial</td>
      <td>317500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1053_528104050</td>
      <td>20</td>
      <td>RL</td>
      <td>114.0</td>
      <td>14803</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>6</td>
      <td>2008</td>
      <td>New</td>
      <td>Partial</td>
      <td>385000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2213_909275160</td>
      <td>20</td>
      <td>RL</td>
      <td>126.0</td>
      <td>13108</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR2</td>
      <td>HLS</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>6</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>153500</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1051_528102030</td>
      <td>20</td>
      <td>RL</td>
      <td>96.0</td>
      <td>12444</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>11</td>
      <td>2008</td>
      <td>New</td>
      <td>Partial</td>
      <td>394617</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1936</th>
      <td>2524_534125210</td>
      <td>190</td>
      <td>RL</td>
      <td>79.0</td>
      <td>13110</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>NaN</td>
      <td>0</td>
      <td>7</td>
      <td>2006</td>
      <td>WD</td>
      <td>Normal</td>
      <td>146500</td>
    </tr>
    <tr>
      <th>1937</th>
      <td>2846_909131125</td>
      <td>190</td>
      <td>RH</td>
      <td>NaN</td>
      <td>7082</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>7</td>
      <td>2006</td>
      <td>WD</td>
      <td>Normal</td>
      <td>160000</td>
    </tr>
    <tr>
      <th>1938</th>
      <td>2605_535382020</td>
      <td>190</td>
      <td>RL</td>
      <td>60.0</td>
      <td>10800</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2006</td>
      <td>ConLD</td>
      <td>Normal</td>
      <td>160000</td>
    </tr>
    <tr>
      <th>1939</th>
      <td>1516_909101180</td>
      <td>190</td>
      <td>RL</td>
      <td>55.0</td>
      <td>5687</td>
      <td>Pave</td>
      <td>Grvl</td>
      <td>Reg</td>
      <td>Bnk</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>3</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>135900</td>
    </tr>
    <tr>
      <th>1940</th>
      <td>1387_905200100</td>
      <td>190</td>
      <td>RL</td>
      <td>60.0</td>
      <td>12900</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>1</td>
      <td>2008</td>
      <td>WD</td>
      <td>Alloca</td>
      <td>95541</td>
    </tr>
  </tbody>
</table>
<p>1941 rows × 81 columns</p>
</div>




```python
housing.describe()
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
      <th>v_MS_SubClass</th>
      <th>v_Lot_Frontage</th>
      <th>v_Lot_Area</th>
      <th>v_Overall_Qual</th>
      <th>v_Overall_Cond</th>
      <th>v_Year_Built</th>
      <th>v_Year_Remod/Add</th>
      <th>v_Mas_Vnr_Area</th>
      <th>v_BsmtFin_SF_1</th>
      <th>v_BsmtFin_SF_2</th>
      <th>...</th>
      <th>v_Wood_Deck_SF</th>
      <th>v_Open_Porch_SF</th>
      <th>v_Enclosed_Porch</th>
      <th>v_3Ssn_Porch</th>
      <th>v_Screen_Porch</th>
      <th>v_Pool_Area</th>
      <th>v_Misc_Val</th>
      <th>v_Mo_Sold</th>
      <th>v_Yr_Sold</th>
      <th>v_SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1941.000000</td>
      <td>1620.000000</td>
      <td>1941.000000</td>
      <td>1941.000000</td>
      <td>1941.000000</td>
      <td>1941.000000</td>
      <td>1941.000000</td>
      <td>1923.000000</td>
      <td>1940.000000</td>
      <td>1940.000000</td>
      <td>...</td>
      <td>1941.000000</td>
      <td>1941.000000</td>
      <td>1941.000000</td>
      <td>1941.000000</td>
      <td>1941.000000</td>
      <td>1941.000000</td>
      <td>1941.000000</td>
      <td>1941.000000</td>
      <td>1941.000000</td>
      <td>1941.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>58.088614</td>
      <td>69.301235</td>
      <td>10284.770222</td>
      <td>6.113344</td>
      <td>5.568264</td>
      <td>1971.321999</td>
      <td>1984.073158</td>
      <td>104.846074</td>
      <td>436.986598</td>
      <td>49.247938</td>
      <td>...</td>
      <td>92.458011</td>
      <td>49.157135</td>
      <td>22.947965</td>
      <td>2.249871</td>
      <td>16.249871</td>
      <td>3.386399</td>
      <td>52.553838</td>
      <td>6.431221</td>
      <td>2006.998454</td>
      <td>182033.238022</td>
    </tr>
    <tr>
      <th>std</th>
      <td>42.946015</td>
      <td>23.978101</td>
      <td>7832.295527</td>
      <td>1.401594</td>
      <td>1.087465</td>
      <td>30.209933</td>
      <td>20.837338</td>
      <td>184.982611</td>
      <td>457.815715</td>
      <td>169.555232</td>
      <td>...</td>
      <td>127.020523</td>
      <td>70.296277</td>
      <td>65.249307</td>
      <td>22.416832</td>
      <td>56.748086</td>
      <td>43.695267</td>
      <td>616.064459</td>
      <td>2.745199</td>
      <td>0.801736</td>
      <td>80407.100395</td>
    </tr>
    <tr>
      <th>min</th>
      <td>20.000000</td>
      <td>21.000000</td>
      <td>1470.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1872.000000</td>
      <td>1950.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2006.000000</td>
      <td>13100.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>20.000000</td>
      <td>58.000000</td>
      <td>7420.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>1953.000000</td>
      <td>1965.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5.000000</td>
      <td>2006.000000</td>
      <td>130000.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>50.000000</td>
      <td>68.000000</td>
      <td>9450.000000</td>
      <td>6.000000</td>
      <td>5.000000</td>
      <td>1973.000000</td>
      <td>1993.000000</td>
      <td>0.000000</td>
      <td>361.500000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>6.000000</td>
      <td>2007.000000</td>
      <td>161900.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>70.000000</td>
      <td>80.000000</td>
      <td>11631.000000</td>
      <td>7.000000</td>
      <td>6.000000</td>
      <td>2001.000000</td>
      <td>2004.000000</td>
      <td>168.000000</td>
      <td>735.250000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>168.000000</td>
      <td>72.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>8.000000</td>
      <td>2008.000000</td>
      <td>215000.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>190.000000</td>
      <td>313.000000</td>
      <td>164660.000000</td>
      <td>10.000000</td>
      <td>9.000000</td>
      <td>2008.000000</td>
      <td>2009.000000</td>
      <td>1600.000000</td>
      <td>5644.000000</td>
      <td>1474.000000</td>
      <td>...</td>
      <td>1424.000000</td>
      <td>742.000000</td>
      <td>1012.000000</td>
      <td>407.000000</td>
      <td>576.000000</td>
      <td>800.000000</td>
      <td>17000.000000</td>
      <td>12.000000</td>
      <td>2008.000000</td>
      <td>755000.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 37 columns</p>
</div>



- Unit of obersvations
    - The dataset looks at information around the selling of a house
- Time period
    - The data ranges from 2006 to 2008 (this is for when the house was sold)
- Sample size
    - There are 1941 observations making up the sample size
- Issues
    - Variables such as v_Alley, v_Pool_QC, v_Fence, v_Misc_Feature have a lot of missing variables which could prove to be an issue in running a regression against those variables


```python
sns.lineplot(data=housing, x="v_Yr_Sold", y="v_SalePrice", hue='v_Sale_Condition')
plt.show()
```


<img src="images/output_5_0.png.jpg?raw=true"/>
    
    



```python
sns.lineplot(data=housing, x="v_Yr_Sold", y="v_SalePrice", hue='v_Lot_Shape')
plt.show()
```


    
<img src="images/output_6_0.png.jpg?raw=true"/>
    


## Part 2: Running Regressions

```
_Insert cells as needed below to run these regressions. Note that $i$ is indexing a given house, and $t$ indexes the year of sale._ 

1. $\text{Sale Price}_{i,t} = \alpha + \beta_1 * \text{v_Lot_Area}$
1. $\text{Sale Price}_{i,t} = \alpha + \beta_1 * log(\text{v_Lot_Area})$
1. $log(\text{Sale Price}_{i,t}) = \alpha + \beta_1 * \text{v_Lot_Area}$
1. $log(\text{Sale Price}_{i,t}) = \alpha + \beta_1 * log(\text{v_Lot_Area})$
1. $log(\text{Sale Price}_{i,t}) = \alpha + \beta_1 * \text{v_Yr_Sold}$
1. $log(\text{Sale Price}_{i,t}) = \alpha + \beta_1 * (\text{v_Yr_Sold==2007})+ \beta_2 * (\text{v_Yr_Sold==2008})$
1. Choose your own adventure: Pick any five variables from the dataset that you think will generate good R2. Use them in a regression of $log(\text{Sale Price}_{i,t})$ 
    - Tip: You can transform/create these five variables however you want, even if it creates extra variables. For example: I'd count Model 6 above as only using one variable: `v_Yr_Sold`.
    - I got an R2 of 0.877 with just "5" variables. How close can you get? I won't be shocked if someone beats that!
    

```


```python
## create new variables -- log of l_SalePrice, log of v_Lot_Area
housing = (housing.assign(l_SalePrice = np.log(housing['v_SalePrice']),
                         l_Lot_Area = np.log(housing['v_Lot_Area'])))
```


```python
# create regressions
#1
reg1 = sm_ols('v_SalePrice ~ v_Lot_Area', data= housing).fit()
#2
reg2 = sm_ols('v_SalePrice ~ l_Lot_Area', data=housing).fit()
#3
reg3 = sm_ols('l_SalePrice ~ v_Lot_Area', data=housing).fit()
#4
reg4 = sm_ols('l_SalePrice ~ l_Lot_Area', data=housing).fit()
#5
reg5 = sm_ols('l_SalePrice ~ v_Yr_Sold', data=housing).fit()
#6
reg6 = sm_ols('l_SalePrice ~ v_Yr_Sold==2007 + v_Yr_Sold==2008', data=housing).fit()
#7
reg7 = sm_ols('l_SalePrice ~ v_Lot_Area + v_Yr_Sold + v_Sale_Type + v_Sale_Condition + v_Lot_Frontage', data=housing).fit()
```


```python
#output table
info_dict={'R-squared' : lambda x: f"{x.rsquared:.2f}",
           'Adj R-squared' : lambda x: f"{x.rsquared_adj:.2f}",
           'No. observations' : lambda x: f"{int(x.nobs):d}"}

#summary col function to show all
print('='*108)
print('                  y = sale price, log(sale price else)')
print(summary_col(results=[reg1,reg2,reg3,reg4,reg5,reg6,reg7], #regressions
                  float_format='%0.2f',
                  stars = True, 
                  model_names=['1','2','3','4','5','6','7'], #model names
                  info_dict=info_dict,
                  regressor_order=['Intercept', 'v_Lot_Area', 'l_Lot_Area', 'v_Yr_Sold', 'v_Sale_Type', 'v_Sale_Condition'
                                   , 'v_Lot_Frontage' ]
                  )
     )
```

    ============================================================================================================
                      y = sale price, log(sale price else)
    
    ================================================================================================
                                     1             2          3        4       5       6        7   
    ------------------------------------------------------------------------------------------------
    Intercept                   154789.55*** -327915.80*** 11.89*** 9.41*** 22.29   12.02*** -31.48 
                                (2911.59)    (30221.35)    (0.01)   (0.15)  (22.94) (0.02)   (22.00)
    v_Lot_Area                  2.65***                    0.00***                           0.00***
                                (0.23)                     (0.00)                            (0.00) 
    l_Lot_Area                               56028.17***            0.29***                         
                                             (3315.14)              (0.02)                          
    v_Yr_Sold                                                               -0.01            0.02*  
                                                                            (0.01)           (0.01) 
    v_Lot_Frontage                                                                           0.00***
                                                                                             (0.00) 
    v_Sale_Type[T.ConLD]                                                                     -0.22**
                                                                                             (0.11) 
    v_Yr_Sold == 2007[T.True]                                                       0.03            
                                                                                    (0.02)          
    v_Sale_Type[T.WD ]                                                                       0.18***
                                                                                             (0.06) 
    v_Sale_Type[T.VWD]                                                                       -0.10  
                                                                                             (0.35) 
    v_Sale_Type[T.Oth]                                                                       0.10   
                                                                                             (0.21) 
    v_Sale_Type[T.New]                                                                       0.57***
                                                                                             (0.19) 
    v_Sale_Type[T.Con]                                                                       0.32   
                                                                                             (0.21) 
    v_Sale_Type[T.ConLw]                                                                     -0.21  
                                                                                             (0.18) 
    v_Sale_Type[T.ConLI]                                                                     -0.12  
                                                                                             (0.21) 
    v_Sale_Type[T.CWD]                                                                       0.34***
                                                                                             (0.12) 
    v_Sale_Condition[T.Partial]                                                              0.22   
                                                                                             (0.18) 
    v_Sale_Condition[T.Normal]                                                               0.19***
                                                                                             (0.04) 
    v_Sale_Condition[T.Family]                                                               0.12*  
                                                                                             (0.07) 
    v_Sale_Condition[T.Alloca]                                                               0.14   
                                                                                             (0.12) 
    v_Sale_Condition[T.AdjLand]                                                              -0.13  
                                                                                             (0.11) 
    v_Yr_Sold == 2008[T.True]                                                       -0.01           
                                                                                    (0.02)          
    R-squared                   0.07         0.13          0.06     0.13    0.00    0.00     0.31   
    R-squared Adj.              0.07         0.13          0.06     0.13    -0.00   0.00     0.31   
    R-squared                   0.07         0.13          0.06     0.13    0.00    0.00     0.31   
    Adj R-squared               0.07         0.13          0.06     0.13    -0.00   0.00     0.31   
    No. observations            1941         1941          1941     1941    1941    1941     1620   
    ================================================================================================
    Standard errors in parentheses.
    * p<.1, ** p<.05, ***p<.01


## Part 3: Regression interpretation

_Insert cells as needed below to answer these questions. Note that $i$ is indexing a given house, and $t$ indexes the year of sale._ 

1. If you didn't use the `summary_col` trick, list $\beta_1$ for Models 1-6 to make it easier on your graders.
1. Interpret $\beta_1$ in Model 2. 
1. Interpret $\beta_1$ in Model 3. 
    - HINT: You might need to print out more decimal places. Show at least 2 non-zero digits. 
1. Of models 1-4, which do you think best explains the data and why?
1. Interpret $\beta_1$ In Model 5
1. Interpret $\alpha$ in Model 6
1. Interpret $\beta_1$ in Model 6
1. Why is the R2 of Model 6 higher than the R2 of Model 5?
1. What variables did you include in Model 7?
1. What is the R2 of your Model 7?
1. Speculate (not graded): Could you use the specification of Model 6 in a predictive regression? 
1. Speculate (not graded): Could you use the specification of Model 5 in a predictive regression? 



```python
#reg1.summary()
#reg2.summary()
#reg3.summary()
#reg4.summary()
#reg5.summary()
#reg6.summary()
#reg7.summary()
reg4.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>l_SalePrice</td>   <th>  R-squared:         </th> <td>   0.135</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.135</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   302.5</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 02 Apr 2023</td> <th>  Prob (F-statistic):</th> <td>4.38e-63</td>
</tr>
<tr>
  <th>Time:</th>                 <td>18:47:15</td>     <th>  Log-Likelihood:    </th> <td> -851.27</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>  1941</td>      <th>  AIC:               </th> <td>   1707.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  1939</td>      <th>  BIC:               </th> <td>   1718.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
       <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>  <td>    9.4051</td> <td>    0.151</td> <td>   62.253</td> <td> 0.000</td> <td>    9.109</td> <td>    9.701</td>
</tr>
<tr>
  <th>l_Lot_Area</th> <td>    0.2883</td> <td>    0.017</td> <td>   17.394</td> <td> 0.000</td> <td>    0.256</td> <td>    0.321</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>84.067</td> <th>  Durbin-Watson:     </th> <td>   0.955</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td> 255.283</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.100</td> <th>  Prob(JB):          </th> <td>3.68e-56</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 4.765</td> <th>  Cond. No.          </th> <td>    164.</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



#### Interpret $\beta_1$ in Model 2. 

- A 1% increase in l_Lot_Area is associated with an increase of 5.603e+04 percentage points in sale price 

#### Interpret $\beta_1$ in Model 3. 

- A 1 unit increase in v_Lot_Area is associated with an proportional increase of 1.309e-05% in sale price

#### Of models 1-4, which do you think best explains the data and why?

- Of models 1-4, the one that explains the data best is most likely model 4. Model 4 has the highest/strong R-Squared at 0.13. Even though all the models have p-values to show that they are statiscally relationships, model 4 shows the relationship between the log of v_SalePrice and log of v_Lot_Area to have the highest R-Sqaured

#### Interpret $\beta_1$ In Model 5

- A 1 unit increase in v_Yr_Sold is associated with a proportional decrease of -0.0051% in sale price

#### Interpret $\alpha$ in Model 6

- The intercept in model 6 is 12.0229 percentage points
- The average value of log sale price is 12.0229 percentage points for group 0

#### Interpret $\beta_1$ in Model 6

- Sale price is about 2.56% higher on average for case when v_Yr_Sold is 2007 then when it is not

#### Why is the R2 of Model 6 higher than the R2 of Model 5?

- The R2 in model 6 is higher than the R2 in model 5 because in model 6 is more specific in what year it regresses against

#### What variables did you include in Model 7?

- In model 7 I included 
    - v_Lot_Area 
    - v_Yr_Sold 
    - v_Sale_Type 
    - v_Sale_Condition
    - v_Lot_Frontage

#### What is the R2 of your Model 7?

- The R2 of model 7 was 0.313

#### Speculate (not graded): Could you use the specification of Model 6 in a predictive regression? 

- Maybe not because that the specified years (2007 & 2008) are around the time of the housing crisis and that would alter the data

#### Speculate (not graded): Could you use the specification of Model 5 in a predictive regression?

- Maybe not either because even though using model 5 would make sense to use for predictions as it contains all historical data, it only has one more additional year in comparions to model 6
