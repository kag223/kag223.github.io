# Interpretting Regression

## Notes

### Goodness of Fit
- use r2 and adjusted r2 to do so


### Intereppreting Regression
- regressions of y on N different variables take the form: 𝑦=𝑎+𝑏1∗𝑋1+𝑏2∗𝑋2+...+𝑏𝑁∗𝑋𝑁+𝑢
- basic interpretation
     - "A 1 unit increase in X..."
     - ... is associated with a b change in y...
     - holding all other X constant"
- focus today is on interpretting
    - $a$
    - any beta, but lets $B_1$
- Intercept:$E(y|X=0)
- Interpretting coefs on X / Beta:
    - "A 1 unit increas in X_i is associated with a B_i change in y holding all other X constant"
    - Filling in the blanks, but depends on what X_i is, and f(y)
    - If X is continuous
        - is x logged and/or y logged
        - copy the table here
    - If X is binary
        - beta compares jump from False to True, whatever that measn for that variables
        - copy table here
    - If X is categorical
        - if x = short medium tall humungous, then regression includes
            - x=medium
            - x=tall
            - x=humungous
            - no variable for short --> short is the "omitted level not in regression"
        - beta compares jump from "omitted level" to the given level
        - copy table here
        
### Vocab
- null hypothesis
    - testable: can we reject the null?
- std err: estimates std deviation of coef
- t-stat: coef/se
- p-value (P>|t|): what is the probability that the non-zero beta is not zero, by random chance?
    - the lower it is, the more "certain" we can be that the relationship is no t zero
    - 1 star means p <10%, 2 means p<5%, 3 means p<1%
    - p<5% has been used as an indicator of a statistically significant relationship
    - if the p-value is not below 5%, treat the coefs as zero
- "economic significance" matters: stat sig but economically trivial



```python
import pandas as pd
from statsmodels.formula.api import ols as sm_ols
import numpy as np
import seaborn as sns
from statsmodels.iolib.summary2 import summary_col # nicer tables

```


```python
url = 'https://github.com/LeDataSciFi/ledatascifi-2023/blob/main/data/Fannie_Mae_Plus_Data.gzip?raw=true'
fannie_mae = pd.read_csv(url,compression='gzip') 
```

## Clean the data and create variables you want


```python
fannie_mae = (fannie_mae
                  # create variables
                  .assign(l_credscore = np.log(fannie_mae['Borrower_Credit_Score_at_Origination']),
                          l_LTV = np.log(fannie_mae['Original_LTV_(OLTV)']),
                          l_int = np.log(fannie_mae['Original_Interest_Rate']),
                          Origination_Date = lambda x: pd.to_datetime(x['Origination_Date']),
                          Origination_Year = lambda x: x['Origination_Date'].dt.year,
                          const = 1
                         )
                  .rename(columns={'Original_Interest_Rate':'int'}) # shorter name will help the table formatting
             )

# create a categorical credit bin var with "pd.cut()"
fannie_mae['creditbins']= pd.cut(fannie_mae['Co-borrower_credit_score_at_origination'],
                                 [0,579,669,739,799,850],
                                 labels=['Very Poor','Fair','Good','Very Good','Exceptional'])

```

## Statsmodels

As before, the psuedocode:
```python
model = sm_ols(<formula>, data=<dataframe>)
result=model.fit()

# you use result to print summary, get predicted values (.predict) or residuals (.resid)
```

Now, let's save each regression's result with a different name, and below this, output them all in one nice table:


```python
# one var: 'y ~ x' means fit y = a + b*X

reg1 = sm_ols('int ~  Borrower_Credit_Score_at_Origination ', data=fannie_mae).fit()

reg1b= sm_ols('int ~  l_credscore  ',  data=fannie_mae).fit()

reg1c= sm_ols('l_int ~  Borrower_Credit_Score_at_Origination  ',  data=fannie_mae).fit()

reg1d= sm_ols('l_int ~  l_credscore  ',  data=fannie_mae).fit()

# multiple variables: just add them to the formula
# 'y ~ x1 + x2' means fit y = a + b*x1 + c*x2
reg2 = sm_ols('int ~  l_credscore + l_LTV ',  data=fannie_mae).fit()

# interaction terms: Just use *
# Note: always include each variable separately too! (not just x1*x2, but x1+x2+x1*x2)
reg3 = sm_ols('int ~  l_credscore + l_LTV + l_credscore*l_LTV',  data=fannie_mae).fit()
      
# categorical dummies: C() 
reg4 = sm_ols('int ~  C(creditbins)  ',  data=fannie_mae).fit()

reg5 = sm_ols('int ~  C(creditbins)  -1', data=fannie_mae).fit()

```

Ok, time to output them:


```python
# now I'll format an output table
# I'd like to include extra info in the table (not just coefficients)
info_dict={'R-squared' : lambda x: f"{x.rsquared:.2f}",
           'Adj R-squared' : lambda x: f"{x.rsquared_adj:.2f}",
           'No. observations' : lambda x: f"{int(x.nobs):d}"}

# q4b1 and q4b2 name the dummies differently in the table, so this is a silly fix
reg4.model.exog_names[1:] = reg5.model.exog_names[1:]  

# This summary col function combines a bunch of regressions into one nice table
print('='*108)
print('                  y = interest rate if not specified, log(interest rate else)')
print(summary_col(results=[reg1,reg1b,reg1c,reg1d,reg2,reg3,reg4,reg5], # list the result obj here
                  float_format='%0.2f',
                  stars = True, # stars are easy way to see if anything is statistically significant
                  model_names=['1','2',' 3 (log)','4 (log)','5','6','7','8'], # these are bad names, lol. Usually, just use the y variable name
                  info_dict=info_dict,
                  regressor_order=[ 'Intercept','Borrower_Credit_Score_at_Origination','l_credscore','l_LTV','l_credscore:l_LTV',
                                  'C(creditbins)[Very Poor]','C(creditbins)[Fair]','C(creditbins)[Good]','C(creditbins)[Vrey Good]','C(creditbins)[Exceptional]']
                  )
     )
```

    ============================================================================================================
                      y = interest rate if not specified, log(interest rate else)
    
    ============================================================================================================
                                            1        2      3 (log) 4 (log)     5         6        7        8   
    ------------------------------------------------------------------------------------------------------------
    Intercept                            11.58*** 45.37*** 2.87***  9.50***  44.13*** -16.81*** 6.65***         
                                         (0.05)   (0.29)   (0.01)   (0.06)   (0.30)   (4.11)    (0.08)          
    Borrower_Credit_Score_at_Origination -0.01***          -0.00***                                             
                                         (0.00)            (0.00)                                               
    l_credscore                                   -6.07***          -1.19*** -5.99*** 3.22***                   
                                                  (0.04)            (0.01)   (0.04)   (0.62)                    
    l_LTV                                                                    0.15***  14.61***                  
                                                                             (0.01)   (0.97)                    
    l_credscore:l_LTV                                                                 -2.18***                  
                                                                                      (0.15)                    
    C(creditbins)[Very Poor]                                                                             6.65***
                                                                                                         (0.08) 
    C(creditbins)[Fair]                                                                         -0.63*** 6.02***
                                                                                                (0.08)   (0.02) 
    C(creditbins)[Good]                                                                         -1.17*** 5.48***
                                                                                                (0.08)   (0.01) 
    C(creditbins)[Exceptional]                                                                  -2.25*** 4.40***
                                                                                                (0.08)   (0.01) 
    C(creditbins)[Very Good]                                                                    -1.65*** 5.00***
                                                                                                (0.08)   (0.01) 
    R-squared                            0.13     0.12     0.13     0.12     0.13     0.13      0.11     0.11   
    R-squared Adj.                       0.13     0.12     0.13     0.12     0.13     0.13      0.11     0.11   
    R-squared                            0.13     0.12     0.13     0.12     0.13     0.13      0.11     0.11   
    Adj R-squared                        0.13     0.12     0.13     0.12     0.13     0.13      0.11     0.11   
    No. observations                     134481   134481   134481   134481   134481   134481    67366    67366  
    ============================================================================================================
    Standard errors in parentheses.
    * p<.1, ** p<.05, ***p<.01


# Today. Work in groups. Refer to the lectures. 

You might need to print out a few individual regressions with more decimals.

1. Interpret coefs in model 1-4
1. Interpret coefs in model 5
1. Interpret coefs in model 6 (and visually?)
1. Interpret coefs in model 7 (and visually? + comp to table)
1. Interpret coefs in model 8 (and visually? + comp to table)
1. Add l_LTV  to Model 8 and interpret (and visually?)






```python
reg1.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>int</td>       <th>  R-squared:         </th>  <td>   0.126</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.126</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>1.938e+04</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 02 Apr 2023</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>13:25:06</td>     <th>  Log-Likelihood:    </th> <td>-2.1575e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>134481</td>      <th>  AIC:               </th>  <td>4.315e+05</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>134479</td>      <th>  BIC:               </th>  <td>4.315e+05</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>      <td> </td>     
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>     
</tr>
</table>
<table class="simpletable">
<tr>
                    <td></td>                      <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>                            <td>   11.5819</td> <td>    0.046</td> <td>  253.270</td> <td> 0.000</td> <td>   11.492</td> <td>   11.671</td>
</tr>
<tr>
  <th>Borrower_Credit_Score_at_Origination</th> <td>   -0.0086</td> <td> 6.14e-05</td> <td> -139.198</td> <td> 0.000</td> <td>   -0.009</td> <td>   -0.008</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>2660.479</td> <th>  Durbin-Watson:     </th> <td>   0.397</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>2660.737</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 0.321</td>  <th>  Prob(JB):          </th> <td>    0.00</td>
</tr>
<tr>
  <th>Kurtosis:</th>       <td> 2.750</td>  <th>  Cond. No.          </th> <td>1.04e+04</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 1.04e+04. This might indicate that there are<br/>strong multicollinearity or other numerical problems.




```python
reg1.summary()
reg1.params[0] + reg1.params[1]*700
reg1b.params[0]
reg1c.summary()
5.5957 * (1+reg1c.params[1])**7
5.5957 * reg1d.params[1]

```




    -6.670224772564381




```python
reg2.params[0] + np.log(700)*reg2.params[1] + reg2.params[2]
```




    5.072916802247792




```python
#at cred = 700, l_LTV = 4.2 avg
reg2.params[0] + np.log(707)*reg2.params[1] + 4.20*reg2.params[2]
```




    5.508163912410515




```python
#model 6
#int = -16 + 3.2logC + 14.61logLTV - 22logClogLTV
```


```python
#model 7
# at 580 cred score --> fair
# left of 580 --> poor
# at 740 --> very good
# int now at 5
```

- Model 1
    - intercept --> an interet of 11.5 percentage points if credit = 0
    - "A 1 unit increase in credit score is associated with a decrease of 0.86 basis points in interest rates"
    - @X=700, E(y) is 5.5957
    - 700 to 707 --> int rate falls ~6 basis points
- Model 2
    - intercept --> an interet of 45.6 percentage points if credit = 1 (log(cred)=0)
    - "A 1% increase in credit score is associated with a decrease of 6.07 basis points in interest rates"
    - @X=700, E(y) is
    - 
- Model 3
    - intercept --> an interest of 2.87 if credit = 0
    - A 1 unit increase in credit score is associated with a proportional decrease of 0.17% in interest rates
    - @X=700, E(y) is 5.5957
    - 700 to 707 --> int rate falls ~6 basis points
- Model 4
    - intercept --> an interest of 9.50 if credit = 1
    - A 1% increase in credit score is associated with a proportional decrease of 1.19% in interest rates
    - @X=700, E(y) is 5.5957
    - 700 to 707 --> int rate falls ~6 basis points
- Model 5
    - intercept --> an intercept of 44.3
    - A 1% increase in credit score is associated with a decrease of 5.59 basis points in interest rates holding LOG(LTV) CONSTANT
    - @X=700, E(y) is 5.5957
    - 700 to 707 --> int rate fall ~6 basis points
- Model 6
    - intercept --> an intercept of -16.81
    - int rate fall ~6 basis points
- Model 7
    - intercept --> an intercept of 6.65
    - at 580 cred score --> fair
    - left of 580 --> poor
    - at 740 --> very good
    - int rate is 1.1 lower than very poor
- Model 8
    - intercept --> no intercept, indicator for very poor included


```python
fannie_mae[['Borrower_Credit_Score_at_Origination','int']].describe()

0.4276/fannie_mae['int'].mean()  #.08
0.4276/fannie_mae['int'].std()   #.33
```




    0.3314997324254188




```python
sm_ols('int ~ Borrower_Credit_Score_at_Origination', 
       data=fannie_mae.eval('Borrower_Credit_Score_at_Origination = Borrower_Credit_Score_at_Origination/1')).fit().summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>int</td>       <th>  R-squared:         </th>  <td>   0.126</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.126</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>1.938e+04</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Wed, 29 Mar 2023</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>13:18:03</td>     <th>  Log-Likelihood:    </th> <td>-2.1575e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>134481</td>      <th>  AIC:               </th>  <td>4.315e+05</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>134479</td>      <th>  BIC:               </th>  <td>4.315e+05</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>      <td> </td>     
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>     
</tr>
</table>
<table class="simpletable">
<tr>
                    <td></td>                      <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>                            <td>   11.5819</td> <td>    0.046</td> <td>  253.270</td> <td> 0.000</td> <td>   11.492</td> <td>   11.671</td>
</tr>
<tr>
  <th>Borrower_Credit_Score_at_Origination</th> <td>   -0.0086</td> <td> 6.14e-05</td> <td> -139.198</td> <td> 0.000</td> <td>   -0.009</td> <td>   -0.008</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>2660.479</td> <th>  Durbin-Watson:     </th> <td>   0.397</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>2660.737</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 0.321</td>  <th>  Prob(JB):          </th> <td>    0.00</td>
</tr>
<tr>
  <th>Kurtosis:</th>       <td> 2.750</td>  <th>  Cond. No.          </th> <td>1.04e+04</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 1.04e+04. This might indicate that there are<br/>strong multicollinearity or other numerical problems.



## Summary
- the goal is not predicting or model fit
- the goal is to understand what variables matter
