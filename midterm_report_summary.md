## Summary

The overall purpose of this project was to compare the sentiment value of a 10K to the returns of that companies stocks. Sentiment dictionaries are used to help analyze the files. 

## Data
### Sample
The sample used in this project are the 10K files of the S&P 500 companies. The variables created include 10 contextual variables (5 topics and then a positive and negative variable for each of those) and then 2 versions of buy and hold. 

### Return Variables
The return variables are to be created using the date from using the business days to find the buy and hold return of the timespan using methods from Assignment 2. The return dates are to be from around the time the 10K was released as to be able to measure the impact it.

### Sentiment Variables
For my contextual variables I created strings of lists related to COVID-19, Ukraine, the encomomy, customers, and sales to find sentiment in the 10Ks based on those. I chose those contextual variables as they are events that have had universal impacts on companies in the past few years. COVID-19 impacted companies largely in the past few years as it resulted in a global lockdown and people had to work from home and businesses were hurt. I wanted to measure this variable as companies are starting to be less remote and more in person and wanted to see how that impacts the company. During 2022 Russia invaded Ukraine and that impacted Europe and the world and since it began in 2022 I wanted to see how that would impact the returns of companies. The economy is also a variable I wanted to measure due to the inflation rising we saw in 2022 and the Fed having raised the basis points. Finally, I chose to measure customers and sales because those are the overall driving points of a business and I was intereseted in seeing how that would impact the returns as well.
