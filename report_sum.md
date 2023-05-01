{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "0fb730e6-fe48-4e65-8ba3-f2f8c27f262f",
   "metadata": {},
   "source": [
    "## Summary"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d461b714-9f5a-4852-af65-7c1f313c4960",
   "metadata": {},
   "source": [
    "The overall purpose of this project was to compare the sentiment value of a 10K to the returns of that companies stocks. Sentiment dictionaries are used to help analyze the files. "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8a6bff94-4b07-4f09-a6b3-28a0500b5214",
   "metadata": {},
   "source": [
    "## Data"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f3d8dd8b-96da-400c-8507-ffb3d19c6298",
   "metadata": {},
   "source": [
    "### Sample"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d5dec388-9f0d-4b19-a023-642215f7792a",
   "metadata": {},
   "source": [
    "The sample used in this project are the 10K files of the S&P 500 companies. The variables created include 10 contextual variables (5 topics and then a positive and negative variable for each of those) and then 2 versions of buy and hold. "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9ff67911-e3c7-4311-bbd3-e9e51d6dd0f7",
   "metadata": {},
   "source": [
    "### Return Variables"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fc19481a-bbf8-478c-a836-7240a13ccb56",
   "metadata": {},
   "source": [
    "The return variables are to be created using the date from using the business days to find the buy and hold return of the timespan using methods from Assignment 2. The return dates are to be from around the time the 10K was released as to be able to measure the impact it."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "90421ce0-7bd8-4cc1-af09-168de83977c9",
   "metadata": {},
   "source": [
    "### Sentiment Variables"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c8118d98-3bb2-49a0-8fa2-3d21c02de08c",
   "metadata": {},
   "source": [
    "For my contextual variables I created strings of lists related to COVID-19, Ukraine, the encomomy, customers, and sales to find sentiment in the 10Ks based on those. I chose those contextual variables as they are events that have had universal impacts on companies in the past few years. COVID-19 impacted companies largely in the past few years as it resulted in a global lockdown and people had to work from home and businesses were hurt. I wanted to measure this variable as companies are starting to be less remote and more in person and wanted to see how that impacts the company. During 2022 Russia invaded Ukraine and that impacted Europe and the world and since it began in 2022 I wanted to see how that would impact the returns of companies. The economy is also a variable I wanted to measure due to the inflation rising we saw in 2022 and the Fed having raised the basis points. Finally, I chose to measure customers and sales because those are the overall driving points of a business and I was intereseted in seeing how that would impact the returns as well."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
