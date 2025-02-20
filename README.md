# STATS 170 Final Project

## Forecasting 2022 U.S. Unemployment Rate Based on Housing Supply Rate and Interest Rate Using Federal Reserve Economic Data 

**Author:** *Ellen Wei, Robin Lee, Xuxin Zhang*

**Date:** *February 27, 2023*

## 1 Introduction

With the number of layoffs in the tech industry escalating throughout the United States following a nation-wide increase in interest rates, we have hypothesized that interest rate is definitely a significant factor that noticeably affects the unemployment rate—raising the question, what other circumstances influence employment? Thus, by applying time series analysis, we attempt to answer the following: *what are other potential factors that impact the unemployment rate?*

In this research assignment, in addition to interest rate, we have picked the monthly supply of new houses as another independent variable. The intuition behind this choice is that we believe when a massive layoff happens—such as the one recently in 2021—people tend to sell their real estate, leading to a significant increase in the housing supply. This is especially true and exemplified during the 2008 Financial Crisis.

As such, we are curious as to whether or not this phenomenon is a common pattern. We have chosen three time series datasets: interest rate, unemployment, and housing supply. Below is a table summarizing the time series datasets we have selected from the Federal Reserve Economic Data (FRED). Data coverage includes all major areas of macroeconomic analysis: growth, inflation, employment, interest rates, exchange rates, production and consumption, income and expenditure, savings and investment, and more

| Variable Code | Description of variable measures, frequency, source | Time Period |
|:--------------|:----------------------------------------------------|:------------|
| MSACSRNSA     | U.S. Census Bureau, Monthly<br>The months' supply is the ratio of new houses for sale to new houses sold. |1963:1-2022:12|
| UNRATENSA     | U.S. Bureau of Labor Statistics, Monthly<br>16+ years old, reside in 50 states or the District of Columbia, do not reside in institutions or are on active duty in the Armed Forces. |1948:1-2022:12|
| FEDFUNDS      | Board of Governors of the Federal Reserve System (US), Monthly<br>Interest rate at which depository institutions trade federal funds with each other overnight. |1954:7-2022:12|

After importing our data into R using the Quandl API, we can display the three time series data on the same plot, as shown in Plot 1. To do this, we first need to make sure that they have the same start time and end time. For simplicity, we picked the time range from `1980-01-01` to `2022-12-01` for all of the three datasets. In this way, we are able to view our full dataset.

Later, we will use part of the full dataset as the training data.

<img width="629" alt="image" src="https://github.com/user-attachments/assets/475b72d1-3b11-44a7-86b9-c104c180eeed" />

From Plot 1, we can see that the unemployment rate has a similar trend compared to both federal funds effective rate and housing supply ratio. To conduct analysis in later parts of this project, we have extracted the last 12 data points as the testing data and used the remaining data points as the training data. That is to say, we use the data from 1980-01 to 2018-01 as the training data and that from `2018-02` to `2019-1` as the testing data.

## 2 Components Feature Analysis

Because we want to see how new housing supply and interest rate will affect unemployment, we have chosen Unemployment Rate as our dependent variable. Based on the dyplot shown earlier (Plot 1), we can see that there is seasonality and no obvious trend in the unemployment data, as indicated by the green line.

### 2.1 Decomposition for Unemployment Training Data

That is to say, we know the seasonality does not increase with the trend. As a result, we can apply additive decomposition, instead of multiplicative decomposition, for our time series.

* The additive model is useful when the seasonal variation is relatively constant over time.
* The multiplicative model is useful when the seasonal variation increases over time.
  
In other words, the additive model is used when the variance of the time series does not change over different values of the time series. On the other hand, if the variance is higher when the time series is higher, then it often means we should use a multiplicative model.

<img width="629" alt="image" src="https://github.com/user-attachments/assets/c21d3687-fa52-4ee6-ab62-74bb94e5a759" />

We first plot the additive decomposition of the training data (Plot 2). We see that the range of fluctuation for the random term is relatively small (from -0.4 to 0.4), implying we do not need any transformation on the raw dataset; as such, an additive decomposition is good enough.

<img width="629" alt="image" src="https://github.com/user-attachments/assets/f456ed3e-b41b-469b-af15-02def33d1895" />

### 2.2 Seasonal Box Plot for Unemployment Training Data

A seasonal box plot is a graphical representation used to display the distribution of a dataset over time, specifically focusing on seasonal patterns. Each box plot within a seasonal box plot represents the distribution of the data for a particular time interval (e.g., each month of the year). The box itself represents the interquartile range (IQR), with the median marked by a line. Seasonal box plots are useful for visualizing seasonal patterns and identifying any variations or trends that occur within specific time intervals.

<img width="629" alt="image" src="https://github.com/user-attachments/assets/ead81411-9d37-4784-8b2a-a063c8d563b0" />

From the boxplot (Plot 3), we can see that the unemployment data does show certain seasonality. In particular, we can see that the unemployment rate drops from January to May, before increasing again in June. However, after this growth in June, the unemployment rate begins to drop again from July to December.

## 3 Autocorrelation Feature Analysis

Next, we plot the ACF and PACF graphs for the training data. The ACF (Autocorrelation Function) and PACF (Partial Autocorrelation Function) plots are graphical tools used in time series analysis to identify the autocorrelation structure of a time series dataset.

These plots help in determining the appropriate parameters for autoregressive integrated moving average (ARIMA) models, which are commonly used for forecasting.

<img width="629" alt="image" src="https://github.com/user-attachments/assets/be07b95b-013e-4b73-89bb-f00d3002bc32" />

Both ACF and PACF plots are crucial for determining the order of autoregressive (AR) and moving average (MA) terms in an ARIMA model. The patterns observed in these plots guide the selection of appropriate model parameters (p, d, q) for fitting the time series data.

Looking at the ACF graph (Plot 4), we notice that the autocorrelations for the random term stay significant until a larger lag, but it gradually cuts off.

<img width="629" alt="image" src="https://github.com/user-attachments/assets/3e91ac5c-497d-4c18-94ca-338ae8d1f6aa" />

For the regular part, it seems like an `ARMA(4, 2)` model. By looking at the ACF graph, the peak cuts off at lag 2 (`MA(2)`), and in the PACF, the peak cuts off at lag 4 — `AR(4)`. For the seasonal part, it looks like the `MA(3)` model.
By comparing the raw time series plots and the ACF plots, we believe an autoregressive process with an order of 1 with a parameter of 0.4 might generate our dependent variables; this is because the ACF graphs are similar.

## Exponential Smoothing Modeling and Forecasting












