# Predict currency exchange rate using RNN - part 1

![Touching the water]({{ '/assets/images/rnn-part-1/cat_touching_water.gif' | relative_url }})

*Fig. 1 The very first micro-step touching the Deep Learning pool*


### Table of contents

* [Introduction](#introduction)
* [Preparations](#preparations) 
* [Fetch and prepare the datasources](#fetch-and-prepare-the-datasources)
* [Build the model](#build-the-model)
	* [Data pre-processing](#data-pre-processing)
	* [Train the model](#train-the-model)
* [Make the prediction and evaluate the result](#make-the-prediction-and-evaluate-the-result)
* [Playing with the configuration](#playing-with-the-configuration)
* [What's next??](#whats-next)

## Introduction

We're going through steps on building a simple model using regression with RNN that can predict the currency exchange rate. In this article, we will use **USD ($)** and **THB (฿)**.

In this scope of the article, I follow the course (and source-code) of [Ligency-Team](https://www.udemy.com/user/ligency-team/). Therefore, I don't go into the detail of what is the RNN model, but about how to apply the concept to resolve a real-life problem. If you're interested in the course, please checkout [Deep Learning A-Z course](https://www.udemy.com/course/deeplearning/) from Ligency-Team.

[The full source code can be found here](https://github.com/HocTran/rnn-currency-exchange-rate).

As a beginner, I also have some troubles while building up and running the solution, so I want to share the setup as well. Of course, a little Python experience will help you go faster, but if you don't have, it's fine; you still get it quickly with basic programming skills, I promise 😉.

This article consists of several steps to go:

* Prepare the working environment.
* Get the real exchange rate data and prepare the data set.
* Build the model.
* Using the trained model to predict the exchange rate and evaluate the result.
* Adjust the model and compare the result.

> Remark1: The article focuses on using RNN to solve a problem, not optimizing the results.

> Remark2: The article is for learning material; the result is not considered a trusted source by any means 😉.


##  Preparations

The code was compiled under **Python v3.9.7**, with the following packages

* numpy (1.19.5)
* pandas (1.3.4)
* matplotlib (3.4.3)
* keras (2.4.3)
* scikit-learn (1.0)
* tensorflow (2.4.1)
* (and auto packages dependencies)

> Skip this step if you're already familiar with the setup.

I had a few troubles setting up my environment; especially I got lots of conflicts while installing the packages. So I end up using [**Anaconda**](https://medium.com/r/?url=https%3A%2F%2Fwww.anaconda.com%2Fproducts%2Findividual) to manage the environment.

See the Anaconda website to learn more about the detail. Here I wrap-up what I did:

* Download and install Anaconda Navigator.
* (Optional) Create a new working environment. This is strongly recommended (from my perspective). The default env contains many other packages you don't need for the project, which may lead to dependency conflicts.
* Then, activate your environment.
* Add a new channel **conda-forge** into the channel list.

    The default package hub doesn't have all packages or versions we need, we'll find them in the channel **conda-forge**.

* Search and install all packages (listed above)
* Launch the code editor from the home page.

    I preferably select Spyder, which I love its feature 'Variable Explorer'. But you can choose any.

Alright, that's it! You're now set for the coding.

> Alternative 1 - You can use Anaconda via the command-line interface.

> Alternative 2 - You can use Google Colab; the env is set and ready from the web browser.


## Fetch and prepare the datasources

In order to feed the train and evaluation, we need data. We will go to the [XE.com](https://medium.com/r/?url=https%3A%2F%2Fwww.xe.com%2Fcurrencycharts%2F).

> I also attached the train and test sets (rate_train.csv and rate_test.csv) in the repo, you can skip this step and go to pre-processing.

From XE.com, in the free charts section, select the source and destination currencies, in our case, they're **USD** and **THB** correspondingly. To make the prediction more precisely, we select the data in the past **10 years**. Then, inspect the network call, and we'll get the raw json for the rates. You can also find the attached json file in the source code `raw_10y.json`.

![Xe Currency Charts]({{ '/assets/images/rnn-part-1/xe_chart.png' | relative_url }})

*Fig. 2 USD to THB exchange rates*

Once we had the raw data, we split it into two parts, one for training and the other one for evaluating.

Open the **update_data_set.py**, which contains a few simple steps to split the data.


```python
#1. Import the libraries
import pandas as pd
import json
from datetime import datetime

#2. Parse the json
file = open('raw_10y.json')
fileData = json.load(file)

first = fileData['batchList'][0]

startTime = first['startTime']
interval = first['interval']
rates = first['rates'][1:]

#3. Mapping data
nRow = len(rates)
dates = []
for i in range(0, nRow):
    timestamp = startTime + interval * i
    dates.append(datetime.fromtimestamp(timestamp / 1000).strftime('%Y.%m.%d'))

dataframe = pd.DataFrame(list(zip(dates, rates)), columns =['date', 'rate'])

#4. Split to train and test set
trainSize = int(nRow * 0.8)
trainSet = dataframe.iloc[:trainSize, :]
testSet = dataframe.iloc[trainSize:, :]

#5. Write to files
trainSet.to_csv('rate_train.csv', index = False)
testSet.to_csv('rate_test.csv', index = False)
```

**Step #1**. Import libraries

* json lib is to parse json file we recorded earlier.
* pandas is to write data into csv format (and read it later).

**Step #2**. Parse the json

* startTime - the time at the beginning, it means 10y ago by the time data was fetched.
* interval - the time interval between every data row. In our case, it's one day.
* rates - they're rate records for the past 10 years. Note, we drop the first value here, as it doesn't present a rate.

**Step #3**. Mapping data. Prepare data into 2 columns of **date** and **rate**

**Step #4**. Split data set into the train and test sets. Here we use 80% of the samples (equivalent to the first 8 years) as the train set. The rest is for the test set (2 years).

**Step #5**. Write the data to the files.

Congratulations 🎉! You now have the training set and evaluating set ready.


## Build the model

Quick recap. Check out this [ultimate article](https://www.superdatascience.com/blogs/the-ultimate-guide-to-recurrent-neural-networks-rnn) about RNN and LSTM.

![RNN]({{ '/assets/images/rnn-part-1/rnn.png' | relative_url }})

*Fig 3. RNN*

Overview of constants

* **TIME_STEPS**. The number of LSTM inputs and outputs.
* **LSTM_UNITS**. The number of neurons in LSTM.
* **DROPOUT**. The rate that LSTM will drop when propagating.
* **EPOCHS**. The number of times model iterate the train.
* **BATCH_SIZE**. The number of data to feed in a batch.

At the later point, you can adjust these values to observe the different results.

### Data pre-processing

Even though, we already have training and test set, we need to do a bit more to make the data is feedable for the model.

During this step, we only work with the train set.

```python

#1. Importing the training set
dataset_train = pd.read_csv('rate_train.csv')
training_set = dataset_train.iloc[:, 1:2].values

#2. Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

#3. Creating a data structure with TIME_STEPS timesteps and 1 output
X_train = []
y_train = []
for i in range(TIME_STEPS, len(training_set)):
    X_train.append(training_set_scaled[i-TIME_STEPS:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

#4 Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
```

**Step #1**. Read data from the set. We again use **pandas** to read the data from the set. The code will get all row values from the rate column in the csv, turn it into the **numpyarray**.

**Step #2**. Next, we scale the numbers into the bound of **[0,1]**. There are a few scalers can use, here we involve the **MinMaxScaler** from **sklearn**.

**Step #3**. Natively, we convert the set into 2 parts, the **X_train** stands for the inputs, the **y_train** stands for the value which the model drives to.

In our RNN model, it works with the time series, as amount of previous steps will be used as inputs for one current step. Therefore each item in **X_train** consists of an array with previous **TIME_STEPS** values. The **y_train** is simply inserting the plain values.

![X_train]({{ '/assets/images/rnn-part-1/x_train.png' | relative_url }})

Alright, pretty much good, but not done yet!

**Step #4**. Reshape.

We need to define the number of indicators for the input. In our case, we only have one column of the rate, so the indicator equals 1. Therefore, in #4, we reshape the **X_train** with the number of indicators.

Done! We have good data we need to feed the system.

### Train the model

```python
#1. Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#2. Initialising the RNN
regressor = Sequential()

#3. Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = LSTM_UNITS, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(DROPOUT))

#3.x Adding a few more granular LSTM layer....
regressor.add(LSTM(units = LSTM_UNITS, return_sequences = True))
regressor.add(Dropout(DROPOUT))

#3.x Adding the last LSTM layer
regressor.add(LSTM(units = LSTM_UNITS))
regressor.add(Dropout(DROPOUT))

#4. Adding the output layer
regressor.add(Dense(units = 1))

#5. Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

#6. Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = EPOCHS, batch_size = BATCH_SIZE)
```

**Step #1**. Using **keras** to import classes support to build the network.

**Step #2**. Initialize the RNN

**Step #3**. Add a number of LSTM layers. Only the first one will get the input shape of the train data, all the following won't need. The last layer doesn't have `return_sequences`.

**Step #4**. Add the output layer with 1 output.

**Step #5**. Compile with selected optimizier and loss function.

**Step #6**. Train the set.

It's time to get a coffee ☕️, and wait for the machine learning...

![waiting_model_training]({{ '/assets/images/rnn-part-1/mr-bean-waiting.gif' | relative_url }})

## Make the prediction and evaluate the result

Phewww! After a while, we finally got the trained regressor and it's ready for prediction.

In this section, we will use the trained regressor to predict the exchange rate in 2 years, and compare it with the real data in the test set. This is probably the most exciting part. OK, let's go 🏃‍♂️!

**Note, it's different from the interpolation that tries to get as close as possible to the real value, our RNN is trying to predict the direction from the values it learned in the past. It completely doesn't know what the real values are when making the prediction. In other words, the real value and the predicted one may be very different!!**

First, we need to load and pre-process the test data. Then, we do almost the same thing with the training set, but apply for the test set. (Load the test set, apply the scaler, and reshape). But then, we don't feed the **X_test** into the RNN, but use the trained regressor to predict the values.

```python
predicted_rate = regressor.predict(X_test)
```

Great, then we got the predicted values!! But wait, if you remember, we have the **MinMaxScaler** applied on the set, which means the outcome will be scaled within **[0,1]** bound. Therefore, we need to inverse it to get the real value. To do that, we reuse the same scaler and do the inverse transformation.

```python
predicted_rate  = sc.inverse_transform(predicted_rate)
```

Finally! Now, it's time to get a comparison. We use matplotlib to visualize the result in the same space, where we can see the differences between real and prediction values.

![Prediction]({{ '/assets/images/rnn-part-1/prediction-4-60-50-0.2-5.png' | relative_url }})

*Fig5. Real and Prediction in a comparison*

Congratulation 🎉, the machine is working! Even though there are a few peaks at some points, generally, the model truly can predict!

## Playing with the configuration

In the source code, we have a few constants defined and adjust them to evaluate and get better (or worse) results. Basically, there are a few points we can think about

* Increase the dataset. Yes, more data will help to train the better model.
* Increase the number of time steps, where the time item will depend furthermore on the previous one, f.ex 100, 200.
* Increase the number of layers and number of its neurons.
* Decrease the rate in the dropout.
* Increase the number of epochs.

Let's see a few adjustments



| ![Prediction]({{ '/assets/images/rnn-part-1/prediction-2-10-10-0.8-5.png'}}) | ![Prediction]({{ '/assets/images/rnn-part-1/prediction-4-30-20-0.1-20.png'}}) |
|:----------------------------------------------------------------------------:|:-----------------------------------------------------------------------------:|
|                             (4, 10, 10, 0.8, 5)                              |                             (4, 30, 20, 0.1, 20)                              |
|![Prediction]({{ '/assets/images/rnn-part-1/prediction-4-60-50-0.1-25.png' | relative_url }})|![Prediction]({{ '/assets/images/rnn-part-1/prediction-4-120-100-0.1-100.png' | relative_url }})|
|(4, 60, 50, 0.1, 25)|(4, 120, 100, 0.1, 100)|

*Fig 6. Configuration = (LSTM layers, TIME_STEPS, LSTM_UNITS, DROPOUT, EPOCH)*


## What's next??

In this article, we built a model that uses the time interval of 1 day. Therefore, it can predict the value for one day using the number of days in the past. However, what if we want to predict the exchange rate next week or next month 🤔. Then, we need a bit more. Let's discover it in part 2 😉.

Diving further! And practising!

![Diving]({{ '/assets/images/rnn-part-1/diving.gif' | relative_url }})