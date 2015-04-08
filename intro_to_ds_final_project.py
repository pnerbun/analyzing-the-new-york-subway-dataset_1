import pandas as pd
import numpy as np
import scipy
import scipy.stats
from ggplot import *
from datetime import *
import matplotlib.pyplot as plt


def mann_whitney(turnstile_data): # Compare 2 samples of non-normal distributions to validate a null hypothesis
    
    with_rain_mean = np.mean(turnstile_data[turnstile_data.rain == 1]['ENTRIESn_hourly']) # Calculate sample mean of entries while raining
    without_rain_mean = np.mean(turnstile_data[turnstile_data.rain == 0]['ENTRIESn_hourly']) # Calculate sample mean of entries while not raining
    print "Mean entries, with rain: {0}\nMean entries, without rain: {1}".format(with_rain_mean, without_rain_mean)

    # Perform Mann-Whitney U test, return U-statistic and p value
    [U, p] = scipy.stats.mannwhitneyu(turnstile_data[turnstile_data.rain == 1]['ENTRIESn_hourly'],
                                      turnstile_data[turnstile_data.rain == 0]['ENTRIESn_hourly'])
    print "Mann-Whitney Test Statistic: {0}\np-Value: {1}".format(U, p * 2)

    alpha = 0.05 # Set significance level (alpha) equal to .05
    if (p * 2) < alpha: # Run two-tailed test to validate hypothesis (multiply p by 2 for 2-tailed)
        print 'Reject the null hypothesis'
    else:
        print 'Fail to reject null hypothesis'

def entries_histogram(turnstile_data): #Plot histogram showing our data follows a non-normal distribution
    plt.figure()
    plt.xlim([0,5500])
    plt.ylim([0,40000])
    df_rain = turnstile_data[turnstile_data['rain'] == 1]['ENTRIESn_hourly']
    df_without_rain = turnstile_data[turnstile_data['rain'] == 0]['ENTRIESn_hourly'] 
    df_without_rain.hist(color = 'yellow', bins = 250)
    df_rain.hist(color = 'blue', bins = 250)
    plt.legend(['no rain', 'rain'])
    plt.xlabel('ENTRIESn_hourly')
    plt.ylabel('Frequency')
    plt.title(r'ENTRIESn_hourly HISTOGRAM (by RAIN), $n = 1000$')
    
def normalize_features(array):

   # Normalizes the features in the data set.

   mu = array.mean()
   sigma = array.std()
   array_normalized = (array - mu)/sigma

   return array_normalized, mu, sigma
 
def compute_cost(features, values, theta): # Compute the cost of a set of parameters, theta, given a list of features and values
    m = len(values) # This is how many data points we have
    sum_of_square_errors = np.square(np.dot(features, theta) - values).sum()
    cost = sum_of_square_errors / (2*m)
    return cost
 
def gradient_descent(features, values, theta, alpha, num_iterations):
    
    #performs gradient descent given a data set with an arbitrary number of features.
    
    m = len(values) # This is how many data points we have.
    cost_history = [] # Tracks how our cost function evolves over our iterations.

    for i in range(num_iterations): # For each iteration we want to update theta and cost
        predicted_values = np.dot(features, theta) # Take the dot product of features and theta
        theta = theta - (alpha/m) * np.dot((predicted_values - values), features) # Update the values of theta
        cost = compute_cost(features, values, theta) # Use compute_cost function from above
        cost_history.append(cost) # Append newest cost from cost history
    return theta, pd.Series(cost_history)

def predictions(dataframe):
    
    # Runs predictions via gradient descent on turnstile dataframe
    
    # Select Features 
    features = dataframe[['rain', 'precipi', 'meanwindspdi', 'meantempi', 'meanpressurei', 'meandewpti']]

    # Add UNIT and Hour to features using dummy variables
    dummy_units = pd.get_dummies(dataframe['UNIT'], prefix='unit')
    dummy_units2 = pd.get_dummies(dataframe['Hour'], prefix='hour')
    features = features.join(dummy_units).join(dummy_units2)
    print len(features)

    # Values
    values = dataframe['ENTRIESn_hourly']
    m = len(values)

    features, mu, sigma = normalize_features(features)
    features['ones'] = np.ones(m) # Add a column of 1s (y intercept)

    features_array = np.array(features)# Convert features to numpy array
    values_array = np.array(values) # Convert values to numpy array

    # Set values for alpha, number of iterations.
    alpha = 0.1 
    num_iterations = 75 

    # Initialize theta, perform gradient descent
    theta_gradient_descent = np.zeros(len(features.columns))
    theta_gradient_descent, cost_history = gradient_descent(features_array,
                                                            values_array,
                                                            theta_gradient_descent,
                                                            alpha,
                                                            num_iterations)
    print theta_gradient_descent
    plot = plot_cost_history(alpha, cost_history)
    predictions = np.dot(features_array, theta_gradient_descent)
    return predictions, plot

def plot_cost_history(alpha, cost_history):
   cost_df = pd.DataFrame({'Cost_History': cost_history,'Iteration': range(len(cost_history))})
   return ggplot(cost_df, aes('Iteration', 'Cost_History')) + geom_point() + ggtitle('Cost History for alpha = %.3f' % alpha )
 

def plot_residuals(turnstile_data, predicted_values):
    plt.figure()
    (turnstile_data['ENTRIESn_hourly'] - predicted_values).hist(bins = 200)
    plt.suptitle('Residual histogram')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.xlim([-15000,15000])
    return plt


def compute_r_squared(data, predictions):
    
    SST = ((data-np.mean(data)) ** 2).sum()
    SSReg = ((predictions-data) ** 2).sum()
    r_squared = 1 - SSReg / SST
    return r_squared

def plot_ridership(turnstile_data):
    
    df['Day'] = df['DATEn'].map(lambda x:datetime.strptime(x, '%Y-%m-%d').strftime('%w'))
    agg = df.groupby(['Day'], as_index=False).aggregate(np.mean)
    
    plot = ggplot(agg, aes(x='Day', y='ENTRIESn_hourly')) + geom_line(color = 'red') +\
           ggtitle('Average NYC Subway ridership by day of week') +\
           scale_x_continuous(name="Weekday", breaks=[0, 1, 2, 3, 4, 5, 6],
                            labels=["Sunday", "Monday","Tuesday","Wednesday", 
                                    "Thursday", "Friday", "Saturday"]) + ylab('Entries (Avg Riders)')
    return plot

    
    
if __name__ == '__main__':
    
    # Load CSV file from folder on C drive
    turnstile_data = pd.read_csv("C:/Users/Home/.spyder2/turnstile_data_master_with_weather.csv")
    df = turnstile_data
    
    entries_histogram(turnstile_data)
    mann_whitney(turnstile_data)
    print "Linear regression predictions via gradient descent:"
    predicted, plot = predictions(df)
    print plot
    raw_input("Press enter to continue")

    print "Plotting residuals:"
    plot_residuals(df, predicted)
    plt.show()
    raw_input("Press enter to continue")

    print "R-squared value:"
    print compute_r_squared(df['ENTRIESn_hourly'], predicted)
    print plot_ridership(turnstile_data)