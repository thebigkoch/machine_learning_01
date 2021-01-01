# This sample project uses TensorFlow to predict the monthly payment for an
# amortized loan.  The formula is:
#
# M = P * [r * ((1 + r) ^ n) / ((1 + r) ^ n - 1)]
#
# M = the total monthly mortgage payment.
# P = the principal loan amount.
# r = the monthly interest rate.
# n = number of payments.

import tensorflow
import logging

logger = tensorflow.get_logger()
logger.setLevel(logging.ERROR)

celsius_q = [-40, -10, 0, 8, 15, 22, 38]
fahrenheit_a = [-40, 14, 32, 46, 59, 72, 100]

def calculate_monthly_payment(principal: float, interest_rate: float, number_of_payments: float) -> float:
    return principal * (interest_rate * ((1 + interest_rate) ^ number_of_payments) / ((1 + interest_rate) ^ number_of_payments - 1))

def main():

    for i,c in enumerate(celsius_q):
        print("{} degrees Celsius = {} degree Fahrenheit".format(c, fahrenheit_a[i]))

    # Create a model with a single layer with a single neuron.
    l0 = tensorflow.keras.layers.Dense(units=1, input_shape=[1])

    # Define the order in which the model will run the various layers.
    model = tensorflow.keras.Sequential([l0])

    # Before it can be executed, the model must be run.
    model.compile(loss='mean_squared_error',
                optimizer=tensorflow.keras.optimizers.Adam(0.1))

    # Then use the fit function to train the model.
    history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
    print("Finished training the model")

    # Use the model to predict a value
    print(model.predict([100.0]))

    # See our layer's weights
    print(l0.get_weights())

if __name__ == "__main__":
    main()
