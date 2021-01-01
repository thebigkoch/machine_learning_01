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

parameters = [(100000, 0.005, 360),
              (150000, 0.005, 360),
              (200000, 0.005, 360),
              (100000, 0.025, 360),
              (400000, 0.020, 240),
              (450000, 0.020, 240),
              (20000, 0.045, 60),
              (45000, 0.035, 60),
              (90000, 0.025, 60)]

payments = [599.55,
            899.33,
            1199.10,
            2500.34,
            8069.63,
            9078.34,
            969.09,
            1803.99,
            2911.81]

def calculate_monthly_payment(principal: float, interest_rate: float, number_of_payments: float) -> float:
    return principal * (interest_rate * ((1.0 + interest_rate) ** number_of_payments) / ((1.0 + interest_rate) ** number_of_payments - 1.0))

def main():
    for i, value in enumerate(parameters):
        principal = value[0]
        interest_rate = value[1]
        number_of_payments = value[2]
        print("Principal = ${:,}".format(principal))
        print("Interest Rate = {:.2f}%".format(interest_rate * 100))
        print("Number of Payments = {}".format(number_of_payments))
        print("Monthly Payment = ${:,.2f}".format(calculate_monthly_payment(principal, interest_rate, number_of_payments)))
        print("-----")

    # Create a model with a single layer with a single neuron.
    l0 = tensorflow.keras.layers.Dense(units=1, input_shape=[3,])

    # Define the order in which the model will run the various layers.
    model = tensorflow.keras.Sequential([l0])

    # Before it can be executed, the model must be run.
    model.compile(loss='mean_squared_error',
                optimizer=tensorflow.keras.optimizers.Adam(0.1))

    # Then use the fit function to train the model.
    history = model.fit(parameters, payments, epochs=500, verbose=False)
    print("Finished training the model")

    # Use the model to predict a value
    print(model.predict([(125000, 0.075, 360)]))

    # See our layer's weights
    print(l0.get_weights())

if __name__ == "__main__":
    main()
