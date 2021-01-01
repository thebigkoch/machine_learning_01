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

def predict_monthly_payment(principal: float, interest_rate: float, number_of_payments: float, model: tensorflow.keras.Sequential) -> float:
    return model.predict([(principal, interest_rate, number_of_payments)])[0][0]

def compare_values(model: tensorflow.keras.Sequential):
    print("Actual = ${:,.2f}, Predicted = ${:,.2f}".format(calculate_monthly_payment(125000, 0.075, 360), predict_monthly_payment(125000, 0.075, 360, model)))
    print("Actual = ${:,.2f}, Predicted = ${:,.2f}".format(calculate_monthly_payment(125000, 0.750, 360), predict_monthly_payment(125000, 0.750, 360, model)))
    print("Actual = ${:,.2f}, Predicted = ${:,.2f}".format(calculate_monthly_payment(150000, 0.075, 360), predict_monthly_payment(150000, 0.075, 360, model)))
    print("Actual = ${:,.2f}, Predicted = ${:,.2f}".format(calculate_monthly_payment(150000, 0.750, 360), predict_monthly_payment(150000, 0.750, 360, model)))
    print("Actual = ${:,.2f}, Predicted = ${:,.2f}".format(calculate_monthly_payment(200000, 0.075, 360), predict_monthly_payment(200000, 0.075, 360, model)))
    print("Actual = ${:,.2f}, Predicted = ${:,.2f}".format(calculate_monthly_payment(200000, 0.750, 360), predict_monthly_payment(200000, 0.750, 360, model)))
    print("Actual = ${:,.2f}, Predicted = ${:,.2f}".format(calculate_monthly_payment(900000, 0.075, 360), predict_monthly_payment(900000, 0.075, 360, model)))
    print("Actual = ${:,.2f}, Predicted = ${:,.2f}".format(calculate_monthly_payment(900000, 0.750, 360), predict_monthly_payment(900000, 0.750, 360, model)))

def print_values(principal: float, interest_rate: float, number_of_payments: float, monthly_payment: float) -> None:
        print("Principal = ${:,}".format(principal))
        print("Interest Rate = {:.2f}%".format(interest_rate * 100))
        print("Number of Payments = {}".format(number_of_payments))
        print("Monthly Payment = ${:,.2f}".format(monthly_payment))
        print("-----")

def main():
    for i, value in enumerate(parameters):
        print_values(value[0], value[1], value[2], calculate_monthly_payment(value[0], value[1], value[2]))

    # Create a model with many layers with three neurons apiece.
    layers = []

    for i in range(9):
        layers.append(tensorflow.keras.layers.Dense(units=3, input_shape=[3,]))

    # The last layer has only one neuron.
    layers.append(tensorflow.keras.layers.Dense(units=1, input_shape=[3,]))

    # Define the order in which the model will run the various layers.
    model = tensorflow.keras.Sequential(layers)

    # Before it can be executed, the model must be run.
    model.compile(loss='mean_squared_error',
                optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.001))

    # Then use the fit function to train the model.
    history = model.fit(parameters, payments, epochs=500, verbose=False)
    print("Finished training the model")

    # See our layer's weights
    for i, layer in enumerate(layers):
        print("{}: {}".format(i, layer.get_weights()))

    # Use the model to predict a value
    compare_values(model)

if __name__ == "__main__":
    main()
