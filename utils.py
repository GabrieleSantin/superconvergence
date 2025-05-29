import numpy as np


#%%
def ker_matrix(X, Y, a, b):
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    s3 = np.sqrt(3)

    X = X[:, np.newaxis]  # shape (N, 1)
    Y = Y[np.newaxis, :]  # shape (1, M)

    X_plus_Y = X + Y
    X_minus_Y = X - Y

    exp_s3_X_plus_Y = np.exp(-s3/2 * X_plus_Y)
    z = s3 * exp_s3_X_plus_Y / 6 / (np.exp(s3 * b) - np.exp(s3 * a))

    # Masks for x <= y and x > y
    mask = (X <= Y)

    # For x <= y, use (a, b)
    left_ab = (
        2 * np.exp(s3 * X_plus_Y) +
        np.exp(s3 * (X + b)) +
        np.exp(s3 * (Y + a)) +
        2 * np.exp(s3 * (a + b))
    )
    right_ab = s3 * (np.exp(s3 * (X + b)) - np.exp(s3 * (Y + a)))

    # For x > y, swap a <-> b
    left_ba = (
        2 * np.exp(s3 * X_plus_Y) +
        np.exp(s3 * (X + a)) +
        np.exp(s3 * (Y + b)) +
        2 * np.exp(s3 * (b + a))  # same as 2 * exp(s3 * (a + b))
    )
    right_ba = s3 * (np.exp(s3 * (X + a)) - np.exp(s3 * (Y + b)))

    # Cos and sin terms
    cos_term = np.cos(X_minus_Y / 2)
    sin_term = np.sin(X_minus_Y / 2)

    # Combine everything using masking
    val = np.where(
        mask,
        z * (left_ab * cos_term - right_ab * sin_term),
        z * (left_ba * cos_term - right_ba * sin_term)
    )

    # Remove the singleton dimensions
    val = np.squeeze(val)


    # Ensure the output is 2D
    if val.ndim == 1:
        val = val.reshape(X.shape[0], Y.shape[1])
    elif val.ndim > 2:
        raise ValueError("Output has more than 2 dimensions.")

    return val



def compute_conv_rates(array_errors, array_input, list_idx_start, list_idx_stop):
    # array_errors: Contains the errors
    # array_input: Contains the number of centers
    # list_idx_start, list_idx_stop: Used to compute an average convergence rate (using different intervals)

    array_input = np.array(array_input)

    list_arrays_rates = []
    list_rates = []

    for idx_run in range(array_errors.shape[0]):
        # Loop over different run

        indices = np.isinf(array_errors[idx_run, :])        # exclude inf values (inf values caused me some problems)
        vector_errors = array_errors[idx_run, ~indices]
        array_input_ = array_input[~indices]

        # Compute several possible decay rates (using different intervals) and finally take the mean value
        array_rates = np.zeros((len(list_idx_start), len(list_idx_stop)))

        for idx1, idx_start in enumerate(list_idx_start):
            for idx2, idx_stop in enumerate(list_idx_stop):
                array_rates[idx1, idx2] = (np.log(vector_errors[idx_stop]) - np.log(vector_errors[idx_start])) \
                                          / (np.log(array_input_[idx_stop]) - np.log(array_input_[idx_start]))

        list_arrays_rates.append(array_rates)
        list_rates.append(np.mean(array_rates))



    return list_rates, list_arrays_rates
