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

