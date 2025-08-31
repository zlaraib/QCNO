import numpy as np

# def calc_mean_and_sigma(counts, shots, measure, df=0):
    
#     # # Directly calculate expectation value of sigma for the first qubit
#     # sigma = 0
#     # for outcome, count in counts.items():
#     #     if outcome[0] == '0':
#     #         sigma += count / shots
#     #     else:
#     #         sigma -= count / shots
#     # print(f"sigma_{measure}_mean_should_be= ",sigma)
#     # # Round sigma to 15 decimal places
#     # sigma = round(sigma, 15)
#     # print(f"sigma (method 1, 15 decimals): {sigma:.15f}")
        
#     #METHOD 2 designed to conduct the chi square test 
#     # Initialize sigma array based on the measure ('X', 'Y', or 'Z') to store sigma_{measure= X, Y, Z} values for all shots
#     sigma = np.zeros(shots, dtype=np.float64)
#     # print("sigma_initial= ",sigma)
#     # Fill the sigma array with values for each shot
#     index = 0
#     for outcome, count in counts.items():
#         value = 1 if outcome[0] == '0' else -1
#         for _ in range(count):
#             if index < shots:
#                 sigma[index] = value
#                 index += 1
#     sigma = sigma.round(decimals=15)

#     # print("sigma_15dp= ",sigma)
#     # Calculate the mean of sigma for this time step
#     mean_sigma = np.mean(sigma)
#     # print(f"Mean of sigma_{measure} is: {mean_sigma}")
    
#     # Calculate the standard deviation using numpy's std function
#     std_sigma_unadjusted = np.std(sigma, ddof=df)  # ddof=0 for population std, ddof=1 for sample std

#     # Adjust the standard deviation to match the formula in the image
#     std_sigma = std_sigma_unadjusted * np.sqrt(1 / shots)

#     # print(f"Standard Deviation of sigma_{measure}: {std_sigma}")

#     sigma = mean_sigma #reassigned the <sigma_x> over all shots for each t in time
    
#     # # Calculate the sum of squared deviations from the mean
#     # sum_squared_deviations = np.sum((sigma - mean_sigma) ** 2)

#     # # Apply the custom formula for standard deviation
#     # std_sigma_custom = np.sqrt(sum_squared_deviations / (shots * (shots - 1)))

#     # print(f"Standard Deviation of sigma_{measure} using custom formula: {std_sigma_custom}")

#     return sigma, std_sigma


def calc_mean_and_sigma(counts, shots, measure, N_sites, df=0):
    # Initialize lists to store the mean and standard deviation for each qubit
    mean_values = []
    std_values = []

    # Loop over all the qubits
    for qubit in range(N_sites):
        # Initialize lists to store the Pauli outcomes for this qubit
        outcomes = []

        # Loop over all the outcomes in the counts
        for outcome, count in counts.items():
            # Ensure the outcome string is padded to the correct length
            outcome = outcome.zfill(N_sites)  # Pad with leading zeros if necessary

            # Map the outcome to +1 or -1 for Z-basis measurement
            if measure == 'Z':
                value = 1 if outcome[qubit] == '0' else -1  # For Z-basis measurement (0 -> +1, 1 -> -1)
            elif measure == 'X':
                # For X-basis, apply Hadamard transformation logic (which corresponds to equal superposition of |0> and |1>)
                value = 1 if outcome[qubit] == '0' else -1
            elif measure == 'Y':
                # For Y-basis, apply phase shift (S gate), so same logic as X, but with a phase difference
                value = 1 if outcome[qubit] == '0' else -1

            # Add the outcome to the list
            outcomes.extend([value] * count)  # Extend the list by the count of this outcome

        # Normalize the outcomes (mean for this qubit)
        mean_sigma = np.mean(outcomes)

        # Calculate the standard deviation
        std_sigma = np.std(outcomes, ddof=df)  # Standard deviation for the Pauli operator outcomes

        # Store the calculated mean and standard deviation for this qubit
        mean_values.append(mean_sigma)
        std_values.append(std_sigma)

        # Print the results for debugging
        # print(f"Qubit {qubit} - Mean: {mean_sigma}, Std Dev: {std_sigma}")

    return mean_values, std_values

