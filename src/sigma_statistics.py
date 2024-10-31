import numpy as np

def calc_mean_and_sigma(counts, shots, measure, df=0):
    
    # Directly calculate expectation value of sigma for the first qubit
    sigma = 0
    for outcome, count in counts.items():
        if outcome[0] == '0':
            sigma += count / shots
        else:
            sigma -= count / shots
    print(f"sigma_{measure}_mean_should_be= ",sigma)
        
    #METHOD 2 designed to conduct the chi square test 
    # Initialize sigma array based on the measure ('X', 'Y', or 'Z') to store sigma_{measure= X, Y, Z} values for all shots
    sigma = np.zeros(shots)
    
    # Fill the sigma array with values for each shot
    index = 0
    for outcome, count in counts.items():
        value = 1 if outcome[0] == '0' else -1
        for _ in range(count):
            if index < shots:
                sigma[index] = value
                index += 1

    # Calculate the mean of sigma for this time step
    mean_sigma = np.mean(sigma)
    print(f"Mean of sigma_{measure} is: {mean_sigma}")
    
    # Calculate the standard deviation using numpy's std function
    std_sigma_unadjusted = np.std(sigma, ddof=df)  # ddof=0 for population std, ddof=1 for sample std

    # Adjust the standard deviation to match the formula in the image
    std_sigma = std_sigma_unadjusted * np.sqrt(1 / shots)

    print(f"Standard Deviation of sigma_{measure}: {std_sigma}")
    
    sigma = mean_sigma #reassigned the <sigma_x> over all shots for each t in time
    
    # # Calculate the sum of squared deviations from the mean
    # sum_squared_deviations = np.sum((sigma - mean_sigma) ** 2)

    # # Apply the custom formula for standard deviation
    # std_sigma_custom = np.sqrt(sum_squared_deviations / (shots * (shots - 1)))

    # print(f"Standard Deviation of sigma_{measure} using custom formula: {std_sigma_custom}")

    return sigma, std_sigma
