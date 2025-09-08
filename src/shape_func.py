import numpy as np

""" 
    This file introduces the different shapes for neutrinos and maps those shapes to
    a dictionary to allow easy switching between shapes in test files. 
"""

# Define shape functions as Python functions
def flat_top(Δp, ξ):
    return np.heaviside(1/2 - abs(ξ), 0)

def triangular(Δp, ξ):
    return (1 - abs(ξ)) * np.heaviside(1 - abs(ξ), 0)

def none(Δp, ξ):
    return 1 

# Define the shape_func function
def shape_func(x, Δp, i, j, L, shape_name, periodic):
    # Define a dictionary mapping shape names to functions
    shapes = {
        "flat_top": flat_top,
        "triangular": triangular,
        "none": none
    }
    
    if shape_name in shapes:
        shape_function = shapes[shape_name]  # Assign the corresponding function to shape_function
        ξ = (x[i] - x[j]) / Δp  # Calculate ξ

        if periodic:
            if x[i] - x[j] > L/2:
                ξ -= L/Δp
            if x[i] - x[j] < -L/2:
                ξ += L/Δp

            if shape_name == "none":
                assert Δp == L, f"AssertionError: Δp ({Δp}) should be equal to L ({L})"
            else:
                assert 2*Δp < L, f"AssertionError: 2*Δp ({2*Δp}) should be less than L ({L})"
            
            assert abs(ξ) <= L/(2*Δp), f"AssertionError: abs(ξ) ({abs(ξ)}) should be less than or equal to L/(2*Δp) ({L/(2*Δp)})"

        return shape_function(Δp, ξ)
    else:
        raise ValueError(f"Unknown shape: {shape_name}")

