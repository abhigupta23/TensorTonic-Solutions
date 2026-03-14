import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    PE = np.zeros((seq_len, d_model))
    
    for i in range(seq_len):
        for j in range(d_model):
            if j%2 == 0:
                w = (base ** ((2*(j/2))/d_model))
                PE[i,j] = np.sin(i/w)

            if j%2 != 0:
                PE[i,j] = np.cos(i/w)

            
    return PE

            
        
    