import tensorflow as tf

def LSTM_step(cell_inputs, cell_states, kernel, recurrent_kernel, bias):
    """
    Run one time step of the cell. That is, given the current inputs and the cell states from the last time step, calculate the current state and cell output.
    You will notice that TensorFlow LSTMCell has a lot of other features. But we will not try them. Focus on the very basic LSTM functionality.
    Hint: In LSTM there exist both matrix multiplication and element-wise multiplication. Try not to mix them.
        
        
    :param cell_inputs: The input at the current time step. The last dimension of it should be 1.
    :param cell_states:  The state value of the cell from the last time step, containing previous hidden state h_tml and cell state c_tml.
    :param kernel: The kernel matrix for the multiplication with cell_inputs
    :param recurrent_kernel: The kernel matrix for the multiplication with hidden state h_tml
    :param bias: Common bias value
    
    
    :return: current hidden state, and a list of hidden state and cell state. For details check TensorFlow LSTMCell class.
    """
    
    ###################################################
    # TODO:      INSERT YOUR CODE BELOW               #
    # params                                          #
    ###################################################
    def sigmoid(x):
        return 1 / (1 + tf.exp(-x))

    def tanh(x):
        return (tf.exp(x) - tf.exp(-x)) / (tf.exp(x) + tf.exp(-x))

    h_tml = cell_states[0]
    c_tml = cell_states[1]

    z = tf.matmul(cell_inputs, kernel)
    z += tf.matmul(h_tml, recurrent_kernel)
    z += bias

    z0, z1, z2, z3 = tf.split(z, 4, axis=1)

    forget = sigmoid(z0)
    input = sigmoid(z1)
    cell_hat = tanh(z2)
    output = sigmoid(z3)
    ct = forget * c_tml + input * cell_hat
    ht = tanh(ct) * output
    return ht, [ht, ct]
    
    ###################################################
    # END TODO                                        #
    ###################################################
