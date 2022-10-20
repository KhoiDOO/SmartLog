import numpy as np

def to_categorical(y, num_classes=None, dtype='float32'):
    """to_categorical _summary_

    Arguments:
        y -- The label of the data set with the shape of [None, 1]

    Keyword Arguments:
        num_classes -- The num_classes in the data set (default: {None})
        dtype -- the type of each element of the label after reshape (default: {'float32'})

    Returns:
        the label of the data set with the shape of [number of samples, number of classes].
    """    
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical