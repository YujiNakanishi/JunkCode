import numpy as np


def decideInitialTemperature(function, sampler, x, p = 0.8, sample_num = 100):
    """
    decide initial temperature.

    input:
        function -> <func> objective function (minimize)
        sampler -> <Sample class>
        x -> <array> initial input
        p -> <float> increasing acceptable probability at initial temperature
        sample_num -> <int> sampling number
    output:
        T -> <float> initial temperature
    
    Note:
        initial temperature is decided by
            T = -df / ln(p),
        where df is average objective function increase.
        This average is calculated by sample (where sample number = sample_num)
    """

    df_list = np.zeros(sample_num)
    f = function(x)

    num = 0
    while num < sample_num:
        x_new = sampler(x); f_new = function(x)

        if f_new > f:
            df_list[num] = f_new - f
            num += 1
        
        x = x_new; f = f_new
    
    df = np.mean(df_list)

    T = -df/np.log(p)

    return T