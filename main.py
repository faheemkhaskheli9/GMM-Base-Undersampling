import numpy as np
from sklearn.mixture import GaussianMixture

def gmm_undersampling(X, y, n=5):
    '''
    This function will use GMM based undersampling, it will reduce number of samples of negative class
    represented by 0 in y variable.

    :param X: Input Feature for downsampling, must contain both positive and negative samples
    :param y: Label for positive and negative samples. 0 for negative and 1 for positive
    :param n: represent number of gaussian components to train gaussian mixture model, from 1 to n
    :return: it return the downsampled negative class and original samples for positive class. datatype as np.array
    '''
    CV = ['full', 'tied', 'diag', 'spherical']

    X0 = np.array(X)[y == 0]
    X1 = np.array(X)[y == 1]

    bic_list = []
    gmm_param = []
    for i in range(1, n):
        for cv in CV:
            gmm = GaussianMixture(n_components=i, covariance_type=cv, random_state=2020).fit(X0)

            bic_list.append(gmm.bic(X0))
            gmm_param.append([i, cv])
    best_bic = 99999
    best_param = []

    for indx, value in enumerate(bic_list):
        if value < best_bic:
            best_bic = value
            best_param = gmm_param[indx]

    best_gmm = GaussianMixture(n_components=best_param[0], covariance_type= best_param[1], random_state=2020).fit(X0)

    pdf_m1 = best_gmm.score_samples(X0)
    pdf_n1 = best_gmm.score_samples(X1)

    cross_edge = max(pdf_n1)

    IR = len(X1) / len(X0)

    pro = (10/IR) / 100
    pdf_m1_sorted = sorted(pdf_m1)
    qty = (len(X0)*pro)//2

    ## following code is not sure if it is correct or not.
    # TODO Verify following code
    # here we have to choose certain samples from negative classes
    # based on pdf_m1_sorted, where value of pdf_m1_sorted is between high and low

    # here i choose the highest value lower then cross edge
    # this become mid_index
    mid_index = 0
    for indx, value in enumerate(pdf_m1_sorted):
        if (value < cross_edge):
            mid_index = indx

    # mid_index - qty euqal low index
    low = pdf_m1_sorted[int(mid_index - qty)]

    # mid_index + qty equal high index
    # if high index is less then total negative samples then its ok
    if (mid_index + qty) < len(pdf_m1):
        high = pdf_m1_sorted[int(mid_index + qty)]

    # otherwise, if high index is greater than quantity
    # change the low index to be lower then normal
    # so that we can get exact amount of samples
    else:
        high = pdf_m1_sorted[len(pdf_m1) - 1]
        low = pdf_m1_sorted[ int((mid_index - qty) - ((mid_index + qty) - len(pdf_m1)))]

    # selecting samples if pdf_m1 is between low and high
    new_x0 = []
    for indx, value in enumerate(pdf_m1):
        if value > low and value < high:
            new_x0.append(X0[indx])

    print(np.array(new_x0).shape)

    return np.concatenate(new_x0, X1)
