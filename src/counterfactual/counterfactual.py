def generate_r(tau, Q, wR, alpha, k):
    '''
    Parameters:

        tau: item center
        Q: item embeddings matrix
        wR, alpha: items weighting factors
        k: recommender list size
    '''
    
    score = (tau @ Q.T) + (wR @ alpha)

    scores_dict = dict(enumerate(score))
    sorted_dict = dict(sorted(scores_dict.items(), key=lambda item:item[1], reverse=True))

    return list(sorted_dict.keys())[:k]

def generate_s(u, r, X, Y, wS, beta, M):
    return [45, 89, 134] 

def C(u, tau, alpha, beta, Q, wR, X, Y, wS, K, M):
    '''
    Etapas:
        - Obter r a partir dos K itens mais próximos de tau
        - Obter s a partir dos M itens com maior prob. a partir de p_s
        - u é obtido automaticamente
    '''
    
    r = generate_r(tau, Q, wR, alpha, K)
    s = generate_s(u, r, X, Y, wS, beta, M)

    return (u, r, s) 