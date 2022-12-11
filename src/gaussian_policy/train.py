import torch 

from counterfactual.counterfactual import C

def sample_user(users):
    n_users = users.size()[0]
    return users[torch.randperm(n_users)[0]] 

def sample_alpha(alpha, n_items):
    return torch.randn(n_items) 

def sample_beta(beta, K):
    return torch.randn(K) 

def learn_gaussian_policy(
                        gaussian_policy, 
                        recommender_model, 
                        optimizer, 
                        users, 
                        n_items, 
                        Alpha, 
                        Beta, 
                        n_episodes, 
                        Q, 
                        wR,
                        wS,
                        X,
                        Y, 
                        K, 
                        M, 
                        T, 
                        Nt):

    gp_loss = 0.
    for episode in range(1, n_episodes + 1):
        episode_loss = 0.
        for t in range(T):
            u = sample_user(users)
            tau = gaussian_policy(u) + Nt

            alpha = sample_alpha(Alpha, n_items)
            beta = sample_beta(Beta, K)
            
            u, r, s = C(u, tau, alpha, beta, Q, wR, X, Y, wS, K, M)

            rec_loss = recommender_model.get_user_loss(u, r, s)

            episode_loss -= rec_loss
        
        gp_loss -= episode_loss

        episode_loss.backward() 
        optimizer.step()
        optimizer.zero_grad()
        
        print('episode {0:d} , total_loss: {1:.4f}, episode_loss: {2:.4f}'.format(
                            episode, gp_loss / episode, episode_loss))