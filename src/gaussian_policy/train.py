import torch 

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
                        alpha, 
                        beta, 
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

    for episode in range(n_episodes):
        gp_loss = 0.
        for t in range(T):
            u = sample_user(users)
            tau = gaussian_policy(u) + Nt
            alpha = sample_alpha(None, n_items)
            beta = sample_beta(None, K)
            
            u, r, s = generated_sample = C(u, tau, alpha, beta, Q, wR, K, M)

            rec_loss = recommender_model.test_loss(generated_sample)
            # gp_loss -= rec_loss
            gp_loss += torch.sum(tau) 
            break

        gp_loss.backward()    
        optimizer.step()
        optimizer.zero_grad()
        
        break