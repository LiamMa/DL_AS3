
def calculate_fid_score(sample_feature_iterator,
                        testset_feature_iterator):
    """
    To be implemented by you!
    """
    p=torch.Tensor(next(sample_feature_iterator))
    q=torch.Tensor(next(testset_feature_iterator))

    mu_p=torch.mean(p,dim=-1)
    mu_q=torch.mean(q,dim=-1)

    mu_p,_=torch.broadcast_tensors(mu_p,p)
    mu_q,_=torch.broadcast_tensors(mu_q,q)

    cov_p=torch.mm(p-mu_p,(p-mu_p).transpose())
    cov_q=torch.mm(q-mu_q,(q-mu_q).transpose())
    sq=2*torch.mm(cov_p,cov_q)**(1/2)

    d=torch.norm(mu_p-mu_q,p=2)**2 +torch.trace(cov_p+cov_q+sq)

    return d





    raise NotImplementedError(
        "TO BE IMPLEMENTED."
        "Part of Assignment 3 Quantitative Evaluations"
    )

