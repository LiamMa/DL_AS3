import argparse
import os
import torchvision
import torchvision.transforms as transforms
import torch
import classify_svhn
from classify_svhn import Classifier
from scipy import linalg
import numpy as np
from scipy import linalg
from logging import warnings

# SVHN_PATH = "svhn"
SVHN_PATH ='data'
PROCESS_BATCH_SIZE = 32


def get_sample_loader(path, batch_size):
    """
    Loads data from `[path]/samples`

    - Ensure that path contains only one directory
      (This is due ot how the ImageFolder dataset loader
       works)
    - Ensure that ALL of your images are 32 x 32.
      The transform in this function will rescale it to
      32 x 32 if this is not the case.

    Returns an iterator over the tensors of the images
    of dimension (batch_size, 3, 32, 32)
    """
    data = torchvision.datasets.ImageFolder(
        path,
        transform=transforms.Compose([
            transforms.Resize((32, 32), interpolation=2),
            classify_svhn.image_transform
        ])
    )
    data_loader = torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        num_workers=2,
    )
    return data_loader


def get_test_loader(batch_size):
    """
    Downloads (if it doesn't already exist) SVHN test into
    [pwd]/svhn.

    Returns an iterator over the tensors of the images
    of dimension (batch_size, 3, 32, 32)
    """
    testset = torchvision.datasets.SVHN(
        SVHN_PATH, split='test',
        download=True,
        transform=classify_svhn.image_transform
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
    )
    return testloader


def extract_features(classifier, data_loader):
    """
    Iterator of features for each image.
    """
    with torch.no_grad():
        for x, _ in data_loader:
            h = classifier.extract_features(x).numpy()
            for i in range(h.shape[0]):
                yield h[i]



def calculate_fid_score(sample_feature_iterator,
                        testset_feature_iterator):
    """
    To be implemented by you!
    p: target--> testset
    q: generator-> sample
    """

    sample_feature_iterator = sample_f

    testset_feature_iterator = test_f

    sample = []
    testset = []


    for i in sample_feature_iterator:
        sample.append(np.array(i).reshape(1, -1))

    for j in testset_feature_iterator:
        testset.append(np.array(j).reshape(1, -1))

    testset = np.concatenate(testset, 0)
    sample = np.concatenate(sample, 0)
    print("testset: ",testset.shape)
    print("sample: ",sample.shape)
    print("testset max: %f  and min: %f "%(np.max(testset),np.min(testset)))
    print("sample max: %f  and min: %f "%(np.max(sample),np.min(sample)))


    mu1 = np.mean(testset, axis=0)
    mu2 = np.mean(sample, axis=0)
    sigma1 = np.cov(testset, rowvar=False)
    sigma2 = np.cov(sample, rowvar=False)

    print(sigma1.shape)
    print(sigma2.shape)


    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    d2 = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    print(diff.dot(diff))
    print(np.trace(sigma1))
    print(np.trace(sigma2))
    print(2 * tr_covmean)

    #
    #
    #
    # p=torch.Tensor(next(sample_feature_iterator))
    # q=torch.Tensor(next(testset_feature_iterator))
    #
    # p=p.view(p.size(0),-1)
    # q=q.view(q.size(0),-1)
    #
    # mu_p=torch.mean(p,dim=0)
    # mu_q=torch.mean(q,dim=0)
    #
    # mu_p,_=torch.broadcast_tensors(mu_p,p)
    # mu_q,_=torch.broadcast_tensors(mu_q,q)
    #
    # cov_p=torch.mm(p-mu_p,(p-mu_p).transpose(0,1))
    # cov_q=torch.mm(q-mu_q,(q-mu_q).transpose(0,1))
    # sq=torch.mm(cov_p,cov_q)
    # sq=torch.clamp(sq,min=0)
    #
    # print(sq.size())
    #
    #
    # # TODO: bug---the sqrt of negative leads to nan
    #
    # d=torch.norm(mu_p-mu_q,p=2)**2 +torch.trace(cov_p)+torch.trace(cov_q)-2*torch.trace(torch.sqrt(sq))
    # print(torch.norm(mu_p-mu_q,p=2)**2)
    # print(torch.trace(cov_p))
    # print(torch.trace(cov_q))
    # print(2*torch.trace(torch.sqrt(sq)))
    #
    #
    # print(cov_p)

    return d2
    
    raise NotImplementedError(
        "TO BE IMPLEMENTED."
        "Part of Assignment 3 Quantitative Evaluations"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Score a directory of images with the FID score.')
    parser.add_argument('--model', type=str, default="svhn_classifier.pt",
                        help='Path to feature extraction model.')
    parser.add_argument('directory', type=str,
                        help='Path to image directory')
    args = parser.parse_args()

    quit = False
    if not os.path.isfile(args.model):
        print("Model file " + args.model + " does not exist.")
        quit = True
    if not os.path.isdir(args.directory):
        print("Directory " + args.directory + " does not exist.")
        quit = True
    if quit:
        exit()
    print("Test")
    classifier = torch.load(args.model, map_location='cpu')
    classifier.eval()

    sample_loader = get_sample_loader(args.directory,
                                      PROCESS_BATCH_SIZE)
    sample_f = extract_features(classifier, sample_loader)

    test_loader = get_test_loader(PROCESS_BATCH_SIZE)
    test_f = extract_features(classifier, test_loader)

    fid_score = calculate_fid_score(sample_f, test_f)
    print("FID score:", fid_score)




# sample_feature_iterator=sample_f
#
# testset_feature_iterator=test_f
#
# sample=[]
# testset=[]
# #
# for i in sample_feature_iterator:
#     sample.append(torch.Tensor(i).view(1,-1))
#
# for j in testset_feature_iterator:
#     testset.append(torch.Tensor(j).view(1,-1))
#
#
#
# sample=torch.cat(sample,dim=0)
# testset=torch.cat(testset,dim=0)
#
# print(sample.size())
# print(testset.size())
#
#
# mu_q=torch.mean(sample,dim=0,keepdim=True)
# print(mu_q.size())
# mu_p=torch.mean(testset,dim=0,keepdim=True)
#
#
# mu_q_,_=torch.broadcast_tensors(mu_q,sample)
# mu_p_,_=torch.broadcast_tensors(mu_p,testset)
#
#
# sample_debias=sample-mu_q_
# testset_debias=testset-mu_p_
#
#
# cov_q=torch.mm(sample_debias.transpose(0,1),sample_debias)
# cov_p=torch.mm(testset_debias.transpose(0,1),testset_debias)
# print("cov_p",cov_p.size())
#
# cov_cov=torch.clamp(torch.mm(cov_p,cov_q),min=0).diagonal()
# cov_cov=2*torch.sqrt(cov_cov)
#
#
# d2=torch.norm((mu_p-mu_q),p=2)**2+torch.trace(cov_p)+torch.trace(cov_q)-torch.sum(cov_cov)






#
# for i in sample_feature_iterator:
#     sample.append(np.array(i).reshape(1,-1))
#
# for j in testset_feature_iterator:
#     testset.append(np.array(j).reshape(1,-1))
#
# testset=np.concatenate(testset,0)
# sample=np.concatenate(sample,0)
#
#
# mu1=np.mean(testset, axis=0)
# mu2=np.mean(sample, axis=0)
# sigma1 = np.cov(testset, rowvar=False)
# sigma2=np.cov(sample, rowvar=False)
#
#
# mu1 = np.atleast_1d(mu1)
# mu2 = np.atleast_1d(mu2)
#
# sigma1 = np.atleast_2d(sigma1)
# sigma2 = np.atleast_2d(sigma2)
#
# assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
# assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"
#
# diff = mu1 - mu2
#
# # product might be almost singular
# covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
# if not np.isfinite(covmean).all():
#     msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
#     warnings.warn(msg)
#     offset = np.eye(sigma1.shape[0]) * eps
#     covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
#
# # numerical error might give slight imaginary component
# if np.iscomplexobj(covmean):
#     if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
#         m = np.max(np.abs(covmean.imag))
#         raise ValueError("Imaginary component {}".format(m))
#     covmean = covmean.real
#
# tr_covmean = np.trace(covmean)
#
# d2=diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
#
#
#
# print(diff.dot(diff))
# print(np.trace(sigma1))
# print(np.trace(sigma2))
# print(2 * tr_covmean)
#
# print("fid score: ",d2)










