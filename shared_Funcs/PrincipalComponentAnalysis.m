function [eigenVectors,eigenvalues,meanX,Xpca] = PrincipalComponentAnalysis (X, ndim)
    % This function implement Principal Component Analysis

    % Parameters:
    % X: Dataset with each image instance as a row
    % N: Reduced Dimensions Size

    % Returns:
    % Xpca: Dataset with N dimensions
    
    %calculate mean over the samples
    meanX = mean(X);

    %subtract mean to each sample
    A = X - meanX;

    % calculate covariance of the previous matrix
    S = cov(A);

    % obtain eigenvalue & eigenvector
    [eigenVectors,D] = eig(S);
    eigenvalues = diag(D);
    % sort eigenvalues in descending order
    eigenvalues = eigenvalues(end:-1:1);
    eigenVectors = fliplr(eigenVectors);

    if nargin < 2 
        eigsum = cumsum(eigenvalues);
        eigsum = eigsum / eigsum(end);
        
        index = find(eigsum >= 0.95);
        ndim = index(1);
    end

    % return only the desired number of dimensions with the higher eignvalues
    % ( higher amount of information)
    eigenVectors = eigenVectors(:,1:ndim);
    eigenvalues = eigenvalues(1:ndim);

    % dataset transformed to the pca space:
    Xpca = A*eigenVectors;

end
