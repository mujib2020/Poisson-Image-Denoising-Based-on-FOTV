function [u,output] = FOTV_denoising_ADMM(f,beta,mu,alpha)
% Fractional-order TV regularized method

% If you use this code or part of this code please cite the following references:
%
% REFERENCES: Chowdhury, M. R. and Zhang, J. and Qin, J. and Lou, Y.; Poisson image denoising based on fractional-order total variation
%                Inverse Problem and Imaging Volume 14, No. 1, 2020, 77-96
%                doi: 10.3934/ipi.2019064
%
%
%     Input :
%             f : noisy image
%             beta: balancing parameter for regularization and data fitting term
%             mu: penalty parameter
%             alpha: order of the derivative
%
%     Output:
%             u: recovered/denoised image
%             output.X: energy
%             output.Z: relative error
%
%   The proposed model:
%
%              min_{u,z} || z ||_1 + beta * sum ( u - f * log(u) ) 
%                                + 0.5 * mu || z - D(u) + Lam/mu ||^2


% Mujibur R. Chowdhury, 12/8/2019.

U = f;

itermax = 1000; % Maximum number of iterations
tol = 1e-5;     % Error tolerance

[m,n] = size(f); % Read the image size

% Initialization
P1 = zeros(m,n);
P2 = zeros(m,n);
Lam1 = P1;
Lam2 = P2;
u = zeros(m,n);

% FOTV derivative
K = 20;
w = zeros(1,K);
w(K) = 1;
for i = 1:(K-1)
    w(K-i) = (-1)^i*gamma(alpha+1)/gamma(alpha-i+1)/factorial(i);
end
[D,Dt] = defDDt(w,alpha);

% Define the denominator for the LS subproblem under FFT
D1 = abs(psf2otf(w, [m,n])).^2; % backward, column
D2 = abs(psf2otf(w',[m,n])).^2; % row
C = D1 + D2;


% Main loop
tstart = tic;
for iter=1:itermax
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % ==================
    %   u-subproblem
    % ==================
    
    u = (beta*f + U.*Dt(Lam1,Lam2))/mu + U.*Dt(P1,P2); %for Poisson noise
    u = fft2(u)./((beta/mu)+ U.*C); %for Poisson noise,not ./
    u = real(ifft2(u));
    
    % ==================
    %   z-subproblem, Shrinkage Step
    % ==================
    
    [D1U,D2U] = D(u);
    Z1 = D1U - Lam1/mu;
    Z2 = D2U - Lam2/mu;
    W = Z1.^2 + Z2.^2;
    W = sqrt(W);
    Max = max(W - 1/mu, 0);
    W(W==0) = 1;
    P1 = Max.*Z1./W;
    P2 = Max.*Z2./W;
    
    % ==================
    %    Update Lamda
    % ==================
    
    Lam1 = Lam1 + mu*(P1 - D1U);
    Lam2 = Lam2 + mu*(P2 - D2U);
    
    output.cpu(iter) = toc(tstart);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Energy
    Unorm = sqrt(D1U.^2+D2U.^2);
    x1 = sum(sum( Unorm + (u-f.*log(u+1e-14))*beta ));
    output.X(iter) = x1;
    
    z1 = norm(u(:)-U(:))/norm(u(:));
    output.Z(iter) = z1;
    
    U = u;
    
    % Breaking point
    if z1 < tol
        break;
    end
    
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% compute difference
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function [D,Dt] = defDDt(w,a)
        % defines finite difference operator D^a
        % and its transpose operator Dt=conj(-1^a)div^a
        
        D = @(U) ForwardD(U,w);
        Dt = @(X,Y) Dive(X,Y,w,a);
        
        function [Dux,Duy] = ForwardD(U,w)
            %             [m,n] = size(U);
            % backward finite difference operator
            Dux = cconv2(U,w); % column differences
            Dux = Dux(:,1:n);
            Duy = cconv2(U,w'); % row differences
            Duy = Duy(1:m,:);
        end
        
        function DtXY = Dive(X,Y,w,a)
            % Transpose of the backward finite difference operator
            d = length(w);
            w = fliplr(w);
            DtX = cconv2(X,w);
            DtX = DtX(:,d:end);
            DtY = cconv2(Y,w');
            DtY = DtY(d:end,:);
            DtXY = conj((-1)^a)*(-1)^a*(DtX+DtY); %%add conj((-1)^a)*(-1)^a
        end
    end

    function x = cconv2(A,w)
        x = imfilter(A,w,'circular','full');
    end
end