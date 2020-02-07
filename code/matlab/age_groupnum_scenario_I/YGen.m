
function [FX, DB] = YGen(deltaX)
%function X = ageGroupPrivacyObf()
pgy_input = csvread('../../python/age_groupnum_scenario_I/tmp/pgy_ageGroup_ygen.csv');
JSD_Mat_input = csvread('../../python/age_groupnum_scenario_I/tmp/JSDM_ageGroup_ygen.csv');
size(pgy_input)
size(JSD_Mat_input)

% set error bound
%deltaX = 0.5;

% learn obfuscation matrix
[ X, v, DB ] = pripmapping(pgy_input', JSD_Mat_input, deltaX);
%pripmapping( pgy, JSD_Mat, deltaX );
FX = full(X);
% visualize obfuscation matrix
% csvwrite(strcat('../python/pgg_ageGroup',num2str(deltaX*100),'.csv'), X);
% figure;
% colormap('jet')
% imagesc(X)
end

function [X, v, DB] = pripmapping( pgy, JSD_Mat, deltaX)
d = deltaX;

cvx_begin
    variable X(length(pgy(:,1)),length(pgy(:,1)))
    pg = sum(pgy', 2);
    pyhat = sum(pgy', 1)*X;
    pgyhat = pgy'*X;
        
    minimize( sum(sum(kl_div(pgyhat,pg*pyhat)./log(2))) )
    % minimize( sum(sum((X*pgy).*(log(X*pgy) - log(sum(X*pgy, 2))))) )
    subject to
        % sum(sum(JSD_Mat.*X)) <= deltaX
        sum(sum( JSD_Mat .* X, 2 ))./size(X,2) <= d
        sum(X) == 1
        X >= 0
        X <= 1
cvx_end

v = sum(sum(kl_div(pgyhat,pg*pyhat)./log(2)));
DB = sum(sum( JSD_Mat .* X, 2 ))./size(X,2)
end