
function [X, DB] = PrivCheck(deltaX)
%function X = ageGroupPrivacyObf()
pgy_input = csvread('../../python/checkin_tradeoff_scenario_II/tmp/pgy_check_in_privcheck.csv');
load('../../python/checkin_tradeoff_scenario_II/tmp/JSDM_girdGroup_privcheck')
%JSD_Mat_input = load('JSDM_ageGroup.csv');
size(pgy_input)
size(JSD_Mat_input_Yang_allObf)

% set error bound
%deltaX = 0.5;

% learn obfuscation matrix
[ X, v, DB ] = pripmapping(pgy_input', JSD_Mat_input_Yang_allObf, deltaX);

%pripmapping( pgy, JSD_Mat, deltaX );0.518890


% visualize obfuscation matrix
% csvwrite(strcat('../python/pgg_ageGroup',num2str(deltaX*100),'.csv'), X);
% figure;
% colormap('jet')
% imagesc(X)
end

function [X, v, DB] = pripmapping( pgy, JSD_Mat, deltaX)
d = deltaX;
size(pgy)
size(JSD_Mat)
size(JSD_Mat(:,:,1))
age_group_num = length(JSD_Mat(1,1,:));
cvx_begin
    variable X(length(pgy(:,1)),length(pgy(:,1)))
    pg = sum(pgy', 2);
    pyhat = sum(pgy', 1)*X;
    pgyhat = pgy'*X;
        
    minimize( sum(sum(kl_div(pgyhat,pg*pyhat)./log(2))) )
    % minimize( sum(sum((X*pgy).*(log(X*pgy) - log(sum(X*pgy, 2))))) )
    subject to
        for i = 1:age_group_num
            sum(sum( JSD_Mat(:,:,i) .* X, 2 ))./size(X,2) <= d
        end
        sum(X) == 1
        X >= 0
        X <= 1
cvx_end

v = sum(sum(kl_div(pgyhat,pg*pyhat)./log(2)));
DB = sum(sum( JSD_Mat(:,:,i) .* X, 2 ))./size(X,2)
end