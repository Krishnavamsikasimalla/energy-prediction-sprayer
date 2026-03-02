% MATLAB ANN Training
data = readtable('../data/segments.csv');
X = table2array(data(:, {'Avg_Speed','Avg_Gradient','Spray_Ratio','Area_acres'}))';
y = table2array(data(:, 'Total_Energy_Wh'))';

net = fitnet([10 10]);
net.divideParam.trainRatio = 0.80;
net.divideParam.valRatio   = 0.10;
net.divideParam.testRatio  = 0.10;
[net, tr] = train(net, X, y);

y_pred = net(X);
fprintf('MAE:  %.3f\nRMSE: %.3f\nR2:   %.4f\n', ...
    mean(abs(y_pred-y)), sqrt(mean((y_pred-y).^2)), ...
    1 - sum((y-y_pred).^2)/sum((y-mean(y)).^2));

save('../outputs/ann_model.mat', 'net');
fprintf('Saved → outputs/ann_model.mat\n');
