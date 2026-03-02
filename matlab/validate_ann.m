% MATLAB ANN Validation
load('../outputs/ann_model.mat');
data = readtable('../data/segments.csv');
X = table2array(data(:, {'Avg_Speed','Avg_Gradient','Spray_Ratio','Area_acres'}))';
y = table2array(data(:, 'Total_Energy_Wh'))';

n = size(X,2);
test_idx = round(0.9*n):n;
y_pred = net(X(:, test_idx));

figure;
plot(y(test_idx), 'b-', 'LineWidth', 2); hold on;
plot(y_pred, 'r--', 'LineWidth', 2);
legend('Actual','Predicted');
title('ANN Validation — Mission 10');
xlabel('Segment'); ylabel('Energy (Wh)');
saveas(gcf, '../outputs/ann_validation_plot.png');
