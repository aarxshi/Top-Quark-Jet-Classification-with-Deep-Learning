%% =========================================================
%   ALL-IN-ONE PLOTTING SCRIPT FOR CNN V1 (FIXED VERSION)
%   Works with:
%     cnn_v1_eval.mat  → Ypred, scores, acc, confMat
%     cnn_v1_data.mat  → Yval, Xval
%     cnn_v1_info.mat  → training info (optional)
% =========================================================
clc; clear;

%% ===== LOAD EVAL DATA =====
load("cnn_v1_eval.mat", "Ypred", "scores", "acc", "confMat");

% Load Yval from the data file
if isfile("cnn_v1_data.mat")
    load("cnn_v1_data.mat", "Yval");   % <-- FIXED
    disp("Loaded Yval from cnn_v1_data.mat.");
else
    error("cnn_v1_data.mat not found. Cannot plot evaluation graphs.");
end

%% ===== TRY LOADING TRAINING INFO (optional) =====
infoExists = false;

if isfile("cnn_v1_info.mat")
    load("cnn_v1_info.mat", "cnn2_info"); 
    info = cnn2_info;
    infoExists = true;
    disp("Loaded training info.");
else
    warning("Training info not found. Accuracy/Loss plots skipped.");
end

%% ===== 1. TRAINING ACCURACY & LOSS =====
if infoExists
    % Accuracy curve
    figure;
    plot(info.TrainingAccuracy, 'LineWidth', 1.6); hold on;
    plot(info.ValidationAccuracy, 'LineWidth', 1.6);
    xlabel("Iteration"); ylabel("Accuracy (%)");
    legend("Training", "Validation");
    title("Training vs Validation Accuracy");
    grid on;
    saveas(gcf, "v1_training_accuracy.png");

    % Loss curve
    figure;
    plot(info.TrainingLoss, 'LineWidth', 1.6); hold on;
    plot(info.ValidationLoss, 'LineWidth', 1.6);
    xlabel("Iteration"); ylabel("Loss");
    legend("Training Loss", "Validation Loss");
    title("Training vs Validation Loss");
    grid on;
    saveas(gcf, "v1_training_loss.png");

    disp("Saved accuracy/loss plots.");
else
    disp("Skipped accuracy/loss plots.");
end

%% ===== 2. CONFUSION MATRIX =====
figure;
cm = confusionchart(Yval, Ypred);
cm.Title = sprintf("Confusion Matrix (Accuracy = %.2f%%)", acc*100);
cm.RowSummary = "row-normalized";
cm.ColumnSummary = "column-normalized";
saveas(gcf, "v1_confusion_matrix.png");
disp("Saved confusion matrix.");

%% ===== 3. ROC CURVE =====
classes = categories(Yval);
positiveClass = classes{2};   % assume 2nd class is signal

[Xroc, Yroc, Troc, AUC] = perfcurve(Yval, scores(:,2), positiveClass);

figure;
plot(Xroc, Yroc, 'LineWidth', 1.8);
xlabel("False Positive Rate");
ylabel("True Positive Rate");
title(sprintf("ROC Curve (AUC = %.4f)", AUC));
grid on;
saveas(gcf, "v1_roc_curve.png");
disp("Saved ROC curve.");

%% ===== 4. SCORE DISTRIBUTION =====
signalScores = scores(Yval == positiveClass, 2);
backgroundScores = scores(Yval ~= positiveClass, 2);

figure;
histogram(signalScores, 40); hold on;
histogram(backgroundScores, 40);
legend("Signal", "Background");
xlabel("Predicted Probability of Signal");
ylabel("Count");
title("Score Distribution (Signal vs Background)");
grid on;
saveas(gcf, "v1_score_distribution.png");
disp("Saved score distribution.");
