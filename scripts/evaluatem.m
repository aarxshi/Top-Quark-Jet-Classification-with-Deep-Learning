% evaluatem.m
% author: aarushi
% loads saved outputs from cnn.m and generates evaluation plots

clc; clear;

% paths, change these if your folder structure is different
EVAL_FILE   = fullfile("data", "cnn_v1_eval.mat");
DATA_FILE   = fullfile("data", "cnn_v1_data.mat");
RESULTS_DIR = "results";

% create results folder if it doesn't exist yet
if ~isfolder(RESULTS_DIR)
    mkdir(RESULTS_DIR);
end

% load evaluation outputs saved by cnn.m
if ~isfile(EVAL_FILE)
    error("eval file not found: %s — run cnn.m first.", EVAL_FILE);
end
data   = load(EVAL_FILE);
Ypred  = data.Ypred;
scores = data.scores;
acc    = data.acc;

% check if training info is available for curve plots
infoExists = isfield(data, "info");
if infoExists
    info = data.info;
end

% load validation labels
if ~isfile(DATA_FILE)
    error("data file not found: %s — run cnn.m first.", DATA_FILE);
end
load(DATA_FILE, "Yval");
disp("loaded Yval from: " + DATA_FILE);

% training accuracy and loss curves (only if info was saved)
if infoExists
    % space validation points evenly across the training x range
    nTrain   = length(info.TrainingAccuracy);
    valIter  = linspace(1, nTrain, length(info.ValidationAccuracy));

    figure;
    plot(info.TrainingAccuracy, "LineWidth", 1.6); hold on;
    plot(valIter, info.ValidationAccuracy, "o-", "LineWidth", 1.6, "MarkerSize", 4);
    xlabel("Iteration"); ylabel("Accuracy (%)");
    legend("Training", "Validation");
    title("Training vs Validation Accuracy");
    grid on;
    saveas(gcf, fullfile(RESULTS_DIR, "v1_training_accuracy.png"));
    disp("saved training accuracy plot.");

    figure;
    plot(info.TrainingLoss, "LineWidth", 1.6); hold on;
    plot(valIter, info.ValidationLoss, "o-", "LineWidth", 1.6, "MarkerSize", 4);
    xlabel("Iteration"); ylabel("Loss");
    legend("Training Loss", "Validation Loss");
    title("Training vs Validation Loss");
    grid on;
    saveas(gcf, fullfile(RESULTS_DIR, "v1_training_loss.png"));
    disp("saved training loss plot.");
else
    disp("training info not found in eval file, skipping accuracy/loss plots.");
end

% confusion matrix
figure;
cm               = confusionchart(Yval, Ypred);
cm.Title         = sprintf("Confusion Matrix (Accuracy = %.2f%%)", acc * 100);
cm.RowSummary    = "row-normalized";
cm.ColumnSummary = "column-normalized";
saveas(gcf, fullfile(RESULTS_DIR, "v1_confusion_matrix.png"));
disp("saved confusion matrix.");

% roc curve
classes       = categories(Yval);
positiveClass = classes{2};

[Xroc, Yroc, ~, AUC] = perfcurve(Yval, scores(:,2), positiveClass);

figure;
plot(Xroc, Yroc, "LineWidth", 1.8);
xlabel("False Positive Rate");
ylabel("True Positive Rate");
title(sprintf("ROC Curve (AUC = %.4f)", AUC));
grid on;
saveas(gcf, fullfile(RESULTS_DIR, "v1_roc_curve.png"));
disp("saved roc curve.");

% score distribution for signal vs background
signalScores     = scores(Yval == positiveClass, 2);
backgroundScores = scores(Yval ~= positiveClass, 2);

figure;
histogram(signalScores,     40, "FaceAlpha", 0.6); hold on;
histogram(backgroundScores, 40, "FaceAlpha", 0.6);
legend("Signal (Top Quark)", "Background (QCD)");
xlabel("Predicted Probability of Signal");
ylabel("Count");
title("Score Distribution (Signal vs Background)");
grid on;
saveas(gcf, fullfile(RESULTS_DIR, "v1_score_distribution.png"));
disp("saved score distribution.");

fprintf("\nevaluation complete. validation accuracy: %.2f%%\n", acc * 100);
