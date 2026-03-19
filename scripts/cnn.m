% cnn.m
% author: aarushi
% trains a cnn to classify top-quark jets vs qcd background jets
% uses parquetDatastore for scalable big data handling

clc; clear;

% config, edit paths here if needed
DATA_FILE  = fullfile("data", "jets90000.parquet.gzip");
MODEL_FILE = fullfile("model", "cnn_model.mat");
DATA_OUT   = fullfile("data", "cnn_v1_data.mat");
SPLIT_OUT  = fullfile("data", "cnn_v1_split.mat");
IMG_SIZE   = [40 20];
VAL_FRAC   = 0.2;
MAX_EPOCHS = 8;
BATCH_SIZE = 256;

% set up datastore for scalable loading (works on large datasets too)
disp("Setting up parquetDatastore...");
ds = parquetDatastore(DATA_FILE, "OutputType", "table", "FileExtensions", ".gzip");

% use tall array to preprocess without loading everything at once
tData = tall(ds);

% gather into memory after tall operations are done
disp("Gathering data from datastore...");
% gather into memory for this dataset size (90k samples);
% pipeline uses parquetDatastore + tall arrays to support scaling to larger datasets
df = gather(tData);
disp("Loaded " + height(df) + " samples.");

% extract features and labels
Xraw = table2array(df(:, 1:800));   % 200 particles x 4 features (E, px, py, pz)
Y    = categorical(df.is_signal_new);
N    = size(Xraw, 1);

% convert raw particle 4-momenta into 40x20 jet images
% each row becomes a calorimeter-like eta-phi grid, normalized per jet
disp("Building jet images...");
Ximg = zeros(IMG_SIZE(1), IMG_SIZE(2), 1, N, "single");

for i = 1:N
    img = reshape(Xraw(i,:), IMG_SIZE);
    img = img ./ (max(img(:)) + 1e-6);   % per-jet normalization to [0,1]
    Ximg(:,:,1,i) = single(img);
end

% split into train and validation sets
cv       = cvpartition(Y, "Holdout", VAL_FRAC);
idxTrain = training(cv);
idxVal   = test(cv);

Xtrain = Ximg(:,:,:,idxTrain);
Ytrain = Y(idxTrain);
Xval   = Ximg(:,:,:,idxVal);
Yval   = Y(idxVal);

fprintf("Train: %d | Val: %d\n", sum(idxTrain), sum(idxVal));

% save data so evaluatem.m and visualize_jets.m can use it
save(DATA_OUT, "Xval", "Yval", "Xtrain", "Ytrain");
disp("Saved data to: " + DATA_OUT);

% define the cnn layers
layers = [
    imageInputLayer([IMG_SIZE 1], "Name", "input")

    convolution2dLayer(5, 32, "Padding", "same")
    reluLayer
    maxPooling2dLayer(2, "Stride", 2)

    convolution2dLayer(3, 64, "Padding", "same")
    reluLayer
    maxPooling2dLayer(2, "Stride", 2)

    convolution2dLayer(3, 128, "Padding", "same")
    reluLayer

    fullyConnectedLayer(128)
    reluLayer
    dropoutLayer(0.3)

    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer
];

% training options, lr decays every 4 epochs by factor 0.3
opts = trainingOptions("adam", ...
    "MaxEpochs",           MAX_EPOCHS, ...
    "MiniBatchSize",       BATCH_SIZE, ...
    "InitialLearnRate",    5e-3, ...
    "LearnRateSchedule",   "piecewise", ...
    "LearnRateDropFactor", 0.3, ...
    "LearnRateDropPeriod", 4, ...
    "L2Regularization",    1e-4, ...
    "Shuffle",             "every-epoch", ...
    "ValidationData",      {Xval, Yval}, ...
    "ValidationFrequency", 50, ...
    "Plots",               "training-progress", ...
    "Verbose",             true, ...
    "ExecutionEnvironment","auto");   % change to "gpu" to enable gpu training (requires parallel computing toolbox)

% train the network, save info for plotting curves later
disp("Training CNN...");
[net, info] = trainNetwork(Xtrain, Ytrain, layers, opts);

% save model and training info separately
save(MODEL_FILE, "net");
save(SPLIT_OUT, "info");
disp("Saved model to: " + MODEL_FILE);
disp("Saved training info to: " + SPLIT_OUT);

% run evaluation and save everything evaluatem.m needs
disp("Running evaluation...");
[Ypred, scores] = classify(net, Xval);
acc     = mean(Ypred == Yval);
confMat = confusionmat(Yval, Ypred);

evalFile = fullfile("data", "cnn_v1_eval.mat");
save(evalFile, "Ypred", "scores", "acc", "confMat", "info");
disp("Saved evaluation outputs to: " + evalFile);

fprintf("Validation Accuracy: %.2f%%\n", acc * 100);
disp("done.");
