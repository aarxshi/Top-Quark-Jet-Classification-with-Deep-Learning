clc; clear;

%load data
df = parquetread("data/jets90000.parquet.gzip");
disp("Loaded " + height(df) + " samples.");

Xraw = table2array(df(:,1:800));   % 200 particles × 4 features
Y = categorical(df.is_signal_new);

N = size(Xraw,1);
%convert into 40×20 jet images
imgSize = [40 20];  % exactly matches your data
Ximg = zeros(imgSize(1), imgSize(2), 1, N, "single");

for i = 1:N
    row = Xraw(i,:);
    % reshape using true structure
    img = reshape(row, imgSize);   % 40×20

    % normalize per image
    img = img ./ max(img(:) + 1e-6);

    Ximg(:,:,1,i) = single(img);
end


%train val split
cv = cvpartition(Y, "Holdout", 0.2);
idxTrain = training(cv);
idxVal = test(cv);

Xtrain = Ximg(:,:,:,idxTrain);
Ytrain = Y(idxTrain);

Xval = Ximg(:,:,:,idxVal);
Yval = Y(idxVal);

fprintf("Train: %d | Val: %d\n", sum(idxTrain), sum(idxVal));

%define cnn
layers = [
    imageInputLayer([40 20 1], 'Name','input')

    convolution2dLayer(5, 32, 'Padding','same')
    reluLayer
    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3, 64, 'Padding','same')
    reluLayer
    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3, 128, 'Padding','same')
    reluLayer

    fullyConnectedLayer(128)
    reluLayer
    dropoutLayer(0.3)

    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer
];

%training options
opts = trainingOptions("adam", ...
    "MaxEpochs", 8, ...   
    "MiniBatchSize", 256, ...
    "Shuffle", "every-epoch", ...
    "ValidationData", {Xval, Yval}, ...
    "Plots", "training-progress", ...
    "Verbose", true);

%train
disp("Training CNN...");
net = trainNetwork(Xtrain, Ytrain, layers, opts);

%save
save("cnn_model.mat","net");

disp("DONE (CNN jet classifier).");
