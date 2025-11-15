function visualize_jets(mode, varargin)
    % load validation jet images & labels from ROOT
    dataFile = "cnn_v1_data.mat";
    if ~isfile(dataFile)
        error("File %s not found in current folder.", dataFile);
    end
    load(dataFile, "Xval", "Yval");

    switch lower(mode)
        case "single"
            idx = varargin{1};
            show_single_jet(Xval, idx, Yval(idx));

        case "average"
            show_average_jets(Xval, Yval);

        case "compare"
            n = varargin{1};
            compare_jets_side_by_side(Xval, Yval, n);

        otherwise
            error("Unknown mode '%s'. Use: single, average, compare.", mode);
    end
end

% internal functions
function show_single_jet(Ximg, idx, label)
% display a single jet image

    jet = Ximg(:,:,1,idx);
    figure;
    imagesc(jet);
    colormap hot;
    colorbar;
    axis equal tight;

    if label == "1"
        title(sprintf("Jet %d — Top Quark Jet", idx));
    else
        title(sprintf("Jet %d — QCD Background Jet", idx));
    end
    xlabel("ϕ bins");
    ylabel("η bins");
end


function show_average_jets(Ximg, Y)
% display average top vs background jets

    sig = mean(Ximg(:,:,1, Y=="1"), 4);
    bkg = mean(Ximg(:,:,1, Y=="0"), 4);

    figure;

    subplot(1,2,1);
    imagesc(sig); colormap hot; colorbar; axis equal tight;
    title("Average Top-Quark Jet");

    subplot(1,2,2);
    imagesc(bkg); colormap hot; colorbar; axis equal tight;
    title("Average QCD Jet");
end


function compare_jets_side_by_side(Ximg, Y, n)
% compare n signal jets to n background jets
    sigIdx = find(Y=="1");
    bkgIdx = find(Y=="0");
    figure;

    for i = 1:n
        % signal
        subplot(n,2,2*i-1);
        imagesc(Ximg(:,:,1, sigIdx(i))); 
        colormap hot; axis equal tight;
        title(sprintf("Signal Jet %d", sigIdx(i)));

        % background
        subplot(n,2,2*i);
        imagesc(Ximg(:,:,1, bkgIdx(i)));
        colormap hot; axis equal tight;
        title(sprintf("Background Jet %d", bkgIdx(i)));

    end
end
