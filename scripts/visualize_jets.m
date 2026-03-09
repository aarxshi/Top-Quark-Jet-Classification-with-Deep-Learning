function visualize_jets(mode, varargin)

    % default dataset
    dataFile = "cnn_v1_data.mat";

    % allow optional dataset override
    if nargin >= 2 && isstring(varargin{end})
        dataFile = varargin{end};
        varargin(end) = [];
    end

    % load validation data
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


%% show one jet
function show_single_jet(Ximg, idx, label)

    jet = Ximg(:,:,1,idx);

    figure;
    imagesc(jet);
    format_jet_plot();

    if label == "1"
        title(sprintf("Jet %d — Top Quark Jet", idx));
    else
        title(sprintf("Jet %d — QCD Background Jet", idx));
    end

    xlabel("ϕ bins");
    ylabel("η bins");
end


%% average jets
function show_average_jets(Ximg, Y)

    sig = mean(Ximg(:,:,1, Y=="1"), 4);
    bkg = mean(Ximg(:,:,1, Y=="0"), 4);

    figure;

    subplot(1,2,1);
    imagesc(sig);
    format_jet_plot();
    title("Average Top-Quark Jet");
    xlabel("ϕ bins");
    ylabel("η bins");

    subplot(1,2,2);
    imagesc(bkg);
    format_jet_plot();
    title("Average QCD Jet");
    xlabel("ϕ bins");
    ylabel("η bins");
end


%% compare jets
function compare_jets_side_by_side(Ximg, Y, n)

    sigIdx = find(Y=="1");
    bkgIdx = find(Y=="0");

    figure;

    for i = 1:n

        % signal jet
        subplot(n,2,2*i-1);
        imagesc(Ximg(:,:,1, sigIdx(i)));
        format_jet_plot();
        title(sprintf("Signal Jet %d", sigIdx(i)));

        % background jet
        subplot(n,2,2*i);
        imagesc(Ximg(:,:,1, bkgIdx(i)));
        format_jet_plot();
        title(sprintf("Background Jet %d", bkgIdx(i)));

    end
end


%% helper function for consistent plots
function format_jet_plot()

    colormap hot;
    colorbar;
    axis equal tight;

end
