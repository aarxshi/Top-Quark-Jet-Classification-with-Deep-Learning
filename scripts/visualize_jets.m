% visualize_jets.m
% author: aarushi
% visualizes jet images from the validation set
%
% usage:
%   visualize_jets("single",  idx)     show one jet by index
%   visualize_jets("average")          mean image for signal vs background
%   visualize_jets("compare", n)       n side-by-side signal/background pairs
%
% optional: pass a custom data file as the last argument
%   visualize_jets("single", 50, "data/cnn_v1_data.mat")

function visualize_jets(mode, varargin)

    % default data file, can be overridden by passing a path as last arg
    dataFile = fullfile("data", "cnn_v1_data.mat");

    if nargin >= 2 && isstring(varargin{end}) && isfile(varargin{end})
        dataFile = varargin{end};
        varargin(end) = [];
    end

    if ~isfile(dataFile)
        error("data file not found: %s — run cnn.m first.", dataFile);
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
            error("unknown mode '%s'. use: single, average, compare.", mode);
    end

end


% show one jet image
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


% show mean jet image for signal and background separately
function show_average_jets(Ximg, Y)
    sig = mean(Ximg(:,:,1, Y=="1"), 4);
    bkg = mean(Ximg(:,:,1, Y=="0"), 4);

    figure;

    subplot(1,2,1);
    imagesc(sig);
    format_jet_plot();
    title("Average Top-Quark Jet");
    xlabel("ϕ bins"); ylabel("η bins");

    subplot(1,2,2);
    imagesc(bkg);
    format_jet_plot();
    title("Average QCD Background Jet");
    xlabel("ϕ bins"); ylabel("η bins");
end


% show n signal/background pairs side by side
function compare_jets_side_by_side(Ximg, Y, n)
    sigIdx = find(Y=="1");
    bkgIdx = find(Y=="0");

    figure;
    for i = 1:n
        subplot(n, 2, 2*i-1);
        imagesc(Ximg(:,:,1, sigIdx(i)));
        format_jet_plot();
        title(sprintf("Signal Jet %d", sigIdx(i)));

        subplot(n, 2, 2*i);
        imagesc(Ximg(:,:,1, bkgIdx(i)));
        format_jet_plot();
        title(sprintf("Background Jet %d", bkgIdx(i)));
    end
end


% shared formatting for all jet plots
function format_jet_plot()
    colormap hot;
    colorbar;
    axis equal tight;
end
