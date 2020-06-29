% requires AKtools toolbox (run AKtoolsStart.m)
% $ svn checkout https://svn.ak.tu-berlin.de/svn/AKtools --username aktools --password ak
%
close all; clear; clc;

%% define parameters
DO_RAW_SIGNAL_PLOT = false;
DO_ZERO_ONE_CHANNEL_PLOT = true; % plot only one channel for 0 degree orientations
DO_PLOT_EXPORT = true;
PLOT_EXPORT_DR = [-29,4];
PLOT_LAST_EXPORT_DR = [-79,4];
PLOT_FIG_SIZE = {20,6};
PLOT_COLOR = 'brgy';
FD_FRAC_SMOOTH = 3;
IN_FILES = {'_target_in.wav','_noise_in.wav'};
OUT_FILES = {'_target_out.wav','_noise_out.wav'};
SIGNAL_DIR = '';
SIGNAL_AZIMUTHS = [0,90];

SIGNAL_SETS = {...
    'rec_110ch_8cm_sh8_0dB_{}deg','rec_110ch_8cm_sh4_0dB_{}deg',...
    'rec_110ch_8cm_sh8_18dB_{}deg','rec_110ch_8cm_sh8_0dB_EQ_{}deg',...
    'rec_32ch_4cm_sh4_0dB_{}deg','rec_32ch_4cm_sh4_0dB_EQ_{}deg',...
    'rec_32ch_8cm_sh4_0dB_{}deg','rec_32ch_8cm_sh8_0dB_{}deg',...
    'rec_32ch_4cm_sh4_18dB_{}deg','rec_110ch_8cm_sh8_0dB_1024_{}deg',...
    'rec_230ch_8cm_sh12_0dB_{}deg','rec_338ch_8cm_sh12_0dB_{}deg',...
    'rec_110ch_8cm_sh8_0dB_1ch_{}deg'};
% IMPORTANT!
% First set is handled as reference and shown in plots as comparison.
% First, second and last set are always plotted for both ears.
% Last set is plotted slightly different.

% DO_PLOT_EXPORT = false;
% SIGNAL_SETS = {...
%     'rec_32ch_8cm_sh4_0dB_{}deg','','rec_32ch_4cm_sh4_0dB_{}deg',''};
% % IMPORTANT!
% % Run as comparison seperately and save plot manually. Keeping the empty
% % entries in SIGNAL_SETS is relevant!
% % saveas(gcf,fullfile(SIGNAL_DIR,'rec_32ch_4cm_sh4_0dB_{}deg_diff_8cm.pdf'))

% DO_PLOT_EXPORT = false;
% PLOT_LAST_EXPORT_DR = PLOT_EXPORT_DR;
% SIGNAL_SETS = {...
%     'rec_110ch_8cm_sh4_0dB_{}deg','rec_32ch_8cm_sh4_0dB_{}deg'};
% % IMPORTANT!
% % Run as comparison seperately and save plot manually.
% % saveas(gcf,fullfile(SIGNAL_DIR,'rec_32ch_8cm_sh4_0dB_{}deg_diff_110ch.pdf'))

% DO_ZERO_ONE_CHANNEL_PLOT = false;
% PLOT_EXPORT_DR = [-41,11];
% PLOT_FIG_SIZE = {30,8};
% SIGNAL_AZIMUTHS = [0,45,90];
% SIGNAL_SETS = {...
%     'rec_32ch_4cm_sh4_0dB_incoherent_{}deg','rec_32ch_4cm_sh4_12dB_incoherent_{}deg',...
%     'rec_32ch_4cm_sh4_0dB_coherent_{}deg','rec_32ch_4cm_sh4_12dB_coherent_{}deg',''};

%%
tic; % start measuring execution time

ref_diff_fd_s = [];
ref_fs = 0;
for s = 1:length(SIGNAL_SETS)
    if isempty(strip(SIGNAL_SETS{s})); continue; end

    % specify plotting parameters
    dr = PLOT_EXPORT_DR;
    do_one_channel_plot = DO_ZERO_ONE_CHANNEL_PLOT;
    if s == length(SIGNAL_SETS)
        dr = PLOT_LAST_EXPORT_DR;
    end
    if s <= 2 || s >= length(SIGNAL_SETS)
        do_one_channel_plot = false;
    end

    in_td = [];
    out_td = [];
    diff_fd_s = [];

    for a = 1:numel(SIGNAL_AZIMUTHS)
        % build file names
        name = strrep(SIGNAL_SETS{s},'{}',int2str(SIGNAL_AZIMUTHS(a)));
        in_files = strcat(name,IN_FILES);
        out_files = strcat(name,OUT_FILES);

        % read recorded files
        for f = 1:numel(in_files)
            try
                % read input signals
                len = size(out_td,1);
                if size(in_td,1) > len
                    in_td = in_td(1:len,:);
                end
                len = size(in_td,1);
                fprintf('reading file "%s" ...\n',in_files{f});
                [td,fs] = audioread(fullfile(SIGNAL_DIR,in_files{f}));
                % check sampling frequency
                if fs ~= ref_fs
                    if ref_fs == 0
                        ref_fs = fs;
                    else
                        disp('mismatched sampling rates.');
                        return;
                    end
                end
                % check signal length
                if len == 0
                    len = size(td,1);
                elseif len > size(td,1)
                    len = size(td,1);
                    in_td = in_td(1:len,:);
                end
                in_td = [in_td,td(1:len,:)];

                % read output signals
                len = size(in_td,1);
                if size(out_td,1) > len
                    out_td = out_td(1:len,:);
                end
                len = size(out_td,1);
                fprintf('reading file "%s" ...\n',out_files{f});
                [td,fs] = audioread(fullfile(SIGNAL_DIR,out_files{f}));
                % check sampling frequency
                if fs ~= ref_fs
                    if ref_fs == 0
                        ref_fs = fs;
                    else
                        disp('mismatched sampling rates.');
                        return;
                    end
                end
                % check signal length
                if len == 0
                    len = size(td,1);
                elseif len > size(td,1)
                    len = size(td,1);
                    out_td = out_td(1:len,:);
                    in_td = in_td(1:len,:);
                end
                out_td = [out_td,td(1:len,:)];
            catch ME
                if strcmp(ME.identifier,'MATLAB:audiovideo:audioread:fileNotFound')
                    warning('files of set "%s" not found.\n',SIGNAL_SETS{s});
                    continue;
                else
                    rethrow(ME)
                end
            end
        end; clear f;

        % plot
        if DO_RAW_SIGNAL_PLOT
            plot_in_out(in_td(:,2*a-1),out_td(:,4*a-3:4*a-2),ref_fs,FD_FRAC_SMOOTH,[name,' TARGET']);
            plot_in_out(in_td(:,2*a),out_td(:,4*a-1:4*a),ref_fs,FD_FRAC_SMOOTH,[name,' NOISE']);
        end
    end; clear a name in_files out_files fs len;

    if isempty(in_td)
        fprintf('... skipped set "%s".\n\n',SIGNAL_SETS{s});
        continue;
    end
    fprintf('truncated all signals to %s.\n',generate_str_length(size(in_td,1),ref_fs));

    % calculate spectra and smooth spectra
    in_fd_s = calculate_fd_smooted(in_td,ref_fs,FD_FRAC_SMOOTH);
    out_fd_s = calculate_fd_smooted(out_td,ref_fs,FD_FRAC_SMOOTH);

    % calculate difference
    for d = 1:size(in_fd_s,2)
        diff_fd_s(:,2*d-1) = out_fd_s(:,2*d-1) ./ in_fd_s(:,d);
        diff_fd_s(:,2*d) = out_fd_s(:,2*d) ./ in_fd_s(:,d);
    end; clear d;
    diff_fd_s(1,diff_fd_s(1,:)==inf) = 1.0; % prevent infinity

    name = [SIGNAL_SETS{s},'_diff','.pdf'];
    fig = AKf(PLOT_FIG_SIZE{:});
    set(fig,'name',name);
    for a = 1:numel(SIGNAL_AZIMUTHS)
        if SIGNAL_AZIMUTHS(a) ~= 0
            do_one_channel_plot = false;
        end
        plot_color = PLOT_COLOR;
        if do_one_channel_plot
            plot_color = [plot_color(2),plot_color(4)];
        end

        subplot(1,numel(SIGNAL_AZIMUTHS),a);
        if do_one_channel_plot; ch=[4*a-2,4*a]; else; ch=4*a-3:4*a; end
        try
            AKp(ifft(diff_fd_s(:,ch)),'m2d','fs',ref_fs,'lw',2,'dr',dr,'c',plot_color);
        catch ME
            if strcmp(ME.identifier,'MATLAB:badsubscript')
                error('insufficient data of set "%s" for azimuth %d deg.\n',SIGNAL_SETS{s},SIGNAL_AZIMUTHS(a));
            else
                rethrow(ME)
            end
        end

        if s == 1
            ref_diff_fd_s = diff_fd_s;
        else
            set(gca,'ColorOrderIndex',1); % restart plotting colors index
            AKp(ifft(ref_diff_fd_s(:,ch)),'m2d','fs',ref_fs,'lw',.5,'dr',dr,'c',plot_color);
        end
        if do_one_channel_plot
            legend({'wanted','noise'},'interpreter','tex','location','southwest','orientation','vertical');
        else
            legend({'wanted L','wanted R','noise L','noise R'},'interpreter','tex','location','southwest','orientation','horizontal','NumColumns',2);
        end
        title(sprintf('difference @ %d deg (1/%d oct. smoothing)',...
            SIGNAL_AZIMUTHS(a),FD_FRAC_SMOOTH));
    end; clear a ch plot_color do_one_channel_plot;
    AKtightenFigure; drawnow;

    if DO_PLOT_EXPORT
        % export plot and data
        fprintf('saving file to "%s" ...\n',name);
        saveas(fig,fullfile(SIGNAL_DIR,name));
    end
    clear name fig;

    fprintf('\n');
end; clear s;

fprintf(' ... finished in %.0fh %.0fm %.0fs.\n',toc/60/60,toc/60,mod(toc,60));

%%
function str = generate_str_length(samples,fs)
    str = sprintf('%d samples / %.3f s',samples,samples/fs);
end

function sig_fd_s = calculate_fd_smooted(sig_td,fs,frac)
    sig_fd = fft(sig_td);
    sig_fd = AKboth2singleSidedSpectrum(sig_fd);
    sig_fd_s = AKfractOctSmooth(sig_fd,'amp',fs,frac);
    sig_fd_s = AKsingle2bothSidedSpectrum(sig_fd_s);
end

function plot_in_out(sig1_td,sig2_td,fs,frac,name)
    fig = AKf(50,30);
    set(fig,'name',name);
    subplot(2,3,1); AKp(sig1_td,'et2d','fs',fs);
    subplot(2,3,2); AKp(sig1_td,'m2d','fs',fs);
    subplot(2,3,3); AKp(sig1_td,'m2d','fs',fs,'lw',2,'frac',frac);
    subplot(2,3,4); AKp(sig2_td,'et2d','fs',fs);
    subplot(2,3,5); AKp(sig2_td,'m2d','fs',fs);
    subplot(2,3,6); AKp(sig2_td,'m2d','fs',fs,'lw',2,'frac',frac);
    AKtightenFigure; drawnow;
end
