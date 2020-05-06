clear; clc; close all;

% define parameters
DO_RAW_SIGNAL_PLOT = false;
DO_NOISE_ONE_CHANNEL_PLOT = true; % only one ear for noise will be plotted
FD_FRAC_SMOOTH = 3;
SIGNAL_DIR = '';
SIGNAL_AZIMUTHS = [0,-90];
SIGNAL_SETS = {... 
    'rec_4096_sh8_0db_{}deg_mic14','rec_4096_sh8_18db_{}deg_mic14',...
    'rec_4096_sh5_0db_{}deg_mic14','rec_1024_sh8_0db_{}deg_mic14'};
% SIGNAL_SETS = {... 
%     'rec_4096_sh8_0db_{}deg_mic14','rec_1mic_4096_sh8_0db_{}deg_mic14'};
% first set is handled as reference and shown in plots as comparison

IN_FILES = {'_drums_in.wav','_noise_in.wav'};
OUT_FILES = {'_drums_out.wav','_noise_out.wav'};

ref_diff_fd_s = [];
ref_fs = 0;
for s = 1:length(SIGNAL_SETS)
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
                        disp('mismathed sampling rates.');
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
                        disp('mismathed sampling rates.');
                        return;
                    end
                end
                % check signal length
                if len == 0
                    len = size(td,1);
                elseif len > size(td,1)
                    len = size(td,1);
                    out_td = out_td(1:len,:);
                end
                out_td = [out_td,td(1:len,:)];
            catch ME
                fprintf('files of set "%s" not found.\n',SIGNAL_SETS{s});
                continue;
            end
        end; clear f;
        
        % plot
        if DO_RAW_SIGNAL_PLOT
            plot_in_out(in_td(:,2*a-1),out_td(:,4*a-3:4*a-2),ref_fs,FD_FRAC_SMOOTH,[name,' DRUMS']);
            plot_in_out(in_td(:,2*a),out_td(:,4*a-1:4*a),ref_fs,FD_FRAC_SMOOTH,[name,' NOISE']);
        end
    end; clear a name in_files out_files fs len;
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
    fig = AKf(30,30);
    set(fig,'name',name);
    for a = 1:numel(SIGNAL_AZIMUTHS)
        subplot(numel(SIGNAL_AZIMUTHS),1,a);
        if DO_NOISE_ONE_CHANNEL_PLOT; ch=4*a-3:4*a-1; else; ch=4*a-3:4*a; end
        AKp(ifft(diff_fd_s(:,ch)),'m2d','fs',ref_fs,'lw',3,'dr',[-30,10]);
        if s == 1
            ref_diff_fd_s = diff_fd_s;
        else
            set(gca,'ColorOrderIndex',1); % restart plotting colors index
%             AKp(ifft(ref_diff_fd_s(:,ch)),'m2d','fs',ref_fs,'dr',[-60,5]);
            AKp(ifft(ref_diff_fd_s(:,ch)),'m2d','fs',ref_fs,'dr',[-30,10]);
        end
        leg = {'drums L','drums R','noise L'};
        if ~DO_NOISE_ONE_CHANNEL_PLOT
            leg = [leg,'noise R'];
        end
        legend(leg,'interpreter','none','location','south','orientation','horizontal');
        title(sprintf('difference @ %d deg',SIGNAL_AZIMUTHS(a)));
    end; clear leg a ch;
    AKtightenFigure;

    % export plot and data
    fprintf('saving file to "%s" ...\n',name);
    saveas(fig,fullfile(SIGNAL_DIR,name));
    clear name fig;
    
    fprintf('\n');
end; clear s;
disp('... done.');

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
    AKtightenFigure;
end