% Izhikevich model neuron, Created by Eugene M. Izhikevich, February 25, 2003
% Statistics and Machine Learning and Signal Processing Toolbox required

function main() %#ok<FNDEF>
    clc; close all; clear all;
   
    %% parameter
    Ne=45; Ni=5; cluster = 4; % number of neurons and clusters: Ne excitatory, Ni inhibitory
    t_end = 30000; % simulation time (ms)
    EPSC_0 = 0.08; IPSC_0 = 0.0025;  % STDP rule coefficient
    downscaling_coeff = 0.75; % global downcaling coefficient
    Wee = 0.7; Wei = 0.7; Wie = 0.5; Wii = 0.5; Wcc = 0.1; Wt = 20; % strength of synaptic connection

    %% set initial connectivity
    [Nneu, Ssum, cluster_map] = S_map(Ne, Ni, cluster, Wcc, Wee, Wie, Wei, Wii, Wt);
    
    %% do simulation without learning
    S = Ssum; S(:,(end-cluster + 1):end) = zeros(Nneu,cluster); % remove thalamic input
    firings = Sim(S,Ne,Ni,cluster,t_end);
    sgtitle("spike train before learning without thalamic input")
    
    %% do simulation and update synaptic interaction matrix (learning)
    S = Ssum;
    firings = Sim(S,Ne,Ni,cluster,t_end);
    sgtitle("spike train during first learning with thalamic input")
    
    S = S_Update(S,firings,Ne,Ni,cluster,cluster_map,EPSC_0,IPSC_0,downscaling_coeff);
    firings = Sim(S,Ne,Ni,cluster,t_end);
    sgtitle("spike train during second learning with thalamic input")
    
    S = S_Update(S,firings,Ne,Ni,cluster,cluster_map,EPSC_0,IPSC_0,downscaling_coeff);
    firings = Sim(S,Ne,Ni,cluster,t_end); 
    sgtitle("spike train during third learning with thalamic input")
    
    %% do simulation without thalamic input after learning
    S(:,(end-cluster + 1):end) = zeros(Nneu,cluster); % remove thalamic input
    Sim(S,Ne,Ni,cluster,t_end);
    sgtitle("spike train after learning without thalamic input")
    
    %% plot heatmap of synaptic connectivity
    figure; heatmap([Ssum(:,1:200),max(S(1,:)) * ones(length(Ssum),5),S],'colormap',spring); %draw initial and final connectivity map at once
    figure; heatmap(S(1:50,1:50),'colormap',spring);%draw heatmap of cluster 1 

    fprintf("simulation finished\n");
    fprintf("simulated time of each trial: %d(ms) \n", t_end);
end

%% set initial synaptic connection
function [Nneu, Ssum, cluster_map] = S_map(Ne, Ni, cluster, Wcc, Wee, Wie, Wei, Wii, Wt)
    %% set synaptic connection of several clusters
    % neuron 1 to 200 are cortical neurons
    % neuron 201 to 204 are thalamic neurons which fire regularly
    Nneu = (Ne + Ni) * cluster + cluster;
    
    % give random input to other cluster if neuron is excitatory
    Ssum = [];
    for idx = 1:1:cluster
        tem_sum = [Wcc * abs(random('Normal',0,1,Nneu,Ne)),zeros(Nneu,Ni)];
        Ssum = [Ssum,tem_sum];
    end
    Ssum = [Ssum,zeros(Nneu,cluster)];
    
    cluster_map = zeros(Nneu,Nneu);
    for i = 1:1:cluster
        % set synaptic connection of one cluster
        S1 = [Wee * abs(random('Normal',0,1,Ne,Ne)),Wie * -abs(random('Normal',0,1,Ne,Ni))];
        S2 = [Wei * abs(random('Normal',0,1,Ni,Ne)),Wii * -abs(random('Normal',0,1,Ni,Ni))];
        
        % substitute values to Ssum
        coord = (i - 1) * (Ne + Ni) + 1;
        Ssum(coord:(coord + Ne + Ni - 1),coord:(coord + Ne + Ni - 1)) = [S1;S2];
        cluster_map(coord:(coord + Ne + Ni - 1),coord:(coord + Ne + Ni - 1)) = ones(Ne+Ni,Ne+Ni);
        
        % give thalamic input to five excitatory neurons
        Ssum(coord:(coord + 4),(Ne+Ni) * cluster + i) = ones(5,1) * Wt;
        
        % give thalamic input to three inhibitory neuorns
        Ssum((coord + Ne):(coord + Ne + 2),(Ne+Ni) * cluster + i) = ones(3,1) * Wt;
    end 

    Ssum(((Ne + Ni)* cluster + 1):Nneu,:) = zeros(cluster,Nneu);
end

%% simulation for three types of neurons interacting each other
function firings = Sim(S,Ne,Ni,cluster,t_end)    
    % parameters
    F_ex = 10;         F_in = 10;       F_thal = 30;

    % Excitatory neurons / Inhibitory neurons
    re=0.5 * ones(Ne,1);      ri=0.5 * ones(Ni,1);
    acluster=[0.02*ones(Ne,1); 0.02+0.08*ri];
    bcluster=[0.2*ones(Ne,1);  0.25-0.05*ri];
    ccluster=[-65+15*re.^2;    -65*ones(Ni,1)];
    dcluster=[8-6*re.^2;       2*ones(Ni,1)];
    a = []; b = []; c = []; d = [];

    for row = 1:1:cluster
        a = [a;acluster];
        b = [b;bcluster];
        c = [c;ccluster];
        d = [d;dcluster];
    end

    a = [a;0.02 * ones(cluster,1)];
    b = [b;0.2 * ones(cluster,1)];
    c = [c;-50 * ones(cluster,1)];
    d = [d; 2 * ones(cluster,1)];

    v=-65*ones((Ne + Ni) * cluster + cluster,1); % Initial values of v
    u=b.*v; % Initial values of u
    %% firings: first column: time (ms)
    % second column: neuron number
    % third column: cluster number
    firings=[]; % spike timings

    
    for t=1:t_end % do simulation for designated time
        % thalamic input + random noise input
        I = [];
        for row = 1:1:cluster
            Icluster = [random('Uniform',0,F_ex,Ne,1);random('Uniform',0,F_in,Ni,1)];
            I = [I;Icluster]; % add input to cortical neuron.
        end
        I = [I;random('Uniform',0,F_thal,cluster,1)]; % add input to thalamic neuron

        % calculate V and firing for 1ms
        fired=find(v>=30); % indices of spikes
        if ~isempty(fired) % if fired initialize values and apply postsynpatic current
            firings=[firings; t+0*fired, fired];
            v(fired)=c(fired);
            u(fired)=u(fired)+d(fired);
            I=I+sum(S(:,fired),2);
        end
        %update membrane potential
        v=v+0.5*(0.04*v.^2+5*v+140-u+I);
        v=v+0.5*(0.04*v.^2+5*v+140-u+I);
        u=u+a.*(b.*v-u);
    end

    firings = Sim_Plot(firings, t_end, Ne, Ni, cluster);
end

function firings = Sim_Plot(firings, t_end, Ne, Ni, cluster)
    %% draw raster plot
    figure; subplot(4,1,1);
    scatter(firings(1:end,1),firings(1:end,2),10,'filled');
    xlabel("time (ms)"); ylabel("neuron number"); xlim([300 800]);
    
    %% draw histogram of firing events
    subplot(4,1,2); histogram(firings(:,1),0:10:t_end);
    xlim([300 800]); xlabel("time (ms)"); ylabel("firing events");
    
    % add cluster number to firing events
    for row = 1:1:length(firings)
        if firings(row,2) <= (Ne+Ni) * cluster
            firings(row,3) = ceil(firings(row,2) / 50);
        else
            firings(row,3) = 0;
        end
    end
    
    % make matrix of firing. each cluster saved in distinct column
    firing_by_cluster = [];
    for clsnum = 1:1:cluster
        idx = 1;
        for row = 1:1:length(firings)
            if firings(row,3) == clsnum
                firing_by_cluster(idx,clsnum) = firings(row,1);
                idx = idx + 1;
            end
        end
    end

    % make matrix of inhibitory neuron firing
    firing_inhi = [];
    idx = 1;
    for row = 1:1:length(firings)
        if rem(firings(row,2)-1,(Ne+Ni)) >= Ne % if neuron is inhibitory neuron
            firing_inhi(idx,1) = firings(row,1);
            idx = idx + 1;
        end
    end
    
    % draw Gaussian filtered spike density function
    subplot(4,1,3); Gaussian_SDF(firing_by_cluster,firing_inhi,4,3000,1000);
    xlim([300 800]); xlabel("time (ms)"); ylabel("spike density function");
    legend("cluster 1","cluster 2","cluster 3","cluster 4",'inhibitory neuron');

    %% draw PETH aligned to thalamic neuron firing.
    subplot(4,1,4); PETH(firings,201,Ne,Ni,cluster,10000);
    xlabel("relative time to neuron 201 firing"); ylabel("firing events");
    xlim([-16 16]); legend("diricet thalamic input","without thalamic input");
end

%% draw gaussian filtered spike density function(SDF)
function Gaussian_SDF(firing_by_cluster,firing_inhi,cluster,t_plot,event_plot)
% cluster: number of cluster. t_plot: time to plot(ms). 
% event_plot: number of events to plot.
    %% cluster plotting
    win_length = 100; time_constant = 10;
    window = gausswin(win_length,time_constant);
    hold on;
    for cluster_num = 1:1:cluster % for each cluster
        temp_firing_by_time = zeros(1,t_plot);
        for idx = 1:1:event_plot % count firing event in 1ms interval
            firing_time = firing_by_cluster(idx,cluster_num);
            temp_firing_by_time(1,firing_time) = temp_firing_by_time(1,firing_time) + 1;
        end
        filtered_firing_by_time = filter(window,1,temp_firing_by_time); % apply Gaussian filter
        time_shift = -(win_length)/2; 
        plot((time_shift + 1):1:(time_shift + length(filtered_firing_by_time)),filtered_firing_by_time); % remove time shift after filtering
    end

    %% inhibitory neuron plotting
    temp_firing_by_time = zeros(1,t_plot);
    for idx = 1:1:event_plot % count firing event in 1ms interval
        firing_time = firing_inhi(idx,1);
        temp_firing_by_time(1,firing_time) = temp_firing_by_time(1,firing_time) + 1;
    end
    filtered_firing_by_time = filter(window,1,temp_firing_by_time);
    time_shift = -(win_length)/2; 
    plot((time_shift + 1):1:(time_shift + length(filtered_firing_by_time)),filtered_firing_by_time); % remove time shift after filtering
end

%% align firing event to specific neuron.
function PETH(firings,standard,Ne,Ni,cluster,event)
    firings = firings(1:event,:); % reduce number of event to save time

    len = length(firings);
    rel_firings = [];

    for idx1 = 1:len % for all firing event
        if firings(idx1,2) == standard % when standard neuron fires
            tem_rel_fire = [];
            % for all firing event find -15ms to +15ms firing event from
            % standard neuron firing.
            for idx2 = 1:len
                rel_time = firings(idx2,1) - firings(idx1,1);
                % if firings of another neuron is within +-15ms, save them.
                if firings(idx2,2) ~= standard && -15 <= rel_time && rel_time <= 15
                    tem_rel_fire = [tem_rel_fire; firings(idx2,:) - firings(idx1,1) * [1,0,0]];
                end
            end
            rel_firings = [rel_firings;tem_rel_fire];
        end
    end

    %% classify cells into engram and non engram cells.
    engram = []; non_engram = [];
    for idx = 1:1:length(rel_firings)
        if 1 <= rel_firings(idx,2) && rel_firings(idx,2) <= 5
            engram = [engram; rel_firings(idx,:)];
        elseif 6 <= rel_firings(idx,2) && rel_firings(idx,2) <= Ne
            non_engram = [non_engram; rel_firings(idx,:)];
        elseif (Ne + 1) <= rel_firings(idx,2) && rel_firings(idx,2) <= (Ne + 3)
            engram = [engram; rel_firings(idx,:)];
        elseif (Ne + 4) <= rel_firings(idx,2) && rel_firings(idx,2) <= (Ne + 5)
            non_engram = [non_engram; rel_firings(idx,:)];
        end
    end
    
    hold on; histogram(engram(:,1),-16:2:16); histogram(non_engram(:,1),-16:2:16); hold off;
end


%% update synpatic interaction following hebbian rule (spike time dependent plasticity)
%idx 1 for post synaptic neuron and idx 2 for presynaptic neuron
function S = S_Update(S,firings,Ne,Ni,cluster,cluster_map,EPSC_0,IPSC_0,downscaling_coeff)
    Noncluster_coeff = 1;
    N_nonthal = (Ne + Ni) * cluster;

    for idx1 = 1:length(firings)
        for idx2 = 1:length(firings)
            rel_firing = firings(idx1,1) - firings(idx2,1);
            if rel_firing < -50 || rel_firing > 50
                continue
            end
            % update synaptic strength between cortical neurons.
            % only cortico-cortical connection updated.
            % auto connection was excluded
            % firings(idx1,2) - postsynaptic neuron number
            % firings(idx2,2) - presynaptic neuron number
            if firings(idx2,2) <= N_nonthal if firings(idx1,2) <= N_nonthal if firings(idx1,2) ~= firings(idx2,2) %#ok<SEPEX,ALIGN>
                        if cluster_map(firings(idx1,2),firings(idx2,2)) == 1
                            if rem(firings(idx2,2)-1,(Ne+Ni)) >= Ne % if neuron 2 is inhibitory neuron
                                new_strength = S(firings(idx1,2),firings(idx2,2)) + STDP_inhi(firings(idx1,1) - firings(idx2,1),IPSC_0);
                                if new_strength < 0 % if postsynaptic current is still negative
                                    S(firings(idx1,2),firings(idx2,2)) = new_strength;
                                end
                            else % if neuron 2 is excitatory neuron
                                new_strength = S(firings(idx1,2),firings(idx2,2)) + STDP_exci(firings(idx1,1) - firings(idx2,1),EPSC_0);
                                if new_strength > 0 % if postsynaptic current is still positive
                                    S(firings(idx1,2),firings(idx2,2)) = new_strength;
                                end
                            end
                        else % if connection is outside cluster, reduce change by multiplying coefficient.
                            cluster_neu_num = rem(firings(idx2,2),(Ne+Ni)); % nueron number inside cluster ex:) nueron number 53 -> 3
                            if cluster_neu_num >= 1 && cluster_neu_num <= Ne % if neuron 2 is excitatory neuron
                                new_strength = Noncluster_coeff * STDP_exci(firings(idx1,1) - firings(idx2,1),EPSC_0);
                                if new_strength > 0
                                    S(firings(idx1,2),firings(idx2,2)) = new_strength;
                                end 
                            end % no update for inhibitory neuron becasue inhibitory neuron connection is mostly local
                        end
            end;end;end
        end
    end

    %% global downscaling
    S = S * downscaling_coeff;
end

%% excitatory STDP function.
function delta_EPSC = STDP_exci(post_min_pre,EPSC_0) %EPSC_0: STDP coefficient
    %% parameters
    tau = 10; % time constant (ms)
    
    delta_EPSC = 0;
    if post_min_pre < 0
        delta_EPSC = -exp(post_min_pre / tau) * EPSC_0;
    elseif post_min_pre > 0
        delta_EPSC = exp(post_min_pre / -tau) * EPSC_0;
    elseif post_min_pre == 0
        delta_EPSC = 0;
    end
end

%% inhibitory STDP function.
function delta_EPSC = STDP_inhi(post_min_pre,EPSC_0) %EPSC_0: STDP coefficient
    %% parameters
    tau = 10; % time constant (ms)
    
    delta_EPSC = 0;
    if post_min_pre < 0
        delta_EPSC = -exp(post_min_pre / tau) * EPSC_0;
    elseif post_min_pre > 0
        delta_EPSC = -exp(post_min_pre / -tau) * EPSC_0;
    elseif post_min_pre == 0
        delta_EPSC = 0;
    end
end
