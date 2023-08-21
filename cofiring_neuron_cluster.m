% Izhikevich model neuron, Created by Eugene M. Izhikevich, February 25, 2003
% Statistics and Machine Learning Toolbox required

function main() %#ok<FNDEF>
    clc; clear; close all;
   
    %% parameter
    % number of neurons and clusters
    Ne=45; Ni=5; cluster = 4;
    % simulation time
    t_end = 300;
    %strength of synaptic connection
    Wee = 2; Wei = 2; Wie = 2; Wii = 2; Wcc = 0.1; Wt = 30; 

    %% mode 1
    % set synaptic connection of several clusters
    % neuron 1 to 200 are cortical neurons
    % neuron 201 to 204 are thalamic neurons which fire regularly

    Nneu = (Ne + Ni) * cluster + cluster;
    Ssum = Wcc * random('Normal',0,0.2,Nneu,Nneu); cluster_map = zeros(Nneu,Nneu);
  
    for i = 1:1:cluster
        % set synaptic connection of one cluster
        S1 = [Wee * abs(random('Normal',0,0.1,Ne,Ne)),Wie * -abs(random('Normal',0,0.1,Ne,Ni))];
        S2 = [Wei * abs(random('Normal',0,0.1,Ni,Ne)),Wii * -abs(random('Normal',0,0.1,Ni,Ni))];
        
        % substitute values to Ssum
        coord = (i - 1) * (Ne + Ni) + 1;
        Ssum(coord:(coord + Ne + Ni - 1),coord:(coord + Ne + Ni - 1)) = [S1;S2];
        cluster_map(coord:(coord + Ne + Ni - 1),coord:(coord + Ne + Ni - 1)) = ones(Ne+Ni,Ne+Ni);

        % give thalamic input to five excitatory neurons
        Ssum(coord:(coord + 4),(Ne+Ni) * cluster + i) = ones(5,1) * Wt;

        % give thalamic input to three inhibitory neuorns
        Ssum((coord + Ne):(coord + Ne + 2),(Ne+Ni) * cluster + i) = ones(3,1) * Wt;
    end 
    
    %% do simulation without learning
    S = Ssum; S(:,(end-cluster + 1):end) = zeros(Nneu,cluster);
    firings = sim(S,Ne,Ni,cluster,t_end);
    sgtitle("spike train before learning without thalamic input")
    
    %% do simulation and update synaptic interaction matrix (learning)
    S = Ssum;
    firings = sim(S,Ne,Ni,cluster,t_end);
    sgtitle("spike train during first learning with thalamic input")
    
    S = S_update(S,firings,Ne,Ni,cluster,cluster_map);
    firings = sim(S,Ne,Ni,cluster,t_end);
    sgtitle("spike train during second learning with thalamic input")
    
    S = S_update(S,firings,Ne,Ni,cluster,cluster_map);
    firings = sim(S,Ne,Ni,cluster,t_end); 
    sgtitle("spike train during third learning with thalamic input")
    
    %% do simulation without thalamic input after learning
    S(:,(end-cluster + 1):end) = zeros(Nneu,cluster);
    sim(S,Ne,Ni,cluster,t_end);
    sgtitle("spike train after learning without thalamic input")
end

%% simulation for three types of neurons interacting each other
function firings = sim(S,Ne,Ni,cluster,t_end)    
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
    firings=[]; % spike timings
    
    for t=1:t_end % do simulation for designated time
        % thalamic input + random noise input
        I = [];
        for row = 1:1:cluster
            Icluster = [random('Uniform',0,F_ex,Ne,1);random('Uniform',0,F_in,Ni,1)];
            I = [I;Icluster];
        end
        I = [I;random('Uniform',0,F_thal,cluster,1)];

        % calculate V and firing for 1ms
        fired=find(v>=30); % indices of spikes
        
        if ~isempty(fired)
            firings=[firings; t+0*fired, fired];
            v(fired)=c(fired);
            u(fired)=u(fired)+d(fired);
            I=I+sum(S(:,fired),2);
        end
        
        v=v+0.5*(0.04*v.^2+5*v+140-u+I);
        v=v+0.5*(0.04*v.^2+5*v+140-u+I);
        u=u+a.*(b.*v-u);
    end
    
    %% draw raster plot
    figure; subplot(4,1,1);
    scatter(firings(1:end,1),firings(1:end,2),10,'filled');
    xlabel("time (ms)"); ylabel("neuron number"); xlim([50 300]);

    %% draw histogram of firing events
    subplot(4,1,2); histogram(firings(:,1),0:10:t_end);
    xlim([50 300]); xlabel("time (ms)"); ylabel("firing events");
    
    % add cluster number to firing events
    for row = 1:1:length(firings)
        if firings(row,2) <= (Ne+Ni) * cluster
            firings(row,3) = ceil(firings(row,2) / 50);
        else
            firings(row,3) = 0;
        end
    end

    % plot each cluster
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

    subplot(4,1,3)
    hist(firing_by_cluster,0:10:t_end);
    xlim([50 300]); xlabel("time (ms)"); ylabel("firing events");
    legend("cluster 1","cluster 2","cluster 3","cluster 4")
    
    %% draw PETH aligned to thalamic neuron firing.
    subplot(4,1,4); PETH(firings,201,Ne,Ni,cluster);
    xlabel("relative time to neuron 201 firing"); ylabel("firing events");
    xlim([-16 16]); ylim([0 50]); legend("diricet thalamic input","without thalamic input") 
end

%% update synpatic interaction following hebbian rule (spike time dependent plasticity)
%idx 1 for post synaptic neuron and idx 2 for presynaptic neuron
function S = S_update(S,firings,Ne,Ni,cluster,cluster_map)
    Noncluster_coeff = 0.1;
    N_nonthal = (Ne + Ni) * cluster;

    for idx1 = 1:length(firings)
        for idx2 = 1:length(firings)
            % update synaptic strength between cortical neurons.
            % only cortico-cortical connection updated.
            % auto connection was excluded
            if firings(idx1,2) ~= firings(idx2,2) if firings(idx2,2) <= N_nonthal if firings(idx1,2) <= N_nonthal %#ok<SEPEX,ALIGN>
                        if cluster_map(firings(idx1,2),firings(idx2,2)) == 1
                            S(firings(idx1,2),firings(idx2,2)) = S(firings(idx1,2),firings(idx2,2)) + STDP(firings(idx1,1) - firings(idx2,1));
                        else % if connection is outside cluster, reduce change by multiplying coefficient.
                            S(firings(idx1,2),firings(idx2,2)) = S(firings(idx1,2),firings(idx2,2)) + Noncluster_coeff * STDP(firings(idx1,1) - firings(idx2,1));
                        end
            end;end;end
        end
    end
end

%% STDP function.
function delta_EPSC = STDP(post_min_pre)
    %% parameters
    EPSC_0 = 2.5; % STDP coefficient
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

%% align firing event to specific neuron.
function PETH(firings,standard,Ne,Ni,cluster)
    rel_firings = [];
    engram = [];
    non_engram = [];

    len = length(firings);
    last_spike = -31;
    for idx = 1: len
        % when neuron No.1 fires at least 30ms after last firing
        if firings(idx,2) == standard && firings(idx,1) > last_spike + 30
            rel_firings = [rel_firings;firings - firings(idx,1) * horzcat(ones(len,1),zeros(len,2))];
            last_spike = firings(idx,1);
        end
    end

    % classify cells into engram and non engram cells.
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