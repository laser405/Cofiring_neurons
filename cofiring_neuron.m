% Created by Eugene M. Izhikevich, February 25, 2003
% Statistics and Machine Learning Toolbox required

%% neuron no 1~10 are excitatory neuron, 11 to 20 are inhibitory neuron.
% neuron no 21 is thalamic neuron with burst firing
% neurons interact each other and change their synaptic strength according
% to Hebbian rule.
% thalamic synapses remain unchanged while cortical interactions change.
% neuron shows cofiring after learning while neurons fire independently
% before learning
% initial synaptic strengths are randomly chosen. 

function main() %#ok<FNDEF>
    clc; clear; close all;
   
    %% parameter
    Ne=10; Ni=10; Nthal = 1; t_end = 300;
    
    %% mode 1
    Wee = 2; Wei = 2; Wie = 2; Wii = 2; Wt = 30;
    S1 = [Wee * abs(random('Normal',0,0.1,Ne,Ne)),Wie * -abs(random('Normal',0,0.1,Ne,Ni)),[Wt * ones(5,Nthal);zeros(Ne-5,Nthal)]];
    S2 = [Wei * abs(random('Normal',0,0.1,Ni,Ne)),Wii * -abs(random('Normal',0,0.1,Ni,Ni)),[Wt * ones(5,Nthal);zeros(Ni-5,Nthal)]];
    S3 = zeros(Nthal,Ne + Ni + Nthal);
    
    %% do simulation without learning
    S = [S1;S2;S3]; S(:,end) = zeros(Ne + Ni + Nthal,1);
    firings = sim(S,Ne,Ni,Nthal,t_end);
    sgtitle("spike train before learning without thalamic input")
    
    %% do simulation and update synaptic interaction matrix (learning)
    S = [S1;S2;S3];
    firings = sim(S,Ne,Ni,Nthal,t_end);
    sgtitle("spike train during first learning with thalamic input")
    
    S = S_update(S,firings,Ne+Ni);
    firings = sim(S,Ne,Ni,Nthal,t_end);
    sgtitle("spike train during second learning with thalamic input")
    
    S = S_update(S,firings,Ne+Ni);
    firings = sim(S,Ne,Ni,Nthal,t_end); 
    sgtitle("spike train during third learning with thalamic input")
    
    %% do simulation without thalamic input after learning
    S(:,end) = zeros(Ne + Ni + Nthal,1);
    sim(S,Ne,Ni,Nthal,t_end);
    sgtitle("spike train after learning without thalamic input")
end

%% simulation for three type of neurons interacting each other
function firings = sim(S,Ne,Ni,Nthal,t_end)    
    % Excitatory neurons / Inhibitory neurons
    F_ex = 10;         F_in = 10;       F_thal = 20;
    re=0.5 * ones(Ne,1);      ri=0.5 * ones(Ni,1);
    a=[0.02*ones(Ne,1); 0.02+0.08*ri;   0.02];
    b=[0.2*ones(Ne,1);  0.25-0.05*ri;   0.2];
    c=[-65+15*re.^2;    -65*ones(Ni,1); -50];
    d=[8-6*re.^2;       2*ones(Ni,1);    2];
    v=-65*ones(Ne+Ni+Nthal,1); % Initial values of v
    u=b.*v; % Initial values of u
    firings=[]; % spike timings
    
    for t=1:t_end % do simulation for designated time
        % thalamic input + random noise input
        I_feed = [random('Uniform',0,F_ex,Ne,1);random('Uniform',0,F_in,Ni,1);random('Uniform',0,F_thal,Nthal,1)];
        I = I_feed;
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
    
    figure; subplot(4,1,1);
    scatter(firings(1:end,1),firings(1:end,2),10,'filled');
    xlabel("time (ms)"); ylabel("neuron number");
    
    subplot(4,1,2); heatmap(S);
    xlabel("presynaptic neuron"); ylabel("postsynaptic neuron");
    
    subplot(4,1,3); histogram(firings(:,1),0:10:t_end);
    xlabel("time (ms)"); ylabel("firing events");
    xlim([0 300]); ylim([0 50]);
    
    subplot(4,1,4); PETH(firings,21);
    xlabel("relative time to neuron 21 firing"); ylabel("firing events");
    xlim([-16 16]); ylim([0 50]); legend("engram","non engram")
end

%% update synpatic interaction following hebbian rule (spike time dependent plasticity)
%idx 1 for post synaptic neuron and idx 2 for presynaptic neuron
function S = S_update(S,firings,N_nonthal)
    for idx1 = 1:length(firings)
        for idx2 = 1:length(firings)
            % update synaptic strength between cortical neurons.
            if firings(idx1,2) ~= firings(idx2,2) if firings(idx2,2) <= N_nonthal if firings(idx1,2) <= N_nonthal %#ok<SEPEX,ALIGN>
                S(firings(idx1,2),firings(idx2,2)) = S(firings(idx1,2),firings(idx2,2)) + STDP(firings(idx1,1) - firings(idx2,1));
            end;end;end
        end
    end
end

%% STDP function.
function delta_EPSC = STDP(post_min_pre)
    EPSC_0 = 3;
    tau = 10;
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
function PETH(firings,standard)
    rel_firings = [];
    engram = [];
    non_engram = [];

    matlen = length(firings);
    last_spike = -31;
    for idx = 1: matlen
        % when neuron No.1 fires at least 30ms after last firing
        if firings(idx,2) == standard && firings(idx,1) > last_spike + 30
            rel_firings = [rel_firings;firings - firings(idx,1) * horzcat(ones(matlen,1),zeros(matlen,1))];
            last_spike = firings(idx,1);
        end
    end

    % classify cells into engram and non engram cells.
    for idx = 1:1:length(rel_firings)
        if 1 <= rel_firings(idx,2) && rel_firings(idx,2) <= 5
            engram = [engram; rel_firings(idx,:)];
        elseif 6 <= rel_firings(idx,2) && rel_firings(idx,2) <= 10
            non_engram = [non_engram; rel_firings(idx,:)];
        elseif 11 <= rel_firings(idx,2) && rel_firings(idx,2) <= 15
            engram = [engram; rel_firings(idx,:)];
        elseif 16 <= rel_firings(idx,2) && rel_firings(idx,2) <= 20
            non_engram = [non_engram; rel_firings(idx,:)];
        end
    end
    
    hold on; histogram(engram(:,1),-16:2:16); histogram(non_engram(:,1),-16:2:16); hold off;
end

%% fire rate function (not used)
function [exc_mean,inh_mean] = firerate(firings,Ne,Ni,t_end) 
    for cell_num = 1:Ne+Ni
        firecount(cell_num,1) = length(find(firings(:,2) == cell_num));
    end
    
    plot(1:Ne+Ni,firecount)
    xlabel("cell number"); ylabel("firing rate (Hz)");
    exc_mean = mean(firecount(1:Ne,1)) / t_end * 1000;
    inh_mean = mean(firecount(Ne+1:Ne+Ni,1)) / t_end * 1000;
    fprintf("Average firing rate of excitatory neurons: %.2fHz\n",exc_mean);
    fprintf("Average firing rate of inhibitory neurons: %.2fHz\n",inh_mean);
end