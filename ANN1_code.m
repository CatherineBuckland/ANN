%% Import data for training the model 

clear all; %#ok<CLALL>

% import empirical input data for training - each site includes tree ring
% growth index, grazing / land use pressure and wildfire occurrence
% for each timestep
input1 = csvread('SiteA_Input_Long.csv');
input2 = csvread('SiteB_Input_Long.csv');
input3 = csvread('SiteC_Input_Long.csv');
input4 = csvread('SiteD_Input_Long.csv');
input5 = csvread('SiteE_Input_Long.csv');
input6 = csvread('SiteF_Input_Long.csv');

%% Import data for future predictions

% import data for Low Grazing scenarios (six different scenarios refer to
% the different combinations with hypothetical climate and wildfire
% conditions)
input7 = csvread('LG_Scenario1.csv');
input8 = csvread('LG_Scenario2.csv');
input9 = csvread('LG_Scenario3.csv');
input10 = csvread('LG_Scenario4.csv');
input11 = csvread('LG_Scenario5.csv');
input12 = csvread('LG_Scenario6.csv');
% import data for Moderate Grazing scenarios
input13 = csvread('MG_Scenario1.csv');
input14 = csvread('MG_Scenario2.csv');
input15 = csvread('MG_Scenario3.csv');
input16 = csvread('MG_Scenario4.csv');
input17 = csvread('MG_Scenario5.csv');
input18 = csvread('MG_Scenario6.csv');
% import data for High Grazing scenarios
input19 = csvread('HG_Scenario1.csv');
input20 = csvread('HG_Scenario2.csv');
input21 = csvread('HG_Scenario3.csv');
input22 = csvread('HG_Scenario4.csv');
input23 = csvread('HG_Scenario5.csv');
input24 = csvread('HG_Scenario6.csv');

%% Define model parameters

best_current_total_score = 1e9; % 'score' assigned to each model performance is progressively assessed against the previous 'best' score - a high base score is set at the beginning

repeats = 5; % set the model to run the desired number of repeats
timesteps = length(input7); % define the length of the new 
data = zeros(timesteps,18,repeats); % create an empty array to be filled with the scenario datasets for all 18 combinations
data_peaks = zeros(repeats,1); % create an empty array for the scores of each model performance

%% Preprocessing of datasets

%smooth tree ring data
t1=1:length(input1(:,1)); 
input1(:,1) = smooth(t1,input1(:,1),0.2,'rloess'); %0.2 as default - change smoothing factor and method based on dataset
input2(:,1) = smooth(t1,input2(:,1),0.2,'rloess');
input3(:,1) = smooth(t1,input3(:,1),0.2,'rloess');
input4(:,1) = smooth(t1,input4(:,1),0.2,'rloess');
input5(:,1) = smooth(t1,input5(:,1),0.2,'rloess');
input6(:,1) = smooth(t1,input6(:,1),0.2,'rloess');

t2=1:length(input7(:,1)); 
input7(:,1) = smooth(t2,input7(:,1),0.2,'rloess'); 
input8(:,1) = smooth(t2,input8(:,1),0.2,'rloess');
input9(:,1) = smooth(t2,input9(:,1),0.2,'rloess');
input10(:,1) = smooth(t2,input10(:,1),0.2,'rloess');
input11(:,1) = smooth(t2,input11(:,1),0.2,'rloess');
input12(:,1) = smooth(t2,input12(:,1),0.2,'rloess'); 
input13(:,1) = smooth(t2,input13(:,1),0.2,'rloess');
input14(:,1) = smooth(t2,input14(:,1),0.2,'rloess');
input15(:,1) = smooth(t2,input15(:,1),0.2,'rloess');
input16(:,1) = smooth(t2,input16(:,1),0.2,'rloess');
input17(:,1) = smooth(t2,input17(:,1),0.2,'rloess');
input18(:,1) = smooth(t2,input18(:,1),0.2,'rloess'); 
input19(:,1) = smooth(t2,input19(:,1),0.2,'rloess');
input20(:,1) = smooth(t2,input20(:,1),0.2,'rloess');
input21(:,1) = smooth(t2,input21(:,1),0.2,'rloess');
input22(:,1) = smooth(t2,input22(:,1),0.2,'rloess');
input23(:,1) = smooth(t2,input23(:,1),0.2,'rloess');
input24(:,1) = smooth(t2,input24(:,1),0.2,'rloess');

%smooth grazing pressure data
t1=1:length(input1(:,3)); 
input1(:,3) = smooth(t1,input1(:,3),0.1,'moving'); 
input2(:,3) = smooth(t1,input2(:,3),0.1,'moving');
input3(:,3) = smooth(t1,input3(:,3),0.1,'moving');
input4(:,3) = smooth(t1,input4(:,3),0.1,'moving');
input5(:,3) = smooth(t1,input5(:,3),0.1,'moving');
input6(:,3) = smooth(t1,input6(:,3),0.1,'moving');

t2=1:length(input7(:,3)); 
input7(:,3) = smooth(t2,input7(:,3),0.1,'moving'); 
input8(:,3) = smooth(t2,input8(:,3),0.1,'moving');
input9(:,3) = smooth(t2,input9(:,3),0.1,'moving');
input10(:,3) = smooth(t2,input10(:,3),0.1,'moving');
input11(:,3) = smooth(t2,input11(:,3),0.1,'moving');
input12(:,3) = smooth(t2,input12(:,3),0.1,'moving'); 
input13(:,3) = smooth(t2,input13(:,3),0.1,'moving');
input14(:,3) = smooth(t2,input14(:,3),0.1,'moving');
input15(:,3) = smooth(t2,input15(:,3),0.1,'moving');
input16(:,3) = smooth(t2,input16(:,3),0.1,'moving');
input17(:,3) = smooth(t2,input17(:,3),0.1,'moving');
input18(:,3) = smooth(t2,input18(:,3),0.1,'moving');
input19(:,3) = smooth(t2,input19(:,3),0.1,'moving'); 
input20(:,3) = smooth(t2,input20(:,3),0.1,'moving');
input21(:,3) = smooth(t2,input21(:,3),0.1,'moving');
input22(:,3) = smooth(t2,input22(:,3),0.1,'moving');
input23(:,3) = smooth(t2,input23(:,3),0.1,'moving');
input24(:,3) = smooth(t2,input24(:,3),0.1,'moving');

% import empirical output data (i.e. sediment deposition metric)
target1 =  csvread('SiteA_Output_Long.csv');
target2 =  csvread('SiteB_Output_Long.csv');
target3 =  csvread('SiteC_Output_Long.csv');
target4 =  csvread('SiteD_Output_Long.csv');
target5 =  csvread('SiteE_Output_Long.csv');
target6 =  csvread('SiteF_Output_Long.csv');

% target training dataset standardised to maximum values - range from 0-1,
% and '0' values removed from training dataset
target1_std = (target1) ./ max(target1); target1_std(1:300)=NaN;
target2_std = (target2) ./ max(target2);
target3_std = (target3) ./ max(target3); target3_std(50:175)=NaN;
target4_std = (target4) ./ max(target4);
target5_std = (target5) ./ max(target5); target5_std(1:200)=NaN;
target6_std = (target6) ./ max(target6);

%% Handle data

columnSamples = false; % samples are by columns.
cellTime = false;     % time-steps in matrix, not cell array.

[x1, ~] = tonndata(input1, columnSamples, cellTime);
[x2, ~] = tonndata(input2, columnSamples, cellTime);
[x3, ~] = tonndata(input3, columnSamples, cellTime);
[x4, ~] = tonndata(input4, columnSamples, cellTime);
[x5, ~] = tonndata(input5, columnSamples, cellTime);
[x6, ~] = tonndata(input6, columnSamples, cellTime);
[x7, ~] = tonndata(input7, columnSamples, cellTime);
[x8, ~] = tonndata(input8, columnSamples, cellTime);
[x9, ~] = tonndata(input9, columnSamples, cellTime);
[x10, ~] = tonndata(input10, columnSamples, cellTime);
[x11, ~] = tonndata(input11, columnSamples, cellTime);
[x12, ~] = tonndata(input12, columnSamples, cellTime);
[x13, ~] = tonndata(input13, columnSamples, cellTime);
[x14, ~] = tonndata(input14, columnSamples, cellTime);
[x15, ~] = tonndata(input15, columnSamples, cellTime);
[x16, ~] = tonndata(input16, columnSamples, cellTime);
[x17, ~] = tonndata(input17, columnSamples, cellTime);
[x18, ~] = tonndata(input18, columnSamples, cellTime);
[x19, ~] = tonndata(input19, columnSamples, cellTime);
[x20, ~] = tonndata(input20, columnSamples, cellTime);
[x21, ~] = tonndata(input21, columnSamples, cellTime);
[x22, ~] = tonndata(input22, columnSamples, cellTime);
[x23, ~] = tonndata(input23, columnSamples, cellTime);
[x24, ~] = tonndata(input24, columnSamples, cellTime);
 
columnSamples = false; % samples are by columns.
cellTime = false;     % time-steps in matrix, not cell array.
[y1, ~] = tonndata(target1_std, columnSamples, cellTime);
[y2, ~] = tonndata(target2_std, columnSamples, cellTime);
[y3, ~] = tonndata(target3_std, columnSamples, cellTime);
[y4, ~] = tonndata(target4_std, columnSamples, cellTime);
[y5, ~] = tonndata(target5_std, columnSamples, cellTime);
[y6, ~] = tonndata(target6_std, columnSamples, cellTime);

%Samples to use for inputs and targets: all 6 sites need to be trained in
%tandem. Individual sites can be excluded from this stage and used for
%cross-validation
x = catsamples( x1, x2, x3, x4, x5, x6, 'pad'); %inputs:  x1, x2, x3, x4, x5, x6
y = catsamples( y1, y2, y3, y4, y5, y6, 'pad'); %targets: y1, y2, y3, y4, y5, y6


%% Train network and make predictions

for i=1:repeats %

disp(['Repeat no. ', num2str(i)]); % code inserted to show how many repeats have been completed

inputDelays = 1:8; % the number of time-step delays to be included
net = timedelaynet(inputDelays, [3 30 30 3], 'trainlm'); % neural network architecture and training algorithm - Levenberg-Marquadt used here

net.trainParam.epochs=300; % number of iterations within each model repeat
net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = 0.5; % fraction of known data to be used in training
net.divideParam.valRatio = 0.05; % fraction of known data to be used in validation
net.divideParam.testRatio = 0.45; % fraction of known data to be used in testing
net.performFcn = 'mse';  % Mean Squared Error

trained_net = train(net,x,y);

% predict training datasets (Sites A-F) (known datasets)
pred_x1 = trained_net(x1);
pred_x2 = trained_net(x2);
pred_x3 = trained_net(x3);
pred_x4 = trained_net(x4);
pred_x5 = trained_net(x5);
pred_x6 = trained_net(x6);

%convert NN cells to numerical matrices
p_x1 = cell2mat(pred_x1);
p_x2 = cell2mat(pred_x2);
p_x3 = cell2mat(pred_x3);
p_x4 = cell2mat(pred_x4);
p_x5 = cell2mat(pred_x5);
p_x6 = cell2mat(pred_x6);

% run model for the 18 hypothetical scenarios
pred_x7 = trained_net(x7);
pred_x8 = trained_net(x8);
pred_x9 = trained_net(x9);
pred_x10 = trained_net(x10);
pred_x11 = trained_net(x11);
pred_x12 = trained_net(x12);
pred_x13 = trained_net(x13);
pred_x14 = trained_net(x14);
pred_x15 = trained_net(x15);
pred_x16 = trained_net(x16);
pred_x17 = trained_net(x17);
pred_x18 = trained_net(x18);
pred_x19 = trained_net(x19);
pred_x20 = trained_net(x20);
pred_x21 = trained_net(x21);
pred_x22 = trained_net(x22);
pred_x23 = trained_net(x23);
pred_x24 = trained_net(x24);

% convert NN cells to numeric matrices 
p_x7 = cell2mat(pred_x7);
p_x8 = cell2mat(pred_x8);
p_x9 = cell2mat(pred_x9);
p_x10 = cell2mat(pred_x10);
p_x11 = cell2mat(pred_x11);
p_x12 = cell2mat(pred_x12);
p_x13 = cell2mat(pred_x13);
p_x14 = cell2mat(pred_x14);
p_x15 = cell2mat(pred_x15);
p_x16 = cell2mat(pred_x16);
p_x17 = cell2mat(pred_x17);
p_x18 = cell2mat(pred_x18);
p_x19 = cell2mat(pred_x19);
p_x20 = cell2mat(pred_x20);
p_x21 = cell2mat(pred_x21);
p_x22 = cell2mat(pred_x22);
p_x23 = cell2mat(pred_x23);
p_x24 = cell2mat(pred_x24);

% store predicted output data in array
data(:,1,i)=p_x7;
data(:,2,i)=p_x8;
data(:,3,i)=p_x9;
data(:,4,i)=p_x10;
data(:,5,i)=p_x11;
data(:,6,i)=p_x12;
data(:,7,i)=p_x13;
data(:,8,i)=p_x14;
data(:,9,i)=p_x15;
data(:,10,i)=p_x16;
data(:,11,i)=p_x17;
data(:,12,i)=p_x18;
data(:,13,i)=p_x19;
data(:,14,i)=p_x20;
data(:,15,i)=p_x21;
data(:,16,i)=p_x22;
data(:,17,i)=p_x23;
data(:,18,i)=p_x24;

%% find peaks to assess model performance

%smoothed versions of predicted outputs
t=1:length(input1(:,3));
smoothed_p_x1 = smooth(t,p_x1,0.1,'loess');
smoothed_p_x2 = smooth(t,p_x2,0.1,'loess');
smoothed_p_x3 = smooth(t,p_x3,0.1,'loess');
smoothed_p_x4 = smooth(t,p_x4,0.1,'loess');
smoothed_p_x5 = smooth(t,p_x5,0.1,'loess');
smoothed_p_x6 = smooth(t,p_x6,0.1,'loess');

%find peaks in target dataset
peaks_target=zeros(50,6);

[~,temp] = findpeaks(target1_std(1:end)); n_temp = length(temp); peaks_target(1:n_temp,1)= temp;
[~,temp] = findpeaks(target2_std(50:end)); n_temp = length(temp); peaks_target(1:n_temp,2)= temp;
[~,temp] = findpeaks(target3_std(1:end)); n_temp = length(temp); peaks_target(1:n_temp,3)= temp;
[~,temp] = findpeaks(target4_std(1:end)); n_temp = length(temp); peaks_target(1:n_temp,4)= temp;
[~,temp] = findpeaks(target5_std(1:end)); n_temp = length(temp); peaks_target(1:n_temp,5)= temp;
[~,temp] = findpeaks(target6_std(1:end)); n_temp = length(temp); peaks_target(1:n_temp,6)= temp;
peaks_target(1,2)=peaks_target(1,2)+50;peaks_target(2,2)=peaks_target(2,2)+50;peaks_target(3,2)=peaks_target(3,2)+50;

peaks_predicted=zeros(50,6);
[~,temp] = findpeaks(smoothed_p_x1); n_temp = length(temp); peaks_predicted(1:n_temp,1)= temp;
[~,temp] = findpeaks(smoothed_p_x2); n_temp = length(temp); peaks_predicted(1:n_temp,2)= temp;
[~,temp] = findpeaks(smoothed_p_x3); n_temp = length(temp); peaks_predicted(1:n_temp,3)= temp;
[~,temp] = findpeaks(smoothed_p_x4); n_temp = length(temp); peaks_predicted(1:n_temp,4)= temp;
[~,temp] = findpeaks(smoothed_p_x5); n_temp = length(temp); peaks_predicted(1:n_temp,5)= temp;
[~,temp] = findpeaks(smoothed_p_x6); n_temp = length(temp); peaks_predicted(1:n_temp,6)= temp;

%find if peak nearby for each site
total_dist_per_site = zeros(6,1);
window_width = 13;

for m=1:6
    sum_dist_per_site = 0;
    n_OSL_peaks = sum(peaks_target(:,m)>0);
    for j = 1: n_OSL_peaks
        
        peak_window = peaks_target(j,m) - window_width : peaks_target(j,m) + window_width;
        matched_predicted_peak = find(ismember(peaks_predicted(:,m),peak_window)==1);
        if matched_predicted_peak > 0
            if matched_predicted_peak > 5
                dist = 1e6;
            else    
                dist = abs(peaks_target(j,m) - peaks_predicted(matched_predicted_peak,m));
            end
        else
            dist = 1e6;
        end
        sum_dist_per_site = sum_dist_per_site + min(dist);
            
    end
     total_dist_per_site(m,1) = sum_dist_per_site;   
end

data_peaks(i)= sum(total_dist_per_site);

end
%% Restore data_cleaned with data

data_cleaned = data;
del_list = [];

cut_off = prctile( (data_peaks(:,1)), 10); % define the cut off percentile of data, e.g. 10 = select top 10% of model outputs

%% Create new dataset of cleaned model outputs
for i=1:repeats
    
    if data_peaks(i) > cut_off
       del_list = [del_list,i];
    end

end 

data_cleaned(:,:,del_list) = [];
del_list = del_list';

%% Calculate MEAN & Standard Error of cleaned data
t=1:length(p_x7);
mean_val = zeros(timesteps,18);
upperm_val = zeros(timesteps,18);
lowerm_val = zeros(timesteps,18);
%%
%calculate mean and CI of data
%for i=1:repeats
    for j=1:timesteps
        for k=1:18
            mean_val(j,k) = mean(data_cleaned(j,k,:));
            upperm_val(j,k) = mean_val(j,k) + (std((data_cleaned(j,k,:))) / sqrt(length(((data_cleaned(j,k,:))))));
            lowerm_val(j,k) = mean_val(j,k) - (std((data_cleaned(j,k,:))) / sqrt(length(((data_cleaned(j,k,:))))));
            disp(length(((data_cleaned(j,k,:)))));
           
        end
    end
    
%% Plot MEAN & Standard error of cleaned data

figure('Name','MEAN Scenario Outputs 1-6, Low, Moderate and High Grazing');
for i = 1:18
    xlim([0 100]);
subplot(3,6,i); plot(t,mean_val(:,i), t,upperm_val(:,i), t,lowerm_val(:,i));
end

t=1:length(x7);
figure('Name','MEAN Climate 1: Scenario 1-3 outputs');
subplot(1,3,1); plot(t,mean_val(:,1),'xb', t,mean_val(:,2),'-g', t,mean_val(:,3),'-r');title('Low Grazing');
xlim([0 100]);
xlim([0 100]);
subplot(1,3,2); plot(t,mean_val(:,7),'xb', t,mean_val(:,8),'-g', t,mean_val(:,9),'-r');title('Moderate Grazing');
xlim([0 100]);
xlim([0 100]);
subplot(1,3,3); plot(t,mean_val(:,13),'xb', t,mean_val(:,14),'-g', t,mean_val(:,15),'-r');title('High Grazing');
xlim([0 100]);
xlim([0 100]);

t=1:length(x7);
figure('Name','MEAN Climate 2: Scenario 4-6 outputs');
subplot(1,3,1); plot(t,mean_val(:,4),'xb', t,mean_val(:,5),'-g', t,mean_val(:,6),'-r');title('Low Grazing');
xlim([0 100]);
xlim([0 100]);
subplot(1,3,2); plot(t,mean_val(:,10),'xb', t,mean_val(:,11),'-g', t,mean_val(:,12),'-r');title('Moderate Grazing');
xlim([0 100]);
xlim([0 100]);
subplot(1,3,3); plot(t,mean_val(:,16),'xb', t,mean_val(:,17),'-g', t,mean_val(:,18),'-r');title('High Grazing');
xlim([0 100]);
xlim([0 100]);
