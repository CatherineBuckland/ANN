%% Import data
clear all; %#ok<CLALL>

% import empirical input data
input1 = csvread('Climate_Training.csv');

% import future prediction data
input2 = csvread('Climate1.csv');
input3 = csvread('Climate2.csv');

repeats = 10; % define number of model repeats 
timesteps = length(input2);
data = zeros(timesteps,2,repeats); 

%loop to run NN N times

for i=1:repeats
%% Preprocess input datasets
% smooth input climate datasets
t=1:length(input1(:,1)); input1(:,1) = smooth(t,input1(:,1),0.2,'rloess');
t=1:length(input1(:,2)); input1(:,2) = smooth(t,input1(:,2),0.2,'rloess');
t=1:length(input1(:,3)); input1(:,3) = smooth(t,input1(:,3),0.2,'rloess');
t=1:length(input2(:,1)); input2(:,1) = smooth(t,input2(:,1),0.2,'rloess');
t=1:length(input2(:,2)); input2(:,2) = smooth(t,input2(:,2),0.2,'rloess');
t=1:length(input2(:,3)); input2(:,3) = smooth(t,input2(:,3),0.2,'rloess');
t=1:length(input3(:,1)); input3(:,1) = smooth(t,input3(:,1),0.2,'rloess');
t=1:length(input3(:,2)); input3(:,2) = smooth(t,input3(:,2),0.2,'rloess');
t=1:length(input3(:,3)); input3(:,3) = smooth(t,input3(:,3),0.2,'rloess');

% import empirical output data
target1 =  csvread('Tr_Training.csv');
t=1:length(target1); target1 = smooth(t,target1,0.1,'rloess');

%% Handle data

columnSamples = false; % samples are by columns.
cellTime = false;     % time-steps in matrix, not cell array.

[x1, ~] = tonndata(input1, columnSamples, cellTime);
[x2, ~] = tonndata(input2, columnSamples, cellTime);
[x3, ~] = tonndata(input3, columnSamples, cellTime);

columnSamples = false; % samples are by columns.
cellTime = false;     % time-steps in matrix, not cell array.
[y1, ~] = tonndata(target1, columnSamples, cellTime);

%Samples to use for inputs and targets:
x = catsamples( x1, 'pad'); %inputs:  x1, 
y = catsamples( y1, 'pad'); %targets: y1, 


%% Train network and make predictions

net = fitnet([9], 'trainlm'); % define network architecture and training algorithm

net.trainParam.epochs=3000; % number of iterations within each model repeat
net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = 0.7; % fraction of known data to be used in training
net.divideParam.valRatio = 0.1; % fraction of known data to be used in validation
net.divideParam.testRatio = 0.2; % fraction of known data to be used in testing
net.performFcn = 'mse';  % Mean Squared Error

trained_net = train(net,x,y);

pred_x1 = trained_net(x1);
pred_x2 = trained_net(x2);
pred_x3 = trained_net(x3);

%

%% Convert and store predictions

p_x2 = cell2mat(pred_x2);
p_x3 = cell2mat(pred_x3);

data(:,1,i)=p_x2;
data(:,2,i)=p_x3;

end %end of i loop to repeat NN

%% Clean out erroneous datasets

data_cleaned = data;
del_list = [];
extreme_values = zeros(timesteps,2);
n_extreme_values = zeros(repeats,1);

for i = 1:repeats
   extreme_values(:,:) = abs(data_cleaned(:,:,i)) > 5000;
   n_extreme_values(i,1) = sum(sum(extreme_values));
   if n_extreme_values(i,1) > 8
       del_list = [del_list, i];
   end
end

n_extreme_values;

data_cleaned(:,:,del_list) = [];

del_list = del_list';

%% Calculate mean and CI of final datasets
t=1:length(p_x2);
mean_val = zeros(timesteps,2);
upper_val = zeros(timesteps,2);
lower_val = zeros(timesteps,2);

    for j=1:timesteps
        for k=1:2
            mean_val(j,k) = mean(data_cleaned(j,k,:));
            upperm_val(j,k) = mean_val(j,k) + (std((data_cleaned(j,k,:))) / sqrt(length(((data_cleaned(j,k,:))))));
            lowerm_val(j,k) = mean_val(j,k) - (std((data_cleaned(j,k,:))) / sqrt(length(((data_cleaned(j,k,:))))));
        end
    end
%end
figure('Name','Climate 1 and 2 predicted tree ring indices');
for i = 1:2
    xlim([0 100])
    ylim([0 2000])
subplot(1,2,i); plot(t,mean_val(:,i), t,upperm_val(:,i), t,lowerm_val(:,i));
    xlim([0 100])
    ylim([0 2000])
end
%}


t=1:length(x);
figure('Name','Trained model tree ring index');
% 
p_x1 = cell2mat(pred_x1);
t_y1=cell2mat(y1);
subplot(1,1,1); plot(t,t_y1,'-b',t,p_x1,'-r');