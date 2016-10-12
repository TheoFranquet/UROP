clear;

load('../IDC _Pure_IC_Engine.mat')

seconds_to_remove  = 20;
seconds_to_predict = 2;
sampling           = 10^-4;

%Brake
brake_data = Brake_Time.Data;
brake_time = Brake_Time.Time;

brake_data(1:seconds_to_remove/sampling) = [];
brake_time(1:seconds_to_remove/sampling) = [];

%Throttle
throttle_data = Throttle_Time.Data;
throttle_time = Throttle_Time.Time;

throttle_data(1:seconds_to_remove/sampling) = [];
throttle_time(1:seconds_to_remove/sampling) = [];

%Clutch/IP
clutch_data = Clutch_IP_ClutchSpeed.Data;
clutch_time = Clutch_IP_ClutchSpeed.Time;

clutch_data(1:seconds_to_remove/sampling) = [];
clutch_time(1:seconds_to_remove/sampling) = [];

%Speed
speed_data = VSpeed_khr.Data;
speed_time = VSpeed_khr.Time;

speed_data(1:seconds_to_remove/sampling) = [];
speed_time(1:seconds_to_remove/sampling) = [];

%Reducing sample rate
nbrSamples = length(brake_data);
reduction  = 100;
index      = 1:reduction:nbrSamples;
sampling   = sampling*reduction;

brake_data    = brake_data(index);
brake_time    = brake_time(index);
throttle_data = throttle_data(index);
throttle_time = throttle_time(index);
clutch_data   = clutch_data(index);
clutch_time   = clutch_time(index);
speed_data    = speed_data(index);
speed_time    = speed_time(index);



%Create input and target for the network
%nbrSample = zeros(nbrSample,nbrIn);
inputs  = [brake_data throttle_data clutch_data speed_data (throttle_data.*clutch_data)];
targets = speed_data;

%Removes appropriate nbr of samples to make the network predict future
%speeds

inputs(end-(seconds_to_predict/sampling):end,:) = [];
targets(1:(seconds_to_predict/sampling)+1,:) = [];

nbrSamples = length(targets);

%Create training inputs and targets that uses less data
reductionProcent = 1;

trainInputs = inputs(1:nbrSamples*reductionProcent,:);
trainTargets = targets(1:nbrSamples*reductionProcent,:);


inputSeries = cell(1,nbrSamples);
targetSeries = cell(1,nbrSamples);
for i=1:nbrSamples
   inputSeries{i} = trainInputs(i,:)';
   targetSeries{i} = trainTargets(i,:);
end

% Create a Nonlinear Autoregressive Network with External Input
inputDelays = 1:4;
feedbackDelays = 1:4;
hiddenLayerSize = 10;
net = narxnet(inputDelays,feedbackDelays,hiddenLayerSize);

% Prepare the Data for Training and Simulation
% The function PREPARETS prepares time series data 
% for a particular network, shifting time by the minimum 
% amount to fill input states and layer states.
% Using PREPARETS allows you to keep your original 
% time series data unchanged, while easily customizing it 
% for networks with differing numbers of delays, with
% open loop or closed loop feedback modes.
[inputs,inputStates,layerStates,targets] = ... 
    preparets(net,inputSeries,{},targetSeries);

%[inputs,inputStates,targets] = prepareNetworkData(inputSeries,targetSeries)
% Set up Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% Train the Network
[net,tr] = train(net,inputs,targets,inputStates,layerStates);

% Test the Network
outputs = net(inputs,inputStates,layerStates);
errors = gsubtract(targets,outputs);
performance = perform(net,targets,outputs)

newTargets = zeros(1,length(targets));
newOutputs  = zeros(1,length(outputs));

for i = 1:length(targets)
   newTargets(i) = targets{i};
   newOutputs(i) = outputs{i};
end
clearvars sum;
%Actual error
newError = sum((newTargets-newOutputs).^2)/(nbrSamples)

%Linear regression
p = polyfit(newTargets,newOutputs,1);
x = linspace(0,max(newTargets),length(newOutputs));

%Covariance coefficient
meanTargets = mean(newTargets);
meanOutputs = mean(newOutputs);
R = (sum((newTargets-meanTargets).*(newOutputs-meanOutputs)))./sqrt(sum((newTargets-meanTargets).^2) .* sum((newOutputs-meanOutputs).^2));

%------Output plot------
figure(2); hold on;
plot(newOutputs);
plot(newTargets,'r')
title('Output')
legend('Net est.','Actual')
ylabel('Velocity / m/s')
xlabel('Time / ms')

%------Covariance plot------
figure(3); hold on;
plot(newTargets,newOutputs,'x');
plot(x,polyval(p,x));
title('Covariance')
xlabel('Targets')
ylabel('Output')
legend('Data','Lin.Reg')
dim = [.2 .5 .3 .3];
str = ['R = '  num2str(R)];
annotation('textbox',dim,'String',str,'FitBoxToText','on');

% View the Network
view(net)

% Plots
% Uncomment these lines to enable various plots.
% figure, plotperform(tr)
% figure, plottrainstate(tr)
% figure, plotregression(targets,outputs)
% figure, plotresponse(targets,outputs)
% figure, ploterrcorr(errors)
% figure, plotinerrcorr(inputs,errors)


% Closed Loop Network
% Use this network to do multi-step prediction.
% The function CLOSELOOP replaces the feedback input with a direct
% connection from the output layer.
netc = closeloop(net);
netc.name = [net.name ' - Closed Loop'];
view(netc)
[xc,xic,aic,tc] = preparets(netc,inputSeries,{},targetSeries);
yc = netc(xc,xic,aic);
closedLoopPerformance = perform(netc,tc,yc)

newTargets = zeros(1,length(targets));
newOutputs  = zeros(1,length(yc));

for i = 1:length(targets)
   newTargets(i) = targets{i};
   newOutputs(i) = yc{i};
end
clearvars sum;
%Actual error
newErrorClosed = sum((newTargets-newOutputs).^2)/(nbrSamples)

%Linear regression
p = polyfit(newTargets,newOutputs,1);
x = linspace(0,max(newTargets),length(newOutputs));

%Covariance coefficient
meanTargets = mean(newTargets);
meanOutputs = mean(newOutputs);
R = (sum((newTargets-meanTargets).*(newOutputs-meanOutputs)))./sqrt(sum((newTargets-meanTargets).^2) .* sum((newOutputs-meanOutputs).^2));

%------Output plot------
figure(4); hold on;
plot(newOutputs);
plot(newTargets,'r')
title('Output')
legend('Net est.','Actual')
ylabel('Velocity / m/s')
xlabel('Time / ms')

%------Covariance plot------
figure(5); hold on;
plot(newTargets,newOutputs,'x');
plot(x,polyval(p,x));
title('Covariance')
xlabel('Targets')
ylabel('Output')
legend('Data','Lin.Reg')
dim = [.2 .5 .3 .3];
str = ['R = '  num2str(R)];
annotation('textbox',dim,'String',str,'FitBoxToText','on');


% Early Prediction Network
% For some applications it helps to get the prediction a 
% timestep early.
% The original network returns predicted y(t+1) at the same 
% time it is given y(t+1).
% For some applications such as decision making, it would 
% help to have predicted y(t+1) once y(t) is available, but 
% before the actual y(t+1) occurs.
% The network can be made to return its output a timestep early 
% by removing one delay so that its minimal tap delay is now 
% 0 instead of 1.  The new network returns the same outputs as 
% the original network, but outputs are shifted left one timestep.
nets = removedelay(net);
nets.name = [net.name ' - Predict One Step Ahead'];
view(nets)
[xs,xis,ais,ts] = preparets(nets,inputSeries,{},targetSeries);
ys = nets(xs,xis,ais);
earlyPredictPerformance = perform(nets,ts,ys)


