clear;
for i = 1:15
    clf(figure(i));
end

load('C:\Users\User\Desktop\NeuralNetwork\IDC _Pure_IC_Engine.mat')

seconds_to_remove  = 20;
seconds_to_predict = 1;
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

%trainInputs = inputs(2000:6500,:);
%trainTargets = targets(2000:6500,:);

%Create net and train it
s       = size(inputs);
nbrIn   = s(2);
s       = size(targets);
nbrOut  = s(2);

nbrHidden  = 10;
nbrDelays  = 4; % 1 is normal, just input, no dependence from previous values
nbrFBDelay = 4;

net = BackPropNetV2(nbrHidden,nbrIn,nbrOut,nbrDelays,nbrFBDelay);

[net, Performance] = net.trainNetwork(trainInputs,trainTargets);

output = net.output(inputs);


%------Post training analyzation------
targets(1:nbrDelays-1) = [];

clearvars sum
%Actual error
error = sum((targets-output).^2)/(nbrSamples);

%Linear regression
p = polyfit(targets,output,1);
x = linspace(0,max(targets),length(output));

%Covariance coefficient
meanTargets = mean(targets);
meanOutputs = mean(output);
R = (sum((targets-meanTargets).*(output-meanOutputs)))./sqrt(sum((targets-meanTargets).^2) .* sum((output-meanOutputs).^2));


%--------------------------------------------------------------------------
%---------------------------PLOTS------------------------------------------
%--------------------------------------------------------------------------

%------Input plot------
figure(1); hold on;
plot(brake_time,brake_data)
plot(throttle_time,throttle_data,'r')
plot(speed_time,speed_data,'k')
plot(clutch_time,clutch_data/80,'g')
legend('Brake','Throttle','Speed','RPM')
title('Inputs')

t = linspace(0,nbrSamples/100,nbrSamples-(nbrDelays-1));
%------Output plot------
figure(2); hold on;
plot(t,output);
plot(t,targets,'r')
title('Output')
legend('Net est.','Actual')
ylabel('Velocity / m/s')
xlabel('Time / s')

%------Covariance plot------
figure(3); hold on;
plot(targets,output,'x');
plot(x,polyval(p,x));
title('Covariance')
xlabel('Targets')
ylabel('Output')
legend('Data','Lin.Reg')
dim = [.2 .5 .3 .3];
str = ['R = '  num2str(R)];
annotation('textbox',dim,'String',str,'FitBoxToText','on');

%------Performance plot------
figure(4); hold on;
plot(Performance(:,1));
plot(Performance(:,2));
plot(Performance(:,3));
title('Performance')
legend('Training','Validation','Test');
xlabel('Iterations')
dim = [.2 .5 .3 .3];
str = ['Actual error = '  num2str(error)];
annotation('textbox',dim,'String',str,'FitBoxToText','on');

%--------------------------------------------------------------------------
%--------------------------------------------------------------------------