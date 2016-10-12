clear;
for i = 1:15
    clf(figure(i));
end

load('C:\Users\User\Desktop\NeuralNetwork\IDC _Pure_IC_Engine.mat')

%------------------------ Simulation Parameters ------------------------%

seconds_to_remove    = 20;
sampling             = 10^-4;
max_error            = 2;
time_window          = 2;
time_before_interupt = 2;


%--------------------------- Formating Inputs ---------------------------%

%Brake
brake_data = Brake_Time.Data;
brake_time = Brake_Time.Time;

brake_data(1:seconds_to_remove/sampling) = [];
brake_time(1:seconds_to_remove/sampling) = [];

%Throttle
throttle_data = Throttle_Time.Data;
throttle_time_1 = Throttle_Time.Time;

throttle_data(1:seconds_to_remove/sampling) = [];
throttle_time_1(1:seconds_to_remove/sampling) = [];

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


%-------------------- Dynamic Index & Initialization --------------------%

%Create dynamic index, for 0.1  millisecond sampling%
%Initialize index%
in = 2*(time_window/sampling)+1;
out = 2*(time_window/sampling) + (time_window/sampling);
dynamic_index = in:out; 
%Initialize error% 
error = 1000;
%Initialize counter%
train_count = 0;
%Initialize error sum%
error_sum = 0;
%Initializing total targets%
total_outputs = [];
%Initializing total output%
total_targets = [];


%----------------------------- Two windows -----------------------------%

%First window (present) used for speed estimation
%Second window (old) used for network training

for i = 0:(time_before_interupt/2)

    brake_data_present    = brake_data(dynamic_index);
    brake_time_present    = brake_time(dynamic_index);
    throttle_data_present = throttle_data(dynamic_index);
    throttle_time_present = throttle_time_1(dynamic_index);
    clutch_data_present   = clutch_data(dynamic_index);
    clutch_time_present   = clutch_time(dynamic_index);
    speed_data_present    = speed_data(dynamic_index);
    speed_time_present    = speed_time(dynamic_index);
    
    
    brake_data_old    = brake_data(in - 2*(time_window/sampling):in - (time_window/sampling));
    brake_time_old    = brake_time(in - 2*(time_window/sampling):in - (time_window/sampling));
    throttle_data_old = throttle_data(in - 2*(time_window/sampling):in - (time_window/sampling));
    throttle_time_old = throttle_time_1(in - 2*(time_window/sampling):in - (time_window/sampling));
    clutch_data_old   = clutch_data(in - 2*(time_window/sampling):in - (time_window/sampling));
    clutch_time_old   = clutch_time(in - 2*(time_window/sampling):in - (time_window/sampling));
    speed_data_old    = speed_data(in - 2*(time_window/sampling):in - (time_window/sampling));
    speed_time_old    = speed_time(in - 2*(time_window/sampling):in - (time_window/sampling));
  
    
%--------------------------- Creating Network ---------------------------%
    
    %Create input and target for the network
    %nbrSample = zeros(nbrSample,nbrIn);
    inputs_present  = [brake_data_present throttle_data_present clutch_data_present speed_data_present (throttle_data_present.*clutch_data_present)];
    inputs_old  = [brake_data_old throttle_data_old clutch_data_old speed_data_old (throttle_data_old.*clutch_data_old)];
    targets = speed_data_present;


    %Create training inputs and targets that uses less data
    reductionProcent = 1;
    
    nbrSamples = length(targets);
    
    trainInputs = inputs_old(1:nbrSamples*reductionProcent,:);
    trainTargets = targets(1:nbrSamples*reductionProcent,:);
    
    %Create net and train it
    s       = size(inputs_old);
    nbrIn   = s(2);
    s       = size(targets);
    nbrOut  = s(2);

    nbrHidden  = 10;
    nbrDelays  = 4; % 1 is normal, just input, no dependence from previous values
    nbrFBDelay = 4;
    
    if i == 0
    net = BackPropNetV2(nbrHidden,nbrIn,nbrOut,nbrDelays,nbrFBDelay);
    end
    
    
%-------------------------- Training Decision --------------------------%
    
    if error > max_error
    [net, Performance] = net.trainNetwork(trainInputs,trainTargets);
    train_count = train_count + 1;
    end
    output = net.output(inputs_present);
    targets(1:nbrDelays-1) = [];
    total_outputs = [total_outputs; output];
    total_targets = [total_targets; targets];
    clearvars sum
    %Actual error
    error = sum((targets-output).^2)/(nbrSamples);
    
    error_sum = error_sum + error;
    

%--------------------- Dynamic Index Incrementation ---------------------%    
    
    in = in + (time_window/sampling);
    out = out + (time_window/sampling);
    dynamic_index = in:out;
end

%---------------------- Post Training Analysation ----------------------%


%Linear regression
p = polyfit(targets,output,1);
x = linspace(0,max(targets),length(output));


total_shifts = time_before_interupt/2 + 1;
disp(['total_shifts = ' num2str(total_shifts)]);
disp(['train_count = ' num2str(train_count)]);
disp(['training_rate = ' num2str((train_count/total_shifts)*100) '%']);
disp(['average error = ' num2str(error_sum/total_shifts)]);


%--------------------------------------------------------------------------
%---------------------------PLOTS------------------------------------------
%--------------------------------------------------------------------------

%------Input plot------
figure(1); hold on;
plot(brake_time,brake_data)
plot(throttle_time_present,throttle_data_present,'r')
plot(speed_time_present,speed_data_present,'k')
plot(clutch_time_present,clutch_data_present,'g')
legend('Brake','Throttle','Speed','RPM')
title('Inputs')

t1 = linspace(0,nbrSamples/10000,nbrSamples-(nbrDelays-1));
%------Output plot------
figure(2); hold on;
plot(t1,output);
plot(t1,targets,'r')
title('Output')
legend('Net est.','Actual')
ylabel('Velocity / m/s')
xlabel('Time / s')

s1 = length(total_outputs);
t2 = linspace(0,s1/10000,s1);
%------General Output plot------
figure(3); hold on;
plot(t2,total_outputs);
plot(t2,total_targets);
title('Complete Output')
xlabel('Time / s')
ylabel('Velocity / m/s')
legend('Net est.','Actual')

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







