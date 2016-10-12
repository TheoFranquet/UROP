function [IHW, HOW, Performance] = TrainBackProp(inIHW, inHOW, trainInputs, validInputs, testInputs, trainTargets, validTargets, testTargets, inputStates, Delays, DelaysFeedback)
 
    %Initial Learning Rate
    LearningRate = 0.01;
    %Numbers of max iterations
    limit = 100;
    %Derivate of activasion function
    DerActFun = @(x) 4.*exp(-2.*x)./((1+exp(-2.*x)).^2); %Derivative of tansig function
    %Parameters for adaptive learning rate
    a = 1.2;    
    b = 0.5;
    %Parameter for momentum
    alpha = 0.5;
    %Regularization parameter
    gamma = 0;
    
    %Starting by determing some constants for the network
    IHW        = inIHW;
    HOW        = inHOW;
    s          = size(trainInputs);
    ActNbrIn   = s(2); % Number of actual inputs
    s          = size(trainTargets);
    NbrOut     = s(2); %Number of outputs
    NbrIn      = (Delays*ActNbrIn)+1+(DelaysFeedback*NbrOut); %Number of inputs with all delays
    s          = size(IHW);
    NbrHidden  = s(1); % Number of hidden units
    InputBias  = -1;
    HiddenBias = -1;
    iterations = 0;
    %Variables used during propagation
    hj = zeros(1,NbrHidden);
    Vj = zeros(1,NbrHidden);
    hi = zeros(1,NbrOut);
   
    if length(trainInputs) ~= length(trainTargets)
       error('Inputs and target not same length') 
    end
    
    nbrTrain = length(trainInputs);
    nbrValid = length(validInputs);
    nbrTest  = length(testInputs);
    
    optIHW = IHW;
    optHOW = HOW;
    
    %Error related
    EpochError    = zeros(1,nbrTrain);
    Performance   = zeros(limit,3) + 1;
    OldPerf       = 0;
    DiffError     = 1;
    MinTrainError = 1000;
    MinValidError = 1000;
    MinTestError  = 1000;
    
    %Weight decay / Regularization
    nbrWeights = numel(HOW) + numel(IHW);
    
    
    while (iterations < limit) && (Performance(1) > 10^-6)  %Main loop that check for total error
        iterations = iterations + 1;
        in = [inputStates InputBias];
        DeltaHOW = 0;
        DeltaIHW = 0;
        
        for i =1:nbrTrain %Internal loop that goes through training patterns
            out = propagate(in);
            
            %Create inputSeries for next iterations
            prevIn = in;
            prevHidOut = [Vj HiddenBias];
            
            in(1:ActNbrIn) = [];
            prevY = in(end-DelaysFeedback:end-1);
            in(end-DelaysFeedback:end) = [];
            if DelaysFeedback > 0
                prevY(1) = [];
                prevY    = [prevY trainTargets(i,:)];
            end
            
            if i+1 <= nbrTrain
                in = [in trainInputs(i+1,:) prevY InputBias];
            end
            
            %-------Backpropagations---------
            %OutputLayer Delta
            OutputDeltas = (trainTargets(i,:)-out);
            
            %Hidden layer deltas
            HiddenDeltas = DerActFun(hj) .* sum(bsxfun(@times,HOW(:,1:end-1),OutputDeltas'),1);
            
            %------Update weights------
            tmp = LearningRate .* (OutputDeltas' * prevHidOut) + (alpha .* DeltaHOW); 
            HOW = (1-(LearningRate*gamma/nbrWeights))*HOW + tmp;
            DeltaHOW = tmp;
            
            tmp = LearningRate .* (HiddenDeltas' * prevIn) + alpha .* DeltaIHW;
            IHW = (1-(LearningRate*gamma/nbrWeights))*IHW + tmp;
            DeltaIHW = tmp;
            
            %Error for previous iterations/weight change
            EpochError(i) = sum(trainTargets(i,:) - out);
        end
        %------Performance calculation for training set------
        Performance(iterations,1) = sum((EpochError.^2))/nbrTrain;
        
        DiffError = abs(Performance(iterations,1)-OldPerf);
        OldPerf = Performance(iterations,1);
        
        %------Performance calculation for validation set------
        %Calculate output from validation set
        validOut = zeros(1,nbrValid-(Delays-1));
        VinputStates = zeros(1,NbrIn-1);
        vIn = validInputs;
        for i = 0:Delays-1
            tmp = 1+i*ActNbrIn;
            VinputStates(1,tmp:tmp+ActNbrIn-1) = vIn(i+1,:); 
        end
        vIn(1:Delays-1,:) = [];

        if DelaysFeedback > 0
            for i = 0:NbrOut - 1
                for k = 0:DelaysFeedback - 1
                    VinputStates(1,1 + ActNbrIn*Delays + i*DelaysFeedback + k) = 0;
                end 
            end
        end
       
        inV = [VinputStates InputBias];
        for i = 1:nbrValid-(Delays-1)
            validOut(i) = propagate(inV);
            %Create inputSeries for next iterations
            inV(1:ActNbrIn) = [];
            prevY = inV(end-DelaysFeedback:end-1);
            inV(end-DelaysFeedback:end) = [];
            if DelaysFeedback > 0
                prevY(1) = [];
                prevY = [prevY validTargets(i,:)];
            end
            
            if i+1 <= nbrValid-(Delays-1)
                inV = [inV vIn(i+1,:) prevY InputBias];
            end
        end
        %Performance from validation set
        vTar = validTargets;
        vTar(1:(Delays-1)) = [];
        
        Performance(iterations,2) = sum(((vTar-(validOut')).^2))/nbrValid;
        
        %--------Performance for test set------
        %Calculate output for test set
        testOut = zeros(1,nbrTest-(Delays-1));
        TinputStates = zeros(1,NbrIn-1);
        tIn = testInputs;
        for i = 0:Delays-1
            tmp = 1+i*ActNbrIn;
            TinputStates(1,tmp:tmp+ActNbrIn-1) = tIn(i+1,:); 
        end
        tIn(1:Delays-1,:) = [];

        if DelaysFeedback > 0
            for i = 0:NbrOut - 1
                for k = 0:DelaysFeedback - 1
                    TinputStates(1,1 + ActNbrIn*Delays + i*DelaysFeedback + k) = 0;
                end 
            end
        end
       
        inT = [TinputStates InputBias];
        for i = 1:nbrTest - (Delays-1)
            testOut(i) = propagate(inT);
            %Create inputSeries for next iterations
            inT(1:ActNbrIn) = [];
            prevY = inT(end-DelaysFeedback:end-1);
            inT(end-DelaysFeedback:end) = [];
            if DelaysFeedback > 0
                prevY(1) = [];
                prevY = [prevY validTargets(i,:)];
            end
            
            if i+1 <= nbrTest-(Delays-1)
                inT = [inT tIn(i+1,:) prevY InputBias];
            end
        end
        %Performance from validation set
        tTar = testTargets;
        tTar(1:(Delays-1)) = [];
        
        Performance(iterations,3) = sum(((tTar-(testOut')).^2))/nbrTest;
        
        %--------------------------------------
        
        %Display the current error
        currentPerf = [Performance(iterations,:) iterations]
        
        %Changing learning rate
        if OldPerf > Performance(iterations,1)
            %Increase learning rate
            LearningRate = a*LearningRate;
        else if OldPerf < Performance(iterations,1)
            %Decrease learningrate
            LearningRate = b*LearningRate;
            end
        end
        
        %Makes sure the optimal weights are chosen in the end
        if Performance(iterations,1) < MinTrainError
            MinTrainError = Performance(iterations,1);
            optIHW = IHW;
            optHOW = HOW;
        end
        
        if Performance(iterations,2) < MinValidError
            MinValidError = Performance(iterations,2);
        end
        
    end
    IHW = optIHW;
    HOW = optHOW;
    
    %Remove part of performance matrix that wasn't used
    if iterations ~= limit
        Performance(iterations+1:end,:) = [];
    end
    iterations
 
    function out = propagate(in)
        %Activasion functions
        hiddenFun = @(x) (2./(1+exp(-2.*x))) - 1;  %Tansig
        outFun = @(x) x;                            %Linear

        %Product of weights and inputs and summation for input to hidden unit
        hj = sum(bsxfun(@times,IHW,in),2)';

        %Activasion function of hidden units
        Vj = hiddenFun(hj);

        %Product of weight and output from hidden units and summation for
        %input to output layer
        hi = sum(bsxfun(@times,HOW,[Vj HiddenBias]),2)';

        %output from output layer
        out = outFun(hi);
    end
end