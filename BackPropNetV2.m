classdef BackPropNetV2
   properties
      IHW %Input hidden weights
      HOW %Hidden output weights
      NbrHidden
      NbrIn
      NbrOut
      ActualNbrIn
      Delay
      DelayFeedback
      HiddenBias = -1
      InputBias = -1
      TrainPercent = 0.7
      ValidPercent = 0.15;
      TestPercent = 0.15;
      InputNormSettings
      TargetNormSettings
      NormMin = -1
      NormMax = 1
   end
   
   methods
       function obj = BackPropNetV2(NbrHidden,NbrIn,NbrOut,Delay,DelayFeedback)
           obj.NbrHidden = NbrHidden;
           obj.ActualNbrIn = NbrIn;
           obj.NbrIn = (Delay*NbrIn)+1+(DelayFeedback*NbrOut); %Makes sure we have correct nbr of inputs with all the delays
           obj.NbrOut = NbrOut;
           obj.Delay = Delay;
           obj.DelayFeedback = DelayFeedback;
           
           %Initiate weights with random values - Will generate values
           %between -1 and 1
%            obj.IHW = rand(obj.NbrHidden,obj.NbrIn).*2 - 1;    
%            obj.HOW = rand(obj.NbrOut,obj.NbrHidden+1).*2 - 1;
           
           %Initaiate weights with smaller standard deviation and mean
           %value 0
           obj.IHW = normrnd(0,1/sqrt(obj.NbrHidden*obj.NbrIn),obj.NbrHidden,obj.NbrIn);    
           obj.HOW = normrnd(0,1/sqrt(obj.NbrHidden*obj.NbrOut),obj.NbrOut,obj.NbrHidden+1);
           
           obj.InputNormSettings  = {};
           obj.TargetNormSettings = {};
       end
       function out = output(obj,input)
            s = size(input);
            out = zeros(s(1)-(obj.Delay-1),obj.NbrOut);
            
            %Preprocess input data
            tmpIn = zeros(size(input));
            
            for i = 1:obj.ActualNbrIn
                tmpIn(:,i) = mapminmax('apply', input(:,i)', obj.InputNormSettings{i});
            end
            
            %Create initial input
            inputStates = zeros(1,obj.NbrIn-1);

            for i = 0:obj.Delay-1
                tmp = 1+i*obj.ActualNbrIn;
                inputStates(1,tmp:tmp+obj.ActualNbrIn-1) = tmpIn(i+1,:); 
            end
            input(1:obj.Delay-1,:) = [];
            tmpIn(1:obj.Delay-1,:) = [];
            
            if obj.DelayFeedback > 0
                for i = 0:obj.NbrOut - 1
                    for k = 0:obj.DelayFeedback - 1
                        inputStates(1,1 + obj.ActualNbrIn*obj.Delay + i*obj.DelayFeedback + k) = 0;
                    end 
                end
            end
            
            in = [inputStates obj.InputBias];
            for i = 1:s(1)-(obj.Delay-1)
                
                out(i,:) = obj.propagate(in);

                %Create inputSeries for next iterations
                
                in(1:obj.ActualNbrIn) = [];
                prevY = in(end-obj.DelayFeedback:end-1);
                in(end-obj.DelayFeedback:end) = [];
                if obj.DelayFeedback > 0
                    prevY(1) = [];
                    prevY = [prevY out(i,:)];
                end

                if i+1 <= s(1)-(obj.Delay-1)
                    in = [in tmpIn(i+1,:) prevY obj.InputBias];
                end
            end
            
            for i = 1:obj.NbrOut
                out(:,i) = mapminmax('reverse',out(:,i)',obj.TargetNormSettings{i});
            end
       end
       function [obj, Performance] = trainNetwork(obj,inputs, targets)
            %Preprocess input and target data
            tmpIn = zeros(size(inputs));
            
            for i = 1:obj.ActualNbrIn
                [tmpIn(:,i), obj.InputNormSettings{i}] = mapminmax(inputs(:,i)',obj.NormMin,obj.NormMax);
            end
            
            tmpTar = zeros(size(targets));
            
            for i = 1:obj.NbrOut
                [tmpTar(:,i), obj.TargetNormSettings{i}] = mapminmax(targets(:,i)',obj.NormMin,obj.NormMax);
            end
            
            %Prepare inputs for training
            inputStates = zeros(1,obj.NbrIn-1);

            for i = 0:obj.Delay-1
                tmp = 1+i*obj.ActualNbrIn;
                inputStates(1,tmp:tmp+obj.ActualNbrIn-1) = tmpIn(i+1,:); 
            end
            inputs(1:obj.Delay-1,:) = [];
            tmpIn(1:obj.Delay-1,:) = [];
            
            if obj.DelayFeedback > 0
                for i = 0:obj.NbrOut - 1
                    for k = 0:obj.DelayFeedback - 1
                        inputStates(1,1 + obj.ActualNbrIn*obj.Delay + i*obj.DelayFeedback + k) = tmpTar(k+1,i+1);
                    end 
                end
            end
            
             targets(1:obj.Delay-1,:) = [];
             tmpTar(1:obj.Delay-1,:) = [];
            
            [trainInd, valInd, testInd] = dividerand(length(tmpIn),obj.TrainPercent,obj.ValidPercent,obj.TestPercent);
            trainInputs = tmpIn;
            validInputs = tmpIn(valInd,:);
            testInputs  = tmpIn(testInd,:);
            
            trainTargets = tmpTar;
            validTargets = tmpTar(valInd,:);
            testTargets  = tmpTar(testInd,:);
            
            [IHWout, HOWout, Performance] = TrainLM(obj.IHW, obj.HOW, trainInputs, validInputs, testInputs,trainTargets...
                                ,validTargets,testTargets,inputStates,obj.Delay,obj.DelayFeedback); 
            obj.IHW = IHWout;
            obj.HOW = HOWout;
       end
       function out = propagate(obj,in)
            %Activasion functions
            hiddenFun = @(x) (2./(1+exp(-2.*x))) - 1;  %Tansig
            %hiddenFun = @(x) 1./(1+exp(-1.*x));         %Sigmoid
            outFun = @(x) x;                            %Linear

            %in = [in obj.InputBias];
            %Product of weights and inputs and summation for input to hidden unit
            hj = sum(bsxfun(@times,obj.IHW,in),2)';

            %Activasion function of hidden units
            Vj = hiddenFun(hj);

            %Product of weight and output from hidden units and summation for
            %input to output layer
            hi = sum(bsxfun(@times,obj.HOW,[Vj obj.HiddenBias]),2)';

            %output from output layer
            out = outFun(hi);
        end
   end
end
