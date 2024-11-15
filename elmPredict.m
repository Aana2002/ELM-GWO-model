function [output]=elmPredict(net,samples)
%% get options
number_neurons=net.Opts.number_neurons; % get number of neurons
ELM_Type=net.Opts.ELM_Type;             % get Application Type
Bn=net.Opts.Bn;                         % transform lables into binary codes
N1=net.min;                             % get denormalizing values
N2=net.max;                             % get denormalizing values
input_weights=net.IW;
B=net.OW;
%% normalization
samples=scaledata(samples,0,1);
%% Activation
H=radbas(input_weights*samples');
%% output
output=(H' * B) ;
%% Adjusting the output according to initial conditions
if ELM_Type=='Regrs'
output=scaledata(output,N1,N2);               % denormalization
else
    if Bn==1
    output=round(scaledata(output,0,1));      % adjust outputs normalization
    else
    output=round(scaledata(output,N1,N2));    % adjust outputs normalization
    end
end
end
