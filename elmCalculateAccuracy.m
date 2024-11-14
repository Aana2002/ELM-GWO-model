function acc = elmCalculateAccuracy(inputs, targets, net, Opts)
    outputs = elmPredict(net, inputs);
    %[~, predicted] = max(outputs, [], 2);
    %[~, actual] = max(targets, [], 2);
    %acc = sum(predicted == actual) / length(actual);
    acc = 0;
    for x = 1:length(outputs)
        if abs(outputs(x) - targets(x)) <= .2 % can change threshold for accuracy
            acc = acc + 1;
        end
    end
    acc = acc / length(outputs);
end
