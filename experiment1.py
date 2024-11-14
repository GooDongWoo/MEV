####################################################################
# Scenario1: Temperature Scaling + entropy -> make dynamic ensemble of models which entropy is less than threshold

#step1: find T vector

#step2: find entropy threshold of total

#step3: sum of softmax vector of each good model(under threshold) -> final inference from softmax vector sum

####################################################################
# Scenario2: MC Dropout -> find confident EE -> static ensemble 

#step1: make MC Dropout model

#step2: find confident EE from experiment

#step3: sum of softmax vector of each good model(under threshold) -> final inference from softmax vector sum