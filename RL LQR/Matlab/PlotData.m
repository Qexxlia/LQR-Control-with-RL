clc;clear;

run = "PPO_20240508-2122__LR=0.0003__NS=1024__BS=64__NE=10__CR=0.2__G=0.99__LSI=0.0";


path = "../models/spacecraft/" + run + "/data/";

files = dir(path);
filenames = {files(:).name};
doubles = str2double(strrep(strrep(filenames(endsWith(filenames, '.csv')), 'state', ''), '.csv', ''));
csvs = path + 'state' + string(doubles(:)) + '.csv';

for i = 1:length(csvs)
    data{i,1} = strrep(strrep(csvs{i}, path + 'state', ''), '.csv', '');
    data{i,2}(:, :) = readtable(csvs{i});
end

[~,sortOrder] = sort(str2double([data{:,1}]));

sortedData = data(sortOrder, :);


figure()
tl = tiledlayout(2,1);

for i = 1:length(sortedData)    
nexttile(1)
cla()
hold on
p1 = plot(sortedData{i,2}, 't', 'x', DisplayName='x');
p2 = plot(sortedData{i,2}, 't', 'y', DisplayName='y');
p3 = plot(sortedData{i,2}, 't', 'z', DisplayName='z');
legend([p1 p2 p3])
ylim([-0.25 1]);
xlim([0 60]);
for j = 10:10:60
    xline(j, HandleVisibility="off", LineStyle=':')
end

nexttile(2)
cla()
hold on
p1 = plot(sortedData{i,2}, 't', 'vx', DisplayName='vx');
p2 = plot(sortedData{i,2}, 't', 'vy', DisplayName='vy');
p3 = plot(sortedData{i,2}, 't', 'vz', DisplayName='vz');
legend([p1 p2 p3])
ylim([-0.3 0.3]);
xlim([0 60]);
for j = 10:10:60
    xline(j, HandleVisibility="off", LineStyle=':')
end

pause(0.1)
end