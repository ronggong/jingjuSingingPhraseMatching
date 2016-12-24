load temp.mat

[y,Fs] = audioread(filename_wav);
start_frame_line    = round(str2double(lineList(1,:))*Fs);
end_frame_line      = round(str2double(lineList(2,:))*Fs);
y_line = y(start_frame_line:end_frame_line);

[vs,~]              = vadsohn(y_line,Fs);

% plot(linspace(0,length(vs)/Fs,length(vs)),vs)

[array_start_time,array_end_time,array_label] = vs2time(vs,Fs);

array_start_time = array_start_time + str2double(lineList(1,:));
array_end_time = array_end_time + str2double(lineList(1,:));

file_name = './lineList_matlab.lab';
Fsavelab(file_name, array_start_time, array_end_time, array_label);
