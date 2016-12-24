function [array_start_time,array_end_time,array_label] = vs2time(vs,sr)
    % convert the output from voicebox VAD array to [[start_time, end_time,0],[start_time,end_time,1],...]
    % 0: silence, 1: non_silence
    
    array_start_time = 1/sr;
    array_end_time = [];
    array_label = vs(1);
    label_old = vs(1);
    for ii = 2:length(vs)-1
        if vs(ii) ~= label_old
            array_end_time = [array_end_time,ii/sr];
            array_start_time = [array_start_time,ii/sr];
            array_label = [array_label,vs(ii)];
            label_old = vs(ii);
        end
    end
    array_end_time = [array_end_time,length(vs)/sr];
    
end