% SPEECHTOOLS: Save A struct into a lab
%
% USAGE
% Fsavelab(file_name, t_start, t_end, label)
%
% INPUT
% file_name : string : full path
% t_start   : vector of times
% t_end     : vector of times
% label     : vector of label
%
% OUPUT
% write the lab file
%
% REQUIRED
%
% AUTHOR
% beller@ircam.fr


function Fsavelab(file_name, t_start, t_end, label)

% data is organized as Floadlab:
outfile = fopen(file_name,'w');

% writing last semiphone in diphone
for l = 1:numel(t_start),
    fprintf(outfile, '%3.7f ', t_start(l));
    fprintf(outfile, '%3.7f ', t_end(l));
    fprintf(outfile, '%d\n' , label(l));
end
fclose(outfile);
fileattrib(file_name,'+w','g');