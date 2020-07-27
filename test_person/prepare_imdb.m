txt = fileread('vi-72.json');
json = jsondecode(txt);  %jsondecode is not available on 2015b. I use Matlab2016b.

file_path = {json.file_path};
caption = {json.captions};
id = [json.id];

disp(numel(file_path));
disp(numel(caption));
disp(numel(id));

% make dictionary
caption_dic = [];
for i = 1:numel(caption)
    caption_dic = cat(1,caption_dic,caption{i});
end

save('caption.mat','caption_dic');
