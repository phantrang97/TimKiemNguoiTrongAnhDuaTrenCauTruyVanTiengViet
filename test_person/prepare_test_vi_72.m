txt = fileread('vi-722.json');
json = jsondecode(txt);  %jsondecode is not available on 2015b. I use Matlab2016b.

img_id = [json.id];
file_path = {json.file_path};
caption = {json.captions};

txt_id = [];
caption_dic = [];
for i = 1:numel(img_id)
    num = numel(json(i).captions);
    txt_id = cat(1,txt_id,repmat(img_id(i),num,1));
    caption_dic = cat(1,caption_dic,caption{i});
end

test_id.img_id = img_id;
test_id.txt_id = txt_id;
test_id.caption_dic=caption_dic;
test_id.file_path=file_path

disp(test_id);
save('test_id_vi_72.mat','test_id');