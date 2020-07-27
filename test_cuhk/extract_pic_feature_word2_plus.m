clear;
netStruct = load('../data/res52_cuhk_batch32_Rankloss_2:1:0.5_margin1/net-epoch-1.mat');
%netStruct = load('../data/res52_cuhk_batch32_Rankloss_2:1:0.5_margin1/net-epoch-30.mat');
%netStruct = load('../data/res52_cuhk_batch32_Rankloss_2:1:0.5_margin1/net-epoch-61.mat');
net = dagnn.DagNN.loadobj(netStruct.net);
clear netStruct;
net.mode = 'test' ;
net.move('gpu') ;
net.removeLayer('RankLoss');
net.conserveMemory = true;
im_mean = reshape(net.meta.normalization.averageImage,1,1,3);

load('./url_data.mat'); %dữ liệu url_data để xếp rank
p = imdb.images.data(imdb.images.set==3); %Lấy tập dữ liệu test
ff = [];
%%------------------------------

for i = 1:numel(p) %numel là mảng
    disp(i);
    str = p{i};
    im = imread(str);
    oim = im;
    f = getFeature2(net,oim,im_mean,'data','fc1_1bn');
    f = sum(sum(f,1),2);
    f2 = getFeature2(net,fliplr(oim),im_mean,'data','fc1_1bn');
    f2 = sum(sum(f2,1),2);
    f = f+f2;
    size4 = size(f,4);
    f = reshape(f,[],size4)';
    f = norm_zzd(f);
    ff = cat(1,ff,f);
end
save('./resnet_cuhk_img.mat','ff','-v7.3');
%}

ff = [];
load('./cuhk_word2.mat'); %thư viện từ để chuyển sang vector, sử dụng bộ từ điển đã train
test_word = wordcnn(:,end-6155:end);

for i = 1:6156
    disp(i);
    content = test_word(:,i);
    %disp('123');
    len = numel(find(content>0));
    %disp('456');
    txtinput = zeros(len,2555,'single');
    %disp('789');
    for k=1:len 
        txtinput(k,content(k))=1;
    end
    %disp('246');
    %transfer it to different location
    win = 57-len;
    input = zeros(56,2555,win,'single');
    for kk = 1:win
        input(kk:kk+len-1,:,kk) = txtinput;
    end
    input = reshape(input,1,56,2555,[]);
    f = getFeature2(net,input,[],'data2','fc5_2bn');
    f = sum(f,4);
    size4 = size(f,4);
    f = reshape(f,[],size4)';
    f = norm_zzd(f);
    ff = cat(1,ff,f);
end
save('./resnet_cuhk_txt.mat','ff','-v7.3');

evaluate;
