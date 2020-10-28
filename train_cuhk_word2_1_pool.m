function train_id_net_vgg16(varargin)
% -------------------------------------------------------------------------
% Part 4.1: prepare the data
% -------------------------------------------------------------------------

imdb = load('./dataset/CUHK-ht/url_data.mat'); %load dữ liệu ảnh train, val, test; đường dẫn ảnh; danh mục nhãn cho ảnh tập train; danh mục nhãn cho mô tả tập train
imdb = imdb.imdb;
load('./dataset/CUHK-ht/cuhk_word2.mat'); %load đặc trưng từ
imdb.charcnn = wordcnn;  %load đặc trưng 56d của toàn bộ câu mô tả (56x80440)
%imdb.charmean = mean(imdb.charcnn(:,:,:,imdb.images.set==1),4);
% -------------------------------------------------------------------------
% Part 4.2: initialize a CNN architecture
% -------------------------------------------------------------------------
net = cuhk_word2_pool();
net.conserveMemory = true;
im_mean = imdb.rgbMean;
net.meta.normalization.averageImage = im_mean;
%net.meta.normalization.charmean = imdb.charmean;
% -------------------------------------------------------------------------
% Part 4.3: train and evaluate the CNN

% -------------------------------------------------------------------------
opts.train.averageImage = net.meta.normalization.averageImage;
opts.train.batchSize = 32; %32 ảnh hay câu mô tả trong mỗi lần tính đạo hàm
opts.train.continue = true;
opts.train.gpus = 1;
opts.train.prefetch = false ;
opts.train.nesterovUpdate = true ;
opts.train.expDir = './data/res52_cuhk_batch32_pool_shift';
opts.train.derOutputs = {'objective_img',1,'objective_txt',1} ;
%opts.train.gamma = 0.9;
opts.train.momentum = 0.9; %một tham số để tạo đà vượt qua local minimum
%opts.train.constraint = 100;
opts.train.learningRate = [0.1*ones(1,180)] ; %learning rate =0.1 
opts.train.weightDecay = 0.0001;
opts.train.numEpochs = numel(opts.train.learningRate) ; %học 180 lần
[opts, ~] = vl_argparse(opts.train, varargin) ;
% Call training function in MatConvNet
[net,info] = cnn_train_dag(net, imdb, @getBatch,opts) ;

% --------------------------------------------------------------------
function inputs = getBatch(imdb,batch,opts)
% --------------------------------------------------------------------
%-- img data
im_url = imdb.images.data(batch) ; %lưu đường dẫn ảnh vào biến im_url
im = vl_imreadjpeg(im_url,'Pack','Resize',[224,224],'Flip',...
    'CropLocation','random','CropSize',[0.8,1],...
    'Interpolation', 'bicubic','NumThreads',16,... %'Brightness', double(0.1*imdb.rgbCovariance),...
    'SubtractAverage',imdb.rgbMean,...
    'CropAnisotropy',[3/4,4/3]); %Resize ảnh về 224x224
oim = im{1}; %bsxfun(@minus,im{1},opts.averageImage);
label_img =  imdb.images.label(batch);

%-- txt data
batchsize = numel(batch);
txt_batch = zeros(1,batchsize);
for i=1:batchsize
  txt_batch(i) = rand_same_class(imdb,label_img(i));  % train txt range 1~68126
end
%label_txt =  imdb.images.label2(txt_batch);
label_txt = label_img;
txt = single(imdb.charcnn(:,txt_batch));
txtinput = zeros(1,56,2555,batchsize,'single');
for i=1:batchsize
    len = numel(find(txt(:,i)>0));
    location = randi(57-len);
    for j=1:56
        v = txt(j,i);
        if(v<=0) 
            break;
        end
       txtinput(1,location,v,i)=1;
       location = location+1;
    end
end
txtinput = gpuArray(txtinput);
%}
%--
inputs = {'data',gpuArray(oim),'data2',txtinput,'label_img',label_img,'label_txt',label_txt};
