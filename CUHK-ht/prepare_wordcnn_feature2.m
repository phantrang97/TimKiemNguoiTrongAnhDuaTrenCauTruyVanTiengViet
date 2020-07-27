%produce fix length word input
%keep 56 word in one scentence
load('./CUHK-PEDES_dictionary.mat');
load('./caption.mat');
w_sum = cellfun(@(x) sum(x),subset.names);

wordcnn = zeros(56,80440,'int16');
str_length = zeros(1,80440);

count=1;

m=0;
for i = 1:numel(caption_dic)
    disp(i);
    word_count=0;
    skip=0;
    tline = caption_dic{i};
    split_tline = strsplit(tline,{'-',' ','.',',','(',')'},'CollapseDelimiters',true);
    j=1;
    while( j<=min(numel(split_tline),56+skip)  )% use sum to do hash
        word = split_tline{j};
        j = j+1;
        sub_index = find(w_sum == sum(word));
        %if(isempty(sub_index))  % Some word is caption
         %   word = lower(word);
          %  sub_index = find(w_sum == sum(word));
        %end
        if(isempty(sub_index))
            skip = skip+1;
            %fprintf('%s\n',word);
            continue;
        end
        ind = cellfun(@(x) strcmp(x,word),subset.names(sub_index));
        if(sum(ind(:))==0)
            skip = skip+1;
            %fprintf('%s\n',word);
            continue;
        end
        index = sub_index(ind==1);
        %tmp = w_features(:,index);
        %sentence = sentence+tmp;
        word_count = word_count+1;
        wordcnn(word_count,i) = index;
    end
    str_length(i) = numel(split_tline);
end

fprintf('%f',mean(str_length));
save('cuhk_word2.mat','wordcnn');