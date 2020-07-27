import os
import json
import codecs

def build_vocab(json_data, count_thr=4,fvocab='cuhk_vi_vocab.json', #4: các từ xuất hiện ít hơn 4 lần sẽ bị bỏ đi
                fcount='cuhk_vi_count.json'):     
    #json_data=json_data[:100]    
    ##### Statistic
    counts = {}
    for item in json_data:
        for txt in item['processed_tokens']: #lấy tất cả các từ trong file caption
            for w in txt:
                counts[w] = counts.get(w, 0) + 1 #đếm số lượng các từ
    cw = sorted([(count,w) for w,count in counts.items()], reverse=True)
    print ('top words and their counts:')
    print ('\n'.join(map(str,cw[:20])))
    
    ###### Show some status
    total_words = sum(counts.values())
    print ('total words:', total_words)
    bad_words = [w for w,n in counts.items() if n <= count_thr]
    vocab = [w for w,n in counts.items() if n > count_thr]
    bad_count = sum(counts[w] for w in bad_words)
    print ('number of bad words: %d/%d = %.2f%%' % 
           (len(bad_words), len(counts), len(bad_words)*100.0/len(counts)))
    print ('number of words in vocab would be %d' % (len(vocab), ))
    print ('number of UNKs: %d/%d = %.2f%%' % 
           (bad_count, total_words, bad_count*100.0/total_words))

    ##### Show distribution
    sent_lengths = {}    
    total_len=0
    for item in json_data:
        for txt in item['processed_tokens']:
            nw = len(txt)
            total_len=total_len+nw
            sent_lengths[nw] = sent_lengths.get(nw, 0) + 1
    max_len = max(sent_lengths.keys())    
    sum_len = sum(sent_lengths.values())
    
    print ('max length sentence in raw data: ', max_len)
    print ('avg length sentence in raw data: ', total_len/sum_len)
    print ('sentence length distribution (count, number of words):')    
    
    for i in range(max_len+1):
        print ('%2d: %10d   %f%%' % 
               (i, sent_lengths.get(i,0), sent_lengths.get(i,0)*100.0/sum_len))
        
    ####### Save vocabulary and some results to file
    if bad_count > 0:
        # additional special UNK token we will use below to map infrequent words to        
        vocab.append('UNK')
    with codecs.open(fcount, 'w', "utf-8") as outfile:
        outfile.write(json.dumps(counts, ensure_ascii=False))     
    with codecs.open(fvocab, 'w', "utf-8") as outfile:
        outfile.write(json.dumps(vocab, ensure_ascii=False))   
    return vocab

#input

def main():
    input_json='./vi2608.json'
    with codecs.open(input_json,'r', "utf-8") as f: #mở file vi2608
        json_data = json.load(f)
    
    vocab=build_vocab(json_data,count_thr=2, #các từ xuất hiện ít hơn 2 lần sẽ bị bỏ đi
            fvocab='./cuhk_vi_vocab.json',                                
            fcount='./cuhk_vi_count.json')

if __name__ == "__main__":
    main()