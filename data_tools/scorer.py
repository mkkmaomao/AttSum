
from rouge import Rouge


special_tokens_id = list(range(33, 48))
special_tokens_id += list(range(58, 65))
special_tokens_id += list(range(91, 97))
special_tokens_id += list(range(123, 127))
special_tokens = [chr(i) for i in special_tokens_id]

def convention_tokenize(text):
    '''
    Word segmentation for special symbols 
    '''
    for st in special_tokens:
        text = f' {st} '.join(text.split(st)).strip()
    tokens = text.split()
    return tokens

def rouge_from_maps(refs, hyps):
    '''
    Calculate the sum of the f-values of the three rouge scores in a weighted manner
    The purpose is to compare with the training results of CodeBERT-text 
    return avg_score
    '''
    rouge = Rouge()
    # hyps/refs: list[[tokens], [tokens], [tokens]]
    hyps = [' '.join(tokens) for tokens in hyps]
    refs = [' '.join(tokens) for tokens in refs]
    print("hyps: ",hyps)
    print("refs: ",refs)
    # hyps/refs: list[str, str, str]
    try:
        scores = rouge.get_scores(hyps, refs, avg=True)
        print("scores done!!!")
    except Exception as e:
        return 0
    rouge1_f = scores['rouge-1']['f']
    rouge2_f = scores['rouge-2']['f']
    rougeL_f = scores['rouge-l']['f']

    rouge1_p = scores['rouge-1']['p']
    rouge2_p = scores['rouge-2']['p']
    rougeL_p = scores['rouge-l']['p']

    rouge1_r = scores['rouge-1']['r']
    rouge2_r = scores['rouge-2']['r']
    rougeL_r= scores['rouge-l']['r']
    print('rouge1_f:', rouge1_f)
    print('rouge2_f:', rouge2_f)
    print('rougeL_f:', rougeL_f)
    print('rouge1_p:', rouge1_p)
    print('rouge2_p:', rouge2_p)
    print('rougeL_p:',rougeL_p)
    print('rouge1_r:', rouge1_r)
    print('rouge2_r:', rouge2_r)
    print('rougeL_r:', rougeL_r)

    avg_f = 0.33333333*rouge1_f + 0.33333333*rouge2_f + 0.33333333*rougeL_f
    return avg_f
    # rouge1_f, rouge2_f, rougeL_f,rouge1_p,rouge2_p, rougeL_p, rouge1_r, rouge2_r, rougeL_r