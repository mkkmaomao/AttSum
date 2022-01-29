
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

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
    rouge = Rouge()
    # hyps/refs: list[[tokens], [tokens], [tokens]]
    # hyps = [' '.join(tokens) for tokens in hyps]
    # refs = [' '.join(tokens) for tokens in refs]

    # hyps/refs: list[str, str, str]
    try:
        scores = rouge.get_scores(hyps, refs, avg=True)
        # print("scores done!!!")
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

    # rouge_f = 0.33333333*rouge1_f + 0.33333333*rouge2_f + 0.33333333*rougeL_f
    # rouge_p = 0.33333333 * rouge1_p + 0.33333333 * rouge2_p + 0.33333333 * rougeL_p
    # rouge_r = 0.33333333 * rouge1_r + 0.33333333 * rouge2_r + 0.33333333 * rougeL_r

    score=[rouge1_f,rouge2_f,rougeL_f,rouge1_p,rouge2_p,rougeL_p,rouge1_r,rouge2_r,rougeL_r]
    return score
    # rouge1_f, rouge2_f, rougeL_f,rouge1_p,rouge2_p, rougeL_p, rouge1_r, rouge2_r, rougeL_r


if __name__ == '__main__':
    pred = open("./val/roBERTa_sub_more_test_pred_1110_6am.txt", "r",encoding="utf-8")
    target = open("./val/roBERTa_sub_more_golden_1110_6am.txt", "r",encoding="utf-8")

    rouge1_f=0
    rouge2_f=0
    rougeL_f=0
    rouge1_p=0
    rouge2_p=0
    rougeL_p=0
    rouge1_r=0
    rouge2_r=0
    rougeL_r=0
    blue_score_total1=0
    blue_score_total2 = 0
    blue_score_total3 = 0
    blue_score_total4 = 0
    count=0

    for pred_title,target_title in zip(pred.readlines(),target.readlines()) :
        count+=1

        score= rouge_from_maps(target_title, pred_title)

        smoothie = SmoothingFunction().method2
        blue_score_total1+=sentence_bleu([target_title.split()], pred_title.split(),weights=(1, 0, 0, 0),smoothing_function=smoothie)
        blue_score_total2+=sentence_bleu([target_title.split()], pred_title.split(),weights=(0,1, 0, 0),smoothing_function=smoothie)
        blue_score_total3+=sentence_bleu([target_title.split()], pred_title.split(),weights=(0, 0,1, 0),smoothing_function=smoothie)
        blue_score_total4+=sentence_bleu([target_title.split()], pred_title.split(),weights=( 0, 0, 0,1),smoothing_function=smoothie)

        rouge1_f+=score[0]
        rouge2_f += score[1]
        rougeL_f += score[2]
        rouge1_p += score[3]
        rouge2_p += score[4]
        rougeL_p += score[5]
        rouge1_r += score[6]
        rouge2_r += score[7]
        rougeL_r += score[8]

    print("rouge1_f ",rouge1_f/count)
    print("rouge2_f ",rouge2_f/count)
    print("rougeL_f ",rougeL_f/count)
    print("rouge1_p ",rouge1_p / count)
    print("rouge2_p ",rouge2_p / count)
    print("rougeL_p ",rougeL_p / count)
    print("rouge1_r ",rouge1_r / count)
    print("rouge2_r ",rouge2_r / count)
    print("rougeL_r ",rougeL_r / count)

    print(blue_score_total1 / count)
    print(blue_score_total2 / count)
    print(blue_score_total3 / count)
    print(blue_score_total4 / count)

