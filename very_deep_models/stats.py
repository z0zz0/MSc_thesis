########################################
#### THESE FUNCTIONS HAVE THE PURPOSE
#### OF CALCULATING CERTAIN SCORES.
#######################################

# get_stats()
########################################################################################################################
## This function gives multi-class
## TP FP TN FN
## Used for calculating precision, sensitivity, specificity, FPR, FNR etc.
def get_stats(true, pred, classes):
    stats = {}
    for c in classes:
        stats[c] = {}
        for situation in ["TP","FP","TN","FN"]: ## True positive, False Positive, True Negative, False Negative
            stats[c][situation] = 0
            
    for c in classes:
        for (current_truth, current_prediction) in zip(true,pred):
            
            if c == current_truth:  ## The class is seen as positive positive
            
                if current_truth == current_prediction: ## TP
                    stats[c]["TP"] += 1
                else:                                  ## FN
                    stats[c]["FN"] += 1
            
            else:  ## The class is seen as negative
                if c == current_prediction:            ## FP
                    stats[c]["FP"] += 1
                else:                                  ## TN
                    stats[c]["TN"] += 1
                
    return stats
#-----------------------------------------------------------------------------------------------------------------------


# get_score()
########################################################################################################################
## Returns a certain score, depending on class.
## Must give dictionary containig TP FN FP TN of the different classes
def get_score(stats, c, score):
    score = score.lower()
    ### precision scoring
    if score == "precision":
        if stats[c]["TP"] == 0 :
            tmp = 0
        else:
            tmp = stats[c]["TP"] / (stats[c]["TP"] + stats[c]["FP"])
        return tmp
    
    ### sensitivity scoring
    elif score == "sensitivity":
        if stats[c]["TP"] == 0 :
            tmp = 0
        else:
            tmp = stats[c]["TP"] / (stats[c]["TP"] + stats[c]["FN"])
        return tmp
    
    ### specificity scoring
    elif score == "specificity":
        if stats[c]["TN"] == 0 :
            tmp = 0
        else:
            tmp = stats[c]["TN"] / (stats[c]["TN"] + stats[c]["FP"])
        return tmp
    
    ### fpr scoring
    elif score == "fpr":
        if stats[c]["FP"] == 0 :
            tmp = 0
        else:
            tmp = stats[c]["FP"] / (stats[c]["FP"] + stats[c]["TN"])
        return tmp
    
    ### fnr scoring
    elif score == "fnr":
        if stats[c]["FN"] == 0 :
            tmp = 0
        else:
            tmp = stats[c]["FN"] / (stats[c]["FN"] + stats[c]["TP"])
        return tmp
    
    ### raise no valid scoring option
    else:
        raise Exception(f"{score} can't be found in the function 'get_score'. The only available scores for now are, [Precision, Sensitivity, Specificity, FPR, FNR]") 
#-----------------------------------------------------------------------------------------------------------------------


# calc_intresting_statistics()
########################################################################################################################
## This function returns a dict containing
## precision, sensitivity, specificity, FPR, FNR etc.
## for every class.
def calc_intresting_statistics(true, pred, classes):
    stats = get_stats(true, pred, classes)
    scores = {}
    for c in classes:
        scores[c] = {}
        for score in ["Precision", "Sensitivity", "Specificity", "FPR", "FNR"]: ## True positive, False Positive, True Negative, False Negative
            scores[c][score] = get_score(stats, c, score)
    return scores
#-----------------------------------------------------------------------------------------------------------------------



