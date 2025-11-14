def classify_scenarios(masks: list, 
                       target_dim: int, 
                       alpha: float = 3):
    """
    target_dim has to be less than the number of masks
    alpha should be greater than 3.
    """
    s_mask = 0
    for idx, mask in enumerate(masks):
        if idx == target_dim:
            s_mask += mask*alpha
        else:
            s_mask += mask
    QQ = s_mask == 0
    QS = (s_mask == 1)|(s_mask == 2)
    SQ = s_mask == alpha
    SS = s_mask > alpha
    return QQ, QS, SQ, SS