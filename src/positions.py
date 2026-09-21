"""Locate task-relevant token positions inside a rendered prompt."""


def token_positions(tok, ex):
    """Key token positions for `ex` (identical in the clean and corrupted prompts)."""
    ids = tok(ex.clean_prompt()).input_ids
    idsc = tok(ex.corrupt_prompt()).input_ids
    vid = {v: tok(v).input_ids[0] for _, v in ex.pairs}
    kid = {k: tok(" " + k).input_ids[0] for k, _ in ex.pairs}

    def hits(t):
        return [i for i, x in enumerate(ids) if x == t]

    pos = {"n_tokens": len(ids), "final": len(ids) - 1}
    vpos = {}
    for i, (_, v) in enumerate(ex.pairs):
        h = hits(vid[v])
        assert len(h) == 1, (v, h)
        vpos[i] = h[0]
    pos["value_pos"] = vpos                        # entry index -> token position
    pos["target_value"] = vpos[ex.target_idx]      # carries the clean answer value
    pos["distract_value"] = vpos[ex.distract_idx]  # carries the swapped-in value
    kpos = {i: hits(kid[k])[0] for i, (k, _) in enumerate(ex.pairs)}   # body occurrence
    pos["key_pos"] = kpos
    pos["target_key"] = kpos[ex.target_idx]
    pos["target_delim"] = pos["target_value"] + 1   # ',' or '.' right after the value
    pos["distract_delim"] = pos["distract_value"] + 1
    qh = hits(kid[ex.query_key])
    pos["query_key_body"] = qh[0]
    pos["query_key_question"] = qh[1] if len(qh) > 1 else qh[0]
    pos["query_key_last"] = qh[-1]
    diff = [i for i, (x, y) in enumerate(zip(ids, idsc)) if x != y]
    assert sorted(diff) == sorted([pos["target_value"], pos["distract_value"]]), (diff, pos)
    return pos
