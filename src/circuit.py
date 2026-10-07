"""The Bind-Query circuit per model (Parts XVI-XVIII): which linear layers play which role."""

CIRCUIT = {
    "0.8B":  dict(bind=[0, 1, 2], query=[12, 13, 14], reader=15),
    "9B":    dict(bind=[0, 1, 2], query=[16, 17, 18], reader=19),
    "G1b":   dict(bind=[0, 1, 2, 3, 4], query=list(range(16, 25)), reader=25),
    "Gtiny": dict(bind=[0, 1, 2, 3, 4], query=list(range(16, 25)), reader=25),
}
ROLES = ("bind", "feeder", "query", "post-reader")


def role(tag, layer):
    """'bind' | 'feeder' | 'query' | 'post-reader' for a linear layer of model `tag`."""
    c = CIRCUIT[tag]
    if layer in c["bind"]:
        return "bind"
    if layer in c["query"]:
        return "query"
    return "feeder" if layer < c["reader"] else "post-reader"
