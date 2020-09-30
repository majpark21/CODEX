import dash_bootstrap_components as dbc

def make_tooltips():
    dict_delay = {'show': 1200, 'hide': 250}
    return [
        dbc.Tooltip(
            'TSNE learning rate for optimization phase. Default: 600. Default should work for most cases.',
            target='input-lr',
            delay=dict_delay
        ),

        dbc.Tooltip(
            'TSNE perplexity, trade-off between local (low value) and global (high value) structures in the data. Default: 50, try: 25-200.',
            target='input-perp',
            delay=dict_delay
        ),

        dbc.Tooltip(
            'TSNE number of iterations for optimization phase. Try a couple of thousands (e.g. 2500). Default to 250 for speed but usually results in bad projection.',
            target='input-niter',
            delay=dict_delay
        ),

        dbc.Tooltip(
            'Rerun TSNE with the current parameters. Can follow progression at the command prompt.',
            target='submit-tsne',
            delay=dict_delay
        )
    ]
