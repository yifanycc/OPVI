
python bnn_tq_run.py --n_round 1000 --hyperType 'OPVI' --batchType 'sub' --stepsize 1e-5 --batchsize_init 30
python bnn_tq_run.py --n_round 1000 --hyperType 'OPVI' --batchType 'static' --stepsize 1e-5 --batchsize_init 30
python bnn_tq_run.py --n_round 1000 --hyperType 'SVGD' --batchType 'static' --stepsize 1e-5 --batchsize_init 30
python bnn_tq_run.py --n_round 1000 --hyperType 'SVGD' --batchType 'static' --stepsize 1e-5 --batchsize_init 30000
python bnn_tq_run.py --n_round 1000 --hyperType 'LD' --batchType 'static' --stepsize 1e-5 --batchsize_init 30
python bnn_tq_run.py --n_round 1000 --hyperType 'LD' --batchType 'static' --stepsize 1e-5 --batchsize_init 30000

python plot.py --n_round 1000